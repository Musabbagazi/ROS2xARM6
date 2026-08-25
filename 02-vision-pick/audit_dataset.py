#!/usr/bin/env python3
"""Find frames where a CUBE IS VISIBLE BUT NOT LABELED.  No arm, no camera.

That one defect matters more than any other in this dataset: a frame
saved with an unlabeled cube teaches the model that cubes are
background - the exact opposite of what the capture is for. It is also
invisible in the numbers (the frame looks like a valid background
example) and only shows up as detections that quietly get worse.

Method: run the trained model over every saved image and compare what
it finds against the label file. A confident detection with no label
under it is a suspect. This is NOT circular - we are not asking the
model to invent labels, only to point at places where the depth
labeler and the model flatly disagree, which is where a human should
look.

Detections touching the frame edge are ignored: capture_dataset drops
cut-off cubes on purpose, so those disagreements are correct.

  python audit_dataset.py           report, and copy suspects for review
  python audit_dataset.py --fix     also delete the suspect frames
                                    (image + label + preview, together)
"""
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
SUSPECT = os.path.join(DATASET, "suspect")

CONF = 0.50          # only confident detections accuse a frame
IOU_MATCH = 0.30     # a detection this close to a label is accounted for
BORDER_PX = 6        # detections touching the edge are legitimately unlabeled
FIX = "--fix" in sys.argv


def load_labels(path, w, h):
    """YOLO label file -> pixel boxes."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            xc, yc, bw, bh = (float(v) for v in p[1:])
            out.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                        (xc + bw / 2) * w, (yc + bh / 2) * h])
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def main():
    from detect_cube import MODEL_FILE
    if not os.path.exists(MODEL_FILE):
        print("No cube_model.pt - train once before auditing.")
        return
    from detect_cube import _get_model
    model = _get_model()

    if os.path.isdir(SUSPECT):
        shutil.rmtree(SUSPECT)
    os.makedirs(SUSPECT)

    total = 0
    suspects = []
    for split in ("train", "val"):
        img_dir = os.path.join(DATASET, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.endswith(".jpg"):
                continue
            total += 1
            stem = name[:-4]
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                continue
            h, w = img.shape[:2]
            labels = load_labels(
                os.path.join(DATASET, "labels", split, stem + ".txt"), w, h)
            res = model.predict(img, conf=CONF, verbose=False)[0]
            missed = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if (x1 <= BORDER_PX or y1 <= BORDER_PX
                        or x2 >= w - BORDER_PX or y2 >= h - BORDER_PX):
                    continue                 # cut off - dropped on purpose
                if max([iou([x1, y1, x2, y2], L) for L in labels] or [0.0]) \
                        < IOU_MATCH:
                    missed.append((x1, y1, x2, y2, float(b.conf[0])))
            if missed:
                suspects.append((split, stem, len(labels), missed))
                ann = img.copy()
                for L in labels:
                    cv2.rectangle(ann, (int(L[0]), int(L[1])),
                                  (int(L[2]), int(L[3])), (0, 255, 0), 2)
                for (x1, y1, x2, y2, c) in missed:
                    cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 2)
                    cv2.putText(ann, "UNLABELED %.2f" % c,
                                (int(x1), max(14, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(ann, "%s  green=label  red=cube with NO label"
                            % stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                cv2.imwrite(os.path.join(SUSPECT, stem + ".png"), ann)

    print("\n===== dataset audit =====")
    print("%d images checked, %d suspect" % (total, len(suspects)))
    for split, stem, n_lab, missed in suspects:
        print("  %-5s %s: %d labelled, %d cube(s) NOT labelled (conf %s)"
              % (split, stem, n_lab, len(missed),
                 ", ".join("%.2f" % m[4] for m in missed)))
    if not suspects:
        print("Nothing to clean - every visible cube is labelled.")
        return
    print("\nReview them: %s" % SUSPECT)
    print("  green = the label that was saved")
    print("  red   = a cube the model sees with NO label under it")

    if not FIX:
        print("\nRe-run with --fix to delete these frames "
              "(image + label + preview).")
        return
    print("\nThis DELETES %d frame(s) from the dataset." % len(suspects))
    if input("Type 'delete' to confirm > ").strip().lower() != "delete":
        print("Nothing deleted.")
        return
    gone = 0
    for split, stem, _n, _m in suspects:
        for p in (os.path.join(DATASET, "images", split, stem + ".jpg"),
                  os.path.join(DATASET, "labels", split, stem + ".txt"),
                  os.path.join(DATASET, "preview", stem + ".png")):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print("  could not remove %s: %s" % (p, e))
        gone += 1
    print("Removed %d frame(s). Retrain with '2 - Train Cube Model'." % gone)


if __name__ == "__main__":
    main()

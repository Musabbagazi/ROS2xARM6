#!/usr/bin/env python3
"""Collect a YOLO training dataset of YOUR cubes - auto-labeled by depth.

No manual labeling: with the camera looking straight down, anything
that sticks up out of the floor plane and is cube-shaped gets its box
drawn automatically from the depth image. YOLO then learns to find the
same cubes in the COLOR image - any color, no HSV filter.

Flow: the arm visits a few heights above the floor (with small XY
shifts so no two frames are identical) and snaps frames. Between
rounds you rearrange / swap the cubes. USE EVERY CUBE COLOR YOU OWN -
the network can only learn colors it has seen. Keep the cubes
SEPARATED (a finger apart): touching or stacked cubes make ambiguous
depth blobs, and those frames are thrown away by the safety filters.

Usage: python capture_dataset.py [robot_ip] [rounds]
Keep the e-stop in hand. Keep your hands OUT of view while frames are
being taken (the script tells you when).
"""
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

import vision_common as vc
import vision3 as v3
from detect_cube import (depth_mm_image, floor_plane, floor_depth_map,
                         floor_depth_at, TOP_BAND_MM, MODEL_FILE, CONF_MIN)

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")

# camera heights (flange Z, mm). Lowest is 400: below that, the top of
# a tall (~78mm) cube gets closer than the D435's ~280mm minimum depth
# range - it would show up UNLABELED and poison the training data. The
# high ones (660/740) match vision_pick3.RISE_HEIGHTS so the model is
# trained to detect from the wide-search vantages too; unreachable
# heights are dropped automatically at startup.
HEIGHTS = [740.0, 660.0, 580.0, 480.0, 400.0]
JITTER = [(0.0, 0.0), (25.0, 10.0), (-20.0, 25.0), (10.0, -25.0)]
# whole ROUNDS go to the val split (never single frames: the 12 frames
# of one round are near-duplicates, and splitting them across train and
# val would make the val score measure memorization, not generalization)
VAL_ROUND_EVERY = 4

POP_MM = 10.0            # must stick out of the floor at least this
MIN_AREA_PX = 250
# stricter than the runtime detector on purpose: two identical touching
# cubes form a 2:1 blob (aspect 0.50); if that were LABELED as one cube,
# YOLO would learn to box touching pairs as a single object and the
# gripper would aim at the seam between them. 0.70 rejects the pair,
# which also skips the whole frame (rejected-flag), keeping training
# data clean. Real straight-down cube tops are near 1.0.
SQUARENESS = 0.70
BORDER_PX = 4            # boxes touching the image edge are dropped
MIN_W_MM, MAX_W_MM = 10.0, 85.0
MIN_H_MM, MAX_H_MM = POP_MM + 2.0, 110.0   # below POP_MM+2 the pop-out
                                           # mask cannot even see it
MAX_LEN_RATIO = 1.30     # real cube tops are near-square in mm; more
                         # elongated = a merged pair of touching cubes,
                         # which must never be labeled as one cube
# A cube-sized blob is only treated as a MERGE (which skips the whole
# frame) on a STRONG signal - these are looser than the clean-cube
# thresholds so that mere depth noise on a single cube's top face still
# labels normally instead of throwing the frame away. Two same-size
# touching cubes give aspect ~0.5; a staggered pair, ratio >~1.5 or
# fill <~0.65 (L-shape). A slightly noisy single cube stays well clear.
MERGE_ASP = 0.60
MERGE_FILL = 0.65
MERGE_RATIO = 1.45
# a pop-out blob at least this big (px) is cube-scale, not a speck: if it
# cannot be cleanly labeled as one cube it SKIPS the frame rather than
# being silently ignored (which would train a visible cube as
# background). Well below a ~38mm cube (~2300px), above depth noise.
SUBSTANTIAL_PX = 600


def label_frame(dmm, intr, expect_floor=None, floor_coef=None):
    """Auto-label one frame from depth.

    Returns (yolo_lines, boxes, rejected) - empty lists mean a valid
    background frame; rejected=True means SOMETHING pops out of the
    floor but could not be labeled (touches the border, merged blob,
    odd size...). Such frames must NOT be saved: the visible object
    would be trained as background and poison the model. Returns
    (None, None, False) if the frame is unusable (no floor estimate, or
    a floor that does not read where it should).

    expect_floor: the depth the floor ought to have at this pose. Every
    label here is derived from "how far does this stick out of the
    floor", so a floor fitted to the WRONG surface silently mislabels
    the whole frame. The dangerous version is finding no pop-out at all
    (the plane locks onto the cube tops, or the surface is transparent
    and the infrared measures something far below): no blobs means
    nothing for the rejected-flag to catch, and a frame full of cubes
    gets saved as a clean BACKGROUND image - which actively teaches the
    model that cubes are not cubes."""
    # a taught floor (transparent surface) replaces the per-frame fit -
    # and needs no plausibility check, since there is no independent
    # measurement to compare it against
    coef = np.asarray(floor_coef, dtype=float) if floor_coef is not None \
        else floor_plane(dmm)            # tilt-tolerant floor plane
    if coef is None:
        return None, None, False
    if floor_coef is None and expect_floor is not None:
        centre = floor_depth_at(coef, dmm.shape[1] / 2.0, dmm.shape[0] / 2.0)
        if abs(centre - expect_floor) > v3.FLOOR_TOL_MM:
            return None, None, False     # not the surface we think it is
    fmap = floor_depth_map(coef, dmm.shape)
    m = ((dmm > 100.0) & (dmm < fmap - POP_MM)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = dmm.shape
    lines, boxes = [], []
    rejected = False
    for c in contours:
        ca = cv2.contourArea(c)
        if ca < MIN_AREA_PX:
            continue                     # a speck, ignore
        cxs, cys = c[:, 0, 0], c[:, 0, 1]
        if (cxs.min() <= 1 or cys.min() <= 1 or
                cxs.max() >= iw - 2 or cys.max() >= ih - 2):
            continue                     # cut off at the frame edge (a
            #                              partial cube / intruding object):
            #                              ignore it, do not drop the frame
        # a substantial (cube-scale) blob that fails to become one clean
        # cube must SKIP the frame, not be silently ignored - else a real
        # but mis-measured/merged cube gets saved as background (poison).
        # A small blob that fails is just noise -> ignore.
        substantial = ca >= SUBSTANTIAL_PX
        blob = np.zeros_like(m)
        cv2.drawContours(blob, [c], 0, 255, -1)
        vals = dmm[(blob > 0) & (dmm > 100.0)]
        if vals.size < 30:
            rejected = rejected or substantial
            continue
        top = float(np.percentile(vals, 25))
        mo = cv2.moments(c)
        bx = mo["m10"] / mo["m00"] if mo["m00"] else float(cxs.mean())
        by = mo["m01"] / mo["m00"] if mo["m00"] else float(cys.mean())
        height = floor_depth_at(coef, bx, by) - top   # local (tilt-correct)

        # measure the TOP FACE only (side faces radially elongate an
        # off-centre cube and would fail squareness on the full blob)
        band = ((blob > 0) & (dmm > top - 8.0) &
                (dmm < top + TOP_BAND_MM)).astype(np.uint8) * 255
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((7, 7),
                                                               np.uint8))
        bcont, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not bcont or cv2.contourArea(max(bcont, key=cv2.contourArea)) \
                < MIN_AREA_PX:
            rejected = rejected or substantial
            continue
        bc = max(bcont, key=cv2.contourArea)
        rect = cv2.minAreaRect(bc)
        (cx, cy), (rw, rl), _ang = rect
        pts = cv2.boxPoints(rect)
        p3 = [rs.rs2_deproject_pixel_to_point(
            intr, [float(px), float(py)], top / 1000.0) for px, py in pts]
        s1 = np.linalg.norm(np.subtract(p3[0], p3[1])) * 1000.0
        s2 = np.linalg.norm(np.subtract(p3[1], p3[2])) * 1000.0
        wmin, wmax = min(s1, s2), max(s1, s2)
        asp = min(rw, rl) / max(rw, rl) if max(rw, rl) else 0.0
        fill = cv2.contourArea(bc) / (rw * rl) if rw * rl else 0.0

        cube_sized = (MIN_W_MM <= wmin <= MAX_W_MM and
                      MIN_H_MM <= height <= MAX_H_MM)
        cube_shaped = (asp >= MERGE_ASP and fill >= MERGE_FILL and
                       wmax <= MERGE_RATIO * wmin)
        if not (cube_sized and cube_shaped):
            # cube-scale but not a clean single cube (merge / mis-measure)
            # -> skip the frame; small artifact -> ignore
            rejected = rejected or substantial
            continue

        # clean cube -> label it, clipping to the frame so an edge cube
        # still yields valid normalised coords instead of dropping it
        x1 = max(0.0, float(pts[:, 0].min()))
        y1 = max(0.0, float(pts[:, 1].min()))
        x2 = min(float(iw), float(pts[:, 0].max()))
        y2 = min(float(ih), float(pts[:, 1].max()))
        if x2 - x1 < 3.0 or y2 - y1 < 3.0:
            continue
        lines.append("0 %.6f %.6f %.6f %.6f"
                     % ((x1 + x2) / 2.0 / iw, (y1 + y2) / 2.0 / ih,
                        (x2 - x1) / iw, (y2 - y1) / ih))
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return lines, boxes, rejected


def yolo_sees_cubes(img):
    """How many cubes the trained model finds in the COLOR image.

    Depth-independent, which is the point: it vetoes the one failure the
    depth guards cannot see. If the floor plane quietly fits to the cube
    TOPS, nothing pops out of it, there are no blobs for the
    rejected-flag to judge, and a frame with cubes plainly in it gets
    saved as an empty BACKGROUND image - teaching the model that cubes
    are not cubes. The floor-depth check misses this when the cubes are
    shorter than the tolerance (a 38mm cube shifts the plane only 38mm).

    Returns 0 when there is no model yet, so a first-ever capture runs
    exactly as it always did."""
    if not os.path.exists(MODEL_FILE):
        return 0
    try:
        from detect_cube import _get_model
        res = _get_model().predict(img, conf=CONF_MIN, verbose=False)[0]
        return len(res.boxes)
    except Exception:
        return 0                # never let the veto break a capture run


def next_index():
    n = 0
    for split in ("train", "val"):
        d = os.path.join(DATASET, "images", split)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("img_") and f.endswith(".jpg"):
                    try:
                        n = max(n, int(f[4:-4]) + 1)
                    except ValueError:
                        pass
    return n


def save_sample(idx, img, lines, boxes, split):
    name = "img_%05d" % idx
    cv2.imwrite(os.path.join(DATASET, "images", split, name + ".jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(DATASET, "labels", split, name + ".txt"),
              "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    ann = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(ann, "%s  %d boxes" % (name, len(boxes)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(DATASET, "preview", name + ".png"), ann)
    try:
        cv2.imshow("auto-label preview (green = training box)", ann)
        cv2.waitKey(1)
    except Exception:
        pass
    return split


def write_yaml():
    with open(os.path.join(DATASET, "data.yaml"), "w") as f:
        f.write("path: %s\n" % DATASET.replace("\\", "/"))
        f.write("train: images/train\nval: images/val\n")
        f.write("names:\n  0: cube\n")


def main():
    if ROUNDS < 2:
        print("Need at least 2 rounds: one whole round is held out for "
              "validation,\nand a lone round would leave the TRAINING "
              "split empty.\nRerun: python capture_dataset.py [robot_ip] "
              "[rounds >= 2]")
        return
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET, "labels", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET, "preview"), exist_ok=True)
    idx = next_index()
    if idx:
        print("Existing dataset found (%d images) - new frames will be "
              "APPENDED.\nDelete the vision\\dataset folder first for a "
              "fresh start." % idx)
    # The floor guard needs a calibrated reference floor. On a first-ever
    # setup there is none yet (capture comes before calibration), so it
    # stays off and capture behaves exactly as it always did.
    floor_ref = v3.load_floor_ref()
    if floor_ref:
        print("(using the floor taught by teach_floor - %.0fmm at the "
              "centre; the floor is not measured live)"
              % floor_ref.get("centre_depth_mm", 0))
    ref = v3.load_grip_ref()
    if floor_ref:
        pass                    # taught floor needs no plausibility check
    elif ref is None:
        print("(no grip_ref.json yet - floor-plausibility guard is OFF; "
              "check dataset\\preview carefully)")
    else:
        print("(floor guard ON: a frame whose floor is more than %.0fmm off "
              "the calibrated %.0fmm is skipped, not saved)"
              % (v3.FLOOR_TOL_MM, ref["floor0"]))

    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
    except RuntimeError as e:
        print("ABORT:", e)
        arm.disconnect()
        return
    scan_angles = vc.ik(arm, vc.SCAN_POSE)
    if scan_angles is None:
        print("ABORT: SCAN pose not reachable")
        arm.disconnect()
        return

    # keep only capture heights the arm can actually reach over the scan
    # spot (the tall ones may be outside the xArm6 envelope)
    heights = [h for h in HEIGHTS
               if vc.ik(arm, [vc.SCAN_POSE[0], vc.SCAN_POSE[1], h,
                              180.0, 0.0, 0.0]) is not None]
    if not heights:
        print("ABORT: no capture height reachable")
        arm.disconnect()
        return
    dropped = [h for h in HEIGHTS if h not in heights]
    if dropped:
        print("(heights out of reach, skipped: %s)"
              % ", ".join("%.0f" % h for h in dropped))

    print("\nPlan: %d rounds x %d heights x %d shifts = %d frames."
          % (ROUNDS, len(heights), len(JITTER),
             ROUNDS * len(heights) * len(JITTER)))
    print("Scatter 2-5 cubes on the floor. Between rounds you will "
          "rearrange/swap them.\nUse EVERY cube color you own.")
    if input("Start? [y/N] ").strip().lower() != "y":
        print("Aborted."); arm.disconnect(); return

    total_boxes = 0
    kept = 0
    try:
        vc.movej(arm, "SCAN pose", scan_angles)
        with v3.Camera() as cam:
            for r in range(1, ROUNDS + 1):
                # whole rounds go to val so no cube arrangement can leak
                # into both splits (frames within a round are near-twins)
                val_round = (r % VAL_ROUND_EVERY == 0) or \
                    (ROUNDS < VAL_ROUND_EVERY and r == ROUNDS)
                split = "val" if val_round else "train"
                input("\n--- Round %d/%d [%s]: rearrange the cubes - keep "
                      "them SEPARATED\n    (a finger apart, never touching "
                      "or stacked), take your hands away,\n    press ENTER "
                      "--- " % (r, ROUNDS, split))
                for hz in heights:
                    for jx, jy in JITTER:
                        pose = [vc.SCAN_POSE[0] + jx, vc.SCAN_POSE[1] + jy,
                                hz, 180.0, 0.0, 0.0]
                        try:
                            vc.moveto(arm, "capture z=%.0f (%+.0f,%+.0f)"
                                      % (hz, jx, jy), pose)
                        except RuntimeError as e:
                            print("  skip pose: %s" % e)
                            continue
                        cam.skip(6)
                        img, depth, intr = cam.frame()
                        expect = None if ref is None else \
                            ref["floor0"] + (hz - vc.SCAN_POSE[2])
                        lines, boxes, rejected = label_frame(
                            depth_mm_image(depth), intr, expect,
                            v3.virtual_floor_coef(floor_ref, hz))
                        if lines is None:
                            print("  frame unusable (floor not found, or not "
                                  "where it should be) - skipped")
                            continue
                        if rejected:
                            print("  frame skipped: an object in view "
                                  "could not be labeled (edge / touching "
                                  "cubes / odd size)")
                            continue
                        if not boxes:
                            n_yolo = yolo_sees_cubes(img)
                            if n_yolo:
                                print("  frame skipped: the model sees %d "
                                      "cube(s) here but depth labeled none "
                                      "- saving it would train cubes as "
                                      "background" % n_yolo)
                                continue
                        save_sample(idx, img, lines, boxes, split)
                        print("  img_%05d [%s]: %d cube(s)"
                              % (idx, split, len(boxes)))
                        idx += 1
                        kept += 1
                        total_boxes += len(boxes)
        vc.movej(arm, "SCAN pose", scan_angles)
    except (RuntimeError, KeyboardInterrupt) as e:
        print("\nSTOPPED:", e)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        arm.disconnect()

    write_yaml()
    print("\n===== dataset ready =====")
    print("%d frames kept, %d cube boxes total -> %s"
          % (kept, total_boxes, DATASET))
    print("Check vision\\dataset\\preview - green boxes must sit ON the "
          "cubes.\nNext: run train_cubes.bat")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the cube YOLO model on the auto-captured dataset.

Heavy color/rotation augmentation is enabled on purpose: the boxes were
labeled from DEPTH, so hue shifting teaches the network that cube COLOR
does not matter.

ADDS TO the existing model by default: training starts from the current
cube_model.pt (not from generic yolov8n) and runs over the WHOLE
dataset folder, which capture_dataset.py appends to rather than
replaces. Both halves matter -

  * warm start alone, on new images only, would make the model FORGET
    the old scenes (it only ever sees the new ones);
  * the whole dataset alone would relearn from scratch every time.

Together the model keeps what it knows and gains the new surface.

Output: vision/cube_model.pt (what detect_cube.py loads). The previous
model is backed up first - an additive run can always be undone.

Usage: python train_cubes.py [epochs] [fresh]
       fresh = ignore the existing model and start from yolov8n
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
MODEL_OUT = os.path.join(HERE, "cube_model.pt")

args = [a.lower() for a in sys.argv[1:]]
FRESH = "fresh" in args
EPOCHS = next((int(a) for a in args if a.isdigit()), 80)


def main():
    yaml = os.path.join(DATASET, "data.yaml")
    train_dir = os.path.join(DATASET, "images", "train")
    val_dir = os.path.join(DATASET, "images", "val")
    if not os.path.exists(yaml) or not os.path.isdir(train_dir):
        print("No dataset found - run capture_dataset.bat first.")
        return
    if not os.listdir(train_dir):
        print("dataset/images/train is EMPTY - run capture_dataset.bat "
              "with at least 2 rounds\n(one whole round is held out for "
              "validation).")
        return
    if not os.path.isdir(val_dir) or not os.listdir(val_dir):
        print("dataset/images/val is EMPTY - run capture_dataset.bat for "
              "at least one more round\n(whole rounds are held out for "
              "validation).")
        return

    import torch
    from ultralytics import YOLO

    if torch.cuda.is_available():
        device = 0
        print("Training on GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNING: no CUDA - training on CPU will be slow.")

    n_train = len(os.listdir(train_dir))
    n_val = len(os.listdir(val_dir))
    if FRESH or not os.path.exists(MODEL_OUT):
        base = "yolov8n.pt"
        print("Training FROM SCRATCH (yolov8n) on %d train / %d val images."
              % (n_train, n_val))
        if FRESH:
            print("('fresh' was requested - the existing model is ignored, "
                  "not extended.)")
    else:
        base = MODEL_OUT
        print("EXTENDING the existing cube_model.pt with %d train / %d val "
              "images." % (n_train, n_val))
        print("(the dataset folder holds the old images AND the new ones, "
              "so nothing is forgotten)")
        # keep the working model: if the extended one turns out worse,
        # this is the only way back
        backup = os.path.join(HERE, "cube_model_prev.pt")
        shutil.copyfile(MODEL_OUT, backup)
        print("previous model backed up -> %s" % backup)

    model = YOLO(base)
    model.train(
        data=yaml,
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device=device,
        project=os.path.join(HERE, "runs"),
        name="cubes",
        exist_ok=True,
        patience=25,
        # color-agnostic + rotation augmentation
        hsv_h=0.45, hsv_s=0.6, hsv_v=0.4,
        degrees=180.0, translate=0.1, scale=0.4,
        flipud=0.5, fliplr=0.5,
        verbose=True,
    )

    best = os.path.join(HERE, "runs", "cubes", "weights", "best.pt")
    if not os.path.exists(best):
        print("Training produced no best.pt - check the output above.")
        return
    shutil.copyfile(best, MODEL_OUT)

    # the val split holds whole rounds from EVERY capture session, old
    # surface included - so this score also tells you whether extending
    # the model cost anything on the ground it already worked on
    metrics = model.val(data=yaml, device=device, verbose=False)
    print("\n===== cube_model.pt saved =====")
    print("mAP50: %.3f | mAP50-95: %.3f"
          % (metrics.box.map50, metrics.box.map))
    if metrics.box.map50 < 0.85:
        print("mAP50 is LOW - capture more rounds (more colors, more "
              "positions) and retrain.")
    if base == MODEL_OUT:
        print("\nThis score covers BOTH the old and the new surface (whole "
              "rounds from every\nsession are in the val split). If it "
              "dropped against your last run, restore\nthe old model: copy "
              "cube_model_prev.pt over cube_model.pt.")
    print("Next: check detection on BOTH floors with '0 - Live Camera "
          "View', then run a pick.")


if __name__ == "__main__":
    main()

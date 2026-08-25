#!/usr/bin/env python3
"""Camera test: find a red cube with the wrist-mounted RealSense D435.

Takes a snapshot, detects the biggest red square-ish object, measures its
real-world width and position using the depth stream, and saves an
annotated image you can look at to judge the detection.

No arm motion. Camera only.

Usage:  python camera_test.py
Output: vision/captures/annotated.png  (+ raw color/depth)
        prints a JSON summary line at the end
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
CAPT = os.path.join(HERE, "captures")
os.makedirs(CAPT, exist_ok=True)

# ---- red color range in HSV (red wraps around 0, so two bands) ----
LOW1,  HIGH1 = (0,   90, 60), (10, 255, 255)
LOW2,  HIGH2 = (170, 90, 60), (180, 255, 255)

MIN_AREA_PX = 400          # ignore specks smaller than this
SQUARENESS  = 0.55         # w/h ratio must be at least this (cube-ish)


def detect_red_cube(color_img, depth_frame, intrinsics):
    """Return dict about the largest red cube-ish blob, or None."""
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(LOW1), np.array(HIGH1)) | \
           cv2.inRange(hsv, np.array(LOW2), np.array(HIGH2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        rect = cv2.minAreaRect(c)          # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), angle = rect
        if min(w, h) / max(w, h) < SQUARENESS:
            continue                        # not square-ish
        if best is None or w * h > best[1][0] * best[1][1]:
            best = rect
    if best is None:
        return None, mask

    (cx, cy), (w_px, h_px), angle = best
    # depth at the center (median of a small window for robustness)
    xs = np.clip(np.arange(int(cx) - 4, int(cx) + 5), 0, 639)
    ys = np.clip(np.arange(int(cy) - 4, int(cy) + 5), 0, 479)
    ds = [depth_frame.get_distance(int(x), int(y)) for x in xs for y in ys]
    ds = [d for d in ds if d > 0]
    if not ds:
        return None, mask
    depth_m = float(np.median(ds))

    # convert the rect's box corners to 3D to measure real width
    box = cv2.boxPoints(best)
    pts3d = [rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)],
                                             depth_m) for px, py in box]
    side1 = np.linalg.norm(np.subtract(pts3d[0], pts3d[1])) * 1000.0
    side2 = np.linalg.norm(np.subtract(pts3d[1], pts3d[2])) * 1000.0
    width_mm = float(min(side1, side2))
    length_mm = float(max(side1, side2))

    center3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    # normalize minAreaRect angle to the angle of the SHORT side vs x-axis
    grip_angle = angle if w_px >= h_px else angle + 90.0
    while grip_angle > 90.0:
        grip_angle -= 180.0
    while grip_angle < -90.0:
        grip_angle += 180.0

    return {
        "pixel": [round(cx, 1), round(cy, 1)],
        "cam_xyz_mm": [round(v * 1000.0, 1) for v in center3d],
        "depth_mm": round(depth_m * 1000.0, 1),
        "width_mm": round(width_mm, 1),
        "length_mm": round(length_mm, 1),
        "width_px": round(min(w_px, h_px), 1),
        "length_px": round(max(w_px, h_px), 1),
        "angle_deg": round(grip_angle, 1),
        "box": box.tolist(),
    }, mask


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        # let auto-exposure settle
        for _ in range(30):
            pipe.wait_for_frames()
        frames = align.process(pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        img = np.asanyarray(color.get_data())

        found, mask = detect_red_cube(img, depth, intr)

        ann = img.copy()
        if found:
            box = np.intp(found["box"])
            cv2.drawContours(ann, [box], 0, (0, 255, 0), 2)
            cx, cy = found["pixel"]
            cv2.circle(ann, (int(cx), int(cy)), 4, (255, 0, 0), -1)
            label = "%.0fmm wide  %.0fmm away  %.0fdeg" % (
                found["width_mm"], found["depth_mm"], found["angle_deg"])
            cv2.putText(ann, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(ann, "NO RED CUBE FOUND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(CAPT, "color.png"), img)
        cv2.imwrite(os.path.join(CAPT, "mask.png"), mask)
        cv2.imwrite(os.path.join(CAPT, "annotated.png"), ann)

        summary = {"found": bool(found)}
        if found:
            f = dict(found)
            f.pop("box", None)
            summary.update(f)
        print("RESULT " + json.dumps(summary))
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()

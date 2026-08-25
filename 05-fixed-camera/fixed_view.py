#!/usr/bin/env python3
"""Watch the fixed-camera detector work. NOTHING MOVES.

No arm connection is made at all, so this is safe to run at any time,
with anyone in the cell.

Two panels:

  LEFT   the camera's own view, with each accepted cube outlined and each
         REJECTED cluster marked with the reason it was refused. The
         reasons are the point: a detector that only shows you what it
         found leaves you guessing about what it did not.

  RIGHT  the same scene rendered as a top-down plan in the ROBOT'S
         frame - the "virtual overhead view" the whole project is built
         on. If the transform is right, sliding a cube across the floor
         slides it across this panel in a straight line, and its position
         here is in the same millimetres the arm is commanded in.

That second panel is the real test of a calibration. A hand-eye fit with
a small residual can still be wrong in a way the residual cannot show -
it was fitted over the volume the CUBE was carried through, and it is
being used over the volume cubes LIE IN. Put a cube in a corner and check
the plan view agrees with a tape measure.

It degrades on purpose rather than refusing:
    no handeye.json  -> camera only, and says what to run
    no floor_base    -> plan view of everything, no cube detection

Usage:  python fixed_view.py
Keys:   q / ESC quit    s  save both panels to captures\\
"""
import os
import sys
import time

import cv2
import numpy as np

import fixed_common as fx
import fixed_detect as fd

PX_PER_MM = 0.6
GREEN, RED, BLUE, GREY, YELLOW = ((0, 220, 0), (0, 0, 235), (235, 90, 0),
                                  (150, 150, 150), (0, 220, 220))
COLOR_BGR = {"red": RED, "blue": BLUE, "unknown": YELLOW}


def project(T, intr, pts_base):
    """Base-frame points back to image pixels, for drawing.

    The inverse of the pipeline: undo the rigid transform, then the
    pinhole. Points behind the camera are returned as NaN so a caller
    cannot draw a wrapped-around ghost."""
    R = np.asarray(T["R"], dtype=float)
    t = np.asarray(T["t"], dtype=float)
    cam = (np.atleast_2d(np.asarray(pts_base, dtype=float)) - t) @ R
    z = cam[:, 2]
    out = np.full((len(cam), 2), np.nan)
    ok = z > 1.0
    out[ok, 0] = intr.ppx + intr.fx * cam[ok, 0] / z[ok]
    out[ok, 1] = intr.ppy + intr.fy * cam[ok, 1] / z[ok]
    return out


def plan_canvas():
    rows = int((fx.WORK_X[1] - fx.WORK_X[0]) * PX_PER_MM)
    cols = int((fx.WORK_Y[1] - fx.WORK_Y[0]) * PX_PER_MM)
    return np.full((rows, cols, 3), 28, np.uint8)


def to_plan(x, y):
    """Base-frame XY to plan-view pixel. +X up the image, +Y to the left,
    which is how this arm's frame reads when you stand behind it."""
    col = int((fx.WORK_Y[1] - y) * PX_PER_MM)
    row = int((fx.WORK_X[1] - x) * PX_PER_MM)
    return col, row


def draw_plan_furniture(plan):
    """Grid, and the arm's usable annulus, so a cube's position can be
    read off rather than guessed."""
    for x in range(int(fx.WORK_X[0]) // 100 * 100, int(fx.WORK_X[1]) + 1,
                   100):
        if fx.WORK_X[0] <= x <= fx.WORK_X[1]:
            _, r = to_plan(x, 0)
            cv2.line(plan, (0, r), (plan.shape[1], r), (48, 48, 48), 1)
            cv2.putText(plan, "x%d" % x, (3, r - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    for y in range(int(fx.WORK_Y[0]) // 100 * 100, int(fx.WORK_Y[1]) + 1,
                   100):
        if fx.WORK_Y[0] <= y <= fx.WORK_Y[1]:
            c, _ = to_plan(0, y)
            cv2.line(plan, (c, 0), (c, plan.shape[0]), (48, 48, 48), 1)
            cv2.putText(plan, "y%d" % y, (c + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    origin = to_plan(0.0, 0.0)
    for radius, shade in ((fx.REACH_MIN_MM, (70, 70, 110)),
                          (fx.REACH_MAX_MM, (70, 110, 70))):
        cv2.circle(plan, origin, int(radius * PX_PER_MM), shade, 1)


def draw_cube_on_plan(plan, cube):
    col, row = to_plan(*cube["center"])
    half = cube["width_mm"] * PX_PER_MM / 2.0
    rect = ((col, row), (half * 2, half * 2),
            -cube["yaw_rect_deg"])       # plan Y runs opposite to base Y
    box = np.intp(cv2.boxPoints(rect))
    shade = COLOR_BGR.get(cube["color"], YELLOW)
    cv2.drawContours(plan, [box], 0, shade, 2)
    cv2.putText(plan, "%.0f,%.0f" % (cube["center"][0], cube["center"][1]),
                (col - 28, row - int(half) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, shade, 1)
    cv2.putText(plan, "%.0fmm h%.0f" % (cube["width_mm"], cube["height_mm"]),
                (col - 28, row + int(half) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, shade, 1)


def draw_cube_on_image(img, cube, T, intr):
    """Outline the cube's top face in the camera image, by projecting the
    base-frame square back through the camera model."""
    cx, cy = cube["center"]
    half = cube["width_mm"] / 2.0
    ang = np.radians(cube["yaw_rect_deg"])
    ca, sa = np.cos(ang), np.sin(ang)
    corners = []
    for dx, dy in ((-half, -half), (half, -half), (half, half),
                   (-half, half)):
        corners.append([cx + dx * ca - dy * sa, cy + dx * sa + dy * ca,
                        cube["top_z"]])
    px = project(T, intr, corners)
    if np.isnan(px).any():
        return
    shade = COLOR_BGR.get(cube["color"], YELLOW)
    cv2.polylines(img, [np.intp(px)], True, shade, 2)
    u, v = int(px[:, 0].mean()), int(px[:, 1].min())
    cv2.putText(img, "%s %.0fmm h%.0f" % (cube["color"], cube["width_mm"],
                                          cube["height_mm"]),
                (u - 50, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, shade, 1)


def banner(img, lines, shade=GREEN):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 20 + i * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, shade, 1)


def main():
    fx.start_log("view")
    T = fx.load_handeye()
    floor = fx.load_floor()
    if T is None:
        print("No handeye.json - showing the raw camera only.")
        print("Run '4 - Calibrate Camera' to give the arm a frame to work "
              "in.")
    elif floor is None:
        print("No floor_base.json - showing the plan view, but cubes cannot")
        print("be found until '5 - Teach Floor' has run.")
    else:
        print("Calibrated %.1fmm RMS, surface at %.1f, %s tilt."
              % (T.get("rms_mm", -1), floor["height_at_centre"],
                 "fitted" if floor.get("tilt_fitted") else "assumed level"))
    print("q / ESC to quit, s to save a snapshot.")

    if not os.path.isdir(fx.CAPT):
        os.makedirs(fx.CAPT)

    cam = fx.Camera()
    cam.start()
    rays = cam.rays()
    last, fps = time.time(), 0.0
    try:
        while True:
            depth_mm, color_bgr, _ = cam.frame()
            now = time.time()
            fps = 0.85 * fps + 0.15 / max(1e-3, now - last)
            last = now

            img = color_bgr.copy()
            plan = plan_canvas()
            draw_plan_furniture(plan)

            if T is None:
                banner(img, ["NOT CALIBRATED", "run '4 - Calibrate Camera'"],
                       YELLOW)
                banner(plan, ["no transform"], GREY)
            else:
                base = fx.apply_transform(T, fx.cloud_cam(depth_mm, rays))
                inside = (depth_mm > 0) & fx.in_work_box(base)
                pts = base[inside]

                if floor is None:
                    # No surface yet: paint every in-box point, coloured by
                    # height, so the transform can still be eyeballed.
                    if len(pts):
                        lo, hi = np.percentile(pts[:, 2], [5, 95])
                        rng = max(1.0, hi - lo)
                        for p in pts[::37]:
                            c, r = to_plan(p[0], p[1])
                            if 0 <= r < plan.shape[0] and 0 <= c < \
                                    plan.shape[1]:
                                g = int(np.clip((p[2] - lo) / rng, 0, 1) * 255)
                                plan[r, c] = (60, g, 255 - g)
                    banner(img, ["NO SURFACE TAUGHT",
                                 "run '5 - Teach Floor'"], YELLOW)
                    banner(plan, ["all points, coloured by height"], GREY)
                else:
                    height = pts[:, 2] - fx.plane_z(floor["coef"],
                                                    pts[:, 0], pts[:, 1])
                    for p in pts[(height > fx.POP_MM) &
                                 (height < fd.BAND_TOP_MM)][::7]:
                        c, r = to_plan(p[0], p[1])
                        if 0 <= r < plan.shape[0] and 0 <= c < plan.shape[1]:
                            plan[r, c] = (70, 70, 70)

                    cubes, rejects = fd.detect_all(depth_mm, color_bgr, rays,
                                                   T, floor)
                    for cube in cubes:
                        draw_cube_on_image(img, cube, T, cam.intr)
                        draw_cube_on_plan(plan, cube)
                    for (x, y), why in rejects:
                        px = project(T, cam.intr,
                                     [[x, y, fx.plane_z(floor["coef"], x, y)
                                       + 15.0]])[0]
                        if not np.isnan(px).any():
                            cv2.circle(img, (int(px[0]), int(px[1])), 7,
                                       GREY, 1)
                            cv2.putText(img, why, (int(px[0]) + 10,
                                                   int(px[1]) + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        GREY, 1)
                        c, r = to_plan(x, y)
                        cv2.drawMarker(plan, (c, r), GREY,
                                       cv2.MARKER_TILTED_CROSS, 9, 1)

                    counts = {}
                    for cube in cubes:
                        counts[cube["color"]] = counts.get(cube["color"],
                                                           0) + 1
                    banner(img, ["%d cube(s): %s   %d refused   %.0f fps"
                                 % (len(cubes),
                                    " ".join("%s %d" % kv
                                             for kv in sorted(counts.items()))
                                    or "-", len(rejects), fps)])
                    banner(plan, ["plan view - robot frame, mm"], GREY)

            pad = np.zeros((img.shape[0], plan.shape[1], 3), np.uint8)
            pad[:plan.shape[0], :] = plan
            cv2.imshow("fixed camera - view (nothing moves)",
                       np.hstack([img, pad]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(fx.CAPT, "view_%s.png" % stamp),
                            np.hstack([img, pad]))
                print("   saved captures\\view_%s.png" % stamp)
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

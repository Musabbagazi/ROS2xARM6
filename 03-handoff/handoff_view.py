#!/usr/bin/env python3
"""Live view of a cube held in your HAND - the arm does NOT move.

This is step one, and it exists so that nothing ever moves toward a
person on the strength of an untested detection. It shows, live:

  * every cube the floor-free detector finds, at any height
  * which one it would go for (the NEAREST to the camera - the one you
    are holding up, not the ones on the floor)
  * where that cube actually is in the arm's own coordinates, and
  * whether that spot is inside the safe hold band

Read the banner. GREEN means the picker would accept this cube right
now; RED says why not, in the same words the picker would use.

The arm is only ever asked where it is (a read), never where to go. If
it cannot be reached the view still works and assumes the wait pose.

Keys (click the window first):
  q / ESC  quit        s  save a snapshot        d  depth panel on/off

Usage: python handoff_view.py [robot_ip]
"""
import os
import sys
import time

import cv2
import numpy as np

import handoff_common as hc
import handoff_detect as hd
import vision3 as v3
import vision_common as vc
from camera_test import CAPT
from detect_cube import MODEL_FILE

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

COLOR_BGR = {"red": (0, 0, 255), "blue": (255, 0, 0),
             "unknown": (0, 255, 255)}
OK_BGR = (0, 220, 0)
BAD_BGR = (0, 0, 255)


def depth_panel(depth_frame):
    from detect_cube import depth_mm_image
    dmm = depth_mm_image(depth_frame)
    vis = np.clip(dmm, 200.0, 900.0)
    vis = ((vis - 200.0) / 700.0 * 255.0).astype(np.uint8)
    vis[dmm <= 0] = 0
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def verdict(ref, gz, width_mm, band):
    """Would the picker take this cube? (ok, message) - the same tests,
    in the same order, so the view cannot promise something the picker
    then refuses."""
    lo, hi = band
    high = hc.above_floor_mm(ref, gz)
    if gz < lo:
        return False, ("TOO LOW - %.0fmm above the floor, hold it higher "
                       "(need %.0fmm)"
                       % (high, hc.HAND_MIN_ABOVE_FLOOR))
    if gz > hi:
        return False, ("TOO CLOSE TO THE CAMERA - lower it, the depth "
                       "sensor cannot measure nearer than %.0fmm"
                       % hc.MIN_CAM_MM)
    if not hc.MIN_CUBE_MM <= width_mm <= hc.MAX_CUBE_MM:
        return False, ("%.0fmm is outside the grippable range %.0f-%.0fmm"
                       % (width_mm, hc.MIN_CUBE_MM, hc.MAX_CUBE_MM))
    return True, "READY - %.0fmm cube, %.0fmm above the floor" % (
        width_mm, high)


def main():
    if not os.path.exists(MODEL_FILE):
        print("cube_model.pt not found in vision\\ - train the v3 model "
              "first ('2 - Train Cube Model').")
        return
    calib, ref = hc.load_calibration()
    if calib is None:
        return

    # read-only: we need to know how high the camera is to turn a depth
    # into a height in the arm's frame. Never enables motion.
    arm, obs_pose, live = None, list(hc.WAIT_POSE), False
    try:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(ROBOT_IP, is_radian=False)
        code, p = arm.get_position(is_radian=False)
        if code == 0:
            obs_pose, live = list(p), True
    except Exception as e:
        print("(arm not reachable: %s)" % e)
    if live:
        print("Reading the arm's real pose - heights below are exact.")
    else:
        print("Arm not connected: assuming it sits at the wait pose "
              "%s.\nHeights are only right once the arm is actually there."
              % hc.WAIT_POSE[:3])

    band = hc.hold_band(ref, obs_pose[2])
    print("\nHold the cube between %.0f and %.0fmm above the floor "
          "(roughly %.0f-%.0f cm)."
          % (hc.above_floor_mm(ref, band[0]), hc.above_floor_mm(ref, band[1]),
             hc.above_floor_mm(ref, band[0]) / 10.0,
             hc.above_floor_mm(ref, band[1]) / 10.0))
    print("Hold it by the SIDES - a finger over the top face is what the "
          "camera measures instead of the cube.\n")

    win = "Handoff view - cube in hand (q quit  s snapshot  d depth)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_depth = True
    last_pose_read = 0.0
    last_print = 0.0
    fps_t, fps_n, fps = time.time(), 0, 0.0

    try:
        with v3.Camera(warmup=20) as cam:
            while True:
                color, depth, intr = cam.frame()
                view = color.copy()

                if arm is not None and time.time() - last_pose_read >= 1.0:
                    last_pose_read = time.time()
                    try:
                        code, p = arm.get_position(is_radian=False)
                        if code == 0:
                            obs_pose = list(p)
                            band = hc.hold_band(ref, obs_pose[2])
                    except Exception:
                        pass

                best, cands, n_boxes, reason = hd.detect_held(color, depth,
                                                              intr)

                for c in cands:
                    chosen = best is not None and c is best
                    bgr = COLOR_BGR.get(c["color"], COLOR_BGR["unknown"])
                    cv2.drawContours(view, [np.intp(c["box"])], 0, bgr,
                                     3 if chosen else 1)
                    px = tuple(int(v) for v in c["pixel"])
                    cv2.putText(view, "%.0fmm @%.0fmm"
                                % (c["width_mm"], c["depth_mm"]),
                                (px[0] - 45, px[1] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)

                if best is None:
                    if n_boxes and reason:
                        msg = "cube seen, but %s" % reason
                    elif n_boxes:
                        msg = "cube seen but its top face cannot be measured"
                    else:
                        msg = "no cube in view"
                    cv2.putText(view, msg, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BAD_BGR, 2)
                    status = msg
                else:
                    gx, gy, gz, yaw = hc.aim_hand(calib, ref, obs_pose, best)
                    ok, why = verdict(ref, gz, best["width_mm"], band)
                    bgr = OK_BGR if ok else BAD_BGR
                    px = tuple(int(v) for v in best["pixel"])
                    cv2.drawMarker(view, px, bgr, cv2.MARKER_CROSS, 26, 2)
                    cv2.putText(view, why, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
                    cv2.putText(view, "arm xy [%.0f, %.0f]  yaw %.0f  "
                                "%s" % (gx, gy, yaw, best["color"]),
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1)
                    if arm is not None:
                        follow = [gx, gy, gz + hc.FOLLOW_H, 180.0, 0.0,
                                  vc.SCAN_POSE[5]]
                        if vc.ik(arm, follow) is None:
                            cv2.putText(view, "OUT OF REACH - move it toward "
                                        "the middle of the cell",
                                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, BAD_BGR, 2)
                    status = "%s | %s | z %.0f (%.0fmm up)" % (
                        why, "%.0fmm" % best["depth_mm"], gz,
                        hc.above_floor_mm(ref, gz))

                cv2.drawMarker(view, (320, 240), (255, 255, 255),
                               cv2.MARKER_CROSS, 20, 1)
                fps_n += 1
                if time.time() - fps_t >= 1.0:
                    fps = fps_n / (time.time() - fps_t)
                    fps_t, fps_n = time.time(), 0
                cv2.putText(view, "%.0f fps   arm z %.0f%s"
                            % (fps, obs_pose[2], "" if live else " (assumed)"),
                            (400, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

                frame = view
                if show_depth:
                    frame = np.hstack([view, depth_panel(depth)])
                cv2.imshow(win, frame)

                if time.time() - last_print >= 1.0:
                    print(status)
                    last_print = time.time()

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("d"):
                    show_depth = not show_depth
                elif k == ord("s"):
                    path = os.path.join(CAPT, "handoff_%d.png"
                                        % int(time.time()))
                    cv2.imwrite(path, frame)
                    print("saved", path)
    finally:
        cv2.destroyAllWindows()
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()

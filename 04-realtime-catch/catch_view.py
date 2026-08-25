#!/usr/bin/env python3
"""Watch the tracker work, with the arm standing still - or absent.

The one thing that cannot be checked by arithmetic is whether a REAL
cube, sliding across a REAL bench under this cell's lighting, produces a
velocity fit worth betting an arm on. This shows exactly that, and moves
nothing.

On screen, per frame:
  * the measured cube top (green box)
  * the fitted velocity as an arrow, drawn from the cube along its path
  * where the fit says it will be in MIN_LEAD_S
  * speed, heading, fit residual, sample count
  * the verdict catch_pick would reach - CATCHABLE, or the exact words
    it would say to refuse

Run this before catch_pick.py, every time the cell changes. If a cube
does not read CATCHABLE here, catch_pick will refuse it too - and this
way you find that out without an arm in the room.

  --ip ADDRESS   read the arm's real pose (READ-ONLY - never moves it)
                 so the millimetre positions are true. Without it the
                 arm is ASSUMED to be at the watch pose, which makes the
                 positions an offset guess; speed, heading and residual
                 are unaffected either way, because they are differences.

Press ESC or q in the image window to quit.

Usage: python catch_view.py [--ip 192.168.1.197]
"""
import sys

import cv2
import numpy as np

import catch_common as cc
import catch_detect as cd
import catch_track as ct
import vision3 as v3
import vision_common as vc

WIN = "catch view - ESC to quit"

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AMBER = (0, 190, 255)
WHITE = (255, 255, 255)


def parse_args(argv):
    ip = None
    if "--ip" in argv:
        i = argv.index("--ip")
        if i + 1 < len(argv):
            ip = argv[i + 1]
    return ip


def connect_readonly(ip):
    """Attach to the arm WITHOUT enabling motion.

    Deliberately does not call vc.setup_arm: that enables the servos and
    the gripper, and this tool has no business doing either. All it wants
    is get_position."""
    try:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(ip, is_radian=False)
        code, pos = arm.get_position(is_radian=False)
        if code != 0:
            print("(could not read the arm - falling back to the assumed "
                  "watch pose)")
            arm.disconnect()
            return None
        print("Reading the arm's pose live. Nothing will be commanded.")
        return arm
    except Exception as e:
        print("(no arm at %s: %s - assuming the watch pose)" % (ip, e))
        return None


def obs_pose(arm):
    if arm is None:
        return list(cc.WATCH_POSE)
    code, pos = arm.get_position(is_radian=False)
    return list(pos) if code == 0 else list(cc.WATCH_POSE)


def draw_arrow(img, calib, ref, track, f):
    """Draw the fitted path in the IMAGE, by projecting two points on it.

    The fit lives in millimetres in the arm's frame, so getting it back
    onto the screen means inverting the same mapping aim_moving uses.
    catch_common.predict_pixel already does that inversion for a
    stationary cube under a moving arm; here the arm is still and the
    CUBE has moved, which is the same algebra with the sign of the
    displacement flipped - a cube 30mm north looks exactly like an arm
    30mm south."""
    seen = track.last_seen
    if seen is None or f is None:
        return
    now = track.t[-1]
    here = track.predict(now, f)
    there = track.predict(now + cc.MIN_LEAD_S, f)
    if here is None or there is None:
        return
    d_cube = [there[0] - here[0], there[1] - here[1]]
    tip = cc.predict_pixel(calib, ref, seen["pixel"], seen["depth_mm"],
                           [-d_cube[0], -d_cube[1]], 0.0)
    p0 = (int(seen["pixel"][0]), int(seen["pixel"][1]))
    p1 = (int(tip[0]), int(tip[1]))
    ok, _why = track.confident(f)
    cv2.arrowedLine(img, p0, p1, GREEN if ok else AMBER, 2, tipLength=0.2)
    cv2.circle(img, p1, 6, GREEN if ok else AMBER, 2)


def main():
    ip = parse_args(sys.argv[1:])
    calib, ref = cc.load_calibration()
    if calib is None:
        return
    arm = connect_readonly(ip) if ip else None
    if arm is None and ip is None:
        print("No --ip given: assuming the arm is at the watch pose %s."
              % cc.WATCH_POSE[:3])

    track = ct.CubeTrack()
    locked_px = None
    misses = 0
    print("Send a cube across the view. ESC or q in the window to quit.")
    try:
        with v3.Camera(floor_ref=None) as cam:
            while True:
                pose = obs_pose(arm)
                color, depth, intr = cam.frame()
                best, cands, n_boxes, reason, t_frame = cd.detect_moving(
                    color, depth, intr, prefer_near=locked_px,
                    near_radius=cd_lock(locked_px))
                img = color.copy()

                if best is None:
                    misses += 1
                    if locked_px is not None and misses >= 4:
                        locked_px = None
                        track.reset()
                    note = reason or ("%d box(es), none measurable" % n_boxes
                                      if n_boxes else "no cube in view")
                    cv2.putText(img, note, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2)
                else:
                    misses = 0
                    locked_px = list(best["pixel"])
                    gx, gy, gz, yaw = cc.aim_moving(calib, ref, pose, best)
                    track.add(t_frame, gx, gy, gz, best)
                    cv2.drawContours(img, [np.intp(best["box"])], 0, GREEN, 2)

                    f = track.fit()
                    draw_arrow(img, calib, ref, track, f)
                    lines = describe(track, f, best, ref, gz)
                    for i, (text, col) in enumerate(lines):
                        cv2.putText(img, text, (10, 30 + 24 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                cv2.imshow(WIN, img)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
    finally:
        cv2.destroyAllWindows()
        if arm is not None:
            arm.disconnect()


def cd_lock(locked_px):
    """LOCK_PX only once there is something to stay locked to."""
    return None if locked_px is None else 200.0


def describe(track, f, best, ref, gz):
    """The lines of text overlaid on the frame, with their colours."""
    out = [("%.0fmm cube, %s, top at %.0fmm"
            % (best["width_mm"], best.get("color", "?"), best["depth_mm"]),
            WHITE)]
    if f is None:
        out.append(("watching...", AMBER))
        return out
    out.append(("%.0f mm/s  heading %+.0f deg  fit %.1fmm  n=%d over %.2fs"
                % (f["speed"], f["heading_deg"], f["resid_mm"], f["n"],
                   f["span_s"]), WHITE))
    lo, hi = cc.catch_band(ref, cc.WATCH_POSE[2], best["height_mm"])
    if not lo <= gz <= hi:
        out.append(("OUT OF BAND - %.0fmm above the floor"
                    % cc.above_floor_mm(ref, gz), RED))
        return out
    ok, why = track.confident(f)
    out.append(("CATCHABLE" if ok else "no - " + why, GREEN if ok else AMBER))
    return out


if __name__ == "__main__":
    main()

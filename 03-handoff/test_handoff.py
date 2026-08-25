#!/usr/bin/env python3
"""Offline checks for the handoff project - no arm, no camera, no model.

Everything here is arithmetic or image processing on synthetic data, so
it can be run any time, on any machine, before anything moves. What it
does NOT prove is the part only hardware can answer: whether a real hand
holding a real cube detects reliably, and whether the blind final dive
lands where the follow step said it would.

Usage: python test_handoff.py
"""
import numpy as np

import handoff_common as hc
import handoff_detect as hd
import vision3 as v3
import vision_common as vc

FAILED = []


def check(name, cond, detail=""):
    print("%-52s %s%s" % (name, "ok" if cond else "FAIL",
                          "" if not detail else "   " + detail))
    if not cond:
        FAILED.append(name)


def fake_ref():
    """The real grip_ref numbers, so the printed heights are the ones the
    arm will actually use."""
    ref = v3.load_grip_ref()
    if ref is not None:
        return ref
    return {"g0": [602.4, 43.1, 200.0, 0.0], "p0": [391.0, 144.0],
            "d0": 507.0, "w0_mm": 37.0, "angle0": 0.0, "h0": 30.8,
            "floor0": 537.8, "stall0": 334, "zC": 142.4, "look_h": 180.0,
            "scan_pose": list(vc.SCAN_POSE)}


def fake_calib():
    c = v3.load_calib3()
    if c is not None:
        return c
    return {"J": [[0.0424, -0.8546], [-0.8629, -0.0028]], "d_ref": 507.0}


def seen(pixel, depth, width=30.0, angle=0.0):
    return {"pixel": list(pixel), "depth_mm": float(depth),
            "width_mm": width, "height_mm": width, "angle_deg": angle,
            "floor_mm": None}


def test_heights(ref):
    fz = hc.floor_grab_z(ref)
    # a 30mm cube on the floor is grabbed at floor_grab_z + h/2, which is
    # the taught reference grab height g0[2]
    on_floor = fz + ref["h0"] / 2.0
    check("floor grab Z matches the taught reference",
          abs(on_floor - ref["g0"][2]) < 2.0,
          "%.1f vs g0 %.1f" % (on_floor, ref["g0"][2]))

    lo, hi = hc.hold_band(ref)
    check("hold band low limit is the floor plus the margin",
          abs(hc.above_floor_mm(ref, lo) - hc.HAND_MIN_ABOVE_FLOOR) < 0.01)
    check("hold band is a usable span (>15cm)", hi - lo > 150.0,
          "%.0f..%.0fmm above the floor"
          % (hc.above_floor_mm(ref, lo), hc.above_floor_mm(ref, hi)))
    # at the top of the band the camera must still be at its near limit
    d_at_hi = hc.WAIT_POSE[2] - hi - 30.0 * hc.GRAB_DEPTH_FRAC + ref["zC"]
    check("top of the band sits exactly at the camera's near limit",
          abs(d_at_hi - hc.MIN_CAM_MM) < 0.5,
          "%.1fmm" % d_at_hi)
    # and at the follow height the cube must still be measurable
    d_follow = hc.FOLLOW_H - 30.0 * hc.GRAB_DEPTH_FRAC + ref["zC"]
    check("cube is still visible from the follow height",
          d_follow > hc.MIN_CAM_MM,
          "%.0fmm of range at follow height" % d_follow)


def test_aim(calib, ref):
    obs = [450.0, 70.0, 580.0, 180.0, 0.0, 0.0]
    s = seen((391.0, 144.0), 507.0)
    gx, gy, gz, yaw = hc.aim_hand(calib, ref, obs, s)
    vx, vy, _vz, vyaw = v3.aim_from_pose(calib, ref, obs, s)
    check("XY matches the proven v3 aim exactly",
          abs(gx - vx) < 1e-9 and abs(gy - vy) < 1e-9,
          "[%.2f, %.2f]" % (gx, gy))
    check("yaw matches the proven v3 aim", abs(yaw - vyaw) < 1e-9)
    # the anchor pixel at the anchor depth must map back to the anchor pose
    check("the calibration anchor maps back onto itself",
          abs(gx - ref["g0"][0]) < 0.5 and abs(gy - ref["g0"][1]) < 0.5,
          "[%.1f, %.1f] vs g0 [%.1f, %.1f]"
          % (gx, gy, ref["g0"][0], ref["g0"][1]))

    # height: no floor anywhere in the answer
    high = hc.aim_hand(calib, ref, obs, seen((320.0, 240.0), 300.0))[2]
    low = hc.aim_hand(calib, ref, obs, seen((320.0, 240.0), 500.0))[2]
    check("a nearer cube grabs higher, one-for-one",
          abs((high - low) - 200.0) < 1e-6, "%.1fmm apart" % (high - low))
    check("the grab bites above mid-height",
          hc.GRAB_DEPTH_FRAC < 0.5)

    # v3 would have CLAMPED a mid-air cube down to the floor window;
    # this is the whole reason the project exists, so prove it
    v_z = v3.aim_from_pose(calib, ref, obs, seen((320.0, 240.0), 300.0))[2]
    check("v3's floor clamp really would have ruined a mid-air grab",
          v_z < high - 50.0,
          "v3 %.0f vs handoff %.0f" % (v_z, high))


def test_predict_pixel(calib, ref):
    obs = [450.0, 70.0, 620.0, 180.0, 0.0, 0.0]
    s = seen((250.0, 300.0), 320.0)
    gx, gy, _gz, _y = hc.aim_hand(calib, ref, obs, s)
    for step in ([40.0, -25.0, 0.0], [-60.0, 10.0, -30.0], [0.0, 0.0, -50.0]):
        p2 = hc.predict_pixel(calib, ref, s["pixel"], s["depth_mm"],
                              step[:2], step[2])
        obs2 = [obs[0] + step[0], obs[1] + step[1], obs[2] + step[2],
                180.0, 0.0, 0.0]
        s2 = seen(p2, s["depth_mm"] - step[2])
        nx, ny, _z, _yy = hc.aim_hand(calib, ref, obs2, s2)
        check("predicted pixel keeps a still cube still  %s" % (step,),
              abs(nx - gx) < 0.01 and abs(ny - gy) < 0.01,
              "off by [%.3f, %.3f]" % (nx - gx, ny - gy))


def test_steps():
    nxt = hc.step_toward([0.0, 0.0, 500.0], [500.0, 0.0, 500.0])
    check("a far XY target is capped to one step",
          abs(nxt[0] - hc.MAX_XY_STEP) < 1e-9)
    nxt = hc.step_toward([0.0, 0.0, 500.0], [0.0, 0.0, 100.0])
    check("a far Z target is capped to one step",
          abs(nxt[2] - (500.0 - hc.MAX_Z_STEP)) < 1e-9)
    nxt = hc.step_toward([0.0, 0.0, 500.0], [3.0, 4.0, 502.0])
    check("a near target is reached in one step",
          abs(nxt[0] - 3.0) < 1e-9 and abs(nxt[2] - 502.0) < 1e-9)
    d = np.hypot(*hc.step_toward([0.0, 0.0, 0.0], [300.0, 400.0, 0.0])[:2])
    check("the XY cap is a distance, not per-axis",
          abs(d - hc.MAX_XY_STEP) < 1e-9, "%.1fmm" % d)


def test_selection():
    near = {"pixel": [100.0, 100.0], "depth_mm": 300.0, "width_mm": 30.0}
    far = {"pixel": [400.0, 400.0], "depth_mm": 520.0, "width_mm": 70.0}
    check("the cube held up (nearest) wins over a bigger one below",
          hd.pick_held([far, near]) is near)
    check("with a lock, the one near that pixel wins",
          hd.pick_held([far, near], prefer_near=[395.0, 402.0]) is far)
    check("a lock with a radius refuses a stranger",
          hd.pick_held([far], prefer_near=[100.0, 100.0],
                       near_radius=50.0) is None)
    check("no candidates -> nothing", hd.pick_held([]) is None)


def test_rect_local():
    """A synthetic scene: a cube top nearer than everything around it."""
    dmm = np.full((200, 200), 900.0, np.float32)      # background, far
    dmm[70:130, 70:130] = 400.0                       # cube top, near
    rect, top, why = hd._rect_local(dmm, [60, 60, 140, 140])
    check("finds a cube top with no floor anywhere in the scene",
          rect is not None and abs(top - 400.0) < 1.0, why or "")
    if rect is not None:
        (cx, cy), (w, h), _a = rect
        check("centre and size are right",
              abs(cx - 100.0) < 3 and abs(cy - 100.0) < 3
              and abs(min(w, h) - 60.0) < 4,
              "centre [%.0f, %.0f] size %.0f" % (cx, cy, min(w, h)))

    # THE dangerous false positive: a big flat surface fills the box and
    # looks like a perfect square top face at a plausible depth. Nothing
    # is behind it, so it is not an object.
    flat = np.full((200, 200), 700.0, np.float32)
    rect, _t, why = hd._rect_local(flat, [60, 60, 140, 140])
    check("a flat surface is not a cube", rect is None, why or "")

    # two cubes touching along one axis merge into a 1.5:1 blob whose
    # centre is the seam - the fingers would close on the gap
    pair = np.full((200, 200), 900.0, np.float32)
    pair[70:130, 70:130] = 400.0
    pair[70:130, 100:160] = 400.0
    rect, _t, why = hd._rect_local(pair, [60, 60, 170, 140])
    check("a merged touching pair is refused, not grabbed at the seam",
          rect is None, why or "")

    # fingers at the EDGE of the top face (how you are told to hold it):
    # nearer than the cube, so the first percentile locks onto them and
    # fails the gates - the percentile ladder then finds the cube itself
    hand = np.full((200, 200), 900.0, np.float32)
    hand[70:130, 70:130] = 400.0
    hand[60:140, 62:76] = 345.0                       # a fingertip, near
    rect, top, why = hd._rect_local(hand, [60, 60, 140, 140])
    check("still finds the cube with a fingertip at its edge",
          rect is not None and abs(top - 400.0) < 12.0,
          why or "top %s" % round(top))

    # THE failure from the first hardware run: the hand makes a BIGGER
    # blob than the cube at a similar depth. Testing only the largest
    # blob threw the cube away and reported the hand's outline instead
    # ("not square", "measures 140mm across"). Every blob is tested now.
    big = np.full((240, 240), 900.0, np.float32)
    big[70:130, 70:130] = 400.0                       # the cube
    big[60:200, 145:175] = 404.0                      # a hand, bigger
    rect, top, why = hd._rect_local(big, [60, 60, 190, 190])
    ok = rect is not None
    check("finds the cube even when the hand is the bigger blob",
          ok and abs(rect[0][0] - 100.0) < 6 and abs(top - 400.0) < 6,
          why or "centre x %.0f (cube is at 100)" % rect[0][0])

    # fingers gripping the SIDES sit below the top face; the depth band
    # is tight enough (10mm) to leave them out even when they touch
    grip = np.full((240, 240), 900.0, np.float32)
    grip[70:130, 70:130] = 400.0
    grip[60:140, 55:72] = 414.0                       # a finger, 14mm down
    grip[60:140, 128:145] = 414.0                     # and the other one
    rect, top, why = hd._rect_local(grip, [60, 60, 140, 140])
    check("fingers on the sides do not merge into the top face",
          rect is not None and abs(min(rect[1]) - 60.0) < 8,
          why or "measured %.0fpx (cube is 60)" % min(rect[1]))

    # a finger across the MIDDLE of the top face must be refused: the
    # visible face is broken in two and any rect fitted to it is wrong,
    # so refusing is the correct answer, not a shortcoming
    over = np.full((200, 200), 900.0, np.float32)
    over[70:130, 70:130] = 400.0
    over[93:110, 40:170] = 330.0
    rect, _t, why = hd._rect_local(over, [60, 60, 140, 140])
    check("refuses a cube with something lying across its top face",
          rect is None, why or "")


def test_reach_hints():
    check("the follow ladder starts at the nominal height and descends",
          hc.FOLLOW_LADDER[0] == hc.FOLLOW_H
          and hc.FOLLOW_LADDER == sorted(hc.FOLLOW_LADDER, reverse=True))
    lowest = hc.FOLLOW_LADDER[-1]
    d = lowest - 30.0 * hc.GRAB_DEPTH_FRAC + 142.4
    check("even the lowest hover can still see the cube",
          d > hc.MIN_CAM_MM, "%.0fmm of range at %.0fmm up" % (d, lowest))
    check("too far out says so", "too far out"
          in hc.out_of_reach_hint(900.0, 0.0))
    check("too close in says so", "base"
          in hc.out_of_reach_hint(100.0, 0.0))
    check("in-range but awkward gets a different hint",
          "hand's width" in hc.out_of_reach_hint(500.0, 0.0))


def test_gripper(ref):
    o = hc.open_pulses(ref, 30.0)
    c = hc.close_pulses(ref, 30.0)
    check("opens wider than it closes", o > c, "%d vs %d" % (o, c))
    check("never commands past the gripper's range", 0 <= c and o <= 850)
    check("opening clears the cube by the intended margin",
          abs((o - (ref["stall0"] + (30.0 - ref["w0_mm"])
                    * hc.PULSES_PER_MM)) / hc.PULSES_PER_MM
              - hc.OPEN_EXTRA_MM) < 0.2)


def test_safety_constants():
    check("handoff speed is well below the floor picker's",
          hc.HANDOFF_SPEED < 60 and hc.HANDOFF_ACC < 1000,
          "%d deg/s, %d deg/s^2" % (hc.HANDOFF_SPEED, hc.HANDOFF_ACC))
    check("the FINGERS are slowed too, not just the arm",
          hc.HANDOFF_GRIPPER_SPEED < 5000,
          "%d of 5000" % hc.HANDOFF_GRIPPER_SPEED)
    # slow_down must actually take effect on the module the moves use
    before = (vc.JOINT_SPEED, vc.JOINT_ACC, vc.GRIPPER_SPEED)
    hc.slow_down()
    check("slow_down really lowers arm AND gripper speed",
          vc.JOINT_SPEED == hc.HANDOFF_SPEED
          and vc.JOINT_ACC == hc.HANDOFF_ACC
          and vc.GRIPPER_SPEED == hc.HANDOFF_GRIPPER_SPEED,
          "was %s, now %s" % (before, (vc.JOINT_SPEED, vc.JOINT_ACC,
                                       vc.GRIPPER_SPEED)))
    check("a single step cannot cross the cell",
          hc.MAX_XY_STEP <= 60.0 and hc.MAX_Z_STEP <= 60.0)
    check("the commit needs several steady samples", hc.STEADY_N >= 2)
    check("the deadband is under the alignment tolerance",
          hc.DEADBAND_MM <= hc.ALIGN_MM)


def main():
    ref, calib = fake_ref(), fake_calib()
    print("=== handoff offline checks ===\n")
    test_heights(ref)
    test_aim(calib, ref)
    test_predict_pixel(calib, ref)
    test_steps()
    test_selection()
    test_rect_local()
    test_reach_hints()
    test_gripper(ref)
    test_safety_constants()
    print("\n%s" % ("ALL PASS" if not FAILED
                    else "FAILED: " + ", ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

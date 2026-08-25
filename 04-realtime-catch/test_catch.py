#!/usr/bin/env python3
"""Offline checks for the catch project - no arm, no camera, no model.

Everything here is arithmetic on synthetic tracks, so it runs any time,
on any machine, before anything moves. It covers the part of this project
that is genuinely new - the velocity fit, the gates that refuse a motion
this cannot catch, and the intercept planner - plus the consistency of
the aim math with the mapping v3 already proved on hardware.

What it does NOT prove is the part only hardware can answer:

  * whether a real cube sliding across this cell fits a line well enough
    to bet on. Run catch_view.py for that - it shows the residual live.
  * whether ArmTiming's default travel rate is anywhere near the truth.
    catch_pick measures it and prints the measurement when it finishes.
  * whether GRIPPER_CLOSE_S matches the real fingers at
    CATCH_GRIPPER_SPEED. Time it once and correct catch_common.
  * whether MAX_CATCH_SPEED is honest. It is derived from geometry and a
    guessed timing jitter, not measured.

Usage: python test_catch.py
"""
import numpy as np

import catch_common as cc
import catch_track as ct
import vision3 as v3
import vision_common as vc

FAILED = []


def check(name, cond, detail=""):
    print("%-56s %s%s" % (name, "ok" if cond else "FAIL",
                          "" if not detail else "   " + detail))
    if not cond:
        FAILED.append(name)


def fake_ref():
    """The real grip_ref numbers when they exist, so the printed heights
    are the ones the arm will actually use."""
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


# A cube CROSSING the cell, not one heading away from the arm. The
# distinction is not cosmetic: on a radial path there are only ~250mm of
# reach left in front of the cube, which is less than the arm needs to
# get ahead of anything but a crawl, so a radial track is refused for
# sound reasons and makes a useless fixture for testing the planner.
CROSS_START = (400.0, -250.0)
CROSS_HEADING = 90.0


def straight_track(speed=80.0, heading=0.0, n=10, dt=0.05, z=200.0,
                   noise=0.0, seed=1, start=CROSS_START):
    """A cube travelling in a straight line at constant speed."""
    rng = np.random.default_rng(seed)
    tr = ct.CubeTrack()
    vx = speed * np.cos(np.radians(heading))
    vy = speed * np.sin(np.radians(heading))
    for i in range(n):
        t = i * dt
        jx, jy = (rng.normal(0.0, noise, 2) if noise else (0.0, 0.0))
        tr.add(t, start[0] + vx * t + jx, start[1] + vy * t + jy, z,
               seen((320, 240), 400))
    return tr


def crossing_track(speed=90.0, **kw):
    """The fixture the planner tests use - a cube crossing the cell."""
    return straight_track(speed=speed, heading=CROSS_HEADING, **kw)


# ------------------------- geometry -------------------------

def test_geometry(ref):
    fz = cc.floor_grab_z(ref)
    on_floor = fz + ref["h0"] / 2.0
    check("floor grab Z matches the taught reference",
          abs(on_floor - ref["g0"][2]) < 2.0,
          "%.1f vs g0 %.1f" % (on_floor, ref["g0"][2]))

    lo, hi = cc.catch_band(ref)
    check("catch band reaches DOWN to the floor (sliding cubes)",
          cc.above_floor_mm(ref, lo) <= 0.0,
          "%.0fmm above the floor" % cc.above_floor_mm(ref, lo))
    check("catch band is a usable span (>20cm)", hi - lo > 200.0,
          "%.0f..%.0fmm above the floor"
          % (cc.above_floor_mm(ref, lo), cc.above_floor_mm(ref, hi)))
    # at the top of the band the camera must be exactly at its near limit
    d_at_hi = (cc.WATCH_POSE[2] - hi - 30.0 * cc.GRAB_DEPTH_FRAC
               + ref["zC"])
    check("top of the band sits at the camera's near limit",
          abs(d_at_hi - cc.MIN_CAM_MM) < 0.5, "%.1fmm" % d_at_hi)
    # and at the hover height the cube must still be measurable
    d_hover = cc.HOVER_H - 30.0 * cc.GRAB_DEPTH_FRAC + ref["zC"]
    check("a cube is still measurable from the hover height",
          d_hover > cc.MIN_CAM_MM + 50.0,
          "%.0fmm vs min %.0f" % (d_hover, cc.MIN_CAM_MM))
    # the thing that forces the whole ambush design: at the GRASP pose
    # the camera is inside its own blind zone
    d_grasp = ref["zC"] - 30.0 * cc.GRAB_DEPTH_FRAC
    check("at the grasp pose the camera IS blind (the design premise)",
          d_grasp < cc.MIN_CAM_MM,
          "%.0fmm vs min %.0f" % (d_grasp, cc.MIN_CAM_MM))


def test_aim_matches_v3(calib, ref):
    """X, Y and yaw must be bit-identical to the mapping v3 proved.

    Only the HEIGHT may differ, and only because v3 clamps its answer
    into a window around the floor - see aim_moving's docstring."""
    worst_xy, worst_yaw = 0.0, 0.0
    for px, py, d in ((320, 240, 500), (120, 90, 430), (540, 400, 610),
                      (400, 200, 380)):
        s = seen((px, py), d, width=32.0, angle=17.0)
        for pose in ([450.0, 70.0, 580.0, 180.0, 0.0, 0.0],
                     [380.0, 150.0, 680.0, 180.0, 0.0, 0.0]):
            ax, ay, _az, ayaw = cc.aim_moving(calib, ref, pose, s)
            bx, by, _bz, byaw = v3.aim_from_pose(calib, ref, pose, s)
            worst_xy = max(worst_xy, abs(ax - bx), abs(ay - by))
            worst_yaw = max(worst_yaw, abs(ayaw - byaw))
    check("aim_moving X/Y identical to v3.aim_from_pose", worst_xy < 1e-9,
          "worst %.2e mm" % worst_xy)
    check("aim_moving yaw identical to v3.aim_from_pose", worst_yaw < 1e-9,
          "worst %.2e deg" % worst_yaw)


def test_aim_height_is_unclamped(calib, ref):
    """A cube held high must produce a HIGH grab, not v3's floor clamp."""
    high = seen((320, 240), 260.0)          # top only 260mm from a high pose
    pose = [450.0, 70.0, 680.0, 180.0, 0.0, 0.0]
    _x, _y, gz, _yaw = cc.aim_moving(calib, ref, pose, high)
    _x2, _y2, gz_v3, _y3 = v3.aim_from_pose(calib, ref, pose, high)
    check("aim_moving does not clamp a high cube to the floor window",
          gz > gz_v3 + 20.0,
          "%.0f vs v3 %.0f (%.0fmm above the floor)"
          % (gz, gz_v3, cc.above_floor_mm(ref, gz)))


# ------------------------- the fit -------------------------

def test_fit_recovers_velocity():
    tr = straight_track(speed=120.0, heading=30.0)
    f = tr.fit()
    check("fit recovers the speed of a clean straight track",
          abs(f["speed"] - 120.0) < 0.5, "%.2f mm/s" % f["speed"])
    check("fit recovers the heading", abs(f["heading_deg"] - 30.0) < 0.5,
          "%.2f deg" % f["heading_deg"])
    check("a clean track has a near-zero residual", f["resid_mm"] < 0.01,
          "%.4f mm" % f["resid_mm"])


def test_fit_predicts_forward():
    tr = straight_track(speed=100.0, heading=0.0, n=10, dt=0.05)
    # last sample at t=0.45; one second later it must be 100mm further on
    x, y, _z = tr.predict(1.45)
    check("prediction extrapolates at the fitted speed",
          abs(x - (CROSS_START[0] + 145.0)) < 0.5, "x=%.1f" % x)


def test_noise_is_tolerated():
    tr = straight_track(speed=100.0, noise=2.0, n=12, seed=7)
    f = tr.fit()
    ok, why = tr.confident(f)
    check("a noisy but straight track is still catchable", ok, why or "")
    check("its residual is reported, not hidden",
          0.0 < f["resid_mm"] < ct.MAX_RESID_MM, "%.2f mm" % f["resid_mm"])


def arc_track(radius=60.0, speed=80.0, n=20, dt=0.06):
    """A cube on a circular arc, at a speed well inside the catchable
    band so a refusal can only have come from the curvature."""
    tr = ct.CubeTrack()
    omega = speed / radius              # rad/s
    for i in range(n):
        t = i * dt
        a = omega * t
        tr.add(t, CROSS_START[0] + radius * np.sin(a),
               CROSS_START[1] + radius * (1.0 - np.cos(a)), 200.0,
               seen((320, 240), 400))
    return tr


def test_curved_path_is_refused():
    """The case a Kalman filter would have tracked confidently.

    Note WHERE the refusal comes from: not from the fit residual, which
    this arc passes comfortably, but from the extrapolated drift measured
    per candidate in the planner. That distinction is the whole point of
    drift_at - see its note."""
    tr = arc_track()
    f = tr.fit()
    check("the curved fixture is inside the catchable speed band",
          cc.MIN_CATCH_SPEED < f["speed"] < cc.MAX_CATCH_SPEED,
          "%.0f mm/s" % f["speed"])
    check("the residual ALONE does not catch this curve (why drift_at "
          "exists)", f["resid_mm"] < ct.MAX_RESID_MM,
          "%.1fmm of residual" % f["resid_mm"])
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, ct.ArmTiming())
    check("a curved path yields NO intercept", plan is None, why or "")
    check("...and the reason names the curve", "curving" in (why or ""),
          why or "")


def test_drift_grows_with_the_horizon():
    """The quantity the residual could not express: the same bend is
    harmless soon and hopeless later, as the square of the lead."""
    tr = arc_track()
    near, far = tr.drift_at(0.5), tr.drift_at(2.0)
    check("drift grows with the square of the lead",
          far > 12.0 * near and near > 0.0,
          "%.1fmm at 0.5s, %.1fmm at 2.0s" % (near, far))


def test_noise_does_not_read_as_a_bend():
    """A straight track carrying ordinary detector noise must show ZERO
    significant turn. Without the sigma test, noise on the two half-fits
    reads as a slow curve and refuses every real cube."""
    tr = straight_track(speed=100.0, heading=CROSS_HEADING, noise=2.0,
                        n=24, dt=0.05, seed=3)
    drift = tr.drift_at(2.5)
    check("noise on a straight track is not mistaken for a bend",
          drift <= ct.DRIFT_BUDGET_FRAC * cc.half_gap_mm(),
          "%.1fmm of drift at 2.5s" % drift)
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, ct.ArmTiming())
    check("...and a noisy straight cube is still caught", plan is not None,
          why or "")


def test_speed_gates():
    slow = straight_track(speed=cc.MIN_CATCH_SPEED - 5.0, n=12)
    ok, why = slow.confident()
    check("a barely-moving cube is handed back to the floor picker",
          not ok and "still" in (why or ""), why or "")

    fast = straight_track(speed=cc.MAX_CATCH_SPEED + 50.0, n=12)
    ok, why = fast.confident()
    check("a cube that is too fast is refused", not ok, why or "")
    check("...and the refusal quotes the limit",
          "%.0f" % cc.MAX_CATCH_SPEED in (why or ""), why or "")


def test_falling_is_refused():
    tr = ct.CubeTrack()
    for i in range(12):
        t = i * 0.05
        tr.add(t, 400.0 + 80.0 * t, 0.0, 300.0 - 200.0 * t,
               seen((320, 240), 400))
    ok, why = tr.confident()
    check("a rising/falling cube is refused", not ok, why or "")


def test_gap_resets_history():
    tr = straight_track(speed=100.0, n=8, dt=0.05)
    n_before = tr.n
    tr.add(tr.t[-1] + ct.MAX_GAP_S + 0.2, 999.0, 999.0, 200.0,
           seen((320, 240), 400))
    check("a gap in the detections drops the old history",
          tr.n == 1 and n_before > 1, "%d -> %d" % (n_before, tr.n))


def test_window_trims():
    tr = straight_track(speed=100.0, n=80, dt=0.05)   # 4s of history
    check("the fit window is bounded", tr.span_s <= ct.WINDOW_S + 0.06,
          "%.2fs over %d samples" % (tr.span_s, tr.n))


def test_short_burst_is_not_confident():
    """Ten samples in 60ms have no baseline - the span gate must catch it
    even though the sample COUNT looks healthy."""
    tr = straight_track(speed=100.0, n=10, dt=0.006)
    ok, why = tr.confident()
    check("many samples over a tiny span are not enough", not ok, why or "")


# ------------------------- the planner -------------------------

ARM_AT = (450.0, 70.0, 680.0)       # the watch pose, where planning starts


def test_intercept_is_ahead_and_reachable():
    tr = crossing_track(n=12, dt=0.05)
    timing = ct.ArmTiming()
    now = tr.t[-1]
    plan, why = ct.plan_intercept(tr, now, ARM_AT, timing)
    check("a clean crossing track produces an intercept",
          plan is not None, why or "")
    if plan is None:
        return
    f = tr.fit()
    here = tr.predict(now, f)
    # projected onto the direction of travel, so this reads the same for
    # a cube crossing as for one going away
    ahead = ((plan["x"] - here[0]) * f["vx"]
             + (plan["y"] - here[1]) * f["vy"]) / f["speed"]
    check("the intercept is AHEAD of the cube, not on it", ahead > 20.0,
          "%.0fmm along its path" % ahead)
    check("the intercept is inside the cell's reach",
          cc.in_reach(plan["x"], plan["y"]),
          "r=%.0f" % np.hypot(plan["x"], plan["y"]))
    check("the arm arrives with the required margin",
          plan["slack_s"] >= cc.ARRIVE_MARGIN_S - 1e-9,
          "%.2fs of slack" % plan["slack_s"])


def test_intercept_respects_reach_test():
    tr = crossing_track(n=12, dt=0.05)
    timing = ct.ArmTiming()
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, timing,
                                  reach_test=lambda x, y, z: False)
    check("an IK refusal everywhere yields no plan", plan is None, "")
    check("...and the reason is about reach",
          "reach" in (why or "") or "wrist" in (why or ""), why or "")


def test_intercept_refuses_when_outrun():
    """A cube the arm cannot get in front of must be refused - not met at
    a point it has already gone past.

    The arm is crippled rather than the cube sped up, so this exercises
    the TIMING branch specifically: every candidate point on this path is
    reachable, so a refusal here can only have come from the clock."""
    tr = crossing_track(n=12, dt=0.05)
    timing = ct.ArmTiming()
    timing.rate = 8.0
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, timing)
    check("a cube the arm cannot beat is refused", plan is None, why or "")
    check("...and the reason blames the clock, not the reach",
          "past me" in (why or ""), why or "")


def test_intercept_needs_confidence():
    tr = crossing_track(n=3, dt=0.05)                 # too few samples
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, ct.ArmTiming())
    check("the planner will not plan from an unconfident fit",
          plan is None, why or "")


def test_intercept_counts_the_descent():
    """The travel estimate must include the drop from the watch height,
    not just the XY reach - under-counting the arm's time is the failure
    that parks the fingers in the cube's path while the arm is still
    moving."""
    tr = crossing_track(n=12, dt=0.05)
    timing = ct.ArmTiming()
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, timing)
    if plan is None:
        check("intercept travel includes the descent", False, why or "no plan")
        return
    flat = float(np.hypot(plan["x"] - ARM_AT[0], plan["y"] - ARM_AT[1]))
    check("intercept travel includes the descent, not just XY",
          plan["travel_mm"] > flat + 50.0,
          "%.0fmm vs %.0fmm flat" % (plan["travel_mm"], flat))


def test_radial_path_is_refused_for_the_right_reason():
    """A cube heading straight away from the arm runs out of cell before
    the arm can get in front of it. That must read as a REACH problem -
    the fix is to send it across, not to send it slower."""
    tr = straight_track(speed=90.0, heading=0.0, n=12, dt=0.05,
                        start=(400.0, 0.0))
    plan, why = ct.plan_intercept(tr, tr.t[-1], ARM_AT, ct.ArmTiming())
    check("a cube heading out of the cell is refused", plan is None,
          why or "")


# ------------------------- arm timing -------------------------

def test_arm_timing_learns_pessimistically():
    t = ct.ArmTiming()
    base = t.estimate(500.0)
    t.record(500.0, base + 1.0)          # a slower move than predicted
    slower = t.estimate(500.0)
    check("a slow move makes the estimate longer", slower > base,
          "%.2fs -> %.2fs" % (base, slower))
    t.record(500.0, 0.5 * base)          # then a fast one
    check("a later FAST move does not make it optimistic again",
          t.estimate(500.0) >= slower - 1e-9, "%.2fs" % t.estimate(500.0))


def test_arm_timing_ignores_tiny_moves():
    t = ct.ArmTiming()
    before = t.rate
    t.record(5.0, 2.0)                   # all ramp, no travel
    check("a tiny move does not teach an absurd rate", t.rate == before,
          "%.0f mm/s" % t.rate)


# ------------------------- grasp choice and widths -------------------------

def test_choose_yaw_prefers_perpendicular():
    # a cube travelling due east: the closing axis should end up near
    # north-south, i.e. 90 away from the heading
    for heading in (0.0, 45.0, 90.0, 137.0, -60.0):
        for yaw in (-40.0, -5.0, 0.0, 12.0, 44.0):
            got = cc.choose_yaw(yaw, heading)
            d = (got + cc.FINGER_AXIS_FROM_YAW_DEG - heading) % 180.0
            if d > 90.0:
                d = 180.0 - d
            if d < 45.0:
                check("choose_yaw picks the perpendicular grip", False,
                      "heading %.0f yaw %.0f -> %.0f (%.0f off)"
                      % (heading, yaw, got, d))
                return
    check("choose_yaw picks the perpendicular grip", True)


def test_choose_yaw_stays_a_valid_grip():
    """Whichever it picks must still be a face-aligned grip: the two
    candidates differ by exactly 90, which a square cube is invariant
    under."""
    bad = []
    for heading in (0.0, 33.0, 91.0, -12.0):
        for yaw in (-30.0, 0.0, 20.0):
            got = cc.choose_yaw(yaw, heading)
            if min(abs(got - yaw), abs(got - yaw - 90.0)) > 1e-9:
                bad.append((yaw, heading, got))
    check("the chosen yaw is always yaw or yaw+90", not bad, str(bad[:2]))


def test_gripper_widths(ref):
    w = 30.0
    op = cc.open_pulses(ref, w)
    cl = cc.close_pulses(ref, w)
    check("the fingers wait wider than they close", op > cl,
          "%d vs %d" % (op, cl))
    check("the waiting gap is the catch's timing budget",
          abs(cc.half_gap_mm(w) - cc.CATCH_OPEN_EXTRA_MM / 2.0) < 1e-9)
    check("it waits wider than either sibling opens",
          cc.CATCH_OPEN_EXTRA_MM > 30.0,
          "%.0fmm vs 30mm" % cc.CATCH_OPEN_EXTRA_MM)
    check("a bigger cube gets a bigger opening",
          cc.open_pulses(ref, 50.0) > op)


def test_max_speed_follows_the_gap():
    """MAX_CATCH_SPEED is not a free parameter - it is the gap divided by
    the timing jitter, halved. If someone widens the fingers, the
    catchable speed must rise with them."""
    expect = 0.5 * (cc.CATCH_OPEN_EXTRA_MM / 2.0) / cc.CLOSE_TIMING_SD_S
    check("MAX_CATCH_SPEED is derived from the finger gap",
          abs(cc.MAX_CATCH_SPEED - expect) < 1e-6,
          "%.0f mm/s" % cc.MAX_CATCH_SPEED)
    check("the catchable band is not empty",
          cc.MAX_CATCH_SPEED > cc.MIN_CATCH_SPEED * 3.0,
          "%.0f..%.0f mm/s" % (cc.MIN_CATCH_SPEED, cc.MAX_CATCH_SPEED))


def main():
    ref, calib = fake_ref(), fake_calib()
    if v3.load_grip_ref() is None:
        print("(no grip_ref.json in vision\\ - using the reference numbers "
              "from the teach; heights below are indicative)\n")

    print("--- geometry ---")
    test_geometry(ref)
    test_aim_matches_v3(calib, ref)
    test_aim_height_is_unclamped(calib, ref)

    print("\n--- the velocity fit ---")
    test_fit_recovers_velocity()
    test_fit_predicts_forward()
    test_noise_is_tolerated()
    test_curved_path_is_refused()
    test_drift_grows_with_the_horizon()
    test_noise_does_not_read_as_a_bend()
    test_speed_gates()
    test_falling_is_refused()
    test_gap_resets_history()
    test_window_trims()
    test_short_burst_is_not_confident()

    print("\n--- the intercept planner ---")
    test_intercept_is_ahead_and_reachable()
    test_intercept_respects_reach_test()
    test_intercept_refuses_when_outrun()
    test_intercept_needs_confidence()
    test_intercept_counts_the_descent()
    test_radial_path_is_refused_for_the_right_reason()
    test_arm_timing_learns_pessimistically()
    test_arm_timing_ignores_tiny_moves()

    print("\n--- the grasp ---")
    test_choose_yaw_prefers_perpendicular()
    test_choose_yaw_stays_a_valid_grip()
    test_gripper_widths(ref)
    test_max_speed_follows_the_gap()

    print("\n%d checks failed." % len(FAILED) if FAILED else "\nAll ok.")
    for f in FAILED:
        print("  - %s" % f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

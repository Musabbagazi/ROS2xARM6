#!/usr/bin/env python3
"""Hand-eye calibration for a fixed camera and a vacuum cup (eye-to-hand).

THE ARM MOVES. Keep the e-stop in hand.

WHAT THIS MEASURES

Twelve numbers - a rotation and a translation - that answer the only
question a fixed camera cannot answer for itself:

    the camera sees a surface HERE; where is that for the arm?

WHY THERE IS NO CHECKERBOARD

The usual eye-to-hand recipe prints a calibration board, waves it about,
and solves AX=XB. It gives you a transform to the FLANGE, which you then
have to compose with a tool offset measured some other way.

There is a better target available here. The cup grips flat surfaces from
directly above, so the calibration target is a flat plate stuck to the
cup, and every pair recorded is

    p_cam  = the camera's measurement of the held plate's top face
    p_base = the FLANGE pose the controller reports at that instant

The plate's top face IS the cup's contact plane. So the fit maps "camera
sees a surface here" straight to "put the flange HERE to put the cup on
it". The flange-to-cup distance is inside the answer, absorbed. It is
never measured, so it can never be measured wrong - and the calibration
is validated by exactly the measurement a pick will use.

WHY A PLATE RATHER THAN A CUBE

Because the cup covers the top of whatever it holds, and the top is the
one surface that matters. On a plate wide enough, the top face stays
visible as a ring all round the cup. See fixed_detect.find_held_target
for why the plate has to be wide compared with the cup's standoff.

WHY IT RUNS TWICE

Pass 1 has no frame to work in yet, so it uses the CENTROID of the
visible coloured surface. That is biased - toward the camera, and away
from the cup's shadow - and the bias CHANGES as the plate moves across
the view. A single pass cannot see this: it fits the bias and reports a
small residual.

Pass 2 re-reads the SAME frames with pass 1's rough transform, which is
enough to say which way is up. In the base frame the plate's outline is
well defined, and a minimum-area rectangle is fixed by the extremes, so
neither the viewing angle nor the shadow in the middle moves its centre.
It costs no extra motion - the frames are already in memory.

WHAT SUCTION REMOVED FROM THIS SCRIPT

The finger version had to ask how tall the calibration cube was, and had
to measure the constant between the base-frame face angle and the
wrist's zero so the fingers could be lined up with a cube's faces. A cup
grips the top and is rotationally symmetric, so neither exists here. One
typed number and one whole measurement step, gone.

Usage:  python fixed_calibrate.py [robot_ip] [--colour red|blue] [--taped]

        --taped  the plate is TAPED to the cup and the cup is never
                 switched on - the tool IO is not touched at all. For a
                 cell whose wrist end-module link is faulty, where any
                 tool-IO transaction latches error 28 and halts motion
                 (see fixed_vacuum.py --watch). The arm drives perfectly
                 while the tool is left alone, and calibration only needs
                 the plate to be rigidly attached - suction is merely the
                 usual way of attaching it. Detachment is then caught by
                 geometry instead of by the vacuum switch: see
                 DETACH_MOVE_MM.

                 One caveat worth knowing: suction squashes the cup's lip
                 slightly, tape does not, so a taped plate can sit a
                 millimetre or two further from the flange than a sucked
                 one. That is a CONSTANT, and it lands entirely in how
                 hard the cup presses at pick time - tune CUP_PRESS_MM if
                 picks press too hard once the wrist is repaired.
"""
import sys
import time

import numpy as np

import fixed_common as fx
import fixed_detect as fd
import vision_common as vc

DEFAULT_IP = "192.168.1.197"

# Where the plate is carried, in the plane. Spread matters more than
# count: a fit from points in a small patch extrapolates badly to the
# corners of the cell.
GRID_X = [370.0, 500.0, 630.0]
GRID_Y = [-240.0, 0.0, 240.0]

# Heights are NOT hard-coded. They are built upward from wherever the arm
# is standing when the script starts, which the operator has jogged to
# just above the floor.
#
# Two reasons, and the second is the one that matters. The obvious one is
# safety: nothing here knows how long the vacuum tool is, so any absolute
# height would be a guess about how close the plate comes to the table.
# The real one is accuracy: cubes are picked just above the floor, and a
# transform fitted only in the upper half of the cell is EXTRAPOLATING
# down to where it is actually used. Starting the grid at the working
# height puts the measurements where the picks are.
GRID_RISE = [0.0, 140.0, 280.0]

# Wrist orientation, held constant. Roll and pitch must stay straight
# down or the flange-to-cup offset stops being purely vertical and the
# trick above breaks. Yaw is held simply because nothing needs it to
# vary - a round cup does not care.
RPY = [180.0, 0.0, 0.0]
CAL_YAW = 0.0

FRAMES_PER_POSE = 6

# How many of those frames are KEPT for pass 2. Not a quality choice, a
# memory one: a frame set is ~2MB, and keeping all six at 27 poses would
# hold ~340MB. Three is enough for a median, and depth goes back to the
# uint16 millimetres the sensor gave in the first place to halve it again.
KEEP_FRAMES = 3

SETTLE_S = 0.35

# A pair further than this from the fitted transform is a mis-detection,
# not a measurement. Dropped and refitted - one bad point drags a
# least-squares fit everywhere.
OUTLIER_MM = 15.0

# Below this the calibration is good. Above it, picks will miss.
GOOD_RMS_MM = 5.0

# ---- proving the plate is still on the cup, WITHOUT the vacuum switch ----
#
# In --taped mode there is no switch to ask, and a plate that has come
# off is the one failure that poisons everything downstream: the camera
# still sees a red plate, the controller still reports a flange pose, and
# every pair from then on is confidently, invisibly wrong.
#
# The replacement needs no calibration at all, because a rigid transform
# PRESERVES DISTANCE. Whatever the camera's position and angle, between
# any two poses the plate must travel exactly as far in the camera's
# frame as the flange travelled in the robot's. A plate lying on the
# floor sits still while the flange moves - which shows up immediately as
# a distance that does not match, and cannot be explained by any camera
# placement.
#
# Two strikes rather than one: a single bad depth median should not end a
# 27-pose run, but two in a row is not noise.
DETACH_MOVE_MM = 30.0       # only judge on moves big enough to be sure of
DETACH_FRAC = 0.4           # the plate must cover at least this much of it
DETACH_STRIKES = 2


def pose_grid(z_low):
    """The grid in snake order, so consecutive poses are neighbours.

    Not cosmetic: movej refuses a move needing more than a 120 degree
    joint sweep, and hopping from one end of the cell to the other is how
    you provoke that."""
    poses = []
    for k, rise in enumerate(GRID_RISE):
        ys = GRID_Y if k % 2 == 0 else GRID_Y[::-1]
        for j, y in enumerate(ys):
            xs = GRID_X if j % 2 == 0 else GRID_X[::-1]
            for x in xs:
                poses.append([x, y, z_low + rise] + RPY[:2] + [CAL_YAW])
    return poses


def collect(arm, cam, poses, colour, use_vacuum=True):
    """Drive the grid, keeping one frame set and one flange reading per
    pose. Returns (samples, skipped, lost_grip)."""
    rays = cam.rays()
    samples, skipped = [], []
    last_good = None            # (flange xyz, plate xyz in camera frame)
    strikes = 0

    for n, pose in enumerate(poses, 1):
        label = "pose %d/%d [%.0f,%.0f,%.0f]" % (n, len(poses), pose[0],
                                                 pose[1], pose[2])
        try:
            vc.moveto(arm, label, pose)
        except vc.SweepRefused as e:
            # A different joint branch, not a fault. Reset through HOME -
            # a configuration every straight-down pose is reachable from -
            # and try this one again before giving up on it.
            print("   %s" % e)
            try:
                vc.movej(arm, "HOME (branch reset)", fx.HOME_DEG)
                vc.moveto(arm, label + " (retry)", pose)
            except RuntimeError as e2:
                skipped.append((pose, str(e2)))
                continue
        except RuntimeError as e:
            skipped.append((pose, str(e)))
            continue

        # If the plate has come off the cup, every reading from here on is
        # of a plate lying on the floor while the flange is somewhere
        # else - a whole set of confidently wrong pairs. Stop instead.
        #
        # Only asked when the cup is actually in use. In --taped mode
        # touching the tool IO is exactly what must not happen, and the
        # distance check below covers the same failure without it.
        if use_vacuum and fx.vacuum_state(arm) != fx.HELD:
            return samples, skipped, True

        time.sleep(SETTLE_S)            # let the wrist stop ringing
        code, actual = arm.get_position(is_radian=False)
        if code != 0:
            skipped.append((pose, "could not read the arm's position"))
            continue

        frames, points = [], []
        for _ in range(FRAMES_PER_POSE):
            depth_mm, color_bgr, _ = cam.frame()
            if len(frames) < KEEP_FRAMES:
                frames.append((depth_mm.astype(np.uint16), color_bgr.copy()))
            got, info = fd.find_held_target(depth_mm, color_bgr, rays, colour)
            if got is not None:
                points.append(got)

        if len(points) < max(2, FRAMES_PER_POSE // 3):
            skipped.append((pose, "the plate was not visible (%d/%d frames)"
                            % (len(points), FRAMES_PER_POSE)))
            continue

        cam_pt = np.median(np.asarray(points), axis=0)
        flange_pt = np.asarray([float(actual[0]), float(actual[1]),
                                float(actual[2])])

        # Did the plate travel as far as the flange did? See
        # DETACH_MOVE_MM - distance is preserved by any rigid transform,
        # so this is true whatever the camera's placement.
        if last_good is not None:
            d_flange = float(np.linalg.norm(flange_pt - last_good[0]))
            d_cam = float(np.linalg.norm(cam_pt - last_good[1]))
            if d_flange >= DETACH_MOVE_MM and d_cam < DETACH_FRAC * d_flange:
                strikes += 1
                print("      ! the flange moved %.0fmm but the plate only "
                      "%.0fmm" % (d_flange, d_cam))
                if strikes >= DETACH_STRIKES:
                    return samples, skipped, True
            else:
                strikes = 0
        last_good = (flange_pt, cam_pt)

        samples.append({
            "flange": [float(actual[0]), float(actual[1]), float(actual[2])],
            "cam_pass1": cam_pt,
            "frames": frames,
        })
        print("      plate seen in %d/%d frames" % (len(points),
                                                    FRAMES_PER_POSE))
    return samples, skipped, False


def refine(samples, T_rough, colour, rays):
    """Pass 2: re-measure every stored frame in the base frame.

    Falls back to the pass-1 centroid where the outline cannot be
    resolved, which is honest - it keeps the pair rather than silently
    dropping a pose, and the outlier pass downstream removes it if it is
    wrong."""
    refined = []
    for s in samples:
        got = []
        for depth_u16, color_bgr in s["frames"]:
            p, _ = fd.find_held_target(depth_u16.astype(np.float32),
                                       color_bgr, rays, colour, T=T_rough)
            if p is not None:
                got.append(p)
        refined.append(np.median(np.asarray(got), axis=0) if got
                       else s["cam_pass1"])
    return refined


def fit_with_outliers(cam_pts, base_pts):
    """Rigid fit, then drop the pairs it cannot explain and fit again."""
    T = fx.fit_rigid(cam_pts, base_pts)
    resid = np.asarray(T["residuals_mm"])
    keep = resid <= max(OUTLIER_MM, 3.0 * T["rms_mm"])
    dropped = int((~keep).sum())
    if dropped and keep.sum() >= 6:
        T = fx.fit_rigid(np.asarray(cam_pts)[keep],
                         np.asarray(base_pts)[keep])
    T["dropped"] = dropped
    return T, keep


def report(T, title, cam_pts=None, base_pts=None):
    print("")
    print("   %s" % title)
    print("      pairs used   %d%s" % (T["n"], "  (%d dropped as outliers)"
                                       % T["dropped"] if T.get("dropped")
                                       else ""))
    print("      RMS error    %.2f mm" % T["rms_mm"])
    print("      worst pair   %.2f mm" % T["max_mm"])
    n = max(int(T["n"]), 3)
    band = 3.0 * 0.11 / (n ** 0.5)      # ~3 sd of the diagnostic's own
    print("      best scale   %.4f  (at n=%d anything within +-%.0f%%"
          % (T["best_scale"], n, band * 100.0))
    print("                   of 1.0 is indistinguishable from noise)")
    if cam_pts is not None and base_pts is not None:
        ratio, pairs = fx.distance_ratio(cam_pts, base_pts)
        if ratio is not None:
            print("      dist ratio   %.4f  over %d pair distances - the"
                  % (ratio, pairs))
            print("                   robust scale check; see")
            print("                   fixed_common.distance_ratio. Off by")
            print("                   more than ~2%% = the camera itself.")


def main():
    args = [a for a in sys.argv[1:]]
    colour = "red"
    if "--colour" in args:
        i = args.index("--colour")
        if i + 1 < len(args):
            colour = args[i + 1].lower()
            del args[i:i + 2]
    taped = "--taped" in args or "--no-vacuum" in args
    positional = [a for a in args if not a.startswith("--")]
    ip = positional[0] if positional else DEFAULT_IP

    fx.start_log("calib")
    print("=" * 62)
    print("   FIXED CAMERA  -  hand-eye calibration (vacuum cup)")
    print("=" * 62)
    print("   THE ARM MOVES. Keep the e-stop in hand.")
    print("")
    if taped:
        print("   TAPED-PLATE MODE: the cup is never switched on and the")
        print("   tool IO is never touched, so a cell whose wrist link is")
        print("   faulty can still be calibrated. The plate is held by")
        print("   tape instead of suction; the arm does not need to grip")
        print("   anything to be measured.")
        print("")
    print("   You need a CALIBRATION PLATE: a flat, rigid, matt %s square,"
          % colour)
    print("   100-150mm across. Card glued to thin ply or plastic is ideal.")
    print("   It must be stiff enough not to sag when held by its middle.")
    print("")
    print("   Before you start:")
    print("     - the camera is on its final, RIGID mount and will not be")
    print("       touched again. Everything below is void if it moves.")
    print("     - the camera and its stand are OUTSIDE the arm's reach.")
    if taped:
        print("     - the plate is TAPED to the cup: centred, and FLAT")
        print("       against the cup's face, exactly where suction would")
        print("       hold it. Double-sided tape on the cup face, or tape")
        print("       from the plate's edges up onto the tool. It must not")
        print("       shift or sag through 27 moves - check by lifting the")
        print("       plate gently by one corner before you start.")
    else:
        print("     - the plate is stuck to the cup, centred, vacuum ON.")
    print("     - NO other %s object is in the camera's view." % colour)
    print("     - the floor is clear.")
    print("     - THE ARM IS ALREADY JOGGED so the plate sits just above")
    print("       the floor (about 20mm). That height becomes the LOWEST")
    print("       calibration level, which is what puts the measurements")
    print("       where the picks actually happen. Use mouse_jog.bat.")
    print("")

    from xarm.wrapper import XArmAPI
    fx.slow_down()
    arm = XArmAPI(ip, is_radian=False)
    cam = fx.Camera()
    try:
        fx.setup_arm(arm, tool_io=not taped)

        code, here = arm.get_position(is_radian=False)
        if code != 0:
            print("   could not read the arm's position.")
            return 1
        z_low = float(here[2])
        poses = pose_grid(z_low)
        print("")
        print("   lowest level = the arm's current height, Z %.1f" % z_low)
        print("   levels       = %s" % [round(z_low + r, 1)
                                        for r in GRID_RISE])
        print("   %d poses at %d deg/s." % (len(poses), fx.PICK_SPEED))

        if taped:
            # Nothing to switch on, and nothing may be read. The plate's
            # attachment is checked by geometry once the arm is moving -
            # see DETACH_MOVE_MM.
            print("")
            print("   The cup stays OFF. The plate is held by its tape,")
            print("   and that it is STILL held is checked from the")
            print("   measurements themselves: between poses the plate")
            print("   must travel as far as the flange does.")
        else:
            # setup_arm deliberately turns the cup OFF, so the plate has
            # to go back on now - and this is also the check that the
            # vacuum works at all before 27 moves are spent finding out
            # it does not.
            print("")
            print("   Turning the vacuum ON to take the plate ...")
            ok, state = fx.vacuum_on(arm)
            if not ok:
                print("   the cup did not report a seal (state: %s)." % state)
                print("   Hold the plate against the cup and try again, or "
                      "check the")
                print("   wiring with 'python fixed_vacuum.py'.")
                return 1
            print("   plate held.")

        try:
            if input("\n   Start? [y/N] ").strip().lower() != "y":
                return 1
        except (EOFError, KeyboardInterrupt):
            return 1

        # Preflight: prove the grid is reachable before moving to any of
        # it, so an unreachable corner is a message rather than a stop
        # half way round.
        ok_poses = [p for p in poses if vc.ik(arm, p) is not None]
        if len(ok_poses) < 8:
            print("   only %d of %d grid poses are reachable from this "
                  "height." % (len(ok_poses), len(poses)))
            print("   Jog the arm nearer the middle of the cell and retry,")
            print("   or edit GRID_X / GRID_Y in this file.")
            return 1
        if len(ok_poses) < len(poses):
            print("   %d of %d poses are out of reach and will be skipped."
                  % (len(poses) - len(ok_poses), len(poses)))

        cam.start()
        print("")
        samples, skipped, lost = collect(arm, cam, ok_poses, colour,
                                         use_vacuum=not taped)
        if lost:
            print("")
            print("   THE PLATE CAME OFF THE CUP part way round, so the run")
            print("   was stopped - readings after that point would pair a")
            print("   plate on the floor with a flange somewhere else.")
            print("   Nothing was saved. %s and run it again."
                  % ("Re-tape it more firmly" if taped else "Check the seal"))
            return 1

        print("")
        print("   %d usable poses, %d skipped." % (len(samples),
                                                   len(skipped)))
        for pose, why in skipped:
            print("      [%.0f,%.0f,%.0f]  %s" % (pose[0], pose[1], pose[2],
                                                  why))
        if len(samples) < 8:
            print("")
            print("   Not enough. A fit needs points spread through the")
            print("   volume; fewer than 8 cannot show whether it is right.")
            print("   Most likely the camera cannot see the plate over most")
            print("   of the cell - check its aim with '3 - Camera View'.")
            return 1

        base_pts = [s["flange"] for s in samples]
        rays = cam.rays()

        T1, _ = fit_with_outliers([s["cam_pass1"] for s in samples],
                                  base_pts)
        report(T1, "pass 1 - visible-surface centroids")

        cam_pts2 = refine(samples, T1, colour, rays)
        T2, _ = fit_with_outliers(cam_pts2, base_pts)
        report(T2, "pass 2 - plate outline centres (this is the one used)",
               cam_pts2, base_pts)

        if T2["rms_mm"] > T1["rms_mm"] + 0.5:
            print("")
            print("   NOTE: pass 2 fitted WORSE than pass 1. That usually")
            print("   means the plate's outline is poorly resolved - a very")
            print("   steep viewing angle, or a plate too small for the")
            print("   cup's shadow. Pass 2 is still the one saved, because")
            print("   pass 1's error is a bias that varies across the cell")
            print("   and this one is noise.")

        fx.save_handeye({
            "R": T2["R"], "t": T2["t"],
            "rms_mm": T2["rms_mm"], "max_mm": T2["max_mm"],
            "best_scale": T2["best_scale"], "n": T2["n"],
            "residuals_mm": T2["residuals_mm"],
            "pass1_rms_mm": T1["rms_mm"],
            "z_low": z_low,
            "tool": "vacuum cup",
            "made": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("Maps a camera-frame point to the FLANGE pose that "
                     "puts the CUP on it. Z is offset from the true base "
                     "frame by the flange-to-cup distance, deliberately - "
                     "see fixed_common, 'the grasp frame'."),
        })

        print("")
        if T2["rms_mm"] <= GOOD_RMS_MM:
            print("   saved %s" % fx.HANDEYE_FILE)
            print("   Next: take the plate off, lay paper over the floor,")
            print("   and run '5 - Teach Floor'.")
        else:
            print("   saved %s  -  BUT %.1fmm RMS is too loose to pick with."
                  % (fx.HANDEYE_FILE, T2["rms_mm"]))
            print("   Likely causes, in the order worth checking:")
            print("     - the plate slipped on the cup part way round;")
            print("     - the camera or its stand moved during the run;")
            print("     - the plate is at a grazing angle from the camera")
            print("       over much of the cell - depth is poor there;")
            print("     - the plate is too small, so the cup's shadow eats")
            print("       its far edge;")
            print("     - something else %s was in view." % colour)
            print("   Per-pair errors: %s" % T2["residuals_mm"])

        vc.movej(arm, "HOME", fx.HOME_DEG)
        print("")
        if taped:
            print("   Peel the plate off the cup by hand - the cup was")
            print("   never switched on, so there is nothing to release.")
        else:
            print("   Releasing the plate - HOLD IT.")
            fx.vacuum_off(arm)
        return 0 if T2["rms_mm"] <= GOOD_RMS_MM else 1
    except KeyboardInterrupt:
        print("\n   stopped by the operator.")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n   FAILED: %s" % e)
        return 1
    finally:
        cam.close()
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

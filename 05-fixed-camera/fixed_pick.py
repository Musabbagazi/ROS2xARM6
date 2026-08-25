#!/usr/bin/env python3
"""Pick cubes with a fixed camera and a vacuum cup. THE ARM MOVES.

This is the stationary-cube picker: the first half of the project. The
cubes are sitting still; what is "real time" about it is that the camera
sees the WHOLE cell continuously, so there is no scan pose, no search
pattern, and no waiting for anything to settle. The arm looks once, is
told where every cube is at once, and goes.

WHAT VANISHED COMPARED WITH vision\\vision_pick3

  the SCAN pose            the camera is not on the wrist; there is
                           nothing to point at anything
  the search pattern       15 vantages existed to cover a cell one
                           wrist-camera view could not. One fixed camera
                           covers it in one frame
  track_until_still        that gate was there because the arm had to
                           commit blind; here nothing is committed blind
  the mid-descent look     same reason. The camera watches the descent
                           from outside it
  the cube-flip bugs       four separate hardware bugs came from the
                           detector re-choosing which cube it meant
                           between frames. Cubes now have positions in a
                           fixed frame, so "the same cube" is a distance,
                           not a guess
  the grasp yaw            a round cup does not care which way a cube is
                           turned. The wrist holds one yaw all run
  the gripper widths       nothing opens or closes to fit a cube

WHAT IS NEW, AND NEITHER IS SMALL

The camera can be knocked. See fixed_common.drift_check - it runs before
the first move and the run stops if the scene has shifted.

A cup can lose its seal WHILE CARRYING, which fingers do not do. The
vacuum switch is therefore re-read after the lift, not only at the pick.

Usage:  python fixed_pick.py [robot_ip] [--dry]
        --dry  looks, reports every pose it WOULD command, and moves
               nothing. Run this first, every time the cell changes.
"""
import os
import sys

import numpy as np

import fixed_common as fx
import fixed_detect as fd
import vision_common as vc

DEFAULT_IP = "192.168.1.197"

# Straight down, above the cube, before descending on it. There is no
# blind zone to clear - the approach exists so the cup comes down
# vertically onto the face rather than sliding in across it.
APPROACH_H = 80.0
LIFT_H = 130.0
PLACE_APPROACH = 60.0

# The wrist yaw for every pose in the run. A round cup on a flat face is
# rotationally symmetric, so this never has to change - which also means
# one less joint to swing and one less way for a move to be refused.
PICK_YAW = 0.0

# WHERE CUBES GO. Base-frame XY only - no height, and no taught pose.
#
# The height is DERIVED from the taught floor, and that is not laziness,
# it is a safety property. A drop pose taught with the two-finger gripper
# (which is what vision\places.json holds) carries that tool's length
# inside its Z, so reusing it with a cup of a different length would
# release the cube at the wrong height - possibly by driving into the
# table. Deriving from the floor cannot make that mistake, and it adapts
# to each cube's height for free.
#
# Both spots are deliberately OUTSIDE the camera's work box, so delivered
# cubes are not seen again and re-picked.
PLACES = {"red": [520.0, -430.0], "blue": [520.0, 430.0]}

# How far above the floor the cube's BOTTOM is when the vacuum is cut.
RELEASE_CLEAR_MM = 5.0

# WHAT GETS PICKED, AND IN WHAT ORDER.
#
# Now BLUE ONLY. Red cubes are seen, measured and deliberately left
# alone - they are not picked last, they are not picked at all.
#
# The difference matters for what the run does when it finds nothing:
# with red still in the list, "no blue left" moved on to red and the
# cell emptied the table. With blue alone, "no blue left" means the job
# is done and the arm stops, however many red cubes are lying there.
#
# Restore  [("red", "RED"), ("blue", "BLUE"), (None, "any remaining")]
# to go back to emptying the table, or pass --colour red|blue|any.
PICK_ORDER = [("blue", "BLUE")]

MAX_CYCLES = 14
MAX_RETRIES = 2

# Detections this far apart across frames are different cubes.
SAME_CUBE_MM = 25.0
# A cube found unpickable is remembered within this radius so the run
# moves on to the others instead of failing on it forever.
AVOID_R_MM = 60.0

FRAMES = 5
MIN_FOUND = 3


# ------------------------- stable detection -------------------------

def stable(cam, rays, T, floor, tcp_xy=None, n=FRAMES, min_found=MIN_FOUND):
    """Cubes seen in at least min_found of n frames, medianed together.

    Clustering ACROSS frames by position is what makes this robust, and
    it is only possible because the positions are in a fixed frame. The
    wrist-mounted pipeline had to median pixels, which meant every arm
    movement contaminated the measurement and a detector that flipped
    between two similar cubes produced a median halfway between them."""
    pool = []
    for _ in range(n):
        depth_mm, color_bgr, _ = cam.frame()
        cubes, _ = fd.detect_all(depth_mm, color_bgr, rays, T, floor,
                                 tcp_xy=tcp_xy)
        pool.extend(cubes)

    clusters = []
    for c in pool:
        for group in clusters:
            ref = group[0]["center"]
            if np.hypot(c["center"][0] - ref[0],
                        c["center"][1] - ref[1]) < SAME_CUBE_MM:
                group.append(c)
                break
        else:
            clusters.append([c])

    out = []
    for group in clusters:
        if len(group) < min_found:
            continue
        colours = [c["color"] for c in group]
        red, blue = colours.count("red"), colours.count("blue")
        if max(red, blue) == 0:
            colour = "unknown"
        else:
            colour = "red" if red >= blue else "blue"

        def med(key):
            return float(np.median([c[key] for c in group]))

        out.append({
            "center": [float(np.median([c["center"][0] for c in group])),
                       float(np.median([c["center"][1] for c in group]))],
            "top_z": med("top_z"),
            "width_mm": med("width_mm"),
            "short_mm": med("short_mm"),
            "height_mm": med("height_mm"),
            "flat_mm": med("flat_mm"),
            "tilt_deg": med("tilt_deg"),
            "color": colour,
            "seen": len(group),
        })
    out.sort(key=lambda c: -c["seen"])
    return out


def not_avoided(cubes, avoid):
    keep = []
    for c in cubes:
        if any(np.hypot(c["center"][0] - a[0], c["center"][1] - a[1])
               < AVOID_R_MM for a in avoid):
            continue
        keep.append(c)
    return keep


# ------------------------- delivery -------------------------

def load_places():
    """Drop spots as base-frame XY, from places_xy.json if it exists.

    Deliberately NOT vision\\places.json: those are full 6-value poses
    taught with the two-finger gripper, so their Z belongs to that tool's
    length. Reusing them here would release cubes at the wrong height."""
    data = fx._load(os.path.join(fx.HERE, "places_xy.json"))
    places = dict(PLACES)
    if isinstance(data, dict):
        for key in ("red", "blue"):
            xy = data.get(key)
            if isinstance(xy, list) and len(xy) == 2:
                try:
                    places[key] = [float(xy[0]), float(xy[1])]
                except (TypeError, ValueError):
                    pass
    return places


def drop_pose(places, floor, colour, height_mm):
    """(pose, label) for releasing a cube of this height.

    The release height is the taught floor plus the cube's own height
    plus a small clearance, so the cube's underside is just above the
    table when the vacuum is cut, whatever size it is."""
    key = colour if colour in places else "red"
    x, y = places[key]
    z = float(fx.plane_z(floor["coef"], x, y)) + height_mm + \
        RELEASE_CLEAR_MM
    label = "%s spot" % key
    if colour not in places:
        label = "%s spot (no %s spot set)" % (key, colour)
    return [x, y, z, 180.0, 0.0, PICK_YAW], label


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


# ------------------------- one cube -------------------------

def pick_one(arm, cam, rays, T, floor, places, want, avoid, dry=False):
    """'picked', 'skip', 'retry' or 'empty'."""
    code, here = arm.get_position(is_radian=False)
    tcp_xy = (here[0], here[1]) if code == 0 else None

    cubes = not_avoided(stable(cam, rays, T, floor, tcp_xy=tcp_xy), avoid)
    cube = fd.pick_one(cubes, want_color=want)
    if cube is None:
        if cubes:
            print("   %d cube(s) visible, none of them %s."
                  % (len(cubes), want))
        return "empty"

    gx, gy = cube["center"]
    gz = fx.grasp_z(cube, floor["coef"])
    grasp = [gx, gy, gz, 180.0, 0.0, PICK_YAW]
    approach = above(grasp, APPROACH_H)
    lift = above(grasp, LIFT_H)
    drop, where = drop_pose(places, floor, cube["color"], cube["height_mm"])

    print("%s cube %.0fmm (h %.0fmm, top flat to %.1fmm, seen %d/%d) at "
          "[%.1f, %.1f] -> cup at z %.1f  (deliver to %s)"
          % (cube["color"], cube["width_mm"], cube["height_mm"],
             cube["flat_mm"], cube["seen"], FRAMES, gx, gy, gz, where))

    if not fx.in_reach(gx, gy):
        print("   skipped: %s" % fx.out_of_reach_hint(gx, gy))
        avoid.append((gx, gy))
        return "skip"

    # Preflight every pose before committing to any of them. An
    # unreachable delivery discovered while carrying a cube is a much
    # worse problem than one discovered now.
    plan = [("approach", approach), ("grasp", grasp), ("lift", lift),
            ("above the drop", above(drop, PLACE_APPROACH)),
            ("the drop", drop)]
    angles = {}
    for label, pose in plan:
        a = vc.ik(arm, pose)
        if a is None:
            print("   skipped: the %s pose is not reachable." % label)
            avoid.append((gx, gy))
            return "skip"
        angles[label] = a

    # Sweep preflight for the DELIVERY only. IK proves a pose exists; it
    # does not prove the arm can swing there inside the guard, and
    # finding that out mid-delivery means faulting while carrying.
    swing = max(abs(a - b) for a, b in zip(angles["lift"], angles["the drop"]))
    if swing > vc.MAX_JOINT_SWEEP:
        print("   skipped: delivering it would need a %.0f deg swing. Move "
              "the drop\n            spot nearer the middle of the cell "
              "(places_xy.json)." % swing)
        avoid.append((gx, gy))
        return "skip"

    if dry:
        print("   [dry] approach %s" % [round(v, 1) for v in approach])
        print("   [dry] cup down %s" % [round(v, 1) for v in grasp])
        print("   [dry] release  %s" % [round(v, 1) for v in drop])
        avoid.append((gx, gy))          # so --dry walks the whole floor
        return "skip"

    def go(label, pose):
        """Pre-grasp move that gives up on this CUBE, not on the run.

        A refused sweep is a property of which joint branch the arm is
        standing in, not a fault - the next cube is very likely fine.
        Only used before anything is held; moves made while carrying a
        cube stay fatal, because dropping it somewhere unplanned is
        worse than stopping."""
        try:
            vc.moveto(arm, label, pose)
            return True
        except vc.SweepRefused as e:
            print("   %s" % e)
            avoid.append((gx, gy))
            return False

    if not go("approach", approach):
        return "skip"
    if not go("cup down onto the cube", grasp):
        return "skip"

    held, state = fx.vacuum_on(arm)
    if not held:
        print("   no seal (%s). Backing off." % state)
        fx.vacuum_off(arm, settle=False)
        vc.moveto(arm, "retreat", approach)
        return "retry"
    print("   sealed.")

    vc.movej(arm, "lift", angles["lift"])

    # A cup can let go on the way up; fingers cannot. Ask again before
    # setting off across the cell with something that may already be
    # back on the floor.
    if fx.vacuum_state(arm) != fx.HELD:
        print("   the seal was lost during the lift - the cube is back on "
              "the floor.")
        fx.vacuum_off(arm, settle=False)
        return "retry"

    vc.movej(arm, "above the drop", angles["above the drop"])
    vc.movej(arm, "the drop", angles["the drop"])
    fx.vacuum_off(arm)
    vc.movej(arm, "retreat", angles["above the drop"])
    print("%s cube delivered to %s." % (cube["color"], where))
    return "picked"


# ------------------------- the run -------------------------

def main():
    args = sys.argv[1:]
    dry = "--dry" in args
    order = PICK_ORDER
    if "--colour" in args:
        i = args.index("--colour")
        if i + 1 < len(args):
            want = args[i + 1].lower()
            if want in ("any", "all"):
                order = [(None, "ANY")]
            elif want in ("red", "blue"):
                order = [(want, want.upper())]
            else:
                print("   --colour takes red, blue or any.")
                return 1
            del args[i:i + 2]
    positional = [a for a in args if not a.startswith("--")]
    ip = positional[0] if positional else DEFAULT_IP

    fx.start_log("pick")
    print("=" * 62)
    print("   FIXED CAMERA  -  pick (stationary cubes, vacuum cup)")
    print("=" * 62)
    print("   Picking %s. Anything else is seen and left alone."
          % " then ".join(lbl for _, lbl in order))

    T = fx.load_handeye()
    floor = fx.load_floor()
    if T is None or floor is None:
        print("   Not set up yet:")
        if T is None:
            print("     handeye.json   missing -> run '4 - Calibrate Camera'")
        if floor is None:
            print("     floor_base.json missing -> run '5 - Teach Floor'")
        return 1

    places = load_places()
    print("   calibration %.1fmm RMS on %d points, %s"
          % (T.get("rms_mm", -1), T.get("n", 0),
             T.get("made", "date unknown")))
    print("   surface at %.1f, %s tilt."
          % (floor["height_at_centre"],
             "measured" if floor.get("tilt_fitted") else "assumed level"))
    print("   cup %.0fmm, pressed %.1fmm into the face; drop spots red %s "
          "blue %s." % (fx.CUP_DIA_MM, fx.CUP_PRESS_MM,
                        places["red"], places["blue"]))
    if dry:
        print("   DRY RUN - the arm will not move.")
    else:
        print("   THE ARM MOVES at %d deg/s. Keep the e-stop in hand."
              % fx.PICK_SPEED)

    from xarm.wrapper import XArmAPI
    if not dry:
        fx.slow_down()
    arm = XArmAPI(ip, is_radian=False)
    # temporal=False because the cubes MOVE. The temporal filter averages
    # each pixel's depth over recent frames, which is free accuracy on a
    # still scene and a lie on a moving one: it smears a travelling cube
    # along its own path, pulling the measured centre backwards toward
    # where it used to be. On a static table it is worth having; here it
    # would quietly bias every grasp in the direction of travel.
    cam = fx.Camera(temporal=False)
    picked = 0
    try:
        if dry:
            # A dry run really does not move, so it does not enable
            # motion or touch the cup either. It still needs a live,
            # fault-free controller: IK is answered by the controller,
            # and a latched fault is worth knowing about now.
            arm.clean_warn()
            arm.clean_error()
            code, ew = arm.get_err_warn_code()
            if code != 0 or ew[0] != 0:
                raise RuntimeError("arm is in error %s - run "
                                   "'0 - RESET EVERYTHING'" % ew)
        else:
            fx.setup_arm(arm)
        cam.start()
        rays = cam.rays()

        # Has the camera been knocked since it was calibrated? Asked
        # before the first move, because after the first move it is too
        # late to ask.
        depth_mm, _, _ = cam.frame()
        base = fx.apply_transform(T, fx.cloud_cam(depth_mm, rays))
        status, why = fx.drift_check(base[(depth_mm > 0) &
                                          fx.in_work_box(base)], floor)
        print("   drift check: %s" % why)
        if status == "fail":
            return 1

        if not dry:
            try:
                if input("\n   Start? [y/N] ").strip().lower() != "y":
                    return 1
            except (EOFError, KeyboardInterrupt):
                return 1
            vc.moveto(arm, "standoff", fx.STANDOFF_POSE)

        avoid = []
        phase = 0
        for cycle in range(1, MAX_CYCLES + 1):
            want, label = order[phase]
            print("")
            print("--- cycle %d (%s) ---" % (cycle, label))

            tries = 0
            while True:
                result = pick_one(arm, cam, rays, T, floor, places, want,
                                  avoid, dry=dry)
                if result != "retry":
                    break
                tries += 1
                if tries >= MAX_RETRIES:
                    print("   giving up on that cube for this run.")
                    break

            if result == "picked":
                picked += 1
            elif result == "empty":
                if phase + 1 < len(order):
                    phase += 1
                    print(">>> no %s cubes left - hunting %s now"
                          % (label, order[phase][1]))
                    continue
                print("No %s cubes left. Anything else on the floor is "
                      "left where it is." % label)
                break

        if not dry:
            fx.vacuum_off(arm, settle=False)
            vc.movej(arm, "HOME", fx.HOME_DEG)
        print("")
        print("Done. %s %d cube(s)." % ("Would have picked" if dry
                                        else "Picked", picked))
        return 0
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

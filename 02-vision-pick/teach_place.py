#!/usr/bin/env python3
"""Teach the colour-sorted drop spots for vision_pick3 - saved to
places.json, which the picker reads (no code editing).

The BLUE drop spot is required for colour sorting; the RED spot defaults
to the existing PLACE and only needs teaching if you want to move it.
The arm goes to a safe height above the current red spot, then you
WASD-jog the (empty) gripper to where a cube should be released and
press ENTER. Keep the e-stop in hand.

Usage: python teach_place.py [robot_ip]
"""
import sys

from xarm.wrapper import XArmAPI

import vision_common as vc
import vision3 as v3
import vision_pick3 as vp

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

APPROACH = 80.0


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


def teach_one(arm, name, start_pose):
    print("\n=== TEACH %s DROP ===" % name.upper())
    vc.moveto(arm, "safe height above the drop area",
              above(start_pose, APPROACH))
    print("Drive the gripper to where a %s cube should be released, then "
          "press ENTER.\n(The gripper can stay open/empty - only the pose "
          "is saved.)" % name)
    final, _ = vc.wasd_jog(arm, title="TEACH %s DROP" % name.upper())
    if final is None or len(final) != 6:
        raise RuntimeError("could not read the taught pose")
    print("%s drop pose: %s" % (name, [round(v, 1) for v in final]))
    if input("Save this as the %s drop spot? [y/N] " % name
             ).strip().lower() != "y":
        print("Not saved.")
        return None
    return [round(float(v), 1) for v in final]


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
    except RuntimeError as e:
        print("ABORT:", e)
        arm.disconnect()
        return

    places = v3.load_places() or {}
    # the red spot defaults to the picker's built-in PLACE_RED
    red_start = places.get("red") or vp.PLACE_RED

    print("\nWhich drop spot do you want to teach?")
    print("  [b] blue only (default)   [r] red only   [a] both")
    choice = input("  b/r/a > ").strip().lower() or "b"

    try:
        if choice in ("r", "a"):
            taught = teach_one(arm, "red", red_start)
            if taught is not None:
                places["red"] = taught
                red_start = taught
        if choice in ("b", "a"):
            taught = teach_one(arm, "blue", red_start)
            if taught is not None:
                places["blue"] = taught
        if not places:
            print("\nNothing taught.")
        else:
            v3.save_places(places)
            print("\n===== places.json saved =====")
            for k in ("red", "blue"):
                if k in places:
                    print("  %-4s -> %s" % (k, places[k]))
            if "blue" not in places:
                print("  blue -> (not set; blue cubes go to the red spot)")
            print("Ready: run vision_pick3.bat")
    except RuntimeError as e:
        print("\nSTOPPED:", e)
    finally:
        try:
            arm.set_mode(0)
            arm.set_state(0)
        except Exception:
            pass
        arm.disconnect()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Free WASD drive for the xArm6, with action recording.

Drive the arm with single keypresses; press SPACE at any pose to record
it. When you finish (ENTER) all recorded poses are printed and saved to
recorded_poses.json next to this script.

Usage: python jog_arm.py [robot_ip]
Keep the e-stop in hand.
"""
import json
import os
import sys
import time

from xarm.wrapper import XArmAPI

import vision_common as vc

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "recorded_poses.json")


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
    except RuntimeError as e:
        print("ABORT:", e)
        arm.disconnect()
        return

    final, waypoints = vc.wasd_jog(arm, title="FREE JOG")
    arm.disconnect()

    print("\nFinal pose:", final)
    if waypoints:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        data = {"recorded": stamp, "poses": waypoints}
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2)
        print("%d waypoint(s) saved to %s" % (len(waypoints), OUT))
        for i, w in enumerate(waypoints, 1):
            print("  pose_%d = %s" % (i, w))
    else:
        print("(no waypoints recorded - press SPACE while driving to "
              "record)")


if __name__ == "__main__":
    main()

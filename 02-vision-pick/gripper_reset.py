#!/usr/bin/env python3
"""Recover a gripper that is stuck / still clamped on a cube.

Diagnoses first (position, error code, status) and prints it, then - only
after you confirm - clears the fault, re-enables the gripper and opens it.
The arm itself is never moved.

Usage: python gripper_reset.py [robot_ip]
Keep the e-stop in hand: the fingers WILL move when you confirm.
"""
import sys
import time

from xarm.wrapper import XArmAPI

import vision_common as vc

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"


def report(arm, when):
    code, ew = arm.get_err_warn_code()
    print("%s:" % when)
    print("   controller  err=%s warn=%s (read code %s)"
          % (ew[0], ew[1], code))
    print("   gripper     pos=%s err=%s status=%s"
          % (vc.gripper_pos(arm), arm.get_gripper_err_code()[1],
             arm.get_gripper_status()[1]))


def main():
    vc.start_log("gripper_reset")
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    time.sleep(0.5)
    try:
        report(arm, "Before")
        pos = vc.gripper_pos(arm)
        if pos is not None and pos > 780:
            print("\nThe gripper is already open - nothing to reset.")
            return
        print("\nIf a cube is clamped between the fingers, take hold of it "
              "now:\nit will drop when the gripper opens.")
        if input("\nClear the fault, re-enable and OPEN the gripper? [y/N] "
                 ).strip().lower() != "y":
            print("Aborted - nothing was moved.")
            return

        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        if not vc.enable_gripper(arm):
            print("!! the gripper refused to enable - check its cable and "
                  "the 24V supply on the end-effector.")
        try:
            vc.grip(arm, "open", 850)
        except RuntimeError as e:
            print("\nFAILED:", e)
            report(arm, "After")
            print("\nThe fingers are jammed mechanically or the gripper has "
                  "lost power.\nPower-cycle the controller, then run this "
                  "again before '4 - Vision Pick'.")
            return
        report(arm, "After")
        print("\nGripper reset. You can run '4 - Vision Pick' now.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

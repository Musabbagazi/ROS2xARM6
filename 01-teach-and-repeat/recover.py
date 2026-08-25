#!/usr/bin/env python3
"""Recover the xArm 6 after a stopped / faulted run.

Clears error+warning, re-enables the arm, and (optionally) opens the
gripper to release a box that got stuck in mid-air.

Usage: python3 ~/recover.py [robot_ip]
Keep the E-STOP in hand.
"""
import sys
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP)

    code, ew = arm.get_err_warn_code()
    print("Current err/warn:", ew)

    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(2000)
    arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])  # gripper weight
    arm.set_state(0)

    code, ew = arm.get_err_warn_code()
    print("After clearing err/warn:", ew, " (want [0, 0])")
    code, pos = arm.get_position(is_radian=False)
    print("Arm is at:", [round(v, 1) for v in pos] if code == 0 else code)

    print("\nThe gripper may still be holding the box in mid-air.")
    ans = input("HOLD THE BOX, then press y to OPEN the gripper: ").strip().lower()
    if ans == "y":
        arm.set_gripper_position(850, wait=True)
        print("Gripper opened.")
    else:
        print("Left the gripper as it was.")

    print("\nFault cleared. You can now run go-home or the fixed pick-and-place.")
    arm.disconnect()


if __name__ == "__main__":
    main()

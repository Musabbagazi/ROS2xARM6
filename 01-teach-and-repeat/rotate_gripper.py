#!/usr/bin/env python3
"""Rotate ONLY the gripper (joint 6) - the rest of the arm stays put.

Type an amount in degrees to rotate the gripper by that much:
    +10   rotate 10 degrees one way
    -5    rotate 5 degrees the other way
    0     show current angle without moving
    q     quit

Usage: python3 ~/rotate_gripper.py [robot_ip]
Keep the e-stop in hand.
"""
import sys
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
SPEED = 20  # deg/s, slow


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
    arm.set_state(0)

    code, joints = arm.get_servo_angle(is_radian=False)
    if code != 0:
        print("Could not read joints (code %s)" % code)
        arm.disconnect()
        return
    print("Gripper rotation (joint 6) is now: %.1f deg" % joints[5])
    print("Type +N / -N to rotate by N degrees, q to quit.\n")

    while True:
        cmd = input("rotate> ").strip().lower()
        if cmd == "q":
            break
        try:
            delta = float(cmd)
        except ValueError:
            print("  Type a number like +10 or -5, or q to quit")
            continue
        code, joints = arm.get_servo_angle(is_radian=False)
        if code != 0:
            print("  read failed (code %s)" % code)
            continue
        target = list(joints[:6])
        new_j6 = target[5] + delta
        if not -360.0 <= new_j6 <= 360.0:
            print("  that would pass the joint limit (+/-360), refusing")
            continue
        target[5] = new_j6
        code = arm.set_servo_angle(angle=target, speed=SPEED, wait=True,
                                   is_radian=False)
        if code != 0:
            print("  move failed (code %s)" % code)
            continue
        code, joints = arm.get_servo_angle(is_radian=False)
        print("  gripper now at %.1f deg" % joints[5])

    arm.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()

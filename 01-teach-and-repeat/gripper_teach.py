#!/usr/bin/env python3
"""Teach the GRIP tightness for the UFACTORY 2-finger gripper.

This gripper is POSITION-controlled (0 = closed .. 850 = open), NOT
force-controlled. "Tightness" = how far you tell it to close against the
box. This helper lets you find the right number for YOUR box, then prints
the GRIP_CLOSED value to paste into pick_and_place.py.

Usage: python3 ~/gripper_teach.py [robot_ip]
Keep the E-STOP in hand. For a FRAGILE box, skip 'm' (auto-measure pushes
firmly for a moment) and just try numbers manually.
"""
import sys
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
OPEN = 850


def read_pos(arm):
    code, pos = arm.get_gripper_position()
    return pos if code == 0 else None


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(2000)

    print("Opening gripper...")
    arm.set_gripper_position(OPEN, wait=True)
    print("\nPut the box between the fingers.\n")
    print("Commands:")
    print("   <number 0..850>  move fingers to that position")
    print("   m                auto-measure the box, suggest a value")
    print("   o                open fully")
    print("   q                quit\n")

    grip = None
    while True:
        cmd = input("grip> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "o":
            arm.set_gripper_position(OPEN, wait=True)
        elif cmd == "m":
            print("  Gently closing on the box to measure it...")
            arm.set_gripper_position(0, wait=True)   # stalls at the box
            pos = read_pos(arm)
            if pos is None:
                print("  Could not read position."); continue
            rec = max(0, int(pos) - 40)   # a bit tighter than 'just touching'
            print("  Fingers stopped at ~%s  (that's the box width)" % pos)
            print("  Suggested GRIP_CLOSED = %d" % rec)
            grip = rec
            arm.set_gripper_position(OPEN, wait=True)
        else:
            try:
                v = max(0, min(850, int(float(cmd))))
            except ValueError:
                print("  Type a number 0..850, or  m / o / q"); continue
            arm.set_gripper_position(v, wait=True)
            pos = read_pos(arm)
            print("  Commanded %d, fingers now at ~%s. Does it hold firmly?"
                  % (v, pos))
            grip = v

    arm.set_gripper_position(OPEN, wait=True)
    arm.disconnect()
    if grip is not None:
        print("\n===== Paste into pick_and_place.py =====")
        print("GRIP_CLOSED = %d" % grip)


if __name__ == "__main__":
    main()

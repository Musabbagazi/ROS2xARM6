#!/usr/bin/env python3
"""Teach helper for the physical xArm 6.

Puts the arm in HAND-GUIDE (manual) mode so you can physically push it to a
pose, then press Enter to print that pose. Use it to capture the PICK and
PLACE coordinates you paste into pick_and_place.py.

Usage: python3 ~/xarm6_teach.py [robot_ip]
Keep the e-stop in hand. The arm goes limp in manual mode - support it.
"""
import sys
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    # Declare the gripper's weight so gravity compensation is right in
    # hand-guide mode and the arm doesn't false-detect collisions.
    arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
    arm.set_state(0)

    input("\nAbout to enable HAND-GUIDE mode. Hold the arm, then press Enter...")
    arm.set_mode(2)      # 2 = manual / free-drive
    arm.set_state(0)
    print("Manual mode ON. Push the arm by hand to a pose.")
    print("Press Enter to record a pose. Type q then Enter to finish.\n")

    recorded = []
    while True:
        cmd = input("[Enter]=record  [q]=quit > ").strip().lower()
        if cmd == "q":
            break
        code, pos = arm.get_position(is_radian=False)  # [x,y,z,roll,pitch,yaw]
        code2, angs = arm.get_servo_angle(is_radian=False)  # 6 joint degrees
        if code == 0:
            recorded.append(pos)
            print("  Cartesian [x,y,z,roll,pitch,yaw] mm/deg:")
            print("    [%s]" % ", ".join("%.1f" % v for v in pos))
            print("  Joints (deg): [%s]\n" % ", ".join("%.2f" % v for v in angs))
        else:
            print("  get_position failed, code:", code)

    # Restore normal position mode before leaving.
    arm.set_mode(0)
    arm.set_state(0)
    arm.disconnect()

    if recorded:
        print("\n===== Copy these into pick_and_place.py =====")
        for i, p in enumerate(recorded, 1):
            print("pose_%d = [%s]" % (i, ", ".join("%.1f" % v for v in p)))


if __name__ == "__main__":
    main()

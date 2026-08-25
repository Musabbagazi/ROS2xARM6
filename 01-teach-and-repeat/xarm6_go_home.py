#!/usr/bin/env python3
"""Move the physical xArm 6 back to the saved HOME position.
Usage: python3 ~/xarm6_go_home.py [robot_ip]
Keep the e-stop in hand. Answer 'y' only when the path is clear.
"""
import sys, math
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
HOME_RAD = [
    0.07306542247533798,   # joint1
    -0.2995672821998596,   # joint2
    -1.7921593189239502,   # joint3
    -0.008517428301274776, # joint4
    2.102301597595215,     # joint5
    0.0,                   # joint6 (gripper set to 0 deg)
]
HOME_DEG = [math.degrees(a) for a in HOME_RAD]

def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    print("Current joints (deg):", arm.angles)
    print("Target  HOME  (deg):", [round(a, 3) for a in HOME_DEG])
    ans = input("Move to HOME now? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted."); arm.disconnect(); return
    code = arm.set_servo_angle(angle=HOME_DEG, speed=20, wait=True, is_radian=False)
    print("set_servo_angle return code:", code, "(0 = success)")
    print("Final joints (deg):", arm.angles)
    arm.disconnect()

if __name__ == "__main__":
    main()

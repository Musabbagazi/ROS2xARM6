#!/usr/bin/env python3
"""Keyboard jog for the fixed-camera cell. THE TOOL IS NEVER TOUCHED.

  W / S   reach further out / back toward the base   (base +X / -X)
  A / D   left / right                               (base +Y / -Y)
  R / F   up / down                                  (base +Z / -Z)
  1..5    step size 0.5 / 1 / 2 / 5 / 10 mm per tick
  SPACE   print the pose
  ESC     quit

WHY THIS EXISTS RATHER THAN mouse_jog.py

mouse_jog calls vision_common.setup_arm, which enables the two-finger
MODBUS GRIPPER. On this cell that is wrong twice over: there is a vacuum
cup on the wrist so the enable is refused and it raises, and - the one
that actually bites - talking to the tool at all latches error 28 while
the wrist's end-module link is faulty, which halts motion instantly. Its
O/P keys then poll the gripper every tick, so jogging would fault
continuously.

This one touches nothing on the tool. The arm drives perfectly while the
tool is left alone (measured: clean for 8s with servos enabled, and
through a full grid of IK questions), so jogging works normally even
with the wrist fault unrepaired.

WHAT IT IS FOR

Getting the arm to the height the calibration starts from. The grid is
built UPWARD from wherever the arm is standing, so the plate wants to be
about 20mm above the floor before '4b - Calibrate Camera (TAPED plate)'
is run. Anything higher and the fit is extrapolating down to where the
picks actually happen - and, as the preflight showed, most of the grid
stops being reachable.

Motion is IK-guarded joint motion, the same as everywhere else in the
pipeline: each step becomes a cartesian target, goes through inverse
kinematics, is sweep-checked, then streamed as a non-blocking joint move.
The arm moves only while a key is HELD.

Usage:  python fixed_jog.py [robot_ip]
Keep the e-stop in hand.
"""
import ctypes
import sys
import time

import fixed_common as fx
import vision_common as vc

DEFAULT_IP = "192.168.1.197"

RATE_HZ = 20.0
STEPS = {"1": 0.5, "2": 1.0, "3": 2.0, "4": 5.0, "5": 10.0}
DEFAULT_STEP = 2.0

# The same fence mouse_jog uses. Z_MIN is the one that matters here:
# jogging DOWN toward the floor is the whole point, and the tool's length
# is not known to any of this, so the fence is what stops a hold-the-key
# mistake from pressing the plate into the table.
Z_MIN, Z_MAX = 120.0, 700.0
R_MIN, R_MAX = 220.0, 620.0

VK_ESCAPE = 0x1B
VK_SPACE = 0x20
user32 = ctypes.windll.user32


def down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def pressed(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x0001)


def key(ch):
    return down(ord(ch.upper()))


def clamp(x, y, z):
    """Keep the target inside the fence. Returns (x, y, z, hit)."""
    hit = False
    if z < Z_MIN:
        z, hit = Z_MIN, True
    elif z > Z_MAX:
        z, hit = Z_MAX, True
    r = (x * x + y * y) ** 0.5
    if r > 1e-6:
        if r > R_MAX:
            x, y, hit = x * R_MAX / r, y * R_MAX / r, True
        elif r < R_MIN:
            x, y, hit = x * R_MIN / r, y * R_MIN / r, True
    return x, y, z, hit


def stream(arm, angles):
    """Non-blocking joint move with vc.movej's sweep guard."""
    c, cur = arm.get_servo_angle(is_radian=False)
    if c == 0 and cur:
        worst = max(abs(a - b) for a, b in zip(angles, cur))
        if worst > vc.MAX_JOINT_SWEEP:
            return "sweep %.0f deg refused" % worst
    code = arm.set_servo_angle(angle=angles, speed=vc.JOINT_SPEED,
                               wait=False, is_radian=False)
    return None if code == 0 else "move rejected (code %s)" % code


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ip = args[0] if args else DEFAULT_IP

    fx.start_log("jog")
    print("=" * 62)
    print("   FIXED CAMERA  -  keyboard jog (the tool is never touched)")
    print("=" * 62)
    print("   THE ARM MOVES while a key is held. Keep the e-stop in hand.")
    print("")

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ip, is_radian=False)
    try:
        fx.setup_arm(arm, tool_io=False)

        code, pose = arm.get_position(is_radian=False)
        if code != 0 or pose is None:
            print("   cannot read the current pose.")
            return 1
        tx, ty, tz = float(pose[0]), float(pose[1]), float(pose[2])
        roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        step = DEFAULT_STEP

        print("""
==== JOG - the arm moves only while a key is HELD ====
   w / s   : further out / back      (X)
   a / d   : left / right            (Y)
   r / f   : up / down               (Z)
   1..5    : step 0.5 / 1 / 2 / 5 / 10 mm
   SPACE   : print the pose    ESC : quit
   fence   : z %.0f-%.0f mm, reach %.0f-%.0f mm
======================================================
   start:  x=%.1f  y=%.1f  z=%.1f     step %.1f mm

   For the calibration, jog DOWN (f) until the taped
   plate is about 20mm above the floor, then quit and
   run '4b - Calibrate Camera (TAPED plate)'.
""" % (Z_MIN, Z_MAX, R_MIN, R_MAX, tx, ty, tz, step))

        period = 1.0 / RATE_HZ
        last_note = ""
        while True:
            t0 = time.time()

            if down(VK_ESCAPE):
                break

            for k, v in STEPS.items():
                if key(k) and v != step:
                    step = v
                    print("   step = %.1f mm" % v)
                    break

            if pressed(VK_SPACE):
                c, pp = arm.get_position(is_radian=False)
                if c == 0:
                    print("   pose: x=%.1f y=%.1f z=%.1f" % (pp[0], pp[1],
                                                             pp[2]))

            dx = (step if key("w") else 0.0) - (step if key("s") else 0.0)
            dy = (step if key("a") else 0.0) - (step if key("d") else 0.0)
            dz = (step if key("r") else 0.0) - (step if key("f") else 0.0)
            if dx == 0.0 and dy == 0.0 and dz == 0.0:
                time.sleep(period)
                continue

            # Re-read rather than integrating our own target: if a move is
            # refused or clamped, the next step must start from where the
            # arm REALLY is, or the target drifts away from the arm and
            # every later step is wrong.
            c, pp = arm.get_position(is_radian=False)
            if c == 0 and pp is not None:
                tx, ty, tz = float(pp[0]), float(pp[1]), float(pp[2])

            nx, ny, nz, hit = clamp(tx + dx, ty + dy, tz + dz)
            angles = vc.ik(arm, [nx, ny, nz, roll, pitch, yaw])
            note = ""
            if angles is None:
                note = "unreachable there"
            else:
                note = stream(arm, angles) or ""
            if hit and not note:
                note = "fence"
            if note and note != last_note:
                print("   (%s)" % note)
            last_note = note

            # The controller faulting mid-jog is worth stopping for: with
            # this cell's wrist fault it should NEVER happen while the
            # tool is untouched, so if it does, something else is wrong.
            c, ew = arm.get_err_warn_code()
            if c == 0 and ew[0]:
                print("")
                print("   THE CONTROLLER FAULTED: error %s. Stopping." % ew[0])
                print("   Nothing here touched the tool, so this is not the")
                print("   wrist fault behaving as measured - check the arm")
                print("   before jogging again.")
                break

            time.sleep(max(0.0, period - (time.time() - t0)))

        code, pp = arm.get_position(is_radian=False)
        if code == 0:
            print("")
            print("   final pose:  x=%.1f  y=%.1f  z=%.1f" % (pp[0], pp[1],
                                                              pp[2]))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print("   FAILED: %s" % e)
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

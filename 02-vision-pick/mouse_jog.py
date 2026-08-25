#!/usr/bin/env python3
"""Mouse teleop for the xArm6 - the TCP follows your mouse in X/Y.

  * Move the mouse           -> arm moves in the base X/Y plane
  * W / S                    -> Z up / Z down
  * HOLD LEFT MOUSE BUTTON   -> deadman: the arm only moves while held
  * Q / E                    -> rotate gripper yaw -/+
  * O / P                    -> gripper more open / more closed
  * 1..5                     -> sensitivity 0.05/0.1/0.25/0.5/1.0 mm per pixel
  * SPACE                    -> record current pose
  * ESC                      -> quit (poses saved to mouse_poses.json)

Motion is IK-guarded joint motion exactly like the rest of the pipeline
(no set_position anywhere): every mouse step is turned into a cartesian
target, run through inverse kinematics, sweep-checked, then streamed as a
non-blocking joint move.  Steps are clamped so a fast mouse flick can
never turn into a large arm jump.

Usage: python mouse_jog.py [robot_ip]
Keep the e-stop in hand.
"""
import ctypes
import json
import os
import sys
import time

from xarm.wrapper import XArmAPI

import vision_common as vc

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mouse_poses.json")

# ---- tuning -------------------------------------------------------------
RATE_HZ = 20.0          # control loop rate
MAX_STEP_MM = 12.0      # per-tick clamp on XY travel (flick guard)
Z_STEP_MM = 4.0         # per-tick Z travel while W/S held
YAW_STEP_DEG = 2.0      # per-tick yaw while Q/E held
DEADBAND_MM = 0.3       # ignore tiny mouse jitter
MAX_QUEUED = 2          # don't let commands pile up behind the mouse

# Mouse -> base-frame mapping.  Push the mouse away from you (screen -Y)
# and the arm reaches further out (+X); mouse right (screen +X) sends the
# TCP to the arm's -Y.  Flip a sign here if it feels backwards.
SIGN_X = -1.0           # applied to screen dy  -> base X
SIGN_Y = -1.0           # applied to screen dx  -> base Y

# ---- workspace fence (mm, base frame) -----------------------------------
Z_MIN, Z_MAX = 120.0, 700.0
R_MIN, R_MAX = 220.0, 620.0     # distance from base axis in the XY plane

SENS = {"1": 0.05, "2": 0.1, "3": 0.25, "4": 0.5, "5": 1.0}

# ---- win32 ---------------------------------------------------------------
VK_LBUTTON = 0x01
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def cursor():
    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def pressed(vk):
    """True once per physical press (consumes the low 'was pressed' bit)."""
    return bool(user32.GetAsyncKeyState(vk) & 0x0001)


def key(ch):
    return down(ord(ch.upper()))


def screen_centre():
    return user32.GetSystemMetrics(0) // 2, user32.GetSystemMetrics(1) // 2


def clamp_target(x, y, z):
    """Keep the target inside the fence; returns (x, y, z, hit_limit)."""
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


def stream_move(arm, angles):
    """Non-blocking joint move with the same sweep guard as vc.movej."""
    c, cur = arm.get_servo_angle(is_radian=False)
    if c == 0 and cur:
        worst = max(abs(a - b) for a, b in zip(angles, cur))
        if worst > vc.MAX_JOINT_SWEEP:
            return "sweep %.0f deg refused" % worst
    code = arm.set_servo_angle(angle=angles, speed=vc.JOINT_SPEED,
                               wait=False, is_radian=False)
    if code != 0:
        return "move rejected (code %s)" % code
    return None


def recover(arm):
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
    except RuntimeError as e:
        print("ABORT:", e)
        arm.disconnect()
        return

    code, pose = arm.get_position(is_radian=False)
    if code != 0 or pose is None:
        print("ABORT: cannot read the current pose")
        arm.disconnect()
        return

    mm_per_px = SENS["3"]
    tx, ty, tz, tyaw = pose[0], pose[1], pose[2], pose[5]
    roll, pitch = pose[3], pose[4]
    waypoints = []
    cx, cy = screen_centre()
    armed_prev = False
    last_note = ""
    period = 1.0 / RATE_HZ

    print("""
==== MOUSE JOG - hold the LEFT MOUSE BUTTON to move ====
   mouse      : X / Y        w / s : Z up / down
   q / e      : yaw -/+      o / p : gripper open / close
   1 2 3 4 5  : 0.05 / 0.1 / 0.25 / 0.5 / 1.0 mm per pixel
   SPACE      : record pose  ESC   : quit
   fence: z %.0f-%.0f mm, reach %.0f-%.0f mm
========================================================
start pose: x=%.1f y=%.1f z=%.1f yaw=%.1f   (%.2f mm/px)
""" % (Z_MIN, Z_MAX, R_MIN, R_MAX, tx, ty, tz, tyaw, mm_per_px))

    try:
        while True:
            t0 = time.time()

            if down(VK_ESCAPE):
                break

            for k, v in SENS.items():
                if key(k):
                    if v != mm_per_px:
                        mm_per_px = v
                        print("  sensitivity = %.2f mm/px" % v)
                    break

            if pressed(VK_SPACE):
                c, pp = arm.get_position(is_radian=False)
                if c == 0:
                    waypoints.append([round(v, 1) for v in pp])
                    print("  recorded waypoint %d: %s"
                          % (len(waypoints), waypoints[-1]))

            if key("o") or key("p"):
                g = vc.gripper_pos(arm)
                if g is not None:
                    tgt = min(850, int(g) + 60) if key("o") \
                        else max(0, int(g) - 60)
                    try:
                        arm.set_gripper_position(tgt, wait=False)
                    except Exception as e:
                        print("  gripper: %s" % e)

            armed = down(VK_LBUTTON)
            if armed != armed_prev:
                # Re-sync to where the arm actually is, so releasing and
                # re-grabbing never replays stale mouse travel.
                c, pp = arm.get_position(is_radian=False)
                if c == 0:
                    tx, ty, tz, tyaw = pp[0], pp[1], pp[2], pp[5]
                    roll, pitch = pp[3], pp[4]
                user32.SetCursorPos(cx, cy)
                print("  %s" % ("ARMED - arm is following the mouse"
                                if armed else "released (arm parked)"))
                armed_prev = armed
            if not armed:
                time.sleep(period)
                continue

            mx, my = cursor()
            dx_px, dy_px = mx - cx, my - cy
            user32.SetCursorPos(cx, cy)      # relative-mouse trick

            dx = SIGN_X * dy_px * mm_per_px
            dy = SIGN_Y * dx_px * mm_per_px
            n = (dx * dx + dy * dy) ** 0.5
            if n > MAX_STEP_MM:
                dx, dy = dx * MAX_STEP_MM / n, dy * MAX_STEP_MM / n
            elif n < DEADBAND_MM:
                dx = dy = 0.0

            dz = 0.0
            if key("w"):
                dz += Z_STEP_MM
            if key("s"):
                dz -= Z_STEP_MM
            dyaw = 0.0
            if key("q"):
                dyaw -= YAW_STEP_DEG
            if key("e"):
                dyaw += YAW_STEP_DEG

            if dx == 0.0 and dy == 0.0 and dz == 0.0 and dyaw == 0.0:
                time.sleep(period)
                continue

            nx, ny, nz, hit = clamp_target(tx + dx, ty + dy, tz + dz)
            nyaw = tyaw + dyaw

            # Don't queue commands faster than the arm consumes them.
            try:
                queued = arm.cmd_num
            except Exception:
                queued = 0
            if queued is not None and queued > MAX_QUEUED:
                time.sleep(period)
                continue

            angles = vc.ik(arm, [nx, ny, nz, roll, pitch, nyaw])
            note = None
            if angles is None:
                note = "unreachable - holding"
            else:
                note = stream_move(arm, angles)
                if note is None:
                    tx, ty, tz, tyaw = nx, ny, nz, nyaw
                    if hit:
                        note = "at workspace limit"

            c, ew = arm.get_err_warn_code()
            if c == 0 and ew[0] != 0:
                print("  arm fault %s - clearing, release the button" % ew[0])
                recover(arm)
                note = None

            if note and note != last_note:
                print("  %s" % note)
            last_note = note or ""

            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            arm.set_state(4)     # stop motion
            arm.set_state(0)
        except Exception:
            pass
        c, pp = arm.get_position(is_radian=False)
        if c == 0:
            print("\nFinal pose: %s" % [round(v, 1) for v in pp])
        arm.disconnect()

    if waypoints:
        with open(OUT, "w") as f:
            json.dump({"recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "poses": waypoints}, f, indent=2)
        print("%d waypoint(s) saved to %s" % (len(waypoints), OUT))


if __name__ == "__main__":
    main()

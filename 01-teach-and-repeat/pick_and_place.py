#!/usr/bin/env python3
"""Pick-and-place for the physical xArm 6 + UFACTORY 2-finger gripper.

Self-recovering design:
  * Startup clears stale controller AND gripper faults (naming what it
    cleared), declares the gripper's weight as TCP payload, and checks
    every waypoint is reachable BEFORE any motion.
  * If a previous run stopped while carrying, startup detects the box
    still in the gripper and offers to DELIVER it to PLACE (finishing the
    job) or release it in place -- it will never drop it at HOME.
  * All moves are joint moves. Each IK solution is sanity-checked against
    the current pose so a flipped-configuration solution (possible on
    firmware 1.6.9, where IK has no reference-seed support) is rejected
    instead of executed as a wild sweep.
  * Gripper commands are verified: a real gripper fault aborts the run
    (a normal clamp-on-the-box reads as success, so no false aborts), a
    grab that catches nothing aborts before the phantom carry, and an
    open that didn't open aborts before the descent.

Sequence: HOME -> open -> above PICK -> PICK -> grab -> lift
       -> above PLACE -> PLACE -> release -> retreat -> HOME.

Usage: python3 ~/pick_and_place.py [robot_ip]
Keep the e-stop in hand. Keep the workspace clear.
"""
import sys
import math
from xarm.wrapper import XArmAPI

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

# ---------------------------------------------------------------------------
# POSITIONS  --  Cartesian [x, y, z, roll, pitch, yaw] in mm and degrees.
# ---------------------------------------------------------------------------
PICK  = [450.3, 138.7, 199.1, 180.0, 0.0, -26.6]      # straightened: perfectly vertical
PLACE = [436.5, -498.4, 407.6, -171.1, -9.8, -93.5]   # TODO: re-teach closer to base, then straighten

# Clearance above each point (mm). PLACE is far out; keep its approach low.
PICK_APPROACH  = 100.0
PLACE_APPROACH = 60.0

# Saved HOME joint angles (radians -> degrees), same as your go-home script.
HOME_RAD = [
    0.07306542247533798, -0.2995672821998596, -1.7921593189239502,
    -0.008517428301274776, 2.102301597595215, 0.0,
]
HOME_DEG = [math.degrees(a) for a in HOME_RAD]

# Gripper (UFACTORY 2-finger). 0 = closed .. 850 = open.
GRIP_OPEN   = 850
GRIP_CLOSED = 329   # measured for this box
GRIP_SPEED  = 3000

# Grip verification margins (pulses; ~10 pulses = 1 mm of finger travel).
EMPTY_GRAB_MARGIN = 10   # fingers within this of GRIP_CLOSED => caught nothing
OPEN_CHECK_MARGIN = 60   # fingers this far from GRIP_OPEN => didn't open

# TCP payload: the controller MUST know what's hanging on the wrist, or it
# mistakes the gripper's own weight for a crash (controller error 31,
# "collision detected") and freezes. UFACTORY xArm Gripper spec: 0.82 kg,
# center of gravity at z=48mm.
GRIPPER_WEIGHT = 0.82              # kg
GRIPPER_COG    = [0.0, 0.0, 48.0]  # mm
BOX_WEIGHT     = 0.0               # kg - set this if your box is heavy (>0.3)

# Joint speed (deg/s) and the largest single-joint change we accept from an
# IK solution. Legit moves in this job stay under ~70 deg per joint; a
# flipped elbow/wrist solution shows up as 150+ deg and must not run.
JOINT_SPEED     = 20
MAX_JOINT_SWEEP = 120.0

ERROR_NAMES = {
    1: "emergency stop pressed", 2: "emergency IO triggered",
    24: "speed exceeds limit", 31: "collision detected",
}


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


def ik(arm, pose):
    """Return joint angles (deg) for a Cartesian pose, or None if unreachable."""
    code, angles = arm.get_inverse_kinematics(pose, input_is_radian=False,
                                              return_is_radian=False)
    return angles if (code == 0 and angles is not None) else None


def fault(arm):
    """Return (err, warn); (0, 0) means healthy."""
    code, ew = arm.get_err_warn_code()
    return (ew[0], ew[1]) if code == 0 else (-1, -1)


def gripper_pos(arm):
    """Current finger position (0..850), or None if unreadable."""
    code, pos = arm.get_gripper_position()
    return pos if (code == 0 and pos is not None) else None


def setup(arm):
    """Clear faults (naming them) and put arm + gripper in a ready state."""
    err, _ = fault(arm)
    if err > 0:
        print("Clearing previous fault: error %s (%s)"
              % (err, ERROR_NAMES.get(err, "controller error")))
    arm.clean_warn()
    arm.clean_error()
    arm.clean_gripper_error()   # controller clear does NOT clear gripper faults
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    if arm.set_gripper_enable(True) != 0:
        raise RuntimeError("gripper did not enable - check the gripper "
                           "cable/power, then retry")
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(GRIP_SPEED)
    # Tell the controller the gripper's weight so it doesn't read that
    # weight as a collision (this caused the err-31 stops).
    arm.set_tcp_load(GRIPPER_WEIGHT, GRIPPER_COG)
    arm.set_state(0)
    err, warn = fault(arm)
    if err != 0:
        raise RuntimeError("arm still in error %s after clearing - check the "
                           "physical arm / e-stop, then retry" % err)


def preflight(arm):
    """IK-check every waypoint BEFORE moving. Raise if any is unreachable."""
    print("Pre-flight reachability check (nothing moves)...")
    points = [
        ("above PICK",  above(PICK, PICK_APPROACH)),
        ("PICK",        PICK),
        ("above PLACE", above(PLACE, PLACE_APPROACH)),
        ("PLACE",       PLACE),
    ]
    bad = []
    for name, p in points:
        ok = ik(arm, p) is not None
        print("   %-12s %s" % (name, "ok" if ok else "NOT reachable"))
        if not ok:
            bad.append(name)
    if bad:
        raise RuntimeError("unreachable: %s -- re-teach it closer to the base "
                           "or lower, or reduce its approach height"
                           % ", ".join(bad))


def movej(arm, label, angles):
    print("-> %s" % label)
    code = arm.set_servo_angle(angle=angles, speed=JOINT_SPEED, wait=True,
                               is_radian=False)
    if code != 0:
        raise RuntimeError("%s move failed (code %s)" % (label, code))
    err, warn = fault(arm)
    if err != 0:
        raise RuntimeError("arm faulted (error %s: %s) during %s"
                           % (err, ERROR_NAMES.get(err, "?"), label))


def moveto(arm, label, pose):
    angles = ik(arm, pose)
    if angles is None:
        raise RuntimeError("%s is not reachable from here" % label)
    # Reject flipped-configuration IK solutions (huge joint sweeps).
    code, cur = arm.get_servo_angle(is_radian=False)
    if code == 0 and cur:
        worst = max(abs(a - b) for a, b in zip(angles, cur))
        if worst > MAX_JOINT_SWEEP:
            raise RuntimeError(
                "%s: IK wants a %.0f deg joint sweep (flipped arm "
                "configuration) - refusing to run it" % (label, worst))
    movej(arm, label, angles)


def grip(arm, label, position):
    """Command the gripper and verify it. Returns final finger position."""
    print("-> gripper %s (%d)" % (label, position))
    code = arm.set_gripper_position(position, wait=True)
    # NOTE: clamping on the box returns 0 (the SDK's stall detector treats
    # it as success), so nonzero here is always a REAL gripper fault.
    if code != 0:
        raise RuntimeError("gripper %s failed (code %s) - gripper fault; "
                           "check it, then re-run" % (label, code))
    return gripper_pos(arm)


def open_gripper(arm, label):
    pos = grip(arm, label, GRIP_OPEN)
    if pos is not None and pos < GRIP_OPEN - OPEN_CHECK_MARGIN:
        raise RuntimeError("gripper did not open (stuck at %.0f) - check for "
                           "jams" % pos)


def grab(arm):
    pos = grip(arm, "close (grab)", GRIP_CLOSED)
    if pos is not None and pos <= GRIP_CLOSED + EMPTY_GRAB_MARGIN:
        raise RuntimeError("grab caught NOTHING (fingers closed all the way "
                           "to %.0f) - is the box at the PICK spot?" % pos)
    arm.set_tcp_load(GRIPPER_WEIGHT + BOX_WEIGHT, GRIPPER_COG)  # now carrying


def deliver(arm):
    """Carry whatever is in the gripper to PLACE, release, retreat, go HOME."""
    arm.set_tcp_load(GRIPPER_WEIGHT + BOX_WEIGHT, GRIPPER_COG)
    moveto(arm, "above PLACE", above(PLACE, PLACE_APPROACH))
    moveto(arm, "PLACE", PLACE)
    open_gripper(arm, "open (release)")
    arm.set_tcp_load(GRIPPER_WEIGHT, GRIPPER_COG)
    moveto(arm, "retreat", above(PLACE, PLACE_APPROACH))
    movej(arm, "HOME", HOME_DEG)


def held_box_recovery(arm):
    """If a previous run stopped mid-carry, the box is still in the gripper.
    Never drop it at HOME - deliver it or release it in place instead.
    Returns True if the program should exit now."""
    pos = gripper_pos(arm)
    if pos is None or pos > GRIP_OPEN - 100:
        return False   # gripper is open - nothing held, proceed normally
    print("\n!! The gripper is not open (at %.0f) - it may still be holding"
          " the box\n   from a stopped run." % pos)
    print("   [d] deliver it to the PLACE spot and finish (arm will MOVE)")
    print("   [r] release it right here - HOLD THE BOX first")
    print("   [q] quit, touch nothing")
    choice = input("   choose d/r/q > ").strip().lower()
    if choice == "d":
        deliver(arm)
        print("\nDelivered. Parked at HOME.")
        return True
    if choice == "r":
        input("   Hold the box, then press Enter... ")
        open_gripper(arm, "open (release here)")
        print("   Released. Put the box back at the PICK spot before the "
              "next run.")
        return False   # scene reset by user; a normal run can follow
    print("   Left everything as is.")
    return True


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)

    try:
        setup(arm)          # self-clears stale faults, arms the gripper
        preflight(arm)      # aborts, with no motion, if anything unreachable
        if held_box_recovery(arm):
            arm.disconnect()
            return
    except RuntimeError as e:
        print("\nABORT (no motion):", e)
        arm.disconnect()
        return

    print("\nPlan: HOME -> open -> above PICK -> PICK -> grab -> lift")
    print("      -> above PLACE -> PLACE -> release -> retreat -> HOME")
    if input("\nRun the pick-and-place now? [y/N] ").strip().lower() != "y":
        print("Aborted."); arm.disconnect(); return

    try:
        movej(arm, "HOME", HOME_DEG)
        open_gripper(arm, "open")

        moveto(arm, "above PICK", above(PICK, PICK_APPROACH))
        moveto(arm, "PICK", PICK)
        grab(arm)
        moveto(arm, "lift", above(PICK, PICK_APPROACH))

        moveto(arm, "above PLACE", above(PLACE, PLACE_APPROACH))
        moveto(arm, "PLACE", PLACE)
        open_gripper(arm, "open (release)")
        arm.set_tcp_load(GRIPPER_WEIGHT, GRIPPER_COG)  # box released
        moveto(arm, "retreat", above(PLACE, PLACE_APPROACH))

        movej(arm, "HOME", HOME_DEG)
        print("\nDone. Parked at HOME.")
    except RuntimeError as e:
        print("\nSTOPPED:", e)
        print("Re-run when ready - if the box is still in the gripper, the "
              "script will offer to deliver or release it safely.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()

"""Move the arm BY HAND, then lock it where you left it.

    python cell_handjog.py
    python cell_handjog.py --vacuum      cup ON, so a cube will stick

THE ARM GOES LIMP. It holds its own weight and nothing more, so it can
be pushed around like a desk lamp - but it will not resist a shove, and
it will sag if you let go somewhere it cannot hold. KEEP THE E-STOP IN
HAND and keep a hand on the arm the whole time it is free.

This tool never COMMANDS a motion. It changes the arm's mode, you do the
moving, and it changes the mode back. The only thing it commands is the
vacuum, and only if you ask for it.

set_mode(2) is the xArm's joint teaching mode: the servos hold the arm's
weight and otherwise get out of the way. Position mode is restored on
every path out of here - finish, skip, Ctrl+C, or an exception - because
an arm left in teaching mode is one that sags the moment someone leans
on it.

WHILE IT IS FREE it prints the flange pose live, including how far the
cup is off vertical. For the calibration grid and for check B you want
the cup pointing straight DOWN, which is RPY near [180, 0, 0] - so aim
for 'off vertical' near 0 before you let go.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
ROBOT_IP = "192.168.1.197"

# The cup points straight down here. A round cup does not care about
# yaw, so only roll and pitch are checked.
DOWN_RPY = (180.0, 0.0)


def off_vertical_deg(roll, pitch):
    """How far the tool axis is from straight down, in degrees.

    Roll and pitch are not independent angles you can just add - this
    builds the tool's Z axis from the reported RPY and measures it
    against gravity, which is the only thing that actually matters to a
    suction cup on a horizontal face.
    """
    # xArm RPY is ZYX, so R = Rz(yaw) Ry(pitch) Rx(roll) and the tool's
    # own Z axis is that matrix's third column. Only its BASE-Z
    # component decides how far off vertical the tool is, and that
    # component works out to cos(pitch)cos(roll) - yaw drops out
    # entirely, which is right: spinning a round cup about its own axis
    # does not tilt it.
    cz = (np.cos(np.radians(pitch)) * np.cos(np.radians(roll)))
    # Straight down is base -Z, so the cup is vertical when cz = -1.
    return float(np.degrees(np.arccos(np.clip(-cz, -1.0, 1.0))))


def free_arm(arm, sensitivity=None):
    """Put the arm into joint teaching mode, and PROVE that it took.

    The first version of this tool announced "ARM IS FREE" whether or not
    the controller had accepted anything, because it ignored every return
    code. A refused mode change then looks exactly like a stiff arm, and
    there is nothing on screen to say which it was.

    So every call is checked and the mode is read BACK. Returns True only
    if the controller reports mode 2.
    """
    code, ew = arm.get_err_warn_code()
    if code == 0 and (ew[0] or ew[1]):
        print("   controller error %s / warning %s - clearing"
              % (ew[0], ew[1]))
        arm.clean_error()
        arm.clean_warn()
        time.sleep(0.2)

    # Teaching mode is refused outright unless the servos are enabled and
    # the arm is in a runnable state, so both are established first and
    # verified rather than assumed.
    c_en = arm.motion_enable(enable=True)
    c_m0 = arm.set_mode(0)
    c_s0 = arm.set_state(0)
    time.sleep(0.3)
    print("   motion_enable %s   set_mode(0) %s   set_state(0) %s"
          % (c_en, c_m0, c_s0))

    if sensitivity is not None:
        try:
            print("   set_teach_sensitivity(%d) %s"
                  % (sensitivity, arm.set_teach_sensitivity(sensitivity)))
        except Exception as e:
            print("   (teach sensitivity not settable: %s)" % e)

    c_m2 = arm.set_mode(2)
    c_s2 = arm.set_state(0)
    time.sleep(0.4)
    code, state = arm.get_state()
    mode = getattr(arm, "mode", None)
    print("   set_mode(2) %s   set_state(0) %s   -> mode %s, state %s"
          % (c_m2, c_s2, mode, state))

    if mode == 2 and state == 0:
        return True

    print("""
   THE ARM DID NOT GO FREE, and here is what that means.

   mode 2 is joint teaching. A non-zero return code above is the
   controller REFUSING, not the arm being stiff. The usual causes,
   in the order worth checking:

     code 1   the arm is in an error state that survived the clear.
              Look at the error/warning line above, and run
              '0 - RESET EVERYTHING' from the Fixed Camera folder.

     code 9   the arm is not enabled, or is in STOP state (4).
              Usually follows an earlier collision or e-stop.

     mode stays 0
              some controller/firmware combinations refuse teaching
              mode while a TOOL LOAD is declared wrong, or while the
              controller is in a state that only a power cycle clears.

   THINGS TO TRY, cheapest first:
     1. release the E-STOP if it is latched down, then run this again
     2. run this again - a first attempt after boot is sometimes
        refused where the second is accepted
     3. power-cycle the control box, then run this again
""")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vacuum", action="store_true",
                    help="switch the cup ON before releasing, so a cube "
                         "can be stuck to it while you position the arm")
    ap.add_argument("--sensitivity", type=int, default=None,
                    help="teach sensitivity 1-5. Higher is easier to "
                         "push. Try 5 if the arm goes free but feels "
                         "heavy.")
    ap.add_argument("--ip", default=ROBOT_IP)
    args = ap.parse_args()

    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    from xarm.wrapper import XArmAPI

    os.makedirs(LOGS, exist_ok=True)

    print("=" * 62)
    print("   HAND JOG  -  THE ARM GOES LIMP")
    print("=" * 62)
    print("""
   It holds its own weight and NOTHING MORE. It will not
   resist a shove, and it will sag if you let go somewhere
   it cannot hold itself.

   KEEP THE E-STOP IN HAND.
   KEEP A HAND ON THE ARM the whole time it is free.

   This tool never commands a motion. You do the moving.
""")

    arm = XArmAPI(args.ip)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)

    code, pose = arm.get_position()
    if code == 0:
        print("   now at  x %.1f  y %.1f  z %.1f   rpy %.1f %.1f %.1f"
              % tuple(pose))

    if args.vacuum:
        # Deliberately BEFORE the mode change. Commanding the tool is a
        # 'set' call whose readiness depends on the controller state;
        # the manual phase that follows only ever reads.
        arm.set_vacuum_gripper(True)
        print("   cup is ON - press a cube onto it once the arm is free.")

    try:
        ans = input("\n   HOLD THE ARM NOW. Release the brakes? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans.strip().lower() != "y":
        print("   nothing done.")
        try:
            if args.vacuum:
                arm.set_vacuum_gripper(False)
            arm.disconnect()
        except Exception:
            pass
        return

    final = None
    try:
        if not free_arm(arm, sensitivity=args.sensitivity):
            try:
                if args.vacuum:
                    arm.set_vacuum_gripper(False)
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
            except Exception:
                pass
            return
        print("\n   ARM IS FREE - the controller confirms mode 2.")
        print("   Move it where you want it.")
        print("   Press ENTER to LOCK it there.  Ctrl+C to abort.\n")

        while True:
            code, pose = arm.get_position()
            if code == 0:
                off = off_vertical_deg(pose[3], pose[4])
                flag = "  <- cup is DOWN" if off < 3.0 else ""
                sys.stdout.write(
                    "\r   x %7.1f  y %7.1f  z %7.1f   rpy %6.1f %6.1f "
                    "%6.1f   off vertical %5.1f deg%s   "
                    % (pose[0], pose[1], pose[2], pose[3], pose[4],
                       pose[5], off, flag))
                sys.stdout.flush()
                final = pose
            if msvcrt is not None and msvcrt.kbhit():
                if msvcrt.getch() in (b"\r", b"\n"):
                    break
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\n   aborted - locking where it stands.")
    finally:
        # ALWAYS. Every path out of here goes through this.
        try:
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            print("\n\n   LOCKED. Position mode restored - you can let go.")
        except Exception:
            print("\n\n   WARNING: could not restore position mode.")
            print("   Do NOT let go. Power-cycle the controller.")

    if final is not None:
        off = off_vertical_deg(final[3], final[4])
        print("   left at x %.1f  y %.1f  z %.1f   rpy %.1f %.1f %.1f"
              % tuple(final))
        print("   cup is %.1f deg off vertical" % off)
        if off > 5.0:
            print("\n   NOTE: for check B and for the calibration grid the "
                  "cup should be\n"
                  "         pointing DOWN. %.1f degrees off is enough to "
                  "change what the\n"
                  "         camera sees of the cube's top face. Worth "
                  "another go." % off)
        with open(os.path.join(LOGS, "handjog.log"), "a",
                  encoding="utf-8") as f:
            f.write("%s  %s  off_vertical %.1f\n"
                    % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                       " ".join("%.1f" % v for v in final), off))

    if args.vacuum:
        print("\n   The cup is STILL ON, holding your cube. Run check B "
              "now.\n"
              "   It stays on until something switches it off.")
    try:
        arm.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Put the cell back into a state where the launchers run again.

Use this after a run stopped badly and every .bat now refuses to move,
with a line like

    ABORT (no motion): arm still in error [13, 0] - check e-stop, retry

That message is not a bug in the script: setup_arm() reads the
controller's error register before it moves anything, and a LATCHED
controller fault makes every launcher stop there. Until the fault is
cleared, nothing runs - which looks exactly like "the arm can no longer
run any .bat file".

What this does, in order:
    1. closes leftover Python windows still holding the arm socket or
       the camera (a crashed run keeps both),
    2. re-plugs the RealSense camera in software (hardware_reset),
    3. pings the controller,
    4. reads the fault WITH ITS OFFICIAL MEANING (from the SDK's own
       code tables), clears it, re-enables the arm and the gripper,
    5. if the fault will not clear, says exactly why and waits while you
       power-cycle, then re-checks.

The arm is NEVER moved by this script. The gripper fingers can open -
you are asked first, and anything clamped will drop.

Usage:  python reset_all.py [robot_ip] [--yes] [--no-kill]
                            [--no-camera] [--no-gripper]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "vision", "logs")

DEFAULT_IP = "192.168.1.197"

# The pick scripts treat 850 as "fully open" and allow a 60-pulse
# shortfall before calling an open move failed.
GRIPPER_OPEN = 850
GRIPPER_TOL = 60
GRIPPER_SPEED = 5000

MAX_CLEAR_TRIES = 3         # software clear attempts per round
MAX_POWER_ROUNDS = 3        # "power-cycle it and I'll look again" rounds


# ----------------------------- logging -----------------------------

class _Tee(object):
    """Mirror a stream to a file - same idea as vision_common.start_log,
    duplicated here on purpose: this script must keep working when the
    vision stack itself is what broke."""

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, s):
        self._s.write(s)
        self._s.flush()
        self._f.write(s)
        self._f.flush()

    def flush(self):
        self._s.flush()
        self._f.flush()

    def isatty(self):
        return self._s.isatty()

    def fileno(self):
        return self._s.fileno()


def start_log():
    try:
        if not os.path.isdir(LOGS):
            os.makedirs(LOGS)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS, "reset_%s.log" % stamp)
        fh = open(path, "w", encoding="utf-8", errors="replace")
    except Exception as e:
        print("(no run log: %s)" % e)
        return None
    fh.write("=== reset_all  %s ===\n" % stamp)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path


def head(n, title):
    print("")
    print("--- %d) %s %s" % (n, title, "-" * max(4, 44 - len(title))))


def ask(question, default_no=True, assume_yes=False):
    """y/N prompt that survives a console with no input attached."""
    if assume_yes:
        print("%s [auto-yes]" % question)
        return True
    try:
        answer = input("%s " % question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        return False
    return answer == "y" if default_no else answer != "n"


# ------------------- 1. leftover Windows scripts -------------------

def project_scripts():
    """Every .py that belongs to this project, by bare filename.

    A launcher does `cd /d "%~dp0.." & python vision_pick3.py`, so a
    stale process's command line holds the bare name, not a path - the
    name is what we have to match on."""
    names = set()
    for sub in ("", "vision", "realtime", "handoff"):
        try:
            entries = os.listdir(os.path.join(HERE, sub))
        except OSError:
            continue
        names.update(n.lower() for n in entries if n.lower().endswith(".py"))
    return names


def _is_python(name):
    """The Store build of Python 3.12 - the one this project runs on -
    reports itself as python3.12.exe, not python.exe, so an exact-name
    filter finds nothing at all."""
    name = (name or "").lower()
    return name.startswith("python") or name == "py.exe"


def _stale_by_psutil(scripts):
    import psutil
    me = os.getpid()
    root = HERE.lower()
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            if not _is_python(p.info["name"]):
                continue
            cmd = p.info["cmdline"] or []
            why = None
            for token in cmd:
                if os.path.basename(token).lower() in scripts:
                    why = os.path.basename(token)
                    break
            if why is None and root in " ".join(cmd).lower():
                why = "(project path on its command line)"
            if why is None:
                try:
                    if p.cwd().lower().startswith(root):
                        why = "(running inside the project folder)"
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                    pass
            if why:
                found.append((p.info["pid"], why, p))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _stale_by_powershell(scripts):
    """Fallback when psutil is missing: ask Windows for the command lines.

    tasklist cannot show them, so there is no plain-cmd version of this."""
    import json
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%' or "
          "Name='py.exe'\" | Select-Object ProcessId,CommandLine | "
          "ConvertTo-Json -Compress")
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.DEVNULL, text=True, timeout=30)
        rows = json.loads(raw) if raw.strip() else []
    except Exception as e:
        print("   could not list Python processes (%s)" % e)
        return []
    if isinstance(rows, dict):
        rows = [rows]
    me = os.getpid()
    root = HERE.lower()
    found = []
    for row in rows:
        pid = row.get("ProcessId")
        cmd = (row.get("CommandLine") or "").lower()
        if not pid or pid == me:
            continue
        why = None
        for token in cmd.replace('"', " ").split():
            if os.path.basename(token) in scripts:
                why = os.path.basename(token)
                break
        if why is None and root in cmd:
            why = "(project path on its command line)"
        if why:
            found.append((pid, why, None))
    return found


def kill_stale():
    """Close crashed runs. They are why a rerun often cannot connect:
    the controller hands out one control connection and the RealSense
    driver one exclusive claim on the camera, and a dead-but-not-gone
    process is still holding both."""
    scripts = project_scripts()
    try:
        import psutil                                    # noqa: F401
        stale = _stale_by_psutil(scripts)
        have_psutil = True
    except ImportError:
        stale = _stale_by_powershell(scripts)
        have_psutil = False

    if not stale:
        print("   no leftover project scripts are running.")
    for pid, why, proc in stale:
        print("   closing PID %s  %s" % (pid, why))
        try:
            if have_psutil and proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
            else:
                subprocess.call(["taskkill", "/PID", str(pid), "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        except Exception as e:
            print("      could not close it: %s" % e)

    # The RealSense viewer takes the camera exclusively too, and it is
    # easy to leave open after checking a stream by hand.
    killed_viewer = subprocess.call(
        ["taskkill", "/IM", "realsense-viewer.exe", "/F"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if killed_viewer:
        print("   closed RealSense Viewer (it holds the camera).")

    print("   NOTE: a UFACTORY Studio tab open in a browser also holds a")
    print("         control connection - close it if the arm ignores us.")
    if stale:
        time.sleep(1.5)      # let the sockets actually drop
    return len(stale)


# --------------------------- 2. camera ---------------------------

def reset_camera():
    """Software-replug the D435. Clears a USB endpoint left wedged by a
    process that died mid-stream, which otherwise shows up much later as
    'failed to set power state' or an empty depth frame."""
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        print("   pyrealsense2 is not installed here (%s)" % e)
        print("   -> python -m pip install --user -r vision\\requirements.txt")
        return False

    try:
        devices = list(rs.context().query_devices())
    except Exception as e:
        print("   could not query the RealSense driver: %s" % e)
        return False

    if not devices:
        print("   NO camera found.")
        print("   -> unplug the D435 and plug it back into a blue USB 3 port")
        print("      (the vision launchers need it; the arm ones do not).")
        return False

    serials = []
    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            serials.append(serial)
            line = "   found %s  serial %s" % (name, serial)
            try:
                usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                line += "  USB %s" % usb
                if usb.startswith("2"):
                    line += "   <- USB 2 port! depth will be poor/absent"
            except Exception:
                pass
            print(line)
        except Exception:
            print("   found a RealSense device (details unreadable)")

    for dev in devices:
        try:
            dev.hardware_reset()
        except Exception as e:
            print("   hardware reset refused: %s" % e)
            return False
    print("   re-plugged in software, waiting for it to come back ...")

    deadline = time.time() + 20
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            back = [d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()]
        except Exception:
            continue
        if not serials or any(s in back for s in serials):
            print("   camera is back.")
            return True
    print("   camera did NOT come back within 20s - unplug and replug it.")
    return False


# --------------------------- 3. network ---------------------------

def ping(ip):
    return subprocess.call(["ping", "-n", "2", "-w", "1000", ip],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


# ----------------------------- 4. arm -----------------------------

def code_text(kind, code):
    """Official title/description for a code, straight out of the SDK's
    own tables - not a guess, and it stays right across SDK updates."""
    from xarm.core.config import x_code
    cls = {"error": x_code.ControllerError, "warn": x_code.ControllerWarn,
           "gripper": x_code.GripperError}[kind]
    info = cls(code)
    return info.title["en"], info.description["en"]


def servo_faults(arm):
    """Per-joint faults. The controller error only says WHICH joint
    (13 = servo 3); this says what happened to it."""
    try:
        code, msgs = arm.get_servo_debug_msg()
    except Exception:
        return []
    if code != 0 or not msgs:
        return []
    return [m for m in msgs if m.get("status") or m.get("code")]


def report(arm, when):
    print("")
    print("   %s:" % when)
    try:
        code, ew = arm.get_err_warn_code()
    except Exception as e:
        print("      could not read the error register: %s" % e)
        return None, None
    err, warn = ew[0], ew[1]
    print("      state=%s  mode=%s  (read code %s)"
          % (arm.state, arm.mode, code))
    title, desc = code_text("error", err)
    print("      error   %-3s %s" % (err, title))
    if err and desc:
        print("              %s" % desc)
    title, desc = code_text("warn", warn)
    print("      warning %-3s %s" % (warn, title))

    for m in servo_faults(arm):
        print("      %-8s code %-3s %s"
              % (m["name"], m["code"], m["title"]))
        if m.get("desc"):
            print("               %s" % m["desc"])

    # A "joint voltage insufficient" fault (servo code 40) is only
    # believable next to the actual rail readings, so print them.
    try:
        volts = arm.voltages
        if volts:
            print("      joint volts %s"
                  % " ".join("%.1f" % v for v in volts))
    except Exception:
        pass

    pos = gripper_pos(arm)
    try:
        gerr = arm.get_gripper_err_code()[1]
    except Exception:
        gerr = None
    gtitle = code_text("gripper", gerr)[0] if gerr else "Normal"
    print("      gripper pos=%s err=%s (%s)" % (pos, gerr, gtitle))
    return err, warn


def gripper_pos(arm):
    # On a gripper fault the SDK returns a bare int instead of
    # (code, pos) - never unpack blindly.
    try:
        ret = arm.get_gripper_position()
    except Exception:
        return None
    if isinstance(ret, (tuple, list)) and len(ret) == 2:
        code, pos = ret
        return pos if (code == 0 and pos is not None) else None
    return None


def clear_fault(arm):
    """The standard clear sequence, repeated: a servo fault sometimes
    only lets go on the second pass, once the joint has re-enabled."""
    for attempt in range(1, MAX_CLEAR_TRIES + 1):
        print("   clear attempt %d/%d ..." % (attempt, MAX_CLEAR_TRIES))
        try:
            arm.clean_warn()
            arm.clean_error()
            arm.clean_gripper_error()
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)
        except Exception as e:
            print("      the controller refused a command: %s" % e)
        time.sleep(0.8)
        try:
            code, ew = arm.get_err_warn_code()
        except Exception:
            continue
        # A leftover WARNING is not what stops the launchers - setup_arm()
        # only refuses on an error - so it must not fail the reset. It is
        # still worth printing: warn 14 ("no solution") means something
        # asked for a pose the arm cannot reach.
        if code == 0 and ew[0] == 0:
            print("   cleared.")
            if ew[1]:
                print("   (warning %s left over: %s)"
                      % (ew[1], code_text("warn", ew[1])[0]))
            return True
    return False


def rearm_gripper(arm, assume_yes=False, skip=False):
    """Re-enable the gripper and (after you say so) open it.

    The gripper silently loses its enable on any controller fault and
    then ACCEPTS position commands while never moving, reporting error
    0 the whole time - so re-enabling is not optional after a fault."""
    if skip:
        print("   skipped (--no-gripper).")
        return True
    try:
        arm.clean_gripper_error()
        ok = arm.set_gripper_enable(True) == 0
        arm.set_gripper_mode(0)
        arm.set_gripper_speed(GRIPPER_SPEED)
    except Exception as e:
        print("   the gripper refused to enable: %s" % e)
        return False
    if not ok:
        print("   the gripper refused the enable - check its cable and the")
        print("   24V supply on the end-effector.")
        return False
    print("   gripper enabled.")

    pos = gripper_pos(arm)
    if pos is not None and pos > GRIPPER_OPEN - GRIPPER_TOL:
        print("   already open (%s) - nothing to do." % pos)
        return True

    print("")
    print("   The fingers are at %s. If a cube is clamped between them,"
          % pos)
    print("   TAKE HOLD OF IT NOW - it will drop when they open.")
    if not ask("   Open the gripper? [y/N]", assume_yes=assume_yes):
        print("   left as it is - the fingers did not move.")
        return True

    for attempt in (1, 2):
        try:
            arm.set_gripper_position(GRIPPER_OPEN, wait=True)
        except Exception as e:
            print("   open failed: %s" % e)
        pos = gripper_pos(arm)
        if pos is None or pos >= GRIPPER_OPEN - GRIPPER_TOL:
            print("   gripper open (%s)." % pos)
            return True
        print("   stuck at %s of %d - re-enabling and retrying [%d/2]"
              % (pos, GRIPPER_OPEN, attempt))
        arm.clean_gripper_error()
        arm.set_gripper_enable(True)
        arm.set_gripper_mode(0)
    print("   the fingers did NOT open (stuck at %s)." % pos)
    print("   They are jammed mechanically or the gripper lost power:")
    print("   free the cube by hand, then power-cycle the controller.")
    return False


def advise(arm, err):
    """What to actually do about a fault that would not clear."""
    print("")
    print("   WHY IT WILL NOT CLEAR, AND WHAT TO DO")
    print("   " + "-" * 42)
    faults = servo_faults(arm)

    if err in (1, 2, 3):
        print("   The emergency stop is still engaged. Twist the red button")
        print("   on the control box (and check the three-state switch and")
        print("   the emergency IO) until it pops out, then run this again.")
    elif 10 <= err <= 17:
        joint = err - 10
        print("   Joint %d's servo faulted. A servo fault is latched in the"
              % joint)
        print("   JOINT, not in the controller - clean_error() cannot reach")
        print("   it, which is why every launcher keeps aborting.")
        if any(m.get("code") == 40 for m in faults):
            print("")
            print("   The joints report code 40, 'Joint Voltage")
            print("   Insufficient': the 48V rail sagged during the move.")
            print("   Usual causes, in order of likelihood:")
            print("     - the e-stop was hit / power dipped mid-move,")
            print("     - acceleration too high for the reach (JOINT_ACC in")
            print("       vision\\vision_common.py is 1000),")
            print("     - a loose power or joint cable.")
        print("")
        print("   POWER-CYCLE THE CONTROL BOX:")
        print("     1. press the e-stop IN,")
        print("     2. wait 5 seconds,")
        print("     3. twist it back OUT,")
        print("     4. wait for the controller to finish booting (~30s,")
        print("        the arm's light goes steady).")
        print("   If it comes back with the same joint faulting, switch the")
        print("   control box off at the mains for 30 seconds instead.")
    elif err in (19, 28):
        print("   The controller lost the end-effector. Check the gripper's")
        print("   cable at the wrist and its 24V supply, then power-cycle.")
    elif err == 31:
        print("   A collision was detected (abnormal current). This one IS")
        print("   software-clearable - if it came straight back, the arm is")
        print("   still pressed against something. Free it by hand or jog it")
        print("   off with mouse_jog.bat, then run this again.")
    elif err == 22:
        print("   The arm is in a self-collision pose. Jog it out by hand")
        print("   (mouse_jog.bat) before anything else will run.")
    elif err == 23:
        print("   A joint is past its angle limit. Clear the error, then")
        print("   unlock and rotate that joint back by hand.")
    else:
        title, desc = code_text("error", err)
        print("   %s" % title)
        if desc:
            print("   %s" % desc)


def reset_arm(ip, assume_yes=False, skip_gripper=False):
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as e:
        print("   the xArm SDK is not installed here (%s)" % e)
        print("   -> python -m pip install --user xarm-python-sdk")
        return False

    for round_no in range(1, MAX_POWER_ROUNDS + 1):
        print("   connecting to %s ..." % ip)
        arm = None
        try:
            arm = XArmAPI(ip, is_radian=False)
            time.sleep(0.5)
            err, warn = report(arm, "Before")
            if err is None:
                return False
            if err == 0 and warn == 0:
                print("")
                print("   no fault to clear - re-enabling anyway so the")
                print("   next launcher starts from a known state.")
            print("")
            ok = clear_fault(arm)
            gripper_ok = rearm_gripper(arm, assume_yes, skip_gripper)
            err, warn = report(arm, "After")
            if ok and err == 0:
                return gripper_ok
            advise(arm, err)
        except Exception as e:
            print("   connection failed: %s" % e)
        finally:
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception:
                    pass

        if round_no == MAX_POWER_ROUNDS or assume_yes:
            return False
        print("")
        if not ask("   Power-cycle it now, then press y + Enter to re-check "
                   "(anything else quits) [y/N]"):
            return False
        print("   waiting 10s for the controller to finish booting ...")
        time.sleep(10)
    return False


# ----------------------------- main -----------------------------

def main():
    args = [a for a in sys.argv[1:]]
    assume_yes = "--yes" in args
    skip_kill = "--no-kill" in args
    skip_camera = "--no-camera" in args
    skip_gripper = "--no-gripper" in args
    positional = [a for a in args if not a.startswith("--")]
    ip = positional[0] if positional else DEFAULT_IP

    log = start_log()
    print("=" * 60)
    print("   xArm 6  -  RESET EVERYTHING")
    print("=" * 60)
    print("   arm %s   project %s" % (ip, HERE))
    if log:
        print("   log %s" % log)
    print("   The arm is NOT moved. The fingers may open (you are asked).")

    head(1, "leftover scripts holding the arm or camera")
    if skip_kill:
        print("   skipped (--no-kill).")
    else:
        kill_stale()

    head(2, "RealSense camera")
    if skip_camera:
        print("   skipped (--no-camera).")
        camera_ok = None
    else:
        camera_ok = reset_camera()

    head(3, "network")
    reachable = ping(ip)
    if reachable:
        print("   %s answers." % ip)
    else:
        print("   %s does NOT answer." % ip)
        print("   -> is the control box powered and booted?")
        print("   -> is the ethernet cable in the controller's LAN port?")
        print("   -> is this PC still on 192.168.1.x ? (a docking station or")
        print("      a Wi-Fi change is enough to lose that route)")
        print("   -> UFACTORY Studio at http://%s:18333 is the quick test."
              % ip)

    head(4, "controller fault")
    arm_ok = False
    if reachable:
        arm_ok = reset_arm(ip, assume_yes, skip_gripper)
    else:
        print("   skipped - the arm is not reachable.")

    print("")
    print("=" * 60)
    print("   RESULT")
    print("=" * 60)
    print("   camera : %s" % ("skipped" if camera_ok is None
                              else "ready" if camera_ok else "NOT ready"))
    print("   network: %s" % ("ready" if reachable else "NOT ready"))
    print("   arm    : %s" % ("ready" if arm_ok else "NOT ready"))
    print("")
    if arm_ok:
        print("   The launchers will run again.")
        print("   The arm is wherever it stopped - send it home first:")
        print("     go_home.bat            (asks y/N before it moves)")
        print("   then carry on with vision\\launchers or realtime\\launchers.")
    else:
        print("   Do the step printed above, then run this again.")
        print("   If it keeps coming back on the same joint, stop and check")
        print("   the cabling - repeated servo faults are hardware talking.")
    print("")
    return 0 if arm_ok else 1


if __name__ == "__main__":
    sys.exit(main())

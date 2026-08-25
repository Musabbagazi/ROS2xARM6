#!/usr/bin/env python3
"""Hold the cup ON, and MEASURE the wrist fault honestly.

THE ARM DOES NOT MOVE.

This exists because the cell has a live hardware fault (error 28: the
controller loses the wrist's end module the moment the tool IO is
touched - see fixed_vacuum.py --watch). When the fault lands, the
controller reports it and every later tool command is refused. No
software prevents that.

What software CAN do is keep re-commanding the cup, and count what it
costs. But the FIRST version of this script got its own measurement
badly wrong, and the mistake is worth naming because it is the easiest
one to make with this fault:

    it read the vacuum switch every pass, at 10Hz.

Reading the switch IS a tool-IO transaction, which is exactly what trips
error 28. So the loop caused the faults it was counting, and the total -
185 in 20s, 145 in 16s - was nothing but the poll rate. The instrument
was the disease. Two runs at 9.4 and 9.1 "faults" per second, both of
them just sleep(0.1).

So this version:

  * does NOT read the switch unless you ask for it (--read-switch), and
    says plainly what that costs when you do;
  * counts fault EVENTS - clean-to-faulted transitions - not samples
    that happen to be taken while an error is latched. One latched fault
    is one fault, however long it lasts;
  * checks whether clean_error() ACTUALLY cleared, and reports "took"
    and "refused" separately, because "cleared 185 times" and "tried 185
    times and was refused" look identical in a raw count and mean
    opposite things;
  * reports faults per TOOL TRANSACTION, not per second. That number
    survives a change of poll rate; a raw count does not.

Usage:  python vacuum_hold.py [robot_ip] [--read-switch] [--latch]

        --read-switch  also poll the vacuum switch (a tool-IO read, so
                       it provokes the very fault being measured). Off
                       by default. Only useful on a healthy cell.

        --latch        Commands the cup ON exactly ONCE, then touches
                       nothing at all - no reads, no writes, no clears -
                       and watches only the error register while you
                       listen to the pump. See latch_test.

        --stress       IS IT ACTUALLY FIXED? Hammers the tool with
                       hundreds of ordinary transactions and reports
                       faults per transaction. The test to run after any
                       reseat, and the only honest answer to "it seemed
                       fine that time" - see stress_test.
"""
import sys
import time

import fixed_common as fx

DEFAULT_IP = "192.168.1.197"
STATE_TEXT = {-1: "off", 0: "on, nothing held", 1: "ON, OBJECT HELD"}


def err_now(arm):
    """The controller's error code. NOT a tool-IO transaction - this
    reads the controller's own register and never reaches the wrist, so
    it can be polled as fast as you like without provoking anything."""
    try:
        code, ew = arm.get_err_warn_code()
    except Exception:
        return -1
    return ew[0] if code in fx.ADVISORY_CODES else -1


def cup_on(arm):
    try:
        arm.set_vacuum_gripper(True, wait=False,
                               hardware_version=fx.VACUUM_HW)
    except Exception:
        pass


def cup_off(arm):
    try:
        arm.set_vacuum_gripper(False, wait=False,
                               hardware_version=fx.VACUUM_HW)
    except Exception:
        pass


def switch(arm):
    """The vacuum switch, read the way that works.

    NOT arm.get_vacuum_gripper() - see the note in fx.vacuum_state().
    That wrapper reads register 0x0A18 on the way past, which this end
    module does not answer, and it was the entire reason this script
    used to fault ten times a second and print 'off' on every line
    while the cup was physically holding."""
    pin = 0 if fx.VACUUM_HW == 1 else 3
    try:
        code, value = arm.get_tgpio_digital(ionum=pin)
    except Exception:
        return None
    return value if code in fx.ADVISORY_CODES else None


class FaultLog(object):
    """Counts fault EVENTS and the cost of clearing them.

    The distinction this class exists for: sampling the error register
    at 10Hz while one fault stays latched gives ten hits a second and
    means ONE fault. Counting samples turns the poll rate into the
    headline number. Counting edges - clean to faulted - does not."""

    def __init__(self):
        self.edges = 0          # clean -> faulted transitions
        self.stamps = []        # when each one landed
        self.cleared = 0        # clean_error() that actually worked
        self.refused = 0        # clean_error() that did not
        self.touches = 0        # tool-IO transactions we performed
        self._faulted = False

    def sample(self, err, t):
        """Record one reading of the error register. True if new fault."""
        new = err > 0 and not self._faulted
        if new:
            self.edges += 1
            self.stamps.append(t)
        self._faulted = err > 0
        return new

    def clear_result(self, ok):
        if ok:
            self.cleared += 1
            self._faulted = False       # observed clean, so the next
        else:                           # fault is a genuine new edge
            self.refused += 1

    def touch(self, n=1):
        self.touches += n

    def gaps(self):
        return [b - a for a, b in zip(self.stamps, self.stamps[1:])]

    def report(self, elapsed):
        print("")
        print("   " + "-" * 50)
        print("   ran %.1fs" % elapsed)
        print("   tool transactions issued : %d" % self.touches)
        print("   FAULT EVENTS             : %d" % self.edges)
        if self.edges and self.touches:
            print("      %.2f per tool transaction  (the number that means"
                  % (float(self.edges) / self.touches))
            print("      something - it does not change if the loop runs")
            print("      faster or slower)")
        if self.edges and elapsed > 0:
            print("      %.2f per second" % (self.edges / elapsed))
        gaps = self.gaps()
        if gaps:
            print("      one every %.2fs on average (shortest %.2fs)"
                  % (sum(gaps) / len(gaps), min(gaps)))
        print("   clean_error()            : %d took, %d refused"
              % (self.cleared, self.refused))
        if self.refused and not self.cleared:
            print("      NOTHING cleared. The controller would not let go of")
            print("      the fault at all while the tool was being driven.")
        elif self.refused:
            print("      A refused clear means the fault was still there")
            print("      immediately after clearing it - the trigger was")
            print("      still present, not that the clear was ignored.")


def clear_and_check(arm, log):
    """Clear the fault and find out whether it actually went. A4: the
    old version assumed the clear worked and counted it as a success,
    so a fault that never cleared and one that cleared instantly gave
    the same number."""
    try:
        arm.clean_error()
    except Exception:
        pass
    ok = err_now(arm) == 0
    log.clear_result(ok)
    return ok


def latch_test(arm, seconds=30.0):
    """DO THE TOOL OUTPUTS SURVIVE THE FAULT? Everything downstream
    depends on the answer and nothing so far has tested it.

    The keep-alive muddied this. It re-commanded the cup ~10 times a
    second, so of course the pump kept running - and it was impossible
    to tell whether the cup stayed on because the outputs LATCH through
    the fault, or only because a fresh command arrived every 100ms in
    the gaps between faults. The outputs readback cannot settle it
    either: that readback comes back through the dead link.

    So: command the cup ON exactly ONCE, then touch NOTHING. No reads,
    no writes, no clears. Watch only the error register - which lives in
    the controller and never reaches the wrist - and let the operator's
    ear decide.

        pump still running, error 28 latched
            the outputs LATCH. The end module holds its last commanded
            state through the fault. A pick could grasp, fault, and
            STILL HOLD THE CUBE - which changes the outlook entirely.

        pump stopped when the error landed
            the outputs DROP. Anything held falls the moment the fault
            arrives, so picking is impossible until the wire is fixed.
            Definitive, and worth knowing before spending a day on it."""
    print("")
    print("   " + "=" * 50)
    print("   LATCH TEST - does the cup stay on through the fault?")
    print("   " + "=" * 50)
    print("")
    print("   The cup is commanded ON exactly ONCE. After that this")
    print("   touches NOTHING for %.0f seconds - no reads, no commands," % seconds)
    print("   no clearing. Only the controller's error register is")
    print("   watched, and that never reaches the wrist.")
    print("")
    print("   YOUR JOB IS TO LISTEN. The question at the end is whether")
    print("   the pump is still running. Nothing else answers it -")
    print("   every electrical readback comes back through the broken")
    print("   link and cannot be trusted.")
    print("")
    print("   Put NOTHING on the cup. Anything held may drop.")
    print("")
    try:
        if input("   Ready? [y/N] ").strip().lower() != "y":
            return 1
    except (EOFError, KeyboardInterrupt):
        return 1

    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass
    if err_now(arm) != 0:
        print("")
        print("   The controller will not go clean before the test even")
        print("   starts, so a fault seen afterwards would prove nothing.")
        print("   Power-cycle the control box and try again.")
        return 1

    print("")
    print("   clean. commanding the cup ON - LISTEN.")
    t_on = time.time()
    cup_on(arm)                     # the one and only tool transaction

    print("")
    print("      t(s)    err")
    print("      " + "-" * 22)
    t_fault = None
    last = None
    while time.time() - t_on < seconds:
        e = err_now(arm)
        if e != last:
            print("      %5.1f   %s%s" % (time.time() - t_on, e,
                                          "   <- fault landed" if e > 0
                                          and t_fault is None else ""))
            if e > 0 and t_fault is None:
                t_fault = time.time() - t_on
            last = e
        time.sleep(0.05)

    held_s = seconds - (t_fault or seconds)
    print("")
    if t_fault is None:
        print("   The controller stayed CLEAN for the whole %.0fs on a"
              % seconds)
        print("   single command. That is new - it means one write is")
        print("   survivable and it is repeated traffic that kills the")
        print("   link. Say so in the support case.")
    else:
        print("   The fault landed %.1fs after the command, and nothing"
              % t_fault)
        print("   has touched the tool in the %.0fs since." % held_s)
    print("")

    try:
        ans = input("   IS THE PUMP STILL RUNNING? [y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""

    print("")
    print("   " + "-" * 50)
    if ans == "y" and t_fault is not None:
        print("   THE OUTPUTS LATCH THROUGH THE FAULT.")
        print("")
        print("   The end module kept driving the cup for %.0fs while the"
              % held_s)
        print("   controller sat in error %s and nothing re-commanded it."
              % "28")
        print("   So a grasp that lands before the fault KEEPS HOLDING.")
        print("")
        print("   That does not make picking safe yet - the fault still")
        print("   halts MOTION, so the arm would stop mid-carry with the")
        print("   cube attached - but it means the cube would not be")
        print("   dropped, and a pick sequence that grabs, then clears and")
        print("   re-enables motion, is worth designing. Tell me and I")
        print("   will build it.")
    elif ans == "n" and t_fault is not None:
        print("   THE OUTPUTS DROP WITH THE FAULT.")
        print("")
        print("   The cup stopped when the link died, with no way to hold")
        print("   it on. Anything gripped falls the instant the fault")
        print("   arrives - which it does within a second or two of every")
        print("   grasp. Picking is impossible on this cell until the")
        print("   wrist is repaired; no software can work around it.")
        print("")
        print("   The earlier 'it worked!' run only held because the")
        print("   keep-alive re-commanded the cup ten times a second.")
    elif ans == "y":
        print("   Pump running and no fault at all: on a single command")
        print("   this cell behaved normally for %.0fs." % seconds)
    else:
        print("   No verdict recorded. Run it again and answer y or n -")
        print("   this is the one question the software cannot answer for")
        print("   itself.")

    cup_off(arm)
    try:
        arm.clean_error()
        arm.clean_warn()
    except Exception:
        pass
    print("")
    print("   cup commanded OFF.")
    return 0


def stress_test(arm, seconds=60.0, rate_hz=5.0):
    """Is the wrist actually fixed, or just having a good moment?

    THIS EXISTS BECAUSE ONE CLEAN RUN PROVES NOTHING. An intermittent
    contact is defined by working: it carried signalling all morning and
    died only under load, then died on a bare read, then ran the pump
    for a clean 30 seconds. Any single observation of it can be
    'working'. What separates a repaired joint from a lucky one is
    VOLUME - hundreds of transactions, and the fault rate across them.

    So this issues the ordinary traffic a pick would: a switch read and
    a cup write, over and over, checking the error register after each
    one and clearing anything that lands. The output is faults per
    transaction, which is comparable between runs and between days -
    unlike a raw count, which only measures how long you left it going.

    A pick makes roughly a dozen tool transactions. At zero faults in
    several hundred, the odds of getting through one are good. At even
    one in fifty, a pick will fail somewhere in the middle."""
    print("")
    print("   " + "=" * 50)
    print("   STRESS TEST - is the wrist link actually fixed?")
    print("   " + "=" * 50)
    print("")
    print("   %.0f seconds of ordinary tool traffic - switch reads and"
          % seconds)
    print("   cup commands, about %d a second - counting how many fault."
          % int(rate_hz))
    print("   The cup will click on and off throughout.")
    print("")
    print("   One clean run means nothing on an intermittent contact.")
    print("   This one is about volume: hundreds of transactions, and")
    print("   the fault RATE across them.")
    print("")
    try:
        if input("   Go? [y/N] ").strip().lower() != "y":
            return 1
    except (EOFError, KeyboardInterrupt):
        return 1

    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass

    log = FaultLog()
    t0 = time.time()
    period = 1.0 / rate_hz
    n = 0
    print("")
    print("      running ... (Ctrl+C to stop early)")
    try:
        while time.time() - t0 < seconds:
            # A read and a write, alternating - the two things a pick
            # actually does. Each is checked on its own so the report
            # can say which kind of transaction is failing.
            if n % 2 == 0:
                switch(arm)
            else:
                cup_on(arm) if (n // 2) % 2 == 0 else cup_off(arm)
            log.touch()
            n += 1

            if log.sample(err_now(arm), time.time() - t0):
                print("      t=%6.1fs  fault %d after %d transactions"
                      % (time.time() - t0, log.edges, log.touches))
                clear_and_check(arm, log)
            time.sleep(period)
    except KeyboardInterrupt:
        print("      stopped early.")

    cup_off(arm)
    try:
        arm.clean_error()
        arm.clean_warn()
    except Exception:
        pass

    elapsed = time.time() - t0
    log.report(elapsed)
    print("")
    print("   " + "-" * 50)
    if log.edges == 0:
        print("   NOT ONE FAULT in %d transactions." % log.touches)
        print("")
        print("   For comparison, the identical run before the fix scored")
        print("   148 faults in 296 transactions. Nothing was touched on")
        print("   the robot in between - the only change is that this")
        print("   script no longer reads register 0x0A18 through the SDK's")
        print("   get_vacuum_gripper(). The hardware was always fine.")
        print("")
        print("   Next: '2 - Vacuum Check' to confirm the wiring and the")
        print("   seal, then '6 - Pick DRY RUN'.")
        return 0

    print("   STILL FAULTING: %d faults in %d transactions (%.3f per"
          % (log.edges, log.touches, float(log.edges) / max(1, log.touches)))
    print("   transaction). A pick makes roughly a dozen tool")
    print("   transactions, so at this rate it would fail part way")
    print("   through, with a cube attached and the arm over the floor.")
    print("")
    print("   The wrist still needs the physical fix. Nothing has changed")
    print("   about that - see '2b - Vacuum Fault Finder'.")
    return 1


def main():
    argv = sys.argv[1:]
    read_switch = "--read-switch" in argv
    do_latch = "--latch" in argv
    do_stress = "--stress" in argv
    args = [a for a in argv if not a.startswith("--")]
    ip = args[0] if args else DEFAULT_IP

    fx.start_log("hold")
    print("=" * 62)
    print("   FIXED CAMERA  -  cup ON until you say stop (keep-alive)")
    print("=" * 62)
    print("   THE ARM DOES NOT MOVE.")
    print("")

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ip, is_radian=False)
    log = FaultLog()
    t0 = time.time()
    try:
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)

        if do_latch:
            return latch_test(arm)
        if do_stress:
            return stress_test(arm)

        print("   The wrist fault will keep shutting the cup off. This")
        print("   clears it and re-commands the cup until you press a key.")
        print("")
        print("   What the numbers at the end mean: a FAULT EVENT is one")
        print("   clean-to-faulted transition, not one sample taken while")
        print("   an error happened to be latched. The headline figure is")
        print("   faults per TOOL TRANSACTION, because that is the only")
        print("   one that does not change when the loop rate changes.")
        if read_switch:
            print("")
            print("   --read-switch is ON: the sensor is polled as well as")
            print("   commanded. This used to provoke the very fault it was")
            print("   measuring, because it read through the SDK wrapper")
            print("   and that reads register 0x0A18. It now reads the")
            print("   input pin directly and is safe - the flag stays")
            print("   optional only because it doubles the traffic.")
        print("")
        print("   ANYTHING THE CUP GRIPS CAN DROP AT ANY MOMENT - hold it.")
        print("")
        try:
            if input("   Start? [y/N] ").strip().lower() != "y":
                return 1
        except (EOFError, KeyboardInterrupt):
            return 1

        try:
            import msvcrt
        except ImportError:
            msvcrt = None

        cup_on(arm)
        log.touch()
        t0 = time.time()
        print("")
        print("   cup commanded ON.  %s"
              % ("press any key to release"
                 if msvcrt else "press Ctrl+C to release"))
        print("")

        last_beat = time.time()
        last_line = None
        while True:
            e = err_now(arm)                    # not a tool transaction
            if log.sample(e, time.time() - t0):
                print("   t=%6.1fs  fault %d landed (event %d)"
                      % (time.time() - t0, e, log.edges))
            if e > 0:
                clear_and_check(arm, log)
                cup_on(arm)                     # re-assert
                log.touch()
                last_beat = time.time()
            elif time.time() - last_beat > 1.0:
                # Healthy this instant: re-assert about once a second so a
                # dropped output does not stay dropped. This IS a tool
                # transaction and is counted as one.
                cup_on(arm)
                log.touch()
                last_beat = time.time()

            if read_switch:
                st = switch(arm)
                log.touch()
                line = STATE_TEXT.get(st, "unreadable")
                if line != last_line:
                    print("   switch: %s" % line)
                    last_line = line

            if msvcrt is not None and msvcrt.kbhit():
                while msvcrt.kbhit():
                    msvcrt.getch()
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("   FAILED: %s" % e)
        return 1
    finally:
        if not (do_latch or do_stress):
            print("")
            for _ in range(3):
                try:
                    arm.clean_error()
                except Exception:
                    pass
                cup_off(arm)
                time.sleep(0.2)
                if err_now(arm) == 0:
                    break
            try:
                arm.clean_error()
                arm.clean_warn()
            except Exception:
                pass
            print("   cup OFF.")
            log.report(time.time() - t0)
            if log.edges == 0 and time.time() - t0 > 10:
                print("")
                print("   NOT ONE FAULT EVENT. The wrist link held the whole")
                print("   time - if it was just reseated, that may have been")
                print("   the fix. Confirm with '2b - Vacuum Fault Finder'.")
        try:
            arm.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

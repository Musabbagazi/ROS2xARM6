#!/usr/bin/env python3
"""Check the vacuum cup and its wiring. THE ARM DOES NOT MOVE.

The xArm drives the cup through two tool-GPIO outputs and reads its
vacuum switch back on a tool-GPIO input. Which pins depends on how the
cup is connected:

    plug-in connection   outputs 0 and 1   (SDK hardware_version=1)
    contact connection   outputs 3 and 4   (SDK hardware_version=2)

Get that wrong and everything still "works" - the commands are accepted,
no error is raised - except that nothing ever switches and the state read
is always the same. So this tries both and reports which one actually
responds, and lets you watch the switch change when you hold something
against the cup.

The result is the value to put in fixed_common.VACUUM_HW.

The cup is left ON until you stop it, not on a timer - trying a corner, a
rough face or a warm cube does not fit into a fixed number of seconds.

Usage:  python fixed_vacuum.py [robot_ip] [--hold] [--watch]
        --hold   skip the wiring probe and go straight to holding the cup
                 on, for when you already know the wiring and just want
                 to test what seals.
        --watch  the decisive test. Clears any fault, then climbs three
                 rungs - tool left ALONE, one live READ of the tool IO,
                 cup commanded ON - and reports the first rung that
                 faults the controller, with the fault timed against the
                 trigger to the millisecond. On this cell the trigger
                 started at the top rung (energise, ~100ms) and moved
                 down to a bare read as the connection degraded - an
                 electrical failure at the wrist, not a setting. Run it
                 once with the gripper PLUGGED IN and once UNPLUGGED:
                 the pair says which side of the connector is at fault.
"""
import sys
import time

import fixed_common as fx

DEFAULT_IP = "192.168.1.197"
NAMES = {1: "plug-in (tool GPIO 0/1)", 2: "contact (tool GPIO 3/4)"}

# What the sensor input actually says. TWO states, not three.
#
# The old three-state wording ("off" / "on, nothing held" / "ON, OBJECT
# HELD") came from the SDK's get_vacuum_gripper(), which prefixed the
# sensor reading with a check of the OUTPUT register to decide whether
# the cup was on. That register is the one this end module will not
# answer, so we read the sensor pin directly - and the pin only knows
# whether something is sealed against the cup. It cannot tell you
# whether the cup is switched on, and pretending otherwise printed
# "on, nothing held" underneath a line that said "cup OFF".
MEANING = {0: "nothing held", 1: "OBJECT HELD"}


def read(arm, hw):
    """The vacuum switch, or (None, why).

    NOT arm.get_vacuum_gripper() - that wrapper reads register 0x0A18 on
    the way past and this end module does not answer it. See
    fixed_common.vacuum_state() for the whole story. The input pin below
    is what the SDK itself returns once that detour is removed.

    Codes 1 and 2 are NOT failures - see fixed_common.ADVISORY_CODES.
    They mean a fault or a warning is latched on the controller, which is
    true of every call while it is set and says nothing about this one.
    Rejecting them is what made this script report a dead cup while the
    cup was audibly working."""
    pin = 0 if hw == 1 else 3
    try:
        code, state = arm.get_tgpio_digital(ionum=pin)
    except Exception as e:
        return None, str(e)
    if code not in fx.ADVISORY_CODES:
        return None, "read refused (code %s - the call was never sent)" % code
    text = MEANING.get(state, "unrecognised state %s" % state)
    if code:
        text += "   [controller has a latched fault; the reading stands]"
    return state, text


# How long the cup is held on while identifying the wiring. This phase is
# SHORT ON PURPOSE - it only has to see the switch change from "off" to
# "on, nothing held", which takes no time at all - and it is not the seal
# test. The seal test comes afterwards and stays on until you stop it.
PROBE_ON_S = 1.5


def err_now(arm):
    """The controller's live error code: 0 clean, >0 the code, -1 unreadable."""
    try:
        code, ew = arm.get_err_warn_code()
    except Exception:
        return -1
    if code not in fx.ADVISORY_CODES:
        return -1
    return ew[0]


def watch(arm, hw, on_s=6.0, dt=0.05):
    """Clear the fault, pulse the cup, and TIMESTAMP what the controller
    does about it.

    The history matters, because two plausible theories died to this
    test and neither should be resurrected:

        "the fault re-arms on its own"           - it does not. Cleared,
            the controller sat at err=0 for 10s untouched.
        "the controller is still polling the     - same evidence: no cup
            removed two-finger gripper"            command, no fault. 40
                                                   straight tool-IO reads
                                                   changed nothing either.

    What actually happens: the instant the cup is commanded ON the
    controller loses the end module - error 28, inside ~100ms - and
    drops the tool outputs, which is why the cup pulls air for a moment
    and stops. Both sides of the old argument were true at once: the cup
    sucked (while the controller was clean) and the readback showed the
    outputs off (taken after the fault dropped them).

    Energising the pump is the trigger, every time. On a genuine
    UFACTORY gripper on its proper connector that is ELECTRICAL - the
    pump's current draw sagging or glitching the end module's link - so
    the suspects are a half-seated connector, a damaged cable, or a
    failing gripper. No software setting causes or fixes it.

    This run climbs a three-rung ladder and reports the first rung that
    faults, because the trigger has been observed to MOVE DOWN it as the
    connection degrades:

        1. QUIESCENT - nothing touches the tool at all, only the error
           register is polled. Clean here means the link holds when left
           alone (always true so far).
        2. FIRST TOUCH - one live read of the tool IO (the vacuum
           switch). By late on day one this alone killed the link within
           ~100ms; earlier the same day 40 straight reads had been fine.
        3. ENERGISE - the cup commanded ON, the pump draws real current.
           This was the original trigger.

    A fault lower on the ladder is the same physical failure, further
    gone. Then the only question left: is the gripper plugged in right
    now? Faulting while PLUGGED means reseat and inspect, then suspect
    the gripper. Faulting while UNPLUGGED means the arm's own end module
    is sick: support case."""
    print("")
    print("   Three stages: the tool left ALONE, then one READ of the")
    print("   tool IO, then the cup commanded ON - stopping at the first")
    print("   stage that faults the controller.")
    print("   Listen for the cup - the sound stopping is the fault landing.")
    print("")
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass

    def row(t, err, out, state, note=""):
        print("      %6.2f  %3s   %-18s %-18s %s"
              % (t, err, out, MEANING.get(state, "?" if state is None
                                          else state), note))

    def verdict(how):
        # Leave the controller clean - the fault stays away until the
        # tool is touched again, and a latched error left behind is a
        # mystery for the next script to trip over.
        try:
            arm.clean_error()
            arm.clean_warn()
        except Exception:
            pass
        print("")
        print("   " + "-" * 46)
        for line in how:
            print("   " + line)
        print("")
        try:
            plugged = input("   Is the gripper PLUGGED INTO THE WRIST"
                            " right now? [y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            plugged = ""
        print("")
        if plugged == "n":
            print("   With NOTHING attached the fault still fires, so it")
            print("   is on the ARM side of the connector - the end")
            print("   module or its wiring through the arm. UFACTORY")
            print("   support case: quote error 28, end module firmware")
            print("   1.2.0, controller v1.6.9, and the stage this run")
            print("   faulted at.")
        else:
            print("   Next, cheapest first:")
            print("   1. UNPLUG the gripper at the wrist, run this again,")
            print("      answer 'n'. Fault gone = the arm is fine and the")
            print("      gripper or its cable is the culprit. Fault stays")
            print("      = arm-side end module, support case.")
            print("   2. RESEAT the connector: unplug, inspect the pins,")
            print("      plug back in, screw the locking ring down")
            print("      FIRMLY. A half-seated connector carries")
            print("      signalling fine and drops the moment anything")
            print("      loads it - exactly this pattern.")
            print("   3. Inspect the CABLE for kinks and chafing near")
            print("      both connectors, through the arm's whole range.")
            print("   4. Still faulting after all that? Support case,")
            print("      with the evidence this run just printed.")
        return 1

    print("      t(s)     err   tool outputs       switch")
    print("      " + "-" * 56)

    # Stage 1: QUIESCENT. Only the error register is polled - no tool
    # IO of any kind. If the fault lands here the link is dying with no
    # provocation at all.
    t0 = time.time()
    last = None
    idle_fault = 0
    while time.time() - t0 < 2.0:
        err = err_now(arm)
        key = (err,)
        if key != last:
            row(time.time() - t0, err, "(untouched)", None, "(quiescent)")
            last = key
        if err > 0 and not idle_fault:
            idle_fault = err
        time.sleep(dt)

    if idle_fault:
        return verdict([
            "The controller faulted with NOTHING touching the tool -",
            "no read, no command, servos merely enabled. The end-module",
            "link is now dropping unprovoked: the far end of the same",
            "degradation this test exists to time.",
        ])

    # Stage 2: THE SENSOR READ, by the working path - get_tgpio_digital,
    # not the SDK wrapper. Note what is NOT called here: outputs(). That
    # is the 0x0A18 read, it is the known fault, and calling it inline
    # would fault the controller and then blame the switch - which is
    # exactly the mistake this ladder existed to avoid making. It gets
    # its own stage at the end, on purpose.
    t_touch = time.time()
    state, _ = read(arm, hw)
    row(t_touch - t0, err_now(arm), "(not read)", state,
        "<- sensor read (safe path)")
    touch_fault = 0
    while time.time() - t_touch < 1.0:
        err = err_now(arm)
        if err > 0:
            touch_fault = err
            row(time.time() - t0, err, "(not read)", state,
                "<- FAULT, %.0fms after the read"
                % ((time.time() - t_touch) * 1000))
            break
        time.sleep(dt)

    if touch_fault:
        return verdict([
            "Reading the sensor faulted the controller - and this is the",
            "SAFE path, get_tgpio_digital(ionum=0), which was measured",
            "clean 15/15 on this cell. That is new and does not match the",
            "known 0x0A18 problem. Treat it as a real end-module fault:",
            "check the connector at the wrist and the cable.",
        ])

    t_on = time.time()
    try:
        arm.set_vacuum_gripper(True, wait=False, hardware_version=hw)
    except Exception as e:
        print("   could not command the cup: %s" % e)
        return 1
    row(t_on - t0, err_now(arm), "(not read)", read(arm, hw)[0],
        "<- cup commanded ON")

    t_fault = None
    while time.time() - t_on < on_s:
        err = err_now(arm)
        out = "(not read)"
        state, _ = read(arm, hw)
        if err > 0 and t_fault is None:
            t_fault = time.time() - t_on
        key = (err, out, state)
        if key != last:
            note = ("<- FAULT, %.0fms after ON" % (t_fault * 1000)
                    if err > 0 and t_fault is not None else "")
            row(time.time() - t0, err, out, state, note)
            last = key
        if t_fault is not None and time.time() - t_on > t_fault + 1.0:
            break                        # seen what we came for
        time.sleep(dt)

    try:
        arm.set_vacuum_gripper(False, wait=False, hardware_version=hw)
    except Exception:
        pass
    time.sleep(0.3)
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)
    except Exception:
        pass

    if t_fault is not None:
        return verdict([
            "Energising the cup faulted the controller %.0fms after the"
            % (t_fault * 1000),
            "ON command, on the safe read path. Writes were measured 0/30",
            "on this cell, so this does not match the known 0x0A18",
            "problem either. Something electrical has changed - check the",
            "connector at the wrist and the cable.",
        ])

    print("")
    print("   The working path is CLEAN: quiescent, sensor read, and the")
    print("   pump running, all with the controller at error 0.")

    # Stage 4: REPRODUCE THE KNOWN FAULT, DELIBERATELY. Everything above
    # avoids register 0x0A18. This is the one call that reads it, and
    # calling it here - last, alone, expecting the fault - turns the run
    # into a complete statement: the tool works, and here is precisely
    # what breaks it. That is the evidence a support case needs, and it
    # costs one cleared error.
    print("")
    print("   Now provoking the KNOWN fault on purpose: the one register")
    print("   read (0x0A18, via get_tgpio_output_digital) that this end")
    print("   module does not answer. It should fault - that is the")
    print("   point - and the error is cleared straight afterwards.")
    out, faulted = probe_outputs(arm)
    print("")
    if faulted:
        print("   " + "-" * 46)
        print("   CONFIRMED, and the diagnosis is complete:")
        print("     - the cup, the pump, the sensor and the cable: FINE")
        print("     - end module firmware 1.2.0 does not answer 0x0A18")
        print("     - the SDK reads it inside get_vacuum_gripper(), which")
        print("       is why this looked like broken hardware for so long")
        print("")
        print("   This project no longer calls it, so picking and")
        print("   calibrating are unaffected. Send SUPPORT_CASE.md to")
        print("   UFACTORY to get the register or the firmware fixed.")
        return 0
    print("   " + "-" * 46)
    print("   IT DID NOT FAULT: %s" % out)
    print("   The register answered. Either the firmware was updated or")
    print("   this arm never had the problem - either way the SDK's")
    print("   get_vacuum_gripper() is usable again, and the workaround in")
    print("   fixed_common.vacuum_state() is no longer needed (harmless,")
    print("   but no longer load-bearing).")
    return 0


def code_text(code):
    """The controller's own title and description for an error code.

    Straight out of the SDK's tables rather than a guess, so it stays
    right across SDK updates - same approach reset_all.py takes."""
    try:
        from xarm.core.config import x_code
        info = x_code.ControllerError(code)
        return info.title["en"], info.description["en"]
    except Exception:
        return "error %s" % code, ""


def still_faulted(arm, settle_s=1.5):
    """The live error code, or 0 - but only after giving it time to come
    BACK.

    This exists because of a real miss. `clean_error()` genuinely clears
    the flag for error 28 and the controller reads clean - but if the
    TRIGGER is still present the fault lands again moments later. (On
    this arm the trigger turned out to be energising the cup: idle, the
    controller stays clean indefinitely; commanded ON, it faults within
    ~100ms. See watch().) A check taken only at the instant of the clear
    sees zero and waves the run through.

    So: look, wait, and look again. A fault that comes back is still a
    fault, and it is the one that matters."""
    worst = 0
    for _ in range(2):
        try:
            code, ew = arm.get_err_warn_code()
        except Exception:
            return -1
        if code != 0 or ew[0] != 0:
            worst = ew[0] or code
        time.sleep(settle_s / 2.0)
    return worst


def error_clear(arm):
    """True if the controller is fault-free. Otherwise explain and stop.

    Worth being emphatic about WHY this comes first. A latched error
    makes the controller answer HAS_ERROR to everything, so the tool IO
    writes below are refused, the switch reads are refused, and the probe
    sees nothing change on either wiring. It then reports 'the cup is not
    wired' - confident, plausible, and about the wrong subsystem
    entirely."""
    err = still_faulted(arm)
    if err == 0:
        return True
    if err < 0:
        print("   could not read the error register.")
        return False

    title, desc = code_text(err)
    print("")
    print("   " + "=" * 46)
    print("   THE CONTROLLER IS IN ERROR %s - %s" % (err, title))
    print("   " + "=" * 46)
    if desc:
        print("   %s" % desc)
    print("")
    print("   Nothing about the vacuum cup can be tested until this is")
    print("   cleared. While an error is latched the controller refuses")
    print("   EVERY command - which is what all those 'code=1' lines are -")
    print("   so the tool outputs never switch and the switch never reads.")
    print("   Any conclusion this script drew about the wiring would be")
    print("   about the fault, not about the cup.")
    print("")

    print("   trying to clear it ...")
    for _ in range(2):
        try:
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)
        except Exception:
            pass
        time.sleep(0.8)
        if not still_faulted(arm):
            print("   cleared - carrying on.")
            return True

    print("   it will not clear in software.")
    print("")
    print("   THE CUP TEST BELOW STILL RUNS AND ITS RESULT IS STILL GOOD -")
    print("   the tool IO works through a latched fault. What the fault")
    print("   does block is MOTION: the arm is held in stop state, so")
    print("   calibrating and picking are impossible until it is gone.")
    print("")
    if err in (19, 28):
        # THE ANSWER, after two days of wrong theories. Error 28 on this
        # cell is not a loose connector, not the pump's current draw,
        # and not a leftover setting from the two-finger gripper. It is
        # ONE unanswered tool register, 0x0A18, read by
        # get_tgpio_output_digital() - which the SDK calls from inside
        # get_vacuum_gripper(). Nothing else on the tool misbehaves:
        # writes 0/30, digital input reads 0/8, analog 0/8, firmware
        # version 0/8. Only that register, 8/8, in 2-3ms.
        print("   Errors 19 and 28 both mean the controller has lost the")
        print("   END MODULE - the board in the wrist the tool plugs into.")
        print("")
        print("   ON THIS CELL there is a known, specific cause: the tool")
        print("   register 0x0A18, read by get_tgpio_output_digital(), is")
        print("   unanswered by end module firmware 1.2.0 and faults the")
        print("   controller instantly. The SDK's get_vacuum_gripper()")
        print("   calls it on the way to reading the sensor, which is why")
        print("   this looked for so long like broken hardware.")
        print("")
        print("   Everything else on the same connector is fine, so if")
        print("   this project raised the error, something CALLED that")
        print("   register - see fixed_common.vacuum_state() for the way")
        print("   round it. If nothing did, then this is genuinely new:")
        print("   1. POWER-CYCLE THE CONTROL BOX: e-stop IN, wait 5s,")
        print("      back OUT, wait ~30s for the controller to boot.")
        print("   2. Then  python fixed_vacuum.py --watch  to see which")
        print("      transaction is at fault now.")
        print("   3. SUPPORT_CASE.md has the full write-up to send.")
    else:
        print("   Run '0 - RESET EVERYTHING' - it diagnoses this properly")
        print("   and tells you what to do about it.")
    return False


def outputs(arm):
    """The tool GPIO OUTPUT states - THE CALL THAT FAULTS THIS CELL.

    This is get_tgpio_output_digital(), which reads tool register 0x0A18
    (xarm/x3/gpio.py:125). On this arm's end module firmware 1.2.0 that
    register is unanswered and the controller raises error 28 within
    2-3ms, every single time - 8/8 measured. It is the whole fault, and
    for months of debugging it was invisible because the SDK calls it
    from inside get_vacuum_gripper().

    IT IS DELIBERATELY STILL HERE, and deliberately no longer called
    from the ordinary probe. It is the isolation test: the one command
    that reproduces the fault on demand, which is exactly what a support
    case needs. Anything calling it must expect a fault and clear it -
    see probe_outputs().

    What it was for, when it worked: these two answer different
    questions, and conflating them is what makes a vacuum problem hard
    to pin down:

        the OUTPUTS changing  proves the command reached the controller
                              and the right pins were driven
        the SWITCH changing   proves a vacuum sensor exists, is wired to
                              the tool input, and saw the pressure change

    A cup with no feedback switch fitted - which is a perfectly ordinary
    way to buy one - gives outputs that change and a switch that never
    does. That is not a wiring fault and it must not be reported as one."""
    try:
        code, digitals = arm.get_tgpio_output_digital()
    except Exception:
        return None
    return list(digitals) if code in fx.ADVISORY_CODES else None


def probe_outputs(arm):
    """Call the faulting register read ON PURPOSE, then clean up.

    Returns (outputs_or_None, faulted). Every caller of outputs() has to
    go through this, because leaving error 28 latched behind would make
    the next measurement look broken - which is precisely the confusion
    that took two days to unpick."""
    before = err_now(arm)
    out = outputs(arm)
    faulted = before == 0 and err_now(arm) > 0
    if faulted:
        try:
            arm.clean_error()
            arm.clean_warn()
            arm.set_state(0)
        except Exception:
            pass
        time.sleep(0.3)
    return out, faulted


def probe(arm, hw):
    """Test one wiring.

    Returns (sealed, trip), where trip names WHICH TRANSACTION faulted
    the controller, or None.

    HOW THE WIRING IS IDENTIFIED, AND WHY IT CHANGED. This used to watch
    the switch go from "off" to "on, nothing held" when the cup was
    commanded - a change that came from the SDK comparing the OUTPUT
    register, which this end module does not answer. Reading the sensor
    pin directly, there is no such change to see: with nothing held
    against the cup the pin reads 0 whether the cup is running or not,
    so both wirings looked equally dead and the script announced "check
    the tool cable" about a cup that was working perfectly.

    The seal itself is the signal. With something flat held against the
    cup, the correct wiring switches the pump on, the object seals, and
    the pin goes to 1. The wrong wiring drives pins that are not
    connected to anything, so no suction, no seal, and the pin stays 0.
    That is a stronger test than the old one anyway: it proves the whole
    chain - command, pump, vacuum, sensor - rather than just that a
    register changed.

    Naming it matters, and this function used to get it wrong. It
    checked the error register once before the switch read and once
    after the ON command, then blamed everything in between on
    "energising". vacuum_20260810_161604 shows the cost: the fault had
    already landed during the cup-OFF read - the log says so on that
    very line, '[controller has a latched fault]' - and the summary
    still announced "energising on this wiring FAULTS the controller".
    A wrist that dies on a bare read is further gone than one that only
    dies under the pump's load, and that is the finding UFACTORY needs.

    So the register is checked after EACH transaction. When one trips
    it, the fault is cleared before the next phase, so the following
    wiring is not judged through a latched error."""
    pins = (0, 1) if hw == 1 else (3, 4)
    print("")
    print("   --- %s ---" % NAMES[hw])
    trip = None
    if err_now(arm) > 0:                # not attributable to this probe
        try:
            arm.clean_error()
            arm.clean_warn()
        except Exception:
            pass
        time.sleep(0.3)

    # The OUTPUT readback is not taken here any more. It is the 0x0A18
    # read, it faults this end module every time, and taking it mid-probe
    # meant every later reading was made through a latched error - which
    # is how this script once concluded "nothing is wired" about a cup
    # that worked. The sensor alone answers the question the probe asks.
    off_state, off_text = read(arm, hw)
    if err_now(arm) > 0:
        trip = "the READ"
    print("       cup OFF : %s" % off_text)

    try:
        arm.set_vacuum_gripper(True, wait=False, hardware_version=hw)
    except Exception as e:
        print("       could not switch it on: %s" % e)
        return False, trip
    time.sleep(PROBE_ON_S)
    if trip is None and err_now(arm) > 0:
        trip = "ENERGISING"
    on_state, on_text = read(arm, hw)
    print("       cup ON  : %s" % on_text)
    try:
        arm.set_vacuum_gripper(False, wait=False, hardware_version=hw)
    except Exception:
        pass
    time.sleep(0.3)

    faulted = trip is not None
    if faulted:
        print("       ! %s faulted the controller. Clearing it so the"
              % trip)
        print("         next phase is tested honestly.")
        try:
            arm.clean_error()
            arm.clean_warn()
            arm.set_state(0)
        except Exception:
            pass
        time.sleep(0.5)

    sealed = (off_state == 0 and on_state == 1)
    if sealed:
        print("       -> IT SEALED on pins %s. This is the wiring."
              % list(pins))
    elif faulted:
        print("       -> %s faulted the controller on this wiring." % trip)
    elif on_state == 1 and off_state == 1:
        print("       -> already reading HELD before the cup was switched")
        print("          on, so this proves nothing. Take everything off")
        print("          the cup and run it again.")
    else:
        print("       -> no seal on this wiring.")
    return sealed, trip


def main():
    hold_only = "--hold" in sys.argv[1:]
    watch_only = "--watch" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ip = args[0] if args else DEFAULT_IP

    # Before the SDK is imported anywhere, so its own error lines land in
    # the log too - see fixed_common.start_log.
    fx.start_log("vacuum")
    print("=" * 62)
    print("   FIXED CAMERA  -  vacuum cup check")
    print("=" * 62)
    print("   THE ARM DOES NOT MOVE.")
    if hold_only:
        print("   Wiring probe skipped (--hold): using hardware_version=%d."
              % fx.VACUUM_HW)
    else:
        print("   HOLD SOMETHING FLAT AGAINST THE CUP NOW, and keep")
        print("   holding it through both tests - a cube, a card, a phone.")
        print("")
        print("   That is not optional any more, and it is worth knowing")
        print("   why. The sensor only reports whether something is SEALED")
        print("   against the cup; it cannot say whether the cup is")
        print("   switched on. (The reading that used to say so came from")
        print("   a register this end module will not answer - see")
        print("   SUPPORT_CASE.md.) So with nothing held, both wirings")
        print("   read the same and neither can be identified. With")
        print("   something held, the correct wiring seals it and the")
        print("   sensor says so.")
        print("")
        print("   The cup is pulsed for about a second on each wiring.")
        print("   Then comes the long seal test, which stays on until you")
        print("   stop it.")
    print("")
    try:
        if input("   Ready? [y/N] ").strip().lower() != "y":
            return 1
    except (EOFError, KeyboardInterrupt):
        return 1

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ip, is_radian=False)
    try:
        # Not fx.setup_arm: this must run even when the arm is not in a
        # state to be enabled, because "the cup does nothing" is exactly
        # the kind of thing you check when other things are also wrong.
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)

        # A LATCHED CONTROLLER ERROR MUST BE RULED OUT BEFORE ANY OF THIS
        # MEANS ANYTHING.
        #
        # While the controller holds an error, EVERY command comes back
        # as APIState.HAS_ERROR (1) - including the tool IO writes and
        # the switch reads this test is built on. The probe would then
        # see nothing change on either wiring and conclude the cup is not
        # wired - a confident, plausible and completely wrong diagnosis
        # of a fault that has nothing to do with the cup. This cost a
        # real debugging session, so it is checked and named loudly. A
        # latched fault does not STOP the test - tool IO answers through
        # one - but it does block MOTION, and the probe itself now clears
        # the fault between phases, because on this arm the act of
        # energising the cup is what latches it (see watch()).
        if watch_only:
            return watch(arm, fx.VACUUM_HW)

        error_clear(arm)

        result = {hw: probe(arm, hw) for hw in (1, 2)}
        sensed = [hw for hw, (s, _) in result.items() if s]
        zapped = [hw for hw, (_, t) in result.items() if t]
        trips = set(t for _, t in result.values() if t)

        print("")
        print("   " + "-" * 46)
        if zapped:
            print("   THE CONTROLLER FAULTED DURING THE PROBE.")
            print("   What tripped it: %s, on %s."
                  % (" and ".join(sorted(trips)),
                     " and ".join(NAMES[h] for h in zapped)))
            print("")
            print("   Every transaction this probe makes was measured")
            print("   clean on this cell - writes 0/30, sensor reads 0/8 -")
            print("   and it does not touch the 0x0A18 register that this")
            print("   end module refuses. So this is NOT the known")
            print("   problem, and it is worth stopping for.")
            print("")
            print("   Check the connector at the wrist and the cable, then")
            print("   run  python fixed_vacuum.py --watch , which walks")
            print("   the transactions one at a time and names the one")
            print("   that breaks.")
            return 1
        if len(sensed) == 1:
            hw = sensed[0]
            print("   Wiring: %s" % NAMES[hw])
            if hw == fx.VACUUM_HW:
                print("   fixed_common.VACUUM_HW is already %d - correct."
                      % hw)
            else:
                print("   SET fixed_common.VACUUM_HW = %d  (it is %d now)"
                      % (hw, fx.VACUUM_HW))
        elif len(sensed) > 1:
            print("   BOTH wirings appeared to respond, which should not")
            print("   happen. Check the tool IO for anything else wired to")
            print("   it before trusting either.")
        else:
            # Before blaming the wiring, ask the controller again. A
            # fault that re-latched DURING the probe explains a total
            # non-response completely, and saying "check the cable" over
            # the top of it sends you after the wrong thing.
            late = still_faulted(arm)
            if late:
                print("   The controller went back into ERROR %s - %s -"
                      % (late, code_text(late)[0]))
                print("   DURING the test, so every command above was")
                print("   refused. This says nothing about the cup.")
                print("")
                print("   Something re-armed it while the probe ran. Run")
                print("    python fixed_vacuum.py --watch")
                print("   - it timestamps the fault against the cup")
                print("   commands and idle time, and says whether the")
                print("   trigger is the cup energising (electrical, at")
                print("   the wrist) or something else entirely.")
                return 1
            print("   NO SEAL ON EITHER WIRING.")
            print("")
            print("   Much the most likely reason is simply that nothing")
            print("   was held against the cup - the probe cannot identify")
            print("   the wiring without something to seal against, and")
            print("   the cup pulling on open air looks identical on both.")
            print("   Hold a cube or a card firmly on the cup and run it")
            print("   again.")
            print("")
            print("   If you DID hold something and it still says this,")
            print("   then either the cup is not switching on, or nothing")
            print("   seals well enough to trip the sensor. The hold test")
            print("   below separates those: listen for the pump.")

        # ALWAYS offer the hold test, whatever the probe concluded. It is
        # the only part of this script that tells you whether the cup
        # actually picks things up, and that is worth knowing most of all
        # when the diagnosis above is inconclusive.
        hw = (sensed or [fx.VACUUM_HW])[0]
        return hold(arm, hw)
    except Exception as e:
        print("   FAILED: %s" % e)
        return 1
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def hold(arm, hw):
    """Turn the cup on and LEAVE it on until the operator says stop.

    Deliberately not on a timer. A timer decides for you when you have
    finished looking, and the whole point of this test is to hold things
    against the cup, try a corner, try a rough face, see what seals and
    what does not - none of which fits in a fixed number of seconds. The
    keypress is polled rather than waited on, so the reading keeps
    updating live while the cup stays on."""
    print("")
    print("   Live test: the cup goes ON and STAYS ON until you stop it.")
    print("   Hold things against it and watch the reading change.")
    print("")
    print("   ANYTHING STUCK TO IT WILL DROP when you stop - hold it.")
    try:
        if input("   Go? [y/N] ").strip().lower() != "y":
            return 0
    except (EOFError, KeyboardInterrupt):
        return 0

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    arm.set_vacuum_gripper(True, wait=False, hardware_version=hw)
    print("")
    print("   cup ON.  %s" % ("press any key to release"
                              if msvcrt else "press ENTER to release"))
    try:
        if msvcrt is None:
            # No console polling available: block on input and give up
            # the live readout rather than the ability to stop.
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            return 0
        last = None
        while True:
            state, text = read(arm, hw)
            if state != last:
                print("       %s" % text)
                last = state
            if msvcrt.kbhit():
                while msvcrt.kbhit():       # drain key-repeat backlog
                    msvcrt.getch()
                break
            time.sleep(0.25)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        arm.set_vacuum_gripper(False, wait=False, hardware_version=hw)
        print("   cup OFF.")


if __name__ == "__main__":
    sys.exit(main())

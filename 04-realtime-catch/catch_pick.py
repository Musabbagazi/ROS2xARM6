#!/usr/bin/env python3
"""Catch a cube that is MOVING - the arm ambushes it, it does not chase.

  vision\\vision_pick3.py   scan -> WAIT for the cube to stop -> aim once
                            -> descend open-loop -> grab
  handoff\\handoff_pick.py  watch -> FOLLOW the cube -> dive while the
                            hand is momentarily steady -> grab
  this one                  watch -> FIT a velocity -> go to a point the
                            cube has not reached yet -> WAIT there, open
                            -> close on the clock -> verify

The last of those is the only one that never needs the cube to hold
still, and the reason it works is that it moves the uncertainty from
POSITION to TIME. The arm is parked and settled before the cube arrives,
so a timing error of dt costs (dt x speed) of displacement - a few
millimetres at the speeds this accepts - instead of costing the whole
grab. See catch_common's module note for why chasing cannot do that.

HOW TO SEND IT A CUBE
  * in a straight line, at a steady speed, roughly 2-25 cm/s
  * across the middle of the cell, not toward the arm's base
  * flat and not tumbling - it must keep one face up
  * let go of it. A cube still in your hand is the handoff project's
    job, and that one is built to keep your fingers out of the grab.

WHAT IT WILL REFUSE, AND SAY SO
  a curved path, a bouncing or decelerating cube, one that is falling,
  one too fast to meet, and one whose path never enters the arm's reach.
  Every refusal names what to change.

STOPPING IT
  Press ESC or q with the CONSOLE window focused, at any time. The arm
  finishes its current step and retreats. The e-stop is still the real
  stop - keep it in hand. NOTE that between the commit and the close the
  arm is deliberately sitting still in the cube's path; that is the
  design, not a fault.

Usage: python catch_pick.py [robot_ip]
"""
import os
import sys
import time

import numpy as np
from xarm.wrapper import XArmAPI

import catch_common as cc
import catch_detect as cd
import catch_track as ct
import vision3 as v3
import vision_common as vc
from detect_cube import MODEL_FILE

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

SESSION_TIMEOUT_S = 90.0        # per catch attempt
MAX_CATCHES = 10
MAX_MISSED = 3

# Radius (px) around the PREDICTED pixel within which a detection is
# still "the cube I am tracking". The prediction already removes the
# shift caused by the arm's own motion, so this only has to cover the
# cube's own travel between two frames plus detection noise. Wider than
# the handoff project's 140: a cube crossing the frame at 25cm/s moves
# further between frames than a hand drifting does.
LOCK_PX = 200.0

# Consecutive empty frames before the lock is dropped and the tracker
# re-acquires whatever is nearest.
RELOCK_MISSES = 4

TALK_EVERY_S = 2.5


def abort_requested():
    """Did the operator press ESC/q in the console? Drains the buffer so
    a held key cannot queue up dozens of aborts."""
    try:
        import msvcrt
    except ImportError:
        return False
    hit = False
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"\x1b", b"q", b"Q"):
            hit = True
    return hit


def reachable(arm, pose):
    return vc.ik(arm, pose) is not None


def goto_watch(arm):
    """Move to the watching pose, resetting through HOME if refused.

    Firmware 1.6.9 seeds IK from the current joints, so a pose can be
    refused for a huge joint sweep that is really just a different
    configuration branch. HOME is a known-good branch the straight-down
    poses are reachable from - the same lesson both siblings learned."""
    try:
        vc.moveto(arm, "watch pose", cc.WATCH_POSE)
    except vc.SweepRefused as e:
        print("  (%s)" % e)
        print("  resetting the arm's configuration through HOME")
        vc.movej(arm, "HOME", cc.HOME_DEG)
        vc.moveto(arm, "watch pose", cc.WATCH_POSE)


def held_cube_recovery(arm):
    """The gripper is not open at startup - it may still hold a cube."""
    pos = vc.gripper_pos(arm)
    if pos is None or pos > 750:
        return False
    print("\n!! The gripper is not open (at %.0f) - it may still be holding "
          "something." % pos)
    print("   [r] open it here   [q] quit")
    if input("   r/q > ").strip().lower() == "r":
        input("   Put a hand under it, then press Enter... ")
        vc.grip(arm, "open (release)", 850)
        return False
    return True


class FrameClock:
    """Converts the camera's frame clock to the wall clock, and back.

    The track lives on the DEPTH FRAME's timestamps, which is right -
    they say when the photons arrived rather than when Python got round
    to looking. But once the arm commits there are no more frames, and
    the close has to be timed against something that keeps running. One
    (frame_t, wall_t) pair taken at the last observation links the two;
    they run at the same rate, so a single offset is all it takes."""

    def __init__(self):
        self.offset = None          # wall = frame + offset

    def mark(self, t_frame):
        self.offset = time.time() - t_frame

    def to_wall(self, t_frame):
        return t_frame if self.offset is None else t_frame + self.offset


def watch_and_catch(arm, cam, calib, ref, timing):
    """One catch attempt. Returns (result, colour).

    result is 'caught', 'timeout', 'abort' or 'missed'. On 'caught' the
    arm is at the lift pose still holding the cube."""
    scan_yaw = vc.SCAN_POSE[5]
    track = ct.CubeTrack()
    clock = FrameClock()
    locked_px = None
    misses = 0
    last_talk = 0.0
    t0 = time.time()

    def talk(msg):
        """Say something to the operator - at most once every few seconds.

        Rate-limited by TIME ALONE. Resetting the timer whenever the
        wording changes defeats the limiter completely: this loop
        alternates between states many times a second, every message
        differs from the last, and every one prints."""
        nonlocal last_talk
        now = time.time()
        if now - last_talk >= TALK_EVERY_S:
            print("   " + msg)
            last_talk = now

    while time.time() - t0 < SESSION_TIMEOUT_S:
        if abort_requested():
            return "abort", None

        code, cur = arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError("could not read the arm position (code %s)"
                               % code)
        color, depth, intr = cam.frame()
        best, _cands, n_boxes, reason, t_frame = cd.detect_moving(
            color, depth, intr, prefer_near=locked_px,
            near_radius=LOCK_PX if locked_px is not None else None)

        if best is None:
            misses += 1
            if locked_px is not None and misses >= RELOCK_MISSES:
                locked_px = None
                track.reset()
                talk("lost it - looking again")
            elif n_boxes and reason:
                talk("I see a cube, but %s" % reason)
            elif n_boxes:
                talk("I see a cube but cannot measure its top face")
            else:
                talk("send a cube across the cell")
            continue
        misses = 0
        locked_px = list(best["pixel"])
        clock.mark(t_frame)

        gx, gy, gz, yaw = cc.aim_moving(calib, ref, cur, best)
        lo, hi = cc.catch_band(ref, cur[2], best["height_mm"])
        width = best["width_mm"]

        # --- gates that do not depend on motion ---
        if not lo <= gz <= hi:
            track.reset()
            if gz < lo:
                talk("that reads below the floor - I will not reach for it")
            else:
                talk("that is too close to the camera to measure - send it "
                     "lower or further out")
            continue
        if not cc.MIN_CUBE_MM <= width <= cc.MAX_CUBE_MM:
            track.reset()
            talk("that reads %.0fmm across - outside what the gripper takes"
                 % width)
            continue

        track.add(t_frame, gx, gy, gz, best)

        # --- can I be somewhere it is going, before it gets there? ---
        def reach_test(px, py, pz):
            return (reachable(arm, [px, py, pz, 180.0, 0.0, yaw])
                    and reachable(arm, [px, py, pz + cc.HOVER_H,
                                        180.0, 0.0, yaw]))

        plan, why = ct.plan_intercept(track, t_frame, cur[:3],
                                      timing, reach_test=reach_test)
        if plan is None:
            talk(why)
            continue

        f = track.fit()
        grasp_yaw = cc.choose_yaw(yaw, f["heading_deg"])
        print("\n   locked on: %.0fmm/s heading %.0f deg, fit %.1fmm over "
              "%d samples" % (plan["speed"], plan["heading_deg"],
                              plan["resid_mm"], f["n"]))
        print("   meeting it at (%.0f, %.0f) in %.2fs - %.2fs to spare"
              % (plan["x"], plan["y"], plan["lead_s"], plan["slack_s"]))
        return (execute_catch(arm, cam, calib, ref, track, clock, plan,
                              grasp_yaw, width, timing),
                best.get("color", "unknown"))

    return "timeout", None


def execute_catch(arm, cam, calib, ref, track, clock, plan, yaw, width,
                  timing):
    """Go to the ambush point, wait, and close on the clock.

    Everything after the descent is blind by necessity - at the grasp
    pose the camera is ~140mm from the cube top, inside the D435's
    minimum range - so this is where the project spends the credibility
    the tracking earned."""
    gx, gy, gz = plan["x"], plan["y"], plan["z"]
    hover = [gx, gy, gz + cc.HOVER_H, 180.0, 0.0, yaw]
    grasp = [gx, gy, gz, 180.0, 0.0, yaw]
    lift = [gx, gy, gz + cc.HOVER_H, 180.0, 0.0, yaw]
    if not reachable(arm, hover) or not reachable(arm, grasp):
        print("   (the meeting point turned out to be unreachable - "
              "backing off)")
        return "missed"

    open_p = cc.open_pulses(ref, width)
    close_p = cc.close_pulses(ref, width)

    # Open BEFORE moving. The fingers have to be waiting wide by the time
    # the arm is in position, and opening is the slowest thing here.
    vc.grip(arm, "open wide to wait", open_p)

    t_start = time.time()
    try:
        vc.moveto(arm, "go to the meeting point", hover)
    except vc.SweepRefused as e:
        print("   (%s - backing off)" % e)
        return "missed"
    timing.record(plan["travel_mm"], time.time() - t_start)

    # One last look before going blind. The arm is now directly over the
    # spot with the cube still outside the camera's near limit, so if it
    # is in frame this re-anchors the arrival time against a fresh
    # measurement instead of an extrapolation made a second ago. If it is
    # not in frame - most often because it has not arrived yet, which is
    # exactly what was planned - the original prediction stands.
    t_arrive = refresh_arrival(cam, calib, ref, arm, track, clock, plan)

    # Is there still time to descend?
    descend_s = timing.estimate(cc.HOVER_H)
    now_w = time.time()
    if now_w + descend_s + cc.GRIPPER_CLOSE_S > t_arrive:
        late = now_w + descend_s + cc.GRIPPER_CLOSE_S - t_arrive
        print("   it beat me to it by %.2fs - letting this one go" % late)
        vc.moveto(arm, "back off", lift)
        return "missed"

    t_desc = time.time()
    try:
        vc.moveto(arm, "drop into its path", grasp)
    except vc.SweepRefused as e:
        print("   (%s - backing off)" % e)
        vc.moveto(arm, "back off", lift)
        return "missed"
    timing.record(cc.HOVER_H, time.time() - t_desc)

    # --- the wait, and the close ---
    # Close so the fingers are ARRIVING as the cube does, not starting to
    # move once it is already through.
    t_close = t_arrive - cc.GRIPPER_CLOSE_S
    waited = wait_until(t_close)
    if waited is None:
        print("   (stopped while waiting - opening and backing off)")
        vc.moveto(arm, "back off", lift)
        return "abort"
    if waited < -0.25:
        # The descent overran the plan badly enough that the cube has
        # already gone past. Closing now grabs air at best.
        print("   the descent took %.2fs too long - it is past me" % -waited)
        vc.moveto(arm, "back off", lift)
        return "missed"

    final = vc.grip(arm, "close (catch it)", close_p)
    held = vc.grasp_status(arm)
    if held == vc.UNKNOWN:
        if final is None:
            vc.moveto(arm, "back off", lift)
            raise RuntimeError("could not read the gripper after closing - "
                               "check it before running again")
        held = vc.EMPTY if final <= close_p + cc.POSITION_MISS_TOL \
            else vc.HOLDING
    if held == vc.EMPTY:
        # never open on one reading - opening is the only action here
        # that cannot be undone
        again = vc.grasp_status(arm)
        pos = vc.gripper_pos(arm)
        if again == vc.HOLDING or (pos is not None
                                   and pos > close_p + cc.POSITION_MISS_TOL):
            print("   (first reading said empty but the fingers are stalled "
                  "at %s - treating it as a hold)" % pos)
            held = vc.HOLDING

    if held == vc.EMPTY:
        print("   caught nothing (fingers reached %s of %d)"
              % (final, close_p))
        vc.grip(arm, "open (missed)", open_p)
        vc.moveto(arm, "back off", lift)
        vc.grip(arm, "open", 850)
        return "missed"

    if final is not None:
        true_w = ref["w0_mm"] + (final - ref["stall0"]) / cc.PULSES_PER_MM
        print("   caught it: fingers stalled at %d -> cube ~%.1fmm (the "
              "camera said %.1fmm)" % (final, true_w, width))
    arm.set_tcp_load(vc.GRIPPER_WEIGHT + cc.BOX_WEIGHT, vc.GRIPPER_COG)
    vc.moveto(arm, "lift clear", lift)
    return "caught"


def refresh_arrival(cam, calib, ref, arm, track, clock, plan):
    """Re-time the cube's arrival from one fresh frame at the hover pose.

    Returns the arrival time on the WALL clock. Falls back to the planned
    time whenever the cube is not measurable from here - which is the
    normal case if it simply has not arrived yet, so this is a bonus and
    never a requirement."""
    planned = clock.to_wall(plan["t_arrive"])
    try:
        code, cur = arm.get_position(is_radian=False)
        if code != 0:
            return planned
        cam.skip(1)                     # one fresh frame after the move
        color, depth, intr = cam.frame()
        best, _c, _n, _why, t_frame = cd.detect_moving(color, depth, intr)
        if best is None:
            return planned
        gx, gy, gz, _yaw = cc.aim_moving(calib, ref, cur, best)
        # Only believe it if it is the cube we were tracking: it has to
        # be near where the fit says that cube is right now. A different
        # cube sitting in the frame would otherwise re-time the catch to
        # something that is not coming.
        px, py, _pz = track.predict(t_frame)
        if np.hypot(gx - px, gy - py) > 60.0:
            return planned
        track.add(t_frame, gx, gy, gz, best)
        clock.mark(t_frame)
        f = track.fit()
        ok, _why = track.confident(f)
        if not ok:
            return planned
        # time at which the fitted path is nearest the ambush point
        vx, vy, speed = f["vx"], f["vy"], f["speed"]
        if speed < 1.0:
            return planned
        dt = ((plan["x"] - px) * vx + (plan["y"] - py) * vy) / (speed ** 2)
        revised = clock.to_wall(t_frame + dt)
        shift = revised - planned
        if abs(shift) > 1.0:
            # a correction that large means the fit changed character,
            # not that the timing was refined; trust the plan
            return planned
        if abs(shift) > 0.02:
            print("   (re-timed by %+.0fms from a fresh look)"
                  % (shift * 1000.0))
        return revised
    except Exception:
        # a refinement is never worth failing a catch over
        return planned


def wait_until(t_wall):
    """Sleep until t_wall, watching for an abort.

    Returns the signed seconds left at the moment it stopped waiting:
    ~0 after a normal wait, and a NEGATIVE number when the deadline had
    already gone by on entry - i.e. how late the arm is. None if the
    operator aborted. The caller needs the lateness to decide whether
    closing is still worth anything."""
    remaining = t_wall - time.time()
    if remaining <= 0.0:
        return remaining
    while True:
        remaining = t_wall - time.time()
        if remaining <= 0.0:
            return remaining
        if abort_requested():
            return None
        time.sleep(min(0.01, remaining))


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


def deliver(arm, color):
    """Put the cube down. Uses the drop spots taught for the floor picker
    (vision/places.json) when they exist, and otherwise hands it back."""
    places = v3.load_places() or {}
    place = places.get(color) if color in ("red", "blue") else None
    if place is None:
        place = places.get("red")
    if place is None:
        print("\n   No drop spot is taught (vision/places.json) - I will "
              "hold it.")
        input("   Put your hand under it and press Enter to release... ")
        vc.grip(arm, "open (hand back)", 850)
        arm.set_tcp_load(vc.GRIPPER_WEIGHT, vc.GRIPPER_COG)
        return
    a = above(place, 60.0)
    if not reachable(arm, a) or not reachable(arm, place):
        print("\n   The drop spot is not reachable from here - I will hold "
              "it instead.")
        input("   Put your hand under it and press Enter to release... ")
        vc.grip(arm, "open (hand back)", 850)
        arm.set_tcp_load(vc.GRIPPER_WEIGHT, vc.GRIPPER_COG)
        return
    vc.moveto(arm, "above the drop spot", a)
    vc.moveto(arm, "drop spot", place)
    vc.grip(arm, "open (release)", 850)
    arm.set_tcp_load(vc.GRIPPER_WEIGHT, vc.GRIPPER_COG)
    vc.moveto(arm, "retreat", a)
    print("   %s cube delivered." % color)


def main():
    log = vc.start_log("catch")
    if log:
        print("(run log: %s)" % log)
    if not os.path.exists(MODEL_FILE):
        print("cube_model.pt not found in vision\\ - train the v3 model "
              "first ('2 - Train Cube Model').")
        return
    calib, ref = cc.load_calibration()
    if calib is None:
        return

    # BEFORE setup_arm, not after: setup_arm enables the gripper, and
    # that is where the finger speed is applied.
    prev_speed = cc.slow_down()

    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
        if not reachable(arm, cc.WATCH_POSE):
            raise RuntimeError("the watch pose %s is not reachable"
                               % cc.WATCH_POSE[:3])
        if held_cube_recovery(arm):
            arm.disconnect()
            return
    except RuntimeError as e:
        print("ABORT (no motion):", e)
        arm.disconnect()
        return

    print("""
================= REAL-TIME CATCH =================
The arm watches a cube MOVE, works out where it is
going, goes to a spot ahead of it, and closes as it
arrives. It never waits for the cube to stop.

Send the cube:
  * in a straight line at a steady speed
  * roughly %.0f to %.0f cm/s
  * across the middle of the cell
  * flat, not tumbling - one face stays up
  * LET GO of it. A cube still in your hand is the
    handoff project's job.

It will refuse a curved, bouncing or falling path
and say what to change. Refusing costs one cube;
guessing costs a collision.

Between committing and closing the arm SITS STILL
in the cube's path with the fingers open. That is
the design - the camera cannot see that close, so
the catch is timed, not watched.

Slowed down for working next to a person:
  arm      %d deg/s   (floor picker: %d)
  fingers  %d         (floor picker: %d)

Press ESC or q in THIS window to stop between
steps. Keep the e-stop in hand - during a single
move it is the only thing that stops the arm.
===================================================""" % (
        cc.MIN_CATCH_SPEED / 10.0, cc.MAX_CATCH_SPEED / 10.0,
        cc.CATCH_SPEED, prev_speed[0],
        cc.CATCH_GRIPPER_SPEED, prev_speed[2]))
    if input("\nStart? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        arm.disconnect()
        return

    timing = ct.ArmTiming()
    done = 0
    try:
        with v3.Camera(floor_ref=None) as cam:
            missed = 0
            for attempt in range(1, MAX_CATCHES + 1):
                goto_watch(arm)
                vc.grip(arm, "open", 850)
                print("\n--- catch %d - send a cube across ---" % attempt)
                result, color = watch_and_catch(arm, cam, calib, ref, timing)
                if result == "caught":
                    deliver(arm, color)
                    done += 1
                    missed = 0
                elif result == "missed":
                    missed += 1
                    if missed >= MAX_MISSED:
                        print("Missed %d in a row - stopping so we can look "
                              "at why." % missed)
                        break
                elif result == "abort":
                    print("Stopped at your request.")
                    break
                else:
                    print("Nothing sent for %.0fs - stopping."
                          % SESSION_TIMEOUT_S)
                    break
        goto_watch(arm)
        vc.movej(arm, "HOME", cc.HOME_DEG)
        print("\nDone. Caught %d moving cube(s). Parked at HOME." % done)
        if timing.samples:
            print("(measured arm travel: %.0fmm/s over %d moves - if that "
                  "differs a lot from the %.0f default, set it in "
                  "catch_track.ArmTiming)"
                  % (timing.rate, timing.samples,
                     ct.ArmTiming.DEFAULT_RATE_MM_S))
    except RuntimeError as e:
        print("\nSTOPPED:", e)
        print("Re-run when ready - the held-cube check runs at startup.")
    except KeyboardInterrupt:
        print("\nSTOPPED: interrupted from the keyboard.")
    except Exception:
        import traceback
        print("\nSTOPPED: unexpected error -")
        traceback.print_exc()
    finally:
        if log:
            print("(run log saved: %s)" % log)
        arm.disconnect()


if __name__ == "__main__":
    main()

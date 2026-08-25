#!/usr/bin/env python3
"""Take a cube out of your HAND - continuous tracking, the arm moves.

How it differs from the floor picker (vision/vision_pick3.py), which
stays exactly as it is:

  floor picker   scan -> WAIT for the cube to stop moving -> aim once ->
                 descend open-loop -> grab
  this one       watch -> FOLLOW the cube continuously at a height where
                 the camera can still see it -> dive only while the hand
                 is momentarily steady -> grab -> verify

The follow height is not a preference, it is the camera's near limit:
the D435 measures nothing closer than ~280mm and the camera sits ~142mm
above the fingers, so the cube disappears from view about 165mm before
the fingers reach it. The arm therefore tracks your hand down to 180mm
above the cube and the last stretch is open-loop. That is why it waits
for your hand to be still before diving, and why the gripper's own
grasp sensor - not the camera - decides whether it actually got it.

HOW TO HOLD IT
  * by the BOTTOM EDGES, fingertips low and clear of the top face.
    The camera measures the nearest surface inside the cube's box, so a
    finger over the top is what it will measure instead of the cube.
  * still-ish. Drifting is fine, it follows; the dive needs about a
    second of calm.
  * inside the height band the script prints at startup.

STOPPING IT
  Press ESC or q with the CONSOLE window focused, at any time. The arm
  finishes its current step (they are short) and retreats. The e-stop is
  still the real stop - keep it in hand.

Usage: python handoff_pick.py [robot_ip]
"""
import os
import sys
import time

import numpy as np
from xarm.wrapper import XArmAPI

import handoff_common as hc
import handoff_detect as hd
import vision3 as v3
import vision_common as vc
from detect_cube import MODEL_FILE

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

SESSION_TIMEOUT_S = 90.0    # per handoff attempt
MAX_HANDOFFS = 10
MAX_MISSED_GRABS = 3

# How many consecutive frames with no cube before the lock is dropped and
# the tracker re-acquires whatever is nearest. Low, because the usual
# reason is that the hand moved out of view - and then following the old
# pixel is worse than looking again.
RELOCK_MISSES = 4

# Radius (px) around the PREDICTED pixel within which a detection is
# still "the cube I am following". The prediction already removes the
# shift caused by the arm's own motion, so this only has to cover hand
# movement between two steps plus detection noise.
LOCK_PX = 140.0

TALK_EVERY_S = 2.5          # do not repeat the same advice every frame


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


def goto_wait(arm):
    """Move to the watching pose, resetting through HOME if refused.

    Same lesson as the floor picker: firmware 1.6.9 seeds IK from the
    current joints, so a pose can be refused for a huge joint sweep that
    is really just a different configuration branch. HOME is a known-good
    branch that the straight-down poses are reachable from."""
    try:
        vc.moveto(arm, "wait pose", hc.WAIT_POSE)
    except vc.SweepRefused as e:
        print("  (%s)" % e)
        print("  resetting the arm's configuration through HOME")
        vc.movej(arm, "HOME", hc.HOME_DEG)
        vc.moveto(arm, "wait pose", hc.WAIT_POSE)


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


def reachable(arm, pose):
    return vc.ik(arm, pose) is not None


def follow_and_grab(arm, cam, calib, ref):
    """One handoff: follow the held cube, then take it.

    Returns (result, colour) where result is 'held', 'timeout', 'abort'
    or 'missed'. On 'held' the arm is at the lift pose, still holding
    the cube, and the colour is what to sort it by."""
    scan_yaw = vc.SCAN_POSE[5]
    locked_px = None            # pixel of the cube we are following
    last_target = None          # its last accepted arm-frame XY
    steady = 0
    misses = 0
    last_talk = 0.0
    announced_h = hc.FOLLOW_H
    t0 = time.time()

    def talk(msg):
        """Say something to the operator - at most once every few seconds.

        Rate-limited by TIME ALONE. Resetting the timer whenever the
        wording changed, which is what this did at first, defeats the
        limiter completely: the loop alternates between two states many
        times a second, every message differs from the last, and every
        single one prints. The result was a wall of contradictory advice
        with the useful line buried in it."""
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
        cam.skip(1)             # one fresh frame after the step
        color, depth, intr = cam.frame()
        best, _cands, n_boxes, reason = hd.detect_held(
            color, depth, intr, prefer_near=locked_px,
            near_radius=LOCK_PX if locked_px is not None else None)

        if best is None:
            steady = 0
            misses += 1
            if locked_px is not None and misses >= RELOCK_MISSES:
                locked_px = None
                last_target = None
                talk("lost track of it - looking again")
            elif n_boxes and reason:
                talk("I see a cube, but %s" % reason)
            elif n_boxes:
                talk("I see a cube but cannot measure its top face")
            else:
                talk("show me a cube")
            continue
        misses = 0

        gx, gy, gz, yaw = hc.aim_hand(calib, ref, cur, best)
        lo, hi = hc.hold_band(ref, cur[2], best["height_mm"])
        width = best["width_mm"]

        # --- the safety gates, before anything moves ---
        if gz < lo:
            steady = 0
            talk("hold it about %.0fcm higher - that low it could be a "
                 "cube on the floor"
                 % ((lo - gz) / 10.0 + 0.5))
            continue
        if gz > hi:
            # The cube is nearer the camera than the camera can measure.
            # That is MY problem before it is yours: back the camera off
            # and the same cube becomes measurable. Only ask you to move
            # it if the arm has run out of height.
            steady = 0
            need = hc.MIN_CAM_MM - best["depth_mm"] + 15.0
            up = min(hc.MAX_Z_STEP, need)
            higher = [cur[0], cur[1], cur[2] + up, 180.0, 0.0, scan_yaw]
            if cur[2] + up <= hc.WAIT_POSE[2] + 0.1 and reachable(arm, higher):
                talk("backing off %.0fcm to get you in focus" % (up / 10.0))
                try:
                    vc.moveto(arm, "back off for focus", higher)
                except vc.SweepRefused:
                    pass
            else:
                talk("lower it about %.0fcm - that close I cannot measure it"
                     % (need / 10.0 + 0.5))
            continue
        if not hc.MIN_CUBE_MM <= width <= hc.MAX_CUBE_MM:
            steady = 0
            talk("that reads %.0fmm across - outside what the gripper takes"
                 % width)
            continue

        # --- how high can I actually hover over it? ---
        # 180mm is what I want (it keeps the blind dive short) but a cube
        # held high, or out toward the edge, can be reachable at the grasp
        # and not 180mm above it - raising the tool moves it FARTHER from
        # the shoulder. Take the highest hover that IS reachable rather
        # than giving up on the cube; the camera can still see it from
        # anywhere above ~60mm, so a lower hover costs a longer blind dive
        # and nothing else.
        follow_h = None
        for h in hc.FOLLOW_LADDER:
            if reachable(arm, [gx, gy, gz + h, 180.0, 0.0, scan_yaw]) and \
                    reachable(arm, [gx, gy, gz + h, 180.0, 0.0, yaw]):
                follow_h = h
                break
        if follow_h is None or not reachable(arm, [gx, gy, gz, 180.0, 0.0,
                                                   yaw]):
            steady = 0
            talk(hc.out_of_reach_hint(gx, gy))
            continue
        if follow_h != hc.FOLLOW_H and follow_h != announced_h:
            print("   (hovering %.0fmm up instead of %.0f - that is as high "
                  "as I can reach over there)" % (follow_h, hc.FOLLOW_H))
            announced_h = follow_h

        follow = [gx, gy, gz + follow_h, 180.0, 0.0, scan_yaw]

        # --- steadiness: is the hand holding still, and am I on it? ---
        moved = 0.0 if last_target is None else float(
            np.hypot(gx - last_target[0], gy - last_target[1]))
        err_xy = float(np.hypot(gx - cur[0], gy - cur[1]))
        err_z = abs(follow[2] - cur[2])
        aligned = err_xy <= hc.ALIGN_MM and err_z <= hc.ALIGN_MM
        if last_target is not None and moved <= hc.STEADY_MM and aligned:
            steady += 1
        else:
            steady = 0
        last_target = (gx, gy, gz)
        locked_px = list(best["pixel"])

        if steady >= hc.STEADY_N:
            print("\n   steady - taking it now, HOLD STILL")
            return (grab_now(arm, ref, gx, gy, gz, yaw, width, follow_h),
                    best.get("color", "unknown"))

        # --- otherwise: one capped step toward the follow pose ---
        if err_xy <= hc.DEADBAND_MM and err_z <= hc.DEADBAND_MM:
            talk("on it - hold still for a moment")
            continue
        talk("following you (%.0fmm out)" % max(err_xy, err_z))
        nxt = hc.step_toward(cur[:3], follow[:3])
        pose = [nxt[0], nxt[1], nxt[2], 180.0, 0.0, scan_yaw]
        if not reachable(arm, pose):
            steady = 0
            talk("cannot step there - move it a little")
            continue
        try:
            vc.moveto(arm, "follow", pose)
        except vc.SweepRefused as e:
            steady = 0
            talk("%s - move it somewhere easier for the arm to face" % e)
            continue

        # The arm just moved, so the cube will appear somewhere else in
        # the image next frame. Predict where, and the lock stays tight
        # (a plain "near the last pixel" test would need a radius so
        # large it stops excluding other cubes).
        locked_px = hc.predict_pixel(calib, ref, best["pixel"],
                                     best["depth_mm"],
                                     [nxt[0] - cur[0], nxt[1] - cur[1]],
                                     nxt[2] - cur[2])
    return "timeout", None


def grab_now(arm, ref, gx, gy, gz, yaw, width, follow_h):
    """Rotate to the grasp angle, dive, close, and ask the gripper what
    happened. The dive is open-loop by necessity (see the module note),
    so it is short, and the verdict comes from the fingers.

    follow_h is the height the loop actually converged at - not always
    the nominal FOLLOW_H, since an awkward spot may only be reachable
    from lower down."""
    align = [gx, gy, gz + follow_h, 180.0, 0.0, yaw]
    grasp = [gx, gy, gz, 180.0, 0.0, yaw]
    lift = [gx, gy, gz + follow_h, 180.0, 0.0, yaw]
    if not reachable(arm, align) or not reachable(arm, grasp):
        print("   (the grasp turned out to be unreachable - backing off)")
        return "missed"

    open_p = hc.open_pulses(ref, width)
    close_p = hc.close_pulses(ref, width)
    # size the opening to the cube instead of leaving it wide: narrower
    # fingers are less likely to meet the hand holding it
    vc.grip(arm, "open to fit", open_p)
    vc.moveto(arm, "align to the grasp angle", align)
    vc.moveto(arm, "take it", grasp)
    final = vc.grip(arm, "close (take it)", close_p)

    held = vc.grasp_status(arm)             # the gripper's own verdict
    if held == vc.UNKNOWN:
        if final is None:
            vc.moveto(arm, "back off", lift)
            raise RuntimeError("could not read the gripper after closing - "
                               "check it before running again")
        held = vc.EMPTY if final <= close_p + hc.POSITION_MISS_TOL \
            else vc.HOLDING
    if held == vc.EMPTY:
        # never open on one reading - opening is the only action here
        # that cannot be undone
        again = vc.grasp_status(arm)
        pos = vc.gripper_pos(arm)
        if again == vc.HOLDING or (pos is not None
                                   and pos > close_p + hc.POSITION_MISS_TOL):
            print("   (first reading said empty but the fingers are stalled "
                  "at %s - treating it as a hold)" % pos)
            held = vc.HOLDING

    if held == vc.EMPTY:
        print("   caught nothing (fingers reached %s of %d) - backing off"
              % (final, close_p))
        vc.grip(arm, "open (missed)", open_p)
        vc.moveto(arm, "back off", lift)
        vc.grip(arm, "open", 850)
        return "missed"

    if final is not None:
        true_w = ref["w0_mm"] + (final - ref["stall0"]) / hc.PULSES_PER_MM
        print("   got it: fingers stalled at %d -> cube ~%.1fmm (the camera "
              "said %.1fmm)" % (final, true_w, width))
    arm.set_tcp_load(vc.GRIPPER_WEIGHT + hc.BOX_WEIGHT, vc.GRIPPER_COG)
    vc.moveto(arm, "lift clear of your hand", lift)
    return "held"


def deliver(arm, color):
    """Put the cube down. Uses the drop spots taught for the floor picker
    (vision/places.json) when they exist - so a red cube handed over goes
    to the red spot - and otherwise hands it back."""
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


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


def main():
    log = vc.start_log("handoff")
    if log:
        print("(run log: %s)" % log)
    if not os.path.exists(MODEL_FILE):
        print("cube_model.pt not found in vision\\ - train the v3 model "
              "first ('2 - Train Cube Model').")
        return
    calib, ref = hc.load_calibration()
    if calib is None:
        return

    # BEFORE setup_arm, not after: setup_arm enables the gripper, and
    # that is where the finger speed is applied. Doing this later would
    # leave the fingers running at the floor picker's full speed next to
    # a hand for the whole run.
    prev_speed = hc.slow_down()

    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
        if not reachable(arm, hc.WAIT_POSE):
            raise RuntimeError("the wait pose %s is not reachable"
                               % hc.WAIT_POSE[:3])
        if held_cube_recovery(arm):
            arm.disconnect()
            return
    except RuntimeError as e:
        print("ABORT (no motion):", e)
        arm.disconnect()
        return

    lo, hi = hc.hold_band(ref)
    print("""
=================== HANDOFF ===================
The arm will FOLLOW a cube held in your hand and
take it once your hand is still for a moment.

Hold the cube:
  * by the BOTTOM EDGES - keep fingertips off the
    top face, that is what the camera measures
  * between %.0f and %.0f cm above the floor
  * near the middle of the cell

The last %.0fmm of the reach is blind (the depth
camera cannot see closer), so the arm dives only
while your hand is steady - and the gripper's own
grasp sensor confirms what it got.

Slowed down for working next to a person:
  arm      %d deg/s   (floor picker: %d)
  fingers  %d         (floor picker: %d)
The arm also STOPS between every step - it never
streams continuous motion.

Press ESC or q in THIS window to stop between
steps. Keep the e-stop in hand - during a single
move it is the only thing that stops the arm.
===============================================""" % (
        hc.above_floor_mm(ref, lo) / 10.0, hc.above_floor_mm(ref, hi) / 10.0,
        hc.FOLLOW_H, hc.HANDOFF_SPEED, prev_speed[0],
        hc.HANDOFF_GRIPPER_SPEED, prev_speed[2]))
    if input("\nStart? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        arm.disconnect()
        return

    done = 0
    try:
        with v3.Camera(floor_ref=None) as cam:
            missed = 0
            for attempt in range(1, MAX_HANDOFFS + 1):
                goto_wait(arm)
                vc.grip(arm, "open", 850)
                print("\n--- handoff %d - hold a cube up where I can see it "
                      "---" % attempt)
                result, color = follow_and_grab(arm, cam, calib, ref)
                if result == "held":
                    deliver(arm, color)
                    done += 1
                    missed = 0
                elif result == "missed":
                    missed += 1
                    if missed >= MAX_MISSED_GRABS:
                        print("Missed %d grabs in a row - stopping so we can "
                              "look at why." % missed)
                        break
                elif result == "abort":
                    print("Stopped at your request.")
                    break
                else:
                    print("Nothing offered for %.0fs - stopping."
                          % SESSION_TIMEOUT_S)
                    break
        goto_wait(arm)
        vc.movej(arm, "HOME", hc.HOME_DEG)
        print("\nDone. Took %d cube(s) from your hand. Parked at HOME."
              % done)
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

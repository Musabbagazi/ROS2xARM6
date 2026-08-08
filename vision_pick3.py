#!/usr/bin/env python3
"""Autonomous cube pick, v3: YOLO detector (any color), dynamic grab
height from depth, and moving-object tolerance.

Cubes are cleared in COLOUR ORDER: every red cube first, then every
blue one, then anything left whose colour could not be called. The arm
only moves on to blue once no red is visible from ANY of its vantages.

Loop: SCAN -> if nothing seen, RISE straight up for a wider view then
sweep a GRID (search) -> track the cube until it has been STILL for ~1s
(it may be moving - keep watching) -> aim with the measured pixel->mm
mapping (generalized to the vantage it was found from) -> descend
half-way OVER the cube -> LOOK AGAIN and correct -> rotate to the grasp
angle -> descend to a grab height computed from DEPTH (works for any
cube size, no taught grab pose) -> grab (width from measured size) ->
verify -> deliver to PLACE -> repeat until no cube -> HOME.

If the cube moves too much mid-descent, or the grab closes on nothing,
the arm retreats, reopens and RETRIES from the scan instead of aborting.

Needs calib3.json + grip_ref.json (run auto_calibrate3.py first) and
cube_model.pt (capture_dataset.py + train_cubes.py, once).

Usage: python vision_pick3.py [robot_ip]
Keep the e-stop in hand. One y/N confirm, then it runs by itself.

By Musab Bagazi and Yazan Bal'fakeeh.
"""
import sys
import time

import numpy as np
from xarm.wrapper import XArmAPI

import vision_common as vc
import vision3 as v3
from detect_cube import GRIPPABLE_MM

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

# colour-sorted drop spots. RED keeps the original PLACE; BLUE must be
# taught/entered (set PLACE_BLUE below). A cube whose colour is unknown,
# or blue while PLACE_BLUE is unset, falls back to the RED spot.
PLACE_RED = [436.5, -498.4, 407.6, -171.1, -9.8, -93.5]
PLACE_BLUE = None            # <-- set to [x, y, z, roll, pitch, yaw]
PLACE_APPROACH = 60.0

# Colour priority: clear every RED cube first, then every BLUE one, then
# whatever is left (cubes the classifier could not call - they would
# otherwise be stranded on the floor, since a colour hunt never matches
# "unknown"). Each entry is (colour_to_hunt, label for messages);
# None hunts any colour. Watching the arm take all the reds before it
# touches a blue is also the plainest proof that colour recognition
# works, which is what this order is for.
PICK_ORDER = [("red", "RED"), ("blue", "BLUE"), (None, "any remaining")]


def same_cube(color_a, color_b):
    """Could these two sightings be the same cube? Colour is a cheap,
    strong identity check when several cubes share the view: red vs blue
    is decisive. "unknown" (a colour the classifier could not call) is
    treated as compatible so it never blocks a legitimate pick."""
    if not color_a or not color_b:
        return True
    if "unknown" in (color_a, color_b):
        return True
    return color_a == color_b


def place_for(color):
    """Drop pose for a cube of this colour, and a label for messages."""
    if color == "blue" and PLACE_BLUE is not None:
        return PLACE_BLUE, "blue spot"
    if color == "blue":
        return PLACE_RED, "RED spot (blue drop not set!)"
    return PLACE_RED, "red spot"

HOME_DEG = [4.186, -17.164, -102.683, -0.488, 120.453, 0.0]

PULSES_PER_MM = 10.0
# Close this far past where the cube is expected to stall the fingers.
# Only has to be generous enough to cover the error in the VISION width
# (worst seen: 3mm over-read on an off-axis cube), because whether the
# grab worked is now answered by the gripper's own grasp sensor, not by
# where the fingers ended up. 100 pulses = 10mm of margin; the squeeze
# is bounded by the gripper's stall protection, as when gripper_teach.py
# closes fully onto a box to measure it.
CLOSE_MARGIN_PULSES = 100
# fallback only, for a gripper too old to report grasp status: fingers
# that got within this many pulses of the commanded position never hit
# anything on the way
POSITION_MISS_TOL = 10
OPEN_EXTRA_MM = 30.0
MIN_CUBE_MM = 15.0
# shared with the detector: it prefers cubes below this width, so the
# two limits must never diverge (a cube "grippable" to the detector but
# over the caller's limit would shadow pickable cubes forever)
MAX_CUBE_MM = GRIPPABLE_MM
BOX_WEIGHT = 0.05

MAX_CORRECTION_SCAN = 30.0  # mid-descent shift over this @ scan = cube moved
MAX_CORRECTION_ELEV = 60.0  # a wider-view find is rougher; the look step
                            # corrects more before calling it a move
MAX_CYCLES = 14            # runaway guard; comfortably clears ~6 cubes
                           # (one delivered per cycle) plus a few skips,
                           # plus one search per colour handover (each
                           # exhausted colour costs a full empty search)
MAX_RETRIES = 3             # consecutive retreat-and-retry attempts

# when the SCAN view is empty, search a wider area: first rise straight
# up in place (same XY) for a wider field of view, then sweep a grid at
# scan height. Every vantage is IK-guarded; unreachable ones are skipped.
# Rise heights should be within the dataset's captured heights so YOLO
# still detects from up there (see capture_dataset.HEIGHTS).
# MEASURED, not guessed - see plan_search.py, which asks the controller
# which poses it can reach (IK query, no motion) and maps each one's view
# onto the floor through the calibration.
#
# What it found: z720+ is NOT reachable at all (the old 740 was a pose
# that never existed), +200mm in X is unreachable, and the old 11-vantage
# list had only 7 reachable ones covering 79.8% of the working area - a
# fifth of the floor was never seen properly, which is why cubes kept
# being "missed" that the camera could see perfectly well up close.
#
# These five tile 100% of the 400x540mm working area counting only the
# MIDDLE 60% of each view - the part where a cube is under the camera
# rather than off to the side, seen at a slant with a ragged top face.
# Fewer vantages AND full coverage, so the sweep is faster too.
#
# SYMMETRIC by construction. These are not flange offsets chosen by eye:
# the CAMERA's ground targets are laid out on an even grid about the
# cell centre [510, 0] - x in {315, 445, 575}, y in {-300, -150, 0,
# +150, +300} - and the flange offset is that target minus the camera's
# fixed offset from the flange ([+67, +34]). Laying out FLANGE positions
# evenly, as before, gives a pattern symmetric about the ARM rather than
# about the cell, which is why the old one leaned to one side.
# x=705 is dropped because the arm cannot reach it at all - that
# asymmetry is the robot's, not a choice.
#
# ORDER: centre outward. The search stops at the first cube it finds, so
# the middle of the cell is checked first and the far edges last; every
# +y vantage still has its -y twin, so coverage is even.
RISE_HEIGHTS = [680.0]              # highest reachable over the scan spot
GRID_OFFSETS = [
    (-72.0, -104.0), (58.0, -104.0),                    # centre pair
    (-72.0, -254.0), (-72.0, 46.0),                     # +/-150 in y
    (58.0, -254.0), (58.0, 46.0),
    (-202.0, -104.0),                                   # near-base row
    (-202.0, -254.0), (-202.0, 46.0),
    (-72.0, -404.0), (-72.0, 196.0),                    # +/-300 in y
    (-202.0, -404.0), (-202.0, 196.0),
]
# at the look pose our cube must appear near the calibrated reference
# pixel; a detection beyond this radius is a NEIGHBOUR, not our cube
LOOK_NEAR_PX = 130.0
# The look pose is close range with the cube near the frame edge and the
# open fingers intruding, so per-frame detection is genuinely flaky
# (seen 2/5). Take more frames and accept fewer hits: the radius +
# colour checks already guarantee it is OUR cube, so 2 good sightings
# are trustworthy, and demanding 3/5 was throwing away valid looks.
LOOK_N = 8
LOOK_MIN_FOUND = 2
AVOID_R = 60.0              # a found cube within this of a skipped one is
                           # treated as the same unpickable cube

# stillness gate: the cube must stay within STILL_MM of its anchor
# sighting (POSITION only) for STILL_SPAN_S seconds before the arm
# commits. Angle is deliberately NOT part of this test: a square cube's
# measured top-face angle jitters by several degrees from depth noise
# even when the cube is perfectly still (the "long side" the angle is
# read from flips between the two near-equal sides), which used to be
# misread as motion. What matters for "settled" is that the cube is not
# being slid across the floor; the grasp angle is read fresh at the grab.
# Time-based on purpose: a sample-count gate would depend on how fast
# the detector happens to run (GPU vs CPU).
STILL_MM = 5.0
# a detection farther than this from the tracked anchor is a DIFFERENT
# cube (the detector picks "largest cube" and flips between similar ones
# frame-to-frame) - NOT our cube moving. Well above same-cube noise
# (~2mm) and below the >=38mm centre spacing of separated cubes.
SAME_CUBE_MM = 30.0
STILL_SPAN_S = 0.8
TRACK_TIMEOUT_S = 45.0
MISS_LIMIT = 2              # captures with no cube -> floor is empty.
                            # Each capture is ~6 frames, and a full
                            # search walks up to 11 vantages, so this is
                            # paid ~11 times on an empty sweep. Two is
                            # enough: every capture is already a median
                            # over 4 frames needing only 2 hits.


def above(pose, dz):
    p = list(pose)
    p[2] += dz
    return p


def safe_lift(arm, dz=150.0):
    """Raise the TCP vertically, keeping orientation, before any big
    move. Returns True only when the lift actually happened."""
    code, p = arm.get_position(is_radian=False)
    if code != 0:
        return False
    try:
        vc.moveto(arm, "lift to safe height",
                  [p[0], p[1], p[2] + dz, p[3], p[4], p[5]])
        return True
    except RuntimeError as e:
        print("(safe lift skipped: %s)" % e)
        return False


def held_box_recovery(arm):
    pos = vc.gripper_pos(arm)
    if pos is None or pos > 750:
        return False
    print("\n!! Gripper is not open (at %.0f) - may still hold a cube." % pos)
    print("   [d] deliver to PLACE and continue  [r] release here  [q] quit")
    choice = input("   d/r/q > ").strip().lower()
    if choice == "d":
        # colour of an already-held cube is unknown here -> default spot
        arm.set_tcp_load(vc.GRIPPER_WEIGHT + BOX_WEIGHT, vc.GRIPPER_COG)
        if not safe_lift(arm):              # never swing from a low pose
            raise RuntimeError("cannot lift safely while holding the cube "
                               "- clear it manually, then rerun")
        vc.moveto(arm, "above PLACE", above(PLACE_RED, PLACE_APPROACH))
        vc.moveto(arm, "PLACE", PLACE_RED)
        vc.grip(arm, "open (release)", 850)
        arm.set_tcp_load(vc.GRIPPER_WEIGHT, vc.GRIPPER_COG)
        vc.moveto(arm, "retreat", above(PLACE_RED, PLACE_APPROACH))
        return False
    if choice == "r":
        input("   Hold it, press Enter... ")
        vc.grip(arm, "open (release here)", 850)
        return False
    return True


def track_until_still(cam, calib, ref, obs_pose, cycle, attempt, label,
                      want=None):
    """Watch from obs_pose until the (possibly moving) cube holds still.

    want ("red"/"blue"/None) restricts this to cubes of one colour, so
    "empty" means "no cube of THAT colour is visible from here" - cubes
    of other colours are simply not seen, and are left for a later pass.

    Locks onto ONE cube: the first detection sets the anchor position;
    a later detection FAR from it (> SAME_CUBE_MM) is a different cube
    (the detector flips between similar cubes) and is ignored, not read
    as motion. The arm commits once the tracked cube has stayed within
    STILL_MM for STILL_SPAN_S. Angle is not part of the test (see the
    STILL_MM comment). Returns (seen, (gx,gy,gz,yaw)) once still, "empty"
    when no cube is seen MISS_LIMIT times in a row, or "moving" on
    timeout."""
    anchor = None       # (gx, gy) of the ONE cube we are tracking
    anchor_px = None    # its image pixel - aims the detector at OUR cube
    anchor_t = 0.0
    misses = 0
    t0 = time.time()
    shot = 0
    while time.time() - t0 < TRACK_TIMEOUT_S:
        shot += 1
        seen = cam.stable("pick%d.%d_%s%d" % (cycle, attempt, label, shot),
                          n=4, min_found=2, settle=2,
                          prefer_near=anchor_px, want_color=want,
                          flange_z=obs_pose[2])
        now = time.time()
        if seen is None:
            misses += 1
            if misses >= MISS_LIMIT:
                return "empty"
            continue
        misses = 0
        gx, gy, gz, yaw = v3.aim_from_pose(calib, ref, obs_pose, seen)
        if anchor is None:
            anchor = (gx, gy)
            anchor_px = list(seen["pixel"])
            anchor_t = now
            continue
        drift = float(np.hypot(gx - anchor[0], gy - anchor[1]))
        if drift > SAME_CUBE_MM:
            continue                 # a DIFFERENT cube this frame - ignore,
            #                          keep waiting for the one we're tracking
        if drift > STILL_MM:
            print("cube moving (%.0fmm) - waiting..." % drift)
            anchor = (gx, gy)        # our cube shifted a little -> re-anchor
            anchor_px = list(seen["pixel"])
            anchor_t = now
        elif now - anchor_t >= STILL_SPAN_S:
            return seen, (gx, gy, gz, yaw)
    return "moving"


def goto_scan(arm):
    """Move to the SCAN pose, resetting through HOME if need be.

    The IK solution is recomputed HERE rather than reused from startup.
    Firmware 1.6.9 seeds IK from the arm's CURRENT joints, so a solution
    found at startup can belong to a different configuration branch than
    the one the arm is standing in later - and the sweep guard then
    measures the gap between two branches instead of the length of the
    move. Seen live: scan pose refused for a "144 deg" swing on a move
    whose base rotation was only 24 deg.

    If the fresh solution is still refused, HOME is a known-good
    configuration that the scan pose is routinely reachable from, so
    passing through it resets the branch."""
    try:
        vc.moveto(arm, "SCAN pose", vc.SCAN_POSE)
    except vc.SweepRefused as e:
        print("  (%s)" % e)
        print("  resetting the arm's configuration through HOME")
        vc.movej(arm, "HOME", HOME_DEG)
        vc.moveto(arm, "SCAN pose", vc.SCAN_POSE)


def search_for_cube(arm, cam, calib, ref, cycle, attempt, avoid,
                    want=None):
    """Look from the SCAN pose; if empty, rise straight up in place for a
    wider field of view, then sweep a grid at scan height. Every vantage
    is IK-guarded (unreachable ones skipped). A cube whose base XY is
    within AVOID_R of an already-skipped one is ignored, so one
    unpickable cube can't block clearing the rest of the floor. Returns
    (obs_pose, seen, (gx,gy,gz,yaw)) on a still cube, or "empty"/"moving".

    want ("red"/"blue"/None) hunts one colour: "empty" then means no
    cube of that colour was visible from ANY vantage, which is what the
    caller needs before it may move on to the next colour. Sweeping the
    whole search this way (rather than filtering after a find) is what
    stops a big blue cube in the middle of the view from hiding the reds
    behind it.
    """
    scan_yaw = vc.SCAN_POSE[5]
    sx, sy, sz = vc.SCAN_POSE[0], vc.SCAN_POSE[1], vc.SCAN_POSE[2]
    stages = [("scan", list(vc.SCAN_POSE))]
    for h in RISE_HEIGHTS:
        stages.append(("rise%d" % int(h), [sx, sy, h, 180.0, 0.0, scan_yaw]))
    for dx, dy in GRID_OFFSETS:
        stages.append(("grid%+d%+d" % (int(dx), int(dy)),
                       [sx + dx, sy + dy, sz, 180.0, 0.0, scan_yaw]))

    seen_but_avoided = 0        # cubes we DID see but already gave up on
    for i, (label, pose) in enumerate(stages):
        if vc.ik(arm, pose) is None:
            if i == 0:
                raise RuntimeError("SCAN pose not reachable")
            continue                        # this vantage is out of reach
        if i == 0:
            goto_scan(arm)
        else:
            try:
                vc.moveto(arm, "search %s" % label, pose)
            except RuntimeError as e:
                print("  search skip %s: %s" % (label, e))
                continue
            print("  wider look from %s" % label)
        tracked = track_until_still(cam, calib, ref, pose, cycle, attempt,
                                    label, want)
        if tracked == "moving":
            return "moving"
        if tracked != "empty":
            seen, target = tracked
            gx, gy = target[0], target[1]
            if any(np.hypot(gx - ax, gy - ay) < AVOID_R for ax, ay in avoid):
                print("  cube at [%.0f,%.0f] was already skipped - "
                      "looking elsewhere" % (gx, gy))
                seen_but_avoided += 1
                continue                    # a different vantage may see
                                            # a different, pickable cube
            return pose, seen, target
    # "nothing there" and "only ones we already gave up on" are very
    # different things, and reporting both as "no cube found" reads as a
    # detection failure when detection worked perfectly
    return "avoided" if seen_but_avoided else "empty"


def pick_one(arm, cam, calib, ref, cycle, attempt, avoid,
             want=None):
    """One search->pick->deliver cycle (attempt numbers the retries so
    saved capture images are not overwritten; avoid is the shared list of
    base-XY spots of cubes found unpickable this run; want is the colour
    being hunted, None for any).
    Returns 'delivered', 'empty', 'retry' or 'skip'. 'empty' means no
    cube OF THAT COLOUR is left anywhere the arm can see."""
    vc.grip(arm, "open", 850)

    found = search_for_cube(arm, cam, calib, ref, cycle, attempt, avoid, want)
    what = "cube" if want is None else "%s cube" % want
    if found == "avoided":
        # detection worked - these are cubes this run already decided it
        # cannot pick (out of reach, undeliverable, wrong size)
        print("The only %ss still on the floor are ones already found "
              "UNPICKABLE this run\n  (out of reach, or the drop would "
              "exceed the joint-sweep guard) - they were seen and "
              "ignored on purpose, not missed." % what)
        return "empty"
    if found == "empty":
        print("No %s found in the scan, the wider views or the grid." % what)
        return "empty"
    if found == "moving":
        print("Cube would not hold still for %.0fs." % TRACK_TIMEOUT_S)
        return "retry"
    obs_pose, seen, (gx, gy, gz, yaw) = found
    from_scan = (abs(obs_pose[0] - vc.SCAN_POSE[0]) < 1.0 and
                 abs(obs_pose[1] - vc.SCAN_POSE[1]) < 1.0 and
                 abs(obs_pose[2] - vc.SCAN_POSE[2]) < 1.0)
    corr_cap = MAX_CORRECTION_SCAN if from_scan else MAX_CORRECTION_ELEV

    # Is the floor where it should be? The grab height is clamped against
    # it, and a floor that reads too deep drags that clamp - and the arm -
    # down with it. Never guess here: a stale floor0 (the surface really
    # moved) and an unreadable floor (transparent, so the infrared
    # measures whatever is underneath) both need a human, and silently
    # substituting the predicted floor would hide the first case while
    # the arm drove into the second.
    if cam.floor_ref is not None:
        # A TAUGHT floor cannot be cross-checked - it is used precisely
        # because this surface returns no depth of its own, so comparing
        # it against expectation would only compare it with itself. What
        # can still be checked is the CUBE: its top must stand a sane
        # distance above that floor. A cube that does not is a detection
        # on something else entirely, and the grab height would be wrong.
        pop = (seen.get("floor_mm") or 0.0) - seen["depth_mm"]
        if not (8.0 <= pop <= MAX_CUBE_MM + 20.0):
            print("cube top stands %.0fmm above the taught floor - that is "
                  "not a cube of ours;\n  skipping. (Re-run teach_floor if "
                  "the floor itself has moved.)" % pop)
            avoid.append((gx, gy))
            return "skip"
        ferr = None
    else:
        ferr = v3.floor_error(ref, seen.get("floor_mm"), obs_pose[2])
    if ferr is not None and abs(ferr) > v3.FLOOR_TOL_MM:
        print("floor reads %.0fmm %s than it should (%.0f vs %.0f expected) "
              "- REFUSING to grab:\n  either the floor moved (re-run "
              "auto_calibrate3) or the camera cannot see it\n  (a "
              "transparent surface reads through to whatever is below). "
              "Run floor_check.bat."
              % (abs(ferr), "deeper" if ferr > 0 else "shallower",
                 seen["floor_mm"], v3.expected_floor(ref, obs_pose[2])))
        avoid.append((gx, gy))
        return "skip"

    width = seen["width_mm"]
    if not MIN_CUBE_MM <= width <= MAX_CUBE_MM:
        print("Cube width %.0fmm outside grippable range - skipping." % width)
        avoid.append((gx, gy))
        return "skip"

    color = seen.get("color", "unknown")
    place, place_label = place_for(color)

    look_h = ref.get("look_h", v3.LOOK_H)
    scan_yaw = vc.SCAN_POSE[5]
    # A cube near the edge of the workspace can be reachable at the GRASP
    # and not 180mm above it: raising the tool moves it FARTHER from the
    # shoulder, so the approach pose leaves the envelope first (seen live
    # at radius 694mm - grasp 697mm from the shoulder, look 703mm, and
    # the reach is 700). Lower the approach rather than give up the cube;
    # it only has to clear the cube itself. The same height is used for
    # the rotate-then-descend and for the lift, so it must stay well
    # above a cube: 70mm clears the tallest we handle.
    for h in (look_h, 140.0, 100.0, 70.0):
        if vc.ik(arm, [gx, gy, gz + h, 180.0, 0.0, scan_yaw]) is not None \
                and vc.ik(arm, [gx, gy, gz + h, 180.0, 0.0, yaw]) is not None:
            if h != look_h:
                print("  (approach lowered to %.0fmm - the usual %.0fmm is "
                      "out of reach at this spot)" % (h, look_h))
            look_h = h
            break
    print("%s cube %.0fmm (h %.0fmm, conf %.2f) found from %s -> arm "
          "[%.1f, %.1f], z %.1f, yaw %.1f  (deliver to %s)"
          % (color, width, seen["height_mm"], seen["conf"],
             "scan" if from_scan else "a wider view (%s)"
             % ("rise" if abs(obs_pose[2] - vc.SCAN_POSE[2]) > 1.0
                else "grid"), gx, gy, gz, yaw, place_label))

    def look_pose():
        # look with the SCAN yaw so the view matches the calibration
        # reference exactly, whatever the cube's rotation is
        return [gx, gy, gz + look_h, 180.0, 0.0, scan_yaw]

    def grasp_pose(z_off=0.0):
        return [gx, gy, gz + z_off, 180.0, 0.0, yaw]

    def go(label, pose):
        """Move there, or give this cube up cleanly.

        A joint-sweep refusal depends on where the arm currently is and
        where this cube sits - not on anything being broken. Letting it
        propagate ended the whole run over one awkward cube, which is
        wrong when the gripper is still empty and the rest of the floor
        is pickable. Only used BEFORE the grab: once a cube is held, a
        failed move must still stop the run."""
        try:
            vc.moveto(arm, label, pose)
            return True
        except vc.SweepRefused as e:
            print("%s - skipping this cube (the arm cannot get there from "
                  "where it is standing)." % e)
            vc.grip(arm, "open", 850)
            avoid.append((gx, gy))
            return False

    for name, pose in (("look pose", look_pose()),
                       ("align", [gx, gy, gz + look_h, 180.0, 0.0, yaw]),
                       ("grasp", grasp_pose()),
                       ("above PLACE", above(place, PLACE_APPROACH)),
                       ("PLACE", place)):
        if vc.ik(arm, pose) is None:
            print("%s not reachable - skipping this cube." % name)
            avoid.append((gx, gy))
            return "skip"

    # The vision width now only sizes the finger travel - how wide to
    # open (so the fingers clear neighbouring cubes) and how far to
    # close. Neither needs better than ~10mm accuracy, and NEITHER is
    # used to judge whether the grab worked: the gripper reports that.
    open_p = min(850, int(ref["stall0"]
                          + (width + OPEN_EXTRA_MM - ref["w0_mm"])
                          * PULSES_PER_MM))
    close_p = max(0, int(ref["stall0"]
                         + (width - ref["w0_mm"]) * PULSES_PER_MM
                         - CLOSE_MARGIN_PULSES))

    vc.grip(arm, "open to fit", open_p)
    if not go("look pose", look_pose()):
        return "skip"

    # ---- confirm / refine the aim at the look pose (half-way down) ----
    if not from_scan:
        # a wider-view (rise/grid) find is rougher: ALWAYS re-detect over
        # the cube and re-aim from this near-nominal vantage - whether or
        # not a p_look0 reference exists. Never grab blind on an elevated
        # estimate. aim_from_pose here is the accurate near-nominal aim.
        # prefer_near: OUR cube appears near the p_look0 reference from
        # here - without it, a larger NEIGHBOR elsewhere in the look view
        # would hijack the detection and fake a huge shift.
        look = cam.stable("pick%d.%d_look" % (cycle, attempt), n=LOOK_N,
                          min_found=LOOK_MIN_FOUND, settle=8,
                          prefer_near=ref.get("p_look0"),
                          near_radius=LOOK_NEAR_PX,
                          flange_z=look_pose()[2])
        if look is not None and not same_cube(color, look.get("color")):
            print("look pose shows a %s cube, but we are tracking a %s one "
                  "- that is a NEIGHBOUR, not our cube."
                  % (look.get("color"), color))
            look = None
        if look is not None:
            look_z = look_pose()[2]         # before gz can change below
            rgx, rgy, rgz, ryaw = v3.aim_from_pose(calib, ref, look_pose(),
                                                   look)
            shift = float(np.hypot(rgx - gx, rgy - gy))
            if shift > corr_cap:
                print("look-pose re-aim shifted %.0fmm (cap %.0f) - not "
                      "trusting it; grabbing on the search aim."
                      % (shift, corr_cap))
            else:
                if shift > 1.0:
                    print("re-aim: [%.1f, %.1f]mm" % (rgx - gx, rgy - gy))
                if v3.floor_is_trustworthy(ref, look.get("floor_mm"),
                                           look_z):
                    gx, gy, gz, yaw = rgx, rgy, rgz, ryaw
                else:
                    # take the position fix but NOT the height: the grab z
                    # from the search vantage was already validated, and
                    # this one rests on a floor reading we do not believe
                    print("   (look-pose floor reading is off - keeping the "
                          "validated grab height)")
                    gx, gy, yaw = rgx, rgy, ryaw
                if vc.ik(arm, grasp_pose()) is None or \
                        vc.ik(arm, [gx, gy, gz + look_h, 180.0, 0.0,
                                    yaw]) is None:
                    vc.grip(arm, "open", 850)
                    print("re-aimed grasp not reachable - skipping.")
                    avoid.append((gx, gy))
                    return "skip"
                # no move to the re-aimed look pose: the align move below
                # goes to the SAME xyz, only with the grasp yaw, so
                # stopping here first is a wasted trip
        else:
            # The arm is already hovering over the computed spot and will
            # descend straight down, so this is no riskier than a scan
            # pick: try the grab and let the grab-verify catch a miss.
            # Retreating here just repeated the same search and looped.
            print("(our cube not confirmed at the look pose - grabbing on "
                  "the search aim; a miss will retry)")
    elif ref.get("p_look0"):
        # scan find: the validated p_look0-anchored mid-descent correction.
        # prefer_near p_look0: our cube must appear there; never let a
        # larger neighbor in the look view fake a >cap "shift" (that
        # caused an endless retreat-rescan-retreat loop on cube 1).
        look = cam.stable("pick%d.%d_look" % (cycle, attempt), n=LOOK_N,
                          min_found=LOOK_MIN_FOUND, settle=8,
                          prefer_near=ref["p_look0"],
                          near_radius=LOOK_NEAR_PX,
                          flange_z=look_pose()[2])
        if look is not None and not same_cube(color, look.get("color")):
            print("look pose shows a %s cube, but we are tracking a %s one "
                  "- ignoring that neighbour."
                  % (look.get("color"), color))
            look = None
        if look is not None:
            dpx = np.array(look["pixel"]) - np.array(ref["p_look0"])
            d_ref = calib.get("d_ref") or ref["d0"]
            dr = (ref["d_look0"] / d_ref) if ref.get("d_look0") else 0.6
            corr = vc.px_to_mm(calib, dpx, depth_ratio=dr)
            mag = float(np.hypot(corr[0], corr[1]))
            if mag > corr_cap:
                # The SCAN aim is the validated one; a big "shift" here
                # almost always means the look step latched onto another
                # cube, and retreating would just repeat identically
                # (that was the infinite loop). Trust the scan aim and
                # let the grab-verify catch it if the cube really moved.
                print("mid-descent shift %.0fmm exceeds the %.0fmm cap - "
                      "not trusting it; grabbing on the scan aim."
                      % (mag, corr_cap))
            elif mag > 1.0:
                print("correction: [%.1f, %.1f]mm" % (corr[0], corr[1]))
                gx += corr[0]
                gy += corr[1]
                if vc.ik(arm, grasp_pose()) is None or \
                        vc.ik(arm, [gx, gy, gz + look_h, 180.0, 0.0,
                                    yaw]) is None:
                    vc.grip(arm, "open", 850)
                    print("corrected grasp not reachable - skipping.")
                    avoid.append((gx, gy))
                    return "skip"
                # the align move below covers this correction (same xyz,
                # grasp yaw) - going there twice just costs time
        else:
            print("(our cube not confirmed at the look pose - continuing "
                  "on the scan aim)")
    # else: scan find with no p_look0 reference -> uncorrected (validated)

    # would carrying the cube from the lift pose to the drop spot exceed
    # the joint-sweep guard? Check BEFORE grabbing (gripper still empty),
    # so an awkward drop spot skips the cube cleanly instead of faulting
    # mid-delivery while holding it. IK-reachable != sweep-reachable.
    a = vc.ik(arm, grasp_pose(look_h))
    b = vc.ik(arm, above(place, PLACE_APPROACH))
    if a is None or b is None or \
            max(abs(x - y) for x, y in zip(a, b)) > vc.MAX_JOINT_SWEEP:
        vc.grip(arm, "open", 850)
        print("delivery to the %s would exceed the joint-sweep guard from "
              "here - skipping this cube." % place_label)
        avoid.append((gx, gy))
        return "skip"

    # rotate to the grasp angle up here, then go straight down
    if not go("align rotation", [gx, gy, gz + look_h, 180.0, 0.0, yaw]):
        return "skip"
    if not go("grasp", grasp_pose()):
        return "skip"
    final = vc.grip(arm, "close (grab)", close_p)
    held = vc.grasp_status(arm)         # the GRIPPER's own verdict

    if held == vc.UNKNOWN:
        # gripper too old to report grasp status (or the read failed):
        # fall back to the finger position - fingers that reached the
        # commanded position had nothing between them
        if final is None:
            # cannot verify either way - do NOT open (we may be holding
            # it); get safe and stop
            safe_lift(arm, look_h)
            raise RuntimeError("could not read the gripper after closing - "
                               "possible gripper fault; check it")
        held = vc.EMPTY if final <= close_p + POSITION_MISS_TOL else vc.HOLDING

    if held == vc.EMPTY:
        # Never open on a single reading: opening is what drops a cube,
        # and it is the one action here that cannot be undone. Re-ask
        # both the sensor and the fingers before letting go.
        again = vc.grasp_status(arm)
        pos = vc.gripper_pos(arm)
        if again == vc.HOLDING or (pos is not None
                                   and pos > close_p + POSITION_MISS_TOL):
            print("   (first reading said empty, but the gripper is stalled "
                  "at %s - treating it as a hold, NOT opening)" % pos)
            held = vc.HOLDING

    if held == vc.EMPTY:
        print("grab caught NOTHING (gripper reports no object; fingers at "
              "%s of %d) - the cube probably moved; retrying."
              % (final, close_p))
        vc.grip(arm, "open (missed)", open_p)
        vc.moveto(arm, "retreat up", grasp_pose(look_h))
        vc.grip(arm, "open", 850)
        return "retry"

    # The stall position is the cube's TRUE width - the gripper measures
    # it directly, at the fingers. Logged next to what vision claimed so
    # the camera's error is visible on every pick (it drives the grasp
    # quality even now that it can no longer cause a false miss).
    if final is not None:
        true_w = ref["w0_mm"] + (final - ref["stall0"]) / PULSES_PER_MM
        print("   (held: fingers stalled at %d -> cube ~%.1fmm, vision said "
              "%.1fmm, off by %+.1fmm)" % (final, true_w, width,
                                           width - true_w))
    arm.set_tcp_load(vc.GRIPPER_WEIGHT + BOX_WEIGHT, vc.GRIPPER_COG)
    vc.moveto(arm, "lift", grasp_pose(look_h))

    vc.moveto(arm, "above PLACE", above(place, PLACE_APPROACH))
    vc.moveto(arm, "PLACE", place)
    vc.grip(arm, "open (release)", 850)
    arm.set_tcp_load(vc.GRIPPER_WEIGHT, vc.GRIPPER_COG)
    vc.moveto(arm, "retreat", above(place, PLACE_APPROACH))
    print("%s cube delivered to %s." % (color, place_label))
    return "delivered"


def main():
    global PLACE_RED, PLACE_BLUE
    log = vc.start_log("pick")
    if log:
        print("(run log: %s)" % log)
    calib = v3.load_calib3()
    ref = v3.load_grip_ref()
    if calib is None or ref is None:
        print("Missing calib3.json / grip_ref.json - run auto_calibrate3.py "
              "first.")
        return
    # colour-sorted drop spots taught with teach_place.bat (places.json)
    places = v3.load_places()
    if places:
        if places.get("red"):
            PLACE_RED = places["red"]
        if places.get("blue"):
            PLACE_BLUE = places["blue"]
    # zC / J / p0 are only meaningful under the SCAN pose they were
    # measured from - refuse to run against a stale calibration
    for name, data in (("calib3.json", calib), ("grip_ref.json", ref)):
        if list(data.get("scan_pose") or []) != list(vc.SCAN_POSE):
            print("%s was made under a different SCAN pose - run "
                  "auto_calibrate3.py again." % name)
            return

    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
        scan_angles = vc.ik(arm, vc.SCAN_POSE)
        if scan_angles is None:
            raise RuntimeError("SCAN pose not reachable")
        if held_box_recovery(arm):
            arm.disconnect()
            return
    except RuntimeError as e:
        print("ABORT (no motion):", e)
        arm.disconnect()
        return

    floor_ref = v3.load_floor_ref()
    if floor_ref:
        print("\nFloor: using the plane taught by teach_floor (%.0fmm at the "
              "centre, tilt %.0fmm)." % (floor_ref.get("centre_depth_mm", 0),
                                         floor_ref.get("tilt_mm", 0)))
        print("       The floor is NOT measured live - re-run teach_floor if "
              "the cell changed.")
    blue = "set" if PLACE_BLUE is not None else "NOT set (blue -> red spot)"
    print("\nColour sort: red -> PLACE_RED, blue -> PLACE_BLUE [%s]." % blue)
    print("Pick order: %s - the arm only starts a colour once none of the "
          "previous one is visible anywhere."
          % " then ".join(lbl for _c, lbl in PICK_ORDER))
    print("Plan: scan -> (if empty) rise for a wider view, then grid -> "
          "wait for the cube to hold still -> aim -> half-way look & "
          "correct -> grab (auto width, depth height) -> deliver by "
          "colour -> repeat -> HOME")
    if input("\nStart? [y/N] ").strip().lower() != "y":
        print("Aborted."); arm.disconnect(); return

    picked = 0
    try:
        with v3.Camera(floor_ref=floor_ref) as cam:
            retries = 0
            cycle = 0
            phase = 0           # index into PICK_ORDER - the colour hunted
            avoid = []          # base-XY of cubes found unpickable this run
            print("\n>>> hunting %s cubes" % PICK_ORDER[0][1])
            while cycle < MAX_CYCLES:
                cycle += 1
                want, want_label = PICK_ORDER[phase]
                print("\n--- cycle %d (%s) ---" % (cycle, want_label))
                result = pick_one(arm, cam, calib, ref, cycle,
                                  retries, avoid, want)
                if result == "delivered":
                    picked += 1
                    retries = 0
                elif result == "retry":
                    retries += 1
                    if retries > MAX_RETRIES:
                        print("Too many consecutive retries - stopping.")
                        break
                    cycle -= 1          # a retry does not consume a cycle
                elif result == "skip":
                    # a detected cube is unpickable and now on the avoid
                    # list; keep going so the rest of the floor still gets
                    # cleared (bounded by MAX_CYCLES)
                    retries = 0
                    continue
                else:
                    # 'empty' for THIS colour only - hand over to the next
                    # colour rather than stopping. The handover consumes a
                    # cycle on purpose: reusing the number would overwrite
                    # this search's capture images, which are the evidence
                    # of what the arm saw when it decided a colour was done.
                    phase += 1
                    if phase >= len(PICK_ORDER):
                        break           # nothing of any colour left
                    retries = 0
                    print(">>> no %s cubes left - hunting %s now"
                          % (want_label, PICK_ORDER[phase][1]))
        vc.movej(arm, "HOME", HOME_DEG)
        print("\nDone. Picked %d cube(s). Parked at HOME." % picked)
        if avoid:
            print("(%d cube(s) were left because they could not be picked "
                  "- out of reach, wrong size, or grip not verifiable.)"
                  % len(avoid))
    except RuntimeError as e:
        print("\nSTOPPED:", e)
        print("Re-run when ready - held-cube recovery runs at startup.")
    except KeyboardInterrupt:
        print("\nSTOPPED: interrupted from the keyboard.")
    except Exception:
        # anything unexpected must still leave a readable record in the
        # log instead of a console window that vanishes
        import traceback
        print("\nSTOPPED: unexpected error -")
        traceback.print_exc()
    finally:
        if log:
            print("(run log saved: %s)" % log)
        arm.disconnect()


if __name__ == "__main__":
    main()

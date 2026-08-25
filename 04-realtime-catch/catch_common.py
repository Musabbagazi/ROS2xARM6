#!/usr/bin/env python3
"""Shared pieces of the CATCH project - taking a cube that is MOVING.

A SEPARATE project from vision\\ and handoff\\. Nothing in either is
modified. What is reused, READ-ONLY, is the v3 calibration - calib3.json
(the pixel->mm Jacobian) and grip_ref.json (the camera/finger constant zC
and the p0<->g0 anchor) - plus cube_model.pt itself. Those describe the
CAMERA MOUNT, the FINGERS and what a cube looks like; none of them says
anything about the cube standing still, so all three carry over unchanged.
No retraining, no new calibration.

WHY THIS CANNOT BE A FLAG ON EITHER SIBLING

  vision\\vision_pick3.py   waits for the cube to STOP (track_until_still)
                            and then descends open-loop onto a target that
                            is not going anywhere.
  handoff\\handoff_pick.py  follows a moving cube, but still commits only
                            during a steady window - it needs the hand to
                            hold still for about a second before it dives.

Both are correct for their job and both end in "the target is stationary
by the time the fingers arrive". This project has no such moment.

THE CONSTRAINT THAT SHAPES EVERYTHING HERE

The D435 measures nothing closer than ~190mm and the camera sits ~142mm
(zC) above the fingers. At the GRASP pose the camera is therefore about
140mm from the cube top - inside its blind zone. The arm cannot watch the
cube arrive; it is blind for the last part of every catch, and that is
geometry, not tuning.

So this project does NOT chase. Chasing spends the whole time budget
closing a gap that keeps moving, and arrives blind at a position guess
whose error is (time error) x (cube speed) PLUS every tracking error
accumulated on the way.

It AMBUSHES instead:

    1. watch the cube from above and fit a velocity to it
    2. pick a point further along its path that the arm can reach
       BEFORE the cube gets there, with margin
    3. go there first, open wide, and sit at grasp height
    4. close on the clock, timed from the last good track

The arm is already in position when the cube arrives, so the only thing
left uncertain is TIMING, not position - and a timing error costs only
(error x speed) of displacement. At the speeds this can work at, that is
a few millimetres. Position error would have cost everything.

WHAT THAT BUYS, AND WHAT IT COSTS

Bought: the blind stretch stops mattering. The arm is not flying through
it - it is parked, waiting.

Cost: the cube has to travel in a roughly straight line at a roughly
constant speed for about a second. A cube that is tumbling, bouncing,
decelerating hard, or being carried by hand along a curve will not be
caught, and the fit residual is what refuses those rather than a failed
grab (see catch_track.confident).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The vision picker is "02-vision-pick" in the repository layout and
# "vision" in the original standalone tree. Accept either, so this
# file works checked out from git and in place on the cell PC.
for _n in ("02-vision-pick", "vision"):
    VISION = os.path.join(os.path.dirname(HERE), _n)
    if os.path.isdir(VISION):
        break
if VISION not in sys.path:
    sys.path.insert(0, VISION)          # reuse vision\ without copying it

import numpy as np                                          # noqa: E402

import vision_common as vc                                  # noqa: E402
import vision3 as v3                                        # noqa: E402


# ------------------------- poses -------------------------

# Where the arm watches from. Straight down, high: the higher the camera,
# the more of the cube's path is in frame before it has to commit, and
# fitting a velocity needs a stretch of path, not a point. Same pose the
# handoff project waits at - proven reachable in this cell by plan_search.
WATCH_POSE = [450.0, 70.0, 680.0, 180.0, 0.0, 0.0]

# The floor picker's HOME - a known-good joint branch that the
# straight-down poses are reachable from. Firmware 1.6.9 seeds IK from
# the current joints, so a pose can be refused for a huge joint sweep
# that is really just a different configuration; resetting through HOME
# fixes that.
HOME_DEG = [4.186, -17.164, -102.683, -0.488, 120.453, 0.0]


# ------------------------- the camera's blind zone -------------------------

# The D435's true minimum range at this resolution (~175mm), not the
# datasheet's recommended 280mm. Same value and same reasoning as the
# handoff project: using the recommended figure as a hard rejection threw
# away perfectly good readings.
MIN_CAM_MM = 190.0

# Height above the grasp at which the arm can still measure the cube.
# depth to the cube top from HOVER_H above the grasp is
# HOVER_H - h*GRAB_DEPTH_FRAC + zC, so at zC=142 and a 30mm cube this
# gives ~310mm of range - comfortably outside MIN_CAM_MM. Tracking
# happens at or above this height; below it the cube stops existing.
HOVER_H = 180.0


# ------------------------- safety -------------------------
#
# This arm moves to a spot a cube is travelling toward, which means it
# moves to a spot a HAND may be travelling toward. Every number below is
# a limit, not a tuning knob.

# Same reduced speeds as the handoff project. The arm is working next to
# a person who is pushing or sliding something toward it.
CATCH_SPEED = 25
CATCH_ACC = 400

# A third of the SDK maximum. The gripper stalls on contact rather than
# crushing, but stall detection is a reaction - the slower the fingers
# arrive, the less travel happens before it triggers.
CATCH_GRIPPER_SPEED = 1500

# Cube sizes the gripper can take.
MIN_CUBE_MM = 15.0
MAX_CUBE_MM = 78.0

# The cell's usable radius, measured by plan_search. Used to keep the
# chosen intercept point inside the arm's reach BEFORE planning a move to
# it, and to tell the operator which way to move rather than that
# something unspecified is wrong.
REACH_MAX_MM = 700.0
REACH_MIN_MM = 250.0

# A cube must be at least this far above the floor before this project
# will treat it as catchable in mid-air. Below it, a cube is either
# sliding ON the floor (fine - see FLOOR_SLIDE_OK) or it is a
# mis-detection of the floor itself.
CATCH_MIN_ABOVE_FLOOR = -5.0

# Where on the cube the fingers close, as a fraction of its height
# measured DOWN FROM THE TOP. 0.5 is the middle. Biased slightly upward
# for the same reason the handoff project does it - more clearance under
# the fingertips - but less aggressively, because here there is no hand
# to avoid and a higher grab has less of the cube inside the fingers when
# it arrives moving.
GRAB_DEPTH_FRAC = 0.45

PULSES_PER_MM = 10.0

# How much WIDER than the cube the fingers wait. Much more generous than
# either sibling's 30mm. The cube is arriving under its own steam and the
# fingers are a fixed obstacle in its path: every extra millimetre of gap
# is timing error the catch can absorb before the cube hits a fingertip
# instead of passing between them.
CATCH_OPEN_EXTRA_MM = 55.0

# Close this far past the expected stall. The gripper's grasp sensor -
# not this number - decides whether the catch worked, so it only has to
# cover the vision width error.
CLOSE_MARGIN_PULSES = 100
POSITION_MISS_TOL = 10
BOX_WEIGHT = 0.05


# ------------------------- what is catchable -------------------------

# Below this the cube is not really moving and this is the wrong tool -
# vision_pick3 already picks a stationary cube, better, with a closed-loop
# mid-descent correction this cannot use. Refusing here keeps the two
# projects from fighting over the same cube.
MIN_CATCH_SPEED = 15.0          # mm/s

# The upper limit is set by timing error, not by the arm. The fingers
# wait CATCH_OPEN_EXTRA_MM/2 clear of the cube on each side, so the catch
# survives a displacement error of about that much; displacement error is
# (timing error) x speed. With CLOSE_TIMING_SD_S of jitter, the speed at
# which the error eats the whole gap is
#
#     v_max = (CATCH_OPEN_EXTRA_MM / 2) / CLOSE_TIMING_SD_S
#
# The default below is that arithmetic with a safety factor of 2. It is
# a STARTING POINT measured from geometry, not from hardware - see the
# calibration note in README.md and lower it if catches miss late.
CLOSE_TIMING_SD_S = 0.08
MAX_CATCH_SPEED = 0.5 * (CATCH_OPEN_EXTRA_MM / 2.0) / CLOSE_TIMING_SD_S

# How long the fingers take to travel from the waiting opening to closed.
# Subtracted from the predicted arrival so the fingers are CLOSING as the
# cube arrives rather than starting to close once it is already through.
# Measured at CATCH_GRIPPER_SPEED; verify with test_catch.py --grip on
# hardware and correct here.
GRIPPER_CLOSE_S = 0.35

# Safety margin between the arm arriving and the cube arriving. The arm
# must be parked and settled, not still moving, when the cube shows up.
ARRIVE_MARGIN_S = 0.60

# How far ahead of the cube to look for an intercept point, as a time.
# The planner walks outward along the path until it finds a point that is
# reachable AND far enough ahead; this bounds that walk.
#
# The ceiling is not arbitrary. The arm needs roughly 2s to cross the
# cell and settle (see catch_track.ArmTiming), plus ARRIVE_MARGIN_S, so
# anything under about 3s of lead can never produce a plan and a ceiling
# near that would refuse every cube. 5s leaves headroom for a slow cube
# on a long crossing path without extrapolating so far that the fit stops
# meaning anything.
MIN_LEAD_S = 0.8
MAX_LEAD_S = 5.0
LEAD_STEP_S = 0.2


# ------------------------- geometry -------------------------

def floor_grab_z(ref):
    """Flange Z at which the fingers would close on the floor itself.

    A cube of height h sitting on the floor is grabbed at
    SCAN_z - floor0 + zC + h/2, so this is that with h = 0. Every
    "how high is it" question in this project is measured from here."""
    return float(vc.SCAN_POSE[2] - ref["floor0"] + ref["zC"])


def above_floor_mm(ref, grab_z):
    """How far above the floor a grab at this flange Z actually is."""
    return float(grab_z - floor_grab_z(ref))


def catch_band(ref, watch_z=None, height_mm=30.0):
    """(lowest, highest) grab Z this project will accept.

    Lower limit: slightly BELOW the floor grab height. Unlike the handoff
    project - which refuses anything near the floor precisely so a
    floor-lying cube can never drag the arm down - a cube SLIDING on the
    floor is a legitimate target here, and it is at floor height by
    definition. The small negative allowance absorbs the height estimate's
    error without opening the door to a grab that drives into the floor.

    Upper limit: the camera's near range. Held too close to the camera
    the depth is simply not measured, so the cube stops existing rather
    than being measured wrongly."""
    wz = WATCH_POSE[2] if watch_z is None else watch_z
    lo = floor_grab_z(ref) + CATCH_MIN_ABOVE_FLOOR
    hi = wz - MIN_CAM_MM - height_mm * GRAB_DEPTH_FRAC + ref["zC"]
    return lo, hi


def aim_moving(calib, ref, obs_pose, seen):
    """Grasp target (gx, gy, gz, yaw) for a cube seen from obs_pose.

    Identical to v3.aim_from_pose in X, Y and yaw, and to
    handoff_common.aim_hand in all four - deliberately, since that
    mapping was verified algebraically and on hardware. The only
    departure from v3 is the height: v3 clamps its answer into a window
    around the FLOOR, and the underlying relation

        Zgrab = Zobs - depth - (height * fraction) + zC

    never mentions the floor - only the clamp did. That clamp's safety
    job is done here by catch_band, checked by the caller.

    Requires obs_pose to be at the SCAN yaw: the camera's offset from the
    flange is a constant only while the wrist is unrotated, which is why
    the tracking loop never turns the wrist until it commits."""
    J = np.asarray(calib["J"], dtype=float)
    d_ref = calib.get("d_ref") or ref["d0"]
    d_obs = seen["depth_mm"]
    off = (d_obs / d_ref) * (J @ (np.asarray(seen["pixel"], dtype=float)
                                  - np.asarray(v3.CENTER)))
    co = v3.camera_offset(calib, ref)
    gx = float(obs_pose[0] + co[0] + off[0])
    gy = float(obs_pose[1] + co[1] + off[1])
    gz = float(obs_pose[2] - d_obs
               - seen["height_mm"] * GRAB_DEPTH_FRAC + ref["zC"])
    yaw = float(ref["g0"][3] - vc.wrap90(seen["angle_deg"] - ref["angle0"]))
    return gx, gy, gz, yaw


def predict_pixel(calib, ref, pixel, depth_mm, d_flange_xy, d_flange_z=0.0):
    """Where a stationary cube's image pixel moves to after the arm moves.

    Inverting the mapping aim_moving uses. From a straight-down pose,

        cube_xy = flange_xy + camera_offset + (d/d_ref) * J * (p - CENTRE)

    so after a flange step of (dx, dy, dz) - which also brings the camera
    dz nearer, d' = d - dz - a cube that has NOT moved must satisfy

        (d'/d_ref) * J * (p' - C) = (d/d_ref) * J * (p - C) - (dx, dy)

    Only the arm's own contribution is removed; whatever is left in the
    residual is the cube actually moving, which is the signal this whole
    project is built on. Without this the tracker could not tell the two
    apart and every arm step would read as cube velocity."""
    J = np.asarray(calib["J"], dtype=float)
    d_ref = calib.get("d_ref") or ref["d0"]
    d_new = depth_mm - d_flange_z
    if d_new <= 1.0:
        return list(pixel)                  # nonsense step; keep the lock
    p = np.asarray(pixel, dtype=float) - np.asarray(v3.CENTER)
    rhs = (depth_mm / d_ref) * (J @ p) - np.asarray(d_flange_xy, dtype=float)
    p_new = np.linalg.solve((d_new / d_ref) * J, rhs)
    return [float(p_new[0] + v3.CENTER[0]), float(p_new[1] + v3.CENTER[1])]


# The fingers close along the base-frame direction given by the grasp
# yaw. Used ONLY to choose between the two equally valid grips a square
# cube offers - see choose_yaw. If this offset is really 90 in the
# physical gripper, the only consequence is that the picker prefers the
# other equivalent face pair, which is still a proper flat-face grip.
# That is why it is safe to leave unverified on hardware.
FINGER_AXIS_FROM_YAW_DEG = 0.0


def choose_yaw(yaw, heading_deg):
    """Which of the cube's two equivalent grips to use.

    A square cube can be gripped across either pair of opposite faces, so
    yaw and yaw+90 are both proper flat-face grips - the calibration's
    wrap90 has already folded the measurement into one of them
    arbitrarily. Given that free choice, take the one whose CLOSING AXIS
    is nearest PERPENDICULAR to the cube's travel.

    Why perpendicular: closing across the direction of travel makes the
    open fingers a GATE the cube passes through, so an early or late
    close still finds the cube somewhere in the channel. Closing ALONG
    the travel makes them a WALL the cube runs into, and then the timing
    error has to be smaller than the gap or the cube strikes a fingertip
    instead of entering between the fingers.

    This is an optimisation on top of a grip that is correct either way,
    which is the whole reason the finger-axis convention it depends on
    can be left unverified."""
    best, best_score = yaw, None
    for cand in (yaw, yaw + 90.0):
        # Angle between the closing AXIS and the travel direction, folded
        # into 0..90. Folded with period 180, NOT through vc.wrap90:
        # wrap90 has period 90, so it maps both candidates to the same
        # number and would score them identically - which is exactly the
        # distinction being drawn here.
        d = (cand + FINGER_AXIS_FROM_YAW_DEG - heading_deg) % 180.0
        if d > 90.0:
            d = 180.0 - d
        # 90 is perpendicular (the gate), 0 is parallel (the wall).
        if best_score is None or d > best_score:
            best, best_score = cand, d
    return float(best)


def in_reach(x, y):
    """Is this XY inside the cell's usable annulus?

    A cheap gate used while WALKING the path for an intercept point, so
    the planner can reject most candidates without an IK round-trip to
    the controller. The real test is still vc.ik on the actual pose."""
    r = float(np.hypot(x, y))
    return REACH_MIN_MM <= r <= REACH_MAX_MM


def out_of_reach_hint(gx, gy):
    """Why the arm cannot get there, in a form the operator can act on."""
    r = float(np.hypot(gx, gy))
    if r > REACH_MAX_MM:
        return ("that path runs about %.0fcm outside the arm's reach - "
                "start it closer in" % ((r - REACH_MAX_MM) / 10.0 + 0.5))
    if r < REACH_MIN_MM:
        return ("that path runs in over the arm's base - start it further "
                "out")
    return ("I cannot face that spot with the wrist pointing down - shift "
            "the path a hand's width and try again")


# ------------------------- gripper widths -------------------------

def open_pulses(ref, width_mm):
    """How far to open to WAIT for a cube of this width.

    Deliberately wider than either sibling opens (CATCH_OPEN_EXTRA_MM),
    because here the gap is the catch's entire tolerance for timing
    error - see MAX_CATCH_SPEED."""
    return min(850, int(ref["stall0"] + (width_mm + CATCH_OPEN_EXTRA_MM
                                         - ref["w0_mm"]) * PULSES_PER_MM))


def close_pulses(ref, width_mm):
    """How far to close. Overshoots the expected stall on purpose."""
    return max(0, int(ref["stall0"] + (width_mm - ref["w0_mm"])
                      * PULSES_PER_MM - CLOSE_MARGIN_PULSES))


def half_gap_mm(width_mm=None):
    """Clearance on EACH side of the cube while the fingers wait.

    This is the catch's whole displacement budget: the cube may arrive
    this far from where it was predicted and still be between the
    fingers. Everything that can go wrong - the curve of the path, the
    timing of the close, the tracking error - is spending from this one
    number, which is why each of them only gets a share.

    Independent of the cube's width by construction: the opening is sized
    as (width + CATCH_OPEN_EXTRA_MM), so the clearance either side is the
    same for every cube. width_mm is accepted only so callers can pass
    what they have."""
    return CATCH_OPEN_EXTRA_MM / 2.0


# ------------------------- arm speed -------------------------

def slow_down():
    """Put the whole process into catch speed - arm AND fingers.

    Sets vision_common's module-level speeds rather than passing a speed
    to a private copy of the move function, so every move in this project
    still goes through the ONE implementation that carries the
    joint-sweep guard, the error check and the latched-warning clear.
    Returns the previous values for the startup banner.

    MUST be called BEFORE setup_arm: the gripper speed is applied by
    enable_gripper(), which setup_arm calls, so setting it afterwards
    would leave the fingers at the floor picker's full 5000."""
    prev = (vc.JOINT_SPEED, vc.JOINT_ACC, vc.GRIPPER_SPEED)
    vc.JOINT_SPEED = CATCH_SPEED
    vc.JOINT_ACC = CATCH_ACC
    vc.GRIPPER_SPEED = CATCH_GRIPPER_SPEED
    return prev


def load_calibration():
    """(calib3, grip_ref) validated against the current SCAN pose, or
    (None, None) with the reason printed."""
    calib = v3.load_calib3()
    ref = v3.load_grip_ref()
    if calib is None or ref is None:
        print("Missing calib3.json / grip_ref.json in vision\\ - run the v3\n"
              "'3 - Auto Calibrate' once before using the catch project.")
        return None, None
    for name, data in (("calib3.json", calib), ("grip_ref.json", ref)):
        if list(data.get("scan_pose") or []) != list(vc.SCAN_POSE):
            print("%s was made under a different SCAN pose - re-run the v3 "
                  "auto calibrate." % name)
            return None, None
    return calib, ref

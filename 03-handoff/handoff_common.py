#!/usr/bin/env python3
"""Shared pieces of the HANDOFF project - taking a cube out of a hand.

This is a SEPARATE project from vision/. Nothing in vision/ is modified:
the v3 floor picker keeps working exactly as it does today. What is
reused, READ-ONLY, is the calibration - calib3.json (the pixel->mm
Jacobian) and grip_ref.json (the camera/finger geometry constant zC and
the p0<->g0 anchor). Those describe the CAMERA MOUNT and the FINGERS,
not the floor, so they are just as valid for a cube held in mid-air.

Why a new project at all, when the aim math is the same:

  1. The v3 detector is FLOOR-GATED. It keeps only depth pixels that
     stand 10mm proud of a fitted (or taught) floor plane, and the picker
     then demands the cube top sit 8..98mm above that floor. A cube held
     300mm up in a hand fails both, and the hand joins the same blob.
     handoff_detect.py replaces the floor reference with a LOCAL one -
     the nearest surface inside each YOLO box - so it works at any height
     over anything.

  2. The v3 control law is aim-once-then-descend, with an explicit gate
     that waits for the cube to STOP MOVING. A hand never stops. The
     handoff picker servos instead: it follows the cube continuously at a
     height where the camera can still see it, and only commits to the
     final descent when the hand is momentarily steady.

The one hard physical limit, and the reason the last part of the reach is
still open-loop: the D435 cannot measure closer than ~280mm, and the
camera sits ~142mm (zC) above the fingers, so once the fingers are within
~165mm of the cube the camera has already lost it. Every straight-down
pipeline here has that constraint - it is why v3's look pose is 180mm up.
So the arm tracks your hand down to ~180mm above the cube and then dives
blind. That dive is why the hand has to be still for about a second, and
why the gripper's own grasp sensor - not vision - decides whether the
grab worked.
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
    sys.path.insert(0, VISION)          # reuse vision/ without copying it

import numpy as np                                          # noqa: E402

import vision_common as vc                                  # noqa: E402
import vision3 as v3                                        # noqa: E402


# ------------------------- poses -------------------------

# Where the arm waits and watches: straight down over the cell, as high
# as it can reach. Height matters here - the higher the camera, the
# higher you can hold the cube and still be measured (see hold_band).
# 680 is the highest pose plan_search proved reachable in this cell;
# 720+ does not exist. Checked against IK at startup anyway.
WAIT_POSE = [450.0, 70.0, 680.0, 180.0, 0.0, 0.0]

# The same HOME the floor picker parks at - a known-good joint
# configuration that the straight-down poses are reachable from, used to
# reset the arm's IK branch when a move is refused for a large sweep.
HOME_DEG = [4.186, -17.164, -102.683, -0.488, 120.453, 0.0]

# The D435's minimum range - nothing closer is measurable, so this sets
# how near the camera may get before it goes blind.
#
# 190, not 280. 280 is the datasheet's RECOMMENDED start of the ideal
# range; the actual min-Z at this resolution is ~175mm. Using the
# recommended figure as a hard rejection threw away perfectly good
# readings and told the operator to "lower it a little" over and over
# while they were holding the cube at an entirely sensible height. It
# also cost 14cm off the top of the hold band for nothing.
MIN_CAM_MM = 190.0

# How far above the grab the arm follows your hand. Must keep the cube
# beyond MIN_CAM_MM: depth to the cube top from FOLLOW_H above the grab
# is FOLLOW_H - h/2 + zC, so at zC=142 and a 30mm cube, 180mm of follow
# height gives ~307mm of range - which is exactly what grip_ref's
# measured d_look0 (303mm) says. Do not lower this without redoing that
# arithmetic; below ~165mm the camera simply stops returning the cube.
FOLLOW_H = 180.0

# Fallbacks when 180mm above the cube is out of reach. Raising the tool
# moves it FARTHER from the shoulder, so a cube can be reachable at the
# grasp and not at the hover above it - the floor picker hit the same
# thing and lowers its approach the same way. With MIN_CAM_MM at 190 the
# camera still sees the cube from ~60mm up, so a lower hover costs a
# longer blind dive and nothing else. Refusing the cube costs the cube.
FOLLOW_LADDER = [180.0, 150.0, 120.0, 95.0]

# The arm's reach, for telling the operator WHICH WAY to move rather
# than that something unspecified is wrong. Measured from the base axis:
# plan_search found the cell's usable radius runs to about 700mm, and
# nothing closer than ~250mm is reachable with the tool pointing down.
REACH_MAX_MM = 700.0
REACH_MIN_MM = 250.0


# ------------------------- safety -------------------------
#
# This project points a robot arm at a human hand. Every number below is
# a limit, not a tuning knob - loosen one only with a reason.

# Reduced from the floor picker's 60 / 1000. The arm is working next to
# a person, and every servo step is short, so acceleration is what you
# actually feel. Slow enough to step out of the way of.
HANDOFF_SPEED = 25
HANDOFF_ACC = 400

# The FINGERS matter more than the arm here. The floor picker runs the
# gripper at 5000, the SDK maximum - fine when the only thing between
# the fingers is a cube on a floor, wrong when a hand is holding it.
# The gripper stalls on contact rather than crushing, but stall
# detection is a reaction: the slower the fingers arrive, the less
# travel happens before it triggers. 1500 is a third of full speed and
# still closes on a cube in well under a second.
HANDOFF_GRIPPER_SPEED = 1500

# A cube must be held at least this far above the floor to count as
# "in a hand". Anything lower is the floor picker's job, and refusing it
# here means a mis-detection on the floor can never drag the arm down.
HAND_MIN_ABOVE_FLOOR = 80.0

# Largest single servo step. The arm is at follow height (well above the
# hand) when these apply, but a capped step still bounds how fast it can
# move toward you if a detection is wrong.
MAX_XY_STEP = 60.0
MAX_Z_STEP = 50.0

# Do not move at all for less than this - stops the arm twitching at
# every millimetre of detection noise, and keeps the log readable.
DEADBAND_MM = 4.0

# Commit gate: the target must sit within STEADY_MM of the previous
# target, and the arm within ALIGN_MM of it, for STEADY_N samples in a
# row. This is what "your hand held still" means operationally.
STEADY_MM = 6.0
ALIGN_MM = 6.0
STEADY_N = 3

# Cube sizes this can take. Same upper bound as the floor picker's
# gripper limit; the lower bound keeps noise blobs out.
MIN_CUBE_MM = 15.0
MAX_CUBE_MM = 78.0

# Where on the cube the fingers close, as a fraction of its height
# measured DOWN FROM THE TOP. The floor picker uses 0.5 - the middle -
# which is right for a cube standing on a floor. Here the bottom of the
# cube is where the hand is, so the grab is deliberately biased upward:
# 0.38 of a 30mm cube puts the fingers ~11mm below the top face and
# ~4mm higher than mid-height, which is 4mm more clearance from
# fingertips at no cost in grip (the fingers are far taller than that).
# Do not push it much below 0.3 or the grab starts riding the top edge.
GRAB_DEPTH_FRAC = 0.38

PULSES_PER_MM = 10.0
OPEN_EXTRA_MM = 30.0
# Close this far past the expected stall. Generous on purpose: the
# gripper's grasp sensor - not this number - decides whether the grab
# worked, so it only has to cover the vision width error.
CLOSE_MARGIN_PULSES = 100
POSITION_MISS_TOL = 10       # fallback when the gripper cannot report
BOX_WEIGHT = 0.05


# ------------------------- geometry -------------------------

def floor_grab_z(ref):
    """Flange Z at which the fingers would close on the floor itself.

    Falls out of the same constants the grab height uses: a cube of
    height h sitting on the floor is grabbed at 580 - floor0 + zC + h/2,
    so this is that expression with h = 0. Everything about "how high is
    it being held" is measured from here."""
    return float(vc.SCAN_POSE[2] - ref["floor0"] + ref["zC"])


def above_floor_mm(ref, grab_z):
    """How far above the floor a grab at this flange Z actually is."""
    return float(grab_z - floor_grab_z(ref))


def hold_band(ref, wait_z=None, height_mm=30.0):
    """(lowest, highest) grab Z the arm can both SEE and accept.

    Lower limit: HAND_MIN_ABOVE_FLOOR above the floor, so a cube on the
    floor is never mistaken for one in a hand.
    Upper limit: the camera's near range. Held too close to the camera
    the depth simply is not measured, so the cube stops existing rather
    than being measured wrongly."""
    wz = WAIT_POSE[2] if wait_z is None else wait_z
    lo = floor_grab_z(ref) + HAND_MIN_ABOVE_FLOOR
    hi = wz - MIN_CAM_MM - height_mm * GRAB_DEPTH_FRAC + ref["zC"]
    return lo, hi


def aim_hand(calib, ref, obs_pose, seen):
    """Grasp target (gx, gy, gz, yaw) for a cube seen from obs_pose.

    Identical to v3.aim_from_pose in X, Y and yaw - deliberately, since
    that mapping was verified algebraically and on hardware. Two things
    differ, both about height:

      * v3 clamps its answer into a window around the FLOOR, which is
        meaningless (and actively wrong) for a cube held in the air. The
        underlying relation

            Zgrab = Zobs - depth - (height * fraction) + zC

        never mentions the floor; only the clamp did. The clamp's safety
        job is done instead by hold_band, checked by the caller.
      * the fraction is GRAB_DEPTH_FRAC rather than v3's implicit 0.5,
        to keep the fingers a little farther from the hand.

    Requires obs_pose to be at the SCAN yaw. The camera's offset from
    the flange is a constant only while the wrist is not rotated - which
    is why the servo loop never turns the wrist until it commits."""
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
    """Where a STATIONARY cube's image pixel moves to after the arm moves.

    Inverting the same mapping aim_hand uses. From a straight-down pose,

        cube_xy = flange_xy + camera_offset + (d/d_ref) * J * (p - CENTRE)

    and the cube has not moved, so after a flange step of (dx, dy, dz)
    - which also brings the camera dz nearer, d' = d - dz - the new
    pixel must satisfy

        (d'/d_ref) * J * (p' - C) = (d/d_ref) * J * (p - C) - (dx, dy)

    Solving that keeps the tracker's lock tight: without it the "is this
    still my cube?" radius would have to be wide enough to swallow the
    arm's own motion, and would stop excluding anything."""
    J = np.asarray(calib["J"], dtype=float)
    d_ref = calib.get("d_ref") or ref["d0"]
    d_new = depth_mm - d_flange_z
    if d_new <= 1.0:
        return list(pixel)                  # nonsense step; keep the lock
    p = np.asarray(pixel, dtype=float) - np.asarray(v3.CENTER)
    rhs = (depth_mm / d_ref) * (J @ p) - np.asarray(d_flange_xy, dtype=float)
    p_new = np.linalg.solve((d_new / d_ref) * J, rhs)
    return [float(p_new[0] + v3.CENTER[0]), float(p_new[1] + v3.CENTER[1])]


def out_of_reach_hint(gx, gy):
    """Why the arm cannot get there, in a form you can act on.

    "I cannot reach there" is true and useless while you are standing
    holding a cube. The radius from the base says which way to move, and
    that is nearly always the answer."""
    r = float(np.hypot(gx, gy))
    if r > REACH_MAX_MM:
        return ("that is about %.0fcm too far out - bring it in toward "
                "the arm" % ((r - REACH_MAX_MM) / 10.0 + 0.5))
    if r < REACH_MIN_MM:
        return ("that is too close in to the arm's base - hold it further "
                "out")
    return ("I cannot reach that spot with the wrist pointing down - "
            "move it a hand's width and I will try again")


def step_toward(cur_xyz, target_xyz, max_xy=None, max_z=None):
    """Next waypoint: target, but no farther than one capped step away.

    XY and Z are capped separately - a hand moving sideways should not
    slow the descent, and a descent should never be turned into a lunge.
    """
    max_xy = MAX_XY_STEP if max_xy is None else max_xy
    max_z = MAX_Z_STEP if max_z is None else max_z
    cx, cy, cz = cur_xyz
    tx, ty, tz = target_xyz
    dx, dy = tx - cx, ty - cy
    dist = float(np.hypot(dx, dy))
    if dist > max_xy:
        dx, dy = dx * max_xy / dist, dy * max_xy / dist
    dz = tz - cz
    dz = max(-max_z, min(max_z, dz))
    return [cx + dx, cy + dy, cz + dz]


# ------------------------- gripper widths -------------------------

def open_pulses(ref, width_mm):
    """How far to open for a cube of this width (plus clearance)."""
    return min(850, int(ref["stall0"] + (width_mm + OPEN_EXTRA_MM
                                         - ref["w0_mm"]) * PULSES_PER_MM))


def close_pulses(ref, width_mm):
    """How far to close. Overshoots the expected stall on purpose."""
    return max(0, int(ref["stall0"] + (width_mm - ref["w0_mm"])
                      * PULSES_PER_MM - CLOSE_MARGIN_PULSES))


# ------------------------- arm speed -------------------------

def slow_down():
    """Put the whole process into handoff speed - arm AND fingers.

    This sets vision_common's module-level speeds, rather than passing a
    speed to a private copy of the move function, so that every move in
    this project still goes through the ONE implementation that carries
    the joint-sweep guard, the error check and the latched-warning
    clear. Returns the previous values for the startup banner; nothing
    in this project restores them, because nothing here should ever run
    fast.

    MUST be called BEFORE setup_arm: the gripper speed is applied by
    enable_gripper(), which setup_arm calls, so setting it afterwards
    would leave the fingers at the floor picker's full 5000 for the
    whole run."""
    prev = (vc.JOINT_SPEED, vc.JOINT_ACC, vc.GRIPPER_SPEED)
    vc.JOINT_SPEED = HANDOFF_SPEED
    vc.JOINT_ACC = HANDOFF_ACC
    vc.GRIPPER_SPEED = HANDOFF_GRIPPER_SPEED
    return prev


def load_calibration():
    """(calib3, grip_ref) validated against the current SCAN pose, or
    (None, None) with the reason printed."""
    calib = v3.load_calib3()
    ref = v3.load_grip_ref()
    if calib is None or ref is None:
        print("Missing calib3.json / grip_ref.json in vision\\ - run the v3\n"
              "'3 - Auto Calibrate' once before using the handoff project.")
        return None, None
    for name, data in (("calib3.json", calib), ("grip_ref.json", ref)):
        if list(data.get("scan_pose") or []) != list(vc.SCAN_POSE):
            print("%s was made under a different SCAN pose - re-run the v3 "
                  "auto calibrate." % name)
            return None, None
    return calib, ref

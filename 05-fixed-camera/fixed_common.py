#!/usr/bin/env python3
"""Shared pieces of the FIXED-CAMERA picker.

A SEPARATE project from vision\\, handoff\\ and realtime\\. Those three all
assume the camera rides the WRIST; this one assumes it does not move at
all. That single difference invalidates all of their calibration and
most of their geometry, which is why this is its own folder rather than
a flag.

WHAT CHANGES WHEN THE CAMERA STOPS MOVING

  eye-in-hand (vision\\)     the mapping from a pixel to a base-frame mm
                            is a FUNCTION OF THE CURRENT FLANGE POSE.
                            calib3's J, grip_ref's zC, camera_offset(),
                            aim_from_pose() and the yaw mirror all exist
                            to evaluate that function.

  eye-to-hand (here)        the camera never moves relative to the base,
                            so there is exactly ONE rigid transform

                                p_base = R . p_cam + t

                            and it never changes. Twelve numbers, measured
                            once.

THE ONE IDEA THIS PROJECT IS BUILT ON

Transform the WHOLE depth image into the base frame before doing any
geometry. After that step "up" is base +Z, always, and it makes no
difference whatsoever where the camera is bolted or at what angle it
looks. A cube's top face is "the points near the top of its cluster"; its
grasp yaw is a minAreaRect of those points in base XY, already in the
degrees the wrist wants.

That is what makes a 45-degree side mount affordable. Working in CAMERA
coordinates, an oblique view wrecks every gate in detect_cube - the top
face is a foreshortened parallelogram, the squareness test fails, the
"floor" is not a constant depth. Working in BASE coordinates, the oblique
view is just a sparser sampling of the same overhead picture.

THE TOOL IS A VACUUM CUP, NOT THE TWO-FINGER GRIPPER

That is a second, independent departure from vision\\, and it simplifies
the geometry rather than complicating it. Fingers grip the SIDES of a
cube at some fraction of its height, so the grasp height depended on how
tall the cube was. A cup grips the TOP - which is precisely the surface
the camera measures - so the grasp height is the measured top face and
nothing else, for a cube of any size. A round cup is also rotationally
symmetric, so the cube's yaw stops mattering at all.

WHAT IS REUSED FROM vision\\, AND WHAT IS NOT

  reused   vision_common's movej (with its joint-sweep guard, per-move
           error check and latched-warning clear), ik, moveto, wrap90 and
           the run log. All of that is about the ARM and is untouched by
           either change.

  NOT      vision_common.setup_arm, enable_gripper, grip, grasp_status.
           Every one of them drives the two-finger modbus gripper, and
           setup_arm RAISES when that gripper does not answer - which is
           what a vacuum tool does. setup_arm below replaces it.

  DEAD     calib3.json entirely (J is a pixel->mm Jacobian for a camera
           at a known height above its target, and there is no such
           height any more), and grip_ref.json entirely (it describes the
           wrist mount and the fingers, and neither exists here now).

Because the camera is off the wrist, vision_pick3, handoff_pick and
catch_pick CANNOT RUN until it is remounted. That is the price of the
single-camera setup and it is not recoverable in software.
"""
import json
import os
import sys
import time

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
import pyrealsense2 as rs                                   # noqa: E402

import vision_common as vc                                  # noqa: E402


HANDEYE_FILE = os.path.join(HERE, "handeye.json")
FLOOR_FILE = os.path.join(HERE, "floor_base.json")
CAPT = os.path.join(HERE, "captures")
LOGS = os.path.join(HERE, "logs")


# ------------------------- run logs -------------------------
#
# Every script here writes one, into fixedcam\logs\ rather than vision\'s
# folder, so a session's evidence is in one place next to the code that
# produced it.
#
# CALL start_log BEFORE IMPORTING THE xArm SDK. The SDK's logger binds a
# handler to sys.stderr when it is imported, so a tee installed
# afterwards never sees its output - and the SDK's '[SDK][ERROR] ...
# code=1' lines are exactly the evidence worth keeping, because they are
# how a latched controller fault announces itself.

class _Tee(object):
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


def start_log(name):
    """Tee stdout+stderr into logs/<name>_<timestamp>.log. Returns the
    path, or None - never fatal, a missing log must not stop a run."""
    try:
        if not os.path.isdir(LOGS):
            os.makedirs(LOGS)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS, "%s_%s.log" % (name, stamp))
        fh = open(path, "w", encoding="utf-8", errors="replace")
    except Exception as e:
        print("(no run log: %s)" % e)
        return None
    fh.write("=== %s  %s ===\n" % (name, stamp))
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print("(log: %s)" % path)
    return path


# ------------------------- the cell -------------------------

# The floor picker's HOME. Firmware 1.6.9 seeds IK from the arm's CURRENT
# joints, so a pose can be refused for a huge joint sweep that is really
# just a different configuration branch; resetting through HOME fixes it.
# (vision\ learned this the hard way - see the stale-IK note in its log.)
HOME_DEG = [4.186, -17.164, -102.683, -0.488, 120.453, 0.0]

# Where the arm parks so it is not standing in the camera's way. High and
# folded back over the base. Checked by IK at startup, never assumed.
STANDOFF_POSE = [300.0, 0.0, 620.0, 180.0, 0.0, 0.0]

# The working area, in base-frame mm. Everything outside this box is
# thrown away before anything else looks at it - which is how the cell
# walls, the bench, the operator and the arm's own upper links stop being
# candidate cubes. Measured extent of this cell (plan_search); widen only
# after checking the arm can actually reach the new corner.
WORK_X = (250.0, 780.0)
WORK_Y = (-400.0, 400.0)

# The arm's usable annulus, for turning "not reachable" into a sentence
# the operator can act on.
REACH_MAX_MM = 700.0
REACH_MIN_MM = 250.0


# ------------------------- what a cube is -------------------------

MIN_CUBE_MM = 15.0
MAX_CUBE_MM = 78.0

# How far above the taught surface a point must sit to count as an
# object. Below this it is the surface itself, or its noise.
POP_MM = 8.0

# Thickness of the "top face" band, measured DOWN from the highest point
# of a cluster. Kept thin: the whole point is to get the flat top and
# leave the side faces out, because side faces are what elongate a rect
# and make a cube measure wider than it is.
TOP_BAND_MM = 6.0


# ------------------------- the suction cup -------------------------
#
# The cup diameter. It is not a comfort setting: a cup can only seal on a
# flat, unbroken patch of surface, so the cube's top face has to be
# comfortably BIGGER than the cup. With 30mm cubes and an 18mm cup there
# is 6mm of face outside the cup on every side, which is enough to
# tolerate the aim error and still land wholly on the face.
CUP_DIA_MM = 18.0

# Flat face required around the cup, per side. A cube whose top face is
# smaller than CUP_DIA_MM + 2*CUP_MARGIN_MM is refused rather than
# attempted - a cup half on and half off an edge does not seal, it just
# hisses.
CUP_MARGIN_MM = 4.0

# How far BELOW the measured top face the flange is commanded, to press
# the cup onto it.
#
# It is small on purpose. The calibration is taken with the target
# already stuck to the cup, so the fitted transform reproduces the
# engaged geometry exactly and in principle no press is needed at all.
# The reason it is not zero: during calibration the target HANGS from the
# cup, so gravity pulls a bellows cup to its longest; pressing down onto
# a cube compresses it. A couple of millimetres covers that difference.
#
# THIS IS THE FIRST NUMBER TO TUNE if picks fail to seal - raise it in
# 1mm steps. The floor clamp below means it can never drive into the
# table however large it is set.
CUP_PRESS_MM = 2.0

# The cup must never be commanded closer than this to the taught surface.
MIN_ABOVE_FLOOR = 3.0

# Vacuum wiring. 1 = plug-in connection (tool GPIO 0 and 1), which is
# UFACTORY's default and what this cell uses; 2 = contact connection
# (GPIO 3 and 4). See fixed_vacuum.py, which probes both.
VACUUM_HW = 1

# How long to wait for the cup to report a seal before calling it a miss.
VACUUM_TIMEOUT_S = 2.0

# How long to wait after cutting the vacuum before moving away, so the
# cube is actually released rather than dragged.
VACUUM_RELEASE_S = 0.4

# Mass and centre of gravity of the vacuum tool, for set_tcp_load. The
# two-finger gripper this cell used before is 0.82kg; SET THIS to the
# vacuum tool's real mass. Declaring it wrong does not break a pick, but
# it degrades the controller's collision detection, which is a safety
# feature and not one to leave guessed.
TOOL_MASS_KG = 0.6
TOOL_COG = [0.0, 0.0, 48.0]


# ------------------------- safety -------------------------
#
# The arm reaches into a cell a person also reaches into, and - unlike
# the floor picker - it is being aimed by a camera on a stand that can be
# knocked. Until the drift check below has hardware behind it, run slow.

PICK_SPEED = 40                 # floor picker runs 60
PICK_ACC = 700                  # floor picker runs 1000


def slow_down():
    """Put the whole process onto this project's speeds.

    Sets vision_common's module-level values rather than passing a speed
    into a private copy of the move function, so every move still goes
    through the ONE implementation carrying the joint-sweep guard, the
    per-move error check and the latched-warning clear."""
    prev = (vc.JOINT_SPEED, vc.JOINT_ACC)
    vc.JOINT_SPEED = PICK_SPEED
    vc.JOINT_ACC = PICK_ACC
    return prev


# ------------------------- the camera -------------------------

# 848x480 is the D435 depth sensor's NATIVE resolution - the one its
# depth quality is specified at. Running 640x480 crops and rescales it,
# throwing away about a quarter of the lateral samples for nothing. At
# ~1m that is the difference between roughly 3.1mm and 2.4mm between
# samples on a horizontal surface, which is roughly 60 points on a 30mm
# cube top instead of 90. Sampling density is the accuracy limit on a
# cube this small, so this is the cheapest improvement available.
DEPTH_W, DEPTH_H, FPS = 848, 480, 30


class Camera(object):
    """One persistent RealSense pipeline.

    COLOUR IS ALIGNED TO DEPTH, not the other way round, which is the
    opposite of every script in vision\\. The reason is arithmetic, not
    taste: this project deprojects EVERY pixel of every frame, and
    rs2_deproject_pixel_to_point called 400000 times in Python is far too
    slow, so the deprojection is done vectorised in numpy. That is only
    exact for a pinhole model with no distortion - and the DEPTH stream is
    rectified, so its intrinsics carry all-zero coefficients, while the
    COLOUR stream's do not. Aligning to depth is what makes the fast path
    the correct path.

    The cost is that colour is resampled instead of depth, which matters
    to nobody here: colour is used only to say "red" or "blue".

    TWO SETTINGS THAT COST NOTHING AND BUY REAL ACCURACY

    high_accuracy    the D435's High Accuracy visual preset. It raises
                     the confidence threshold, so it returns FEWER depth
                     pixels and the ones it returns are much better. For
                     measuring the flat top of a small cube that is the
                     right trade every time - a cube top is opaque and
                     well lit by the projector, so it survives the
                     threshold while the noisy speculars do not.

    temporal         averages each pixel's depth over recent frames.
                     Free accuracy on a scene that is not moving, which
                     is exactly this half of the project.

                     IT MUST BE TURNED OFF FOR MOVING CUBES. A temporal
                     filter on something in motion smears it along its
                     own path - it would quietly corrupt the one
                     measurement the moving picker exists to make.

    Hole filling is deliberately NOT used at any point: it invents depth
    where the sensor reported none, and invented depth under a cube is
    indistinguishable from a cube that is not there."""

    def __init__(self, width=DEPTH_W, height=DEPTH_H, fps=FPS, warmup=30,
                 high_accuracy=True, temporal=True):
        self.width, self.height, self.fps = width, height, fps
        self.warmup = warmup
        self.high_accuracy = high_accuracy
        self.pipe = None
        self.align = None
        self.intr = None
        self._rays = None           # cached (H,W,2) unit-ray table
        self._temporal = rs.temporal_filter() if temporal else None

    def start(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16, self.fps)
        profile = self.pipe.start(cfg)

        if self.high_accuracy:
            # Best effort, and genuinely optional. The preset raises the
            # depth confidence threshold, which is worth having, but a
            # firmware or SDK build that does not expose the option is a
            # reason to carry on with the camera's default - not a reason
            # to refuse to run. Checked with supports() rather than
            # caught as an exception so the normal case does not print
            # something that reads like a fault.
            sensor = profile.get_device().first_depth_sensor()
            if sensor.supports(rs.option.visual_preset):
                try:
                    sensor.set_option(
                        rs.option.visual_preset,
                        float(rs.rs400_visual_preset.high_accuracy))
                    print("   depth preset: High Accuracy")
                except Exception as e:
                    print("   (High Accuracy preset refused: %s)" % e)
            else:
                print("   depth preset: camera default (this build does "
                      "not expose High Accuracy)")
            # Full laser power is exposed far more widely than the preset
            # is, and on a matt indoor surface it is most of the same
            # benefit: more projected texture, so more pixels pass the
            # confidence test.
            if sensor.supports(rs.option.laser_power):
                try:
                    rng = sensor.get_option_range(rs.option.laser_power)
                    sensor.set_option(rs.option.laser_power, rng.max)
                except Exception:
                    pass

        self.align = rs.align(rs.stream.depth)
        for _ in range(self.warmup):
            self.pipe.wait_for_frames()
        frames = self.align.process(self.pipe.wait_for_frames())
        self.intr = frames.get_depth_frame().profile \
            .as_video_stream_profile().intrinsics
        coeffs = list(getattr(self.intr, "coeffs", []) or [])
        if any(abs(c) > 1e-6 for c in coeffs):
            # Not fatal - print it rather than refuse, since a future
            # firmware could populate these harmlessly - but the fast
            # deprojection silently ignores them, so say so out loud.
            print("   NOTE: the depth stream reports distortion %s; the "
                  "vectorised deprojection ignores it." % coeffs)

        # Warm the temporal filter up before anyone reads a frame: its
        # first few outputs are still converging, and a calibration point
        # taken from a half-converged frame is a silent outlier.
        for _ in range(8):
            self.frame()
        depth_mm, _, _ = self.frame()
        valid = float((depth_mm > 0).mean()) * 100.0
        print("   depth %dx%d, %.0f%% of pixels have a reading"
              % (self.intr.width, self.intr.height, valid))
        if valid < 45.0:
            print("   NOTE: that is low. On this cell's glass floor most "
                  "of the missing\n"
                  "         pixels are the floor itself, which is "
                  "expected - but if CUBE\n"
                  "         TOPS are coming out sparse, try the lights "
                  "OFF (this camera's\n"
                  "         depth is measurably better in the dark) or "
                  "move the camera closer.")
        return self

    def close(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def frame(self):
        """One aligned frame set.

        Returns (depth_mm, color_bgr, stamp_s):
            depth_mm  (H,W) float32, 0 where the sensor returned nothing
            color_bgr (H,W,3) uint8, aligned to the depth pixels
            stamp_s   the DEPTH frame's own timestamp in seconds, not the
                      time this returned - the difference is the pipeline
                      latency, and it is the whole measurement when the
                      cube is moving.
        """
        frames = self.align.process(self.pipe.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if self._temporal is not None:
            depth = self._temporal.process(depth).as_depth_frame()
        scale = depth.get_units() * 1000.0          # z16 units -> mm
        depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * scale
        color_bgr = np.asanyarray(color.get_data())
        return depth_mm, color_bgr, depth.get_timestamp() / 1000.0

    def rays(self):
        """Cached ray table for this camera - see ray_table."""
        if self._rays is None:
            self._rays = ray_table(self.intr)
        return self._rays


def ray_table(intr):
    """(H,W,2) table of (x/z, y/z) for every pixel.

    Deprojection is then just multiplying by Z, which is the whole reason
    this is fast enough to run on every frame. A module function rather
    than a method so the offline tests can build a synthetic scene
    through exactly this formula instead of a copy of it."""
    u = (np.arange(intr.width, dtype=np.float32) - intr.ppx) / intr.fx
    v = (np.arange(intr.height, dtype=np.float32) - intr.ppy) / intr.fy
    return np.stack(np.meshgrid(u, v), axis=-1)


def cloud_cam(depth_mm, rays):
    """(H,W,3) camera-frame points in mm. Invalid pixels come out as 0.

    RealSense camera frame: +X right, +Y down, +Z forward along the
    optical axis. None of that matters downstream - the calibration finds
    whatever rotation relates it to the base - but it is worth naming so
    nobody 'fixes' a sign here."""
    z = depth_mm[..., None]
    return np.concatenate([rays * z, z], axis=-1).astype(np.float32)


# ------------------------- the transform -------------------------

def apply_transform(T, pts):
    """p_base = R . p_cam + t, for a single point or an (...,3) array."""
    R = np.asarray(T["R"], dtype=np.float32)
    t = np.asarray(T["t"], dtype=np.float32)
    return np.asarray(pts, dtype=np.float32) @ R.T + t


def fit_rigid(cam_pts, base_pts):
    """Least-squares RIGID transform mapping camera points onto base
    points (Kabsch / Umeyama with the scale held at 1).

    SCALE IS DELIBERATELY NOT FITTED. Both frames are already metric
    millimetres, so a free scale has nothing legitimate to absorb - it
    would instead soak up depth bias into a multiplier, which flatters
    the residual while making the error grow with distance from the
    calibration cloud. Fixing it at 1 keeps the reported residual an
    honest millimetre error.

    The best-fit scale is still COMPUTED and returned, purely as a
    diagnostic: it should come out within about 1% of 1.0. If it does
    not, something is wrong in units or in the depth itself, and the
    residual alone would not have told you.

    Needs at least 3 non-collinear correspondences; 12+ spread over the
    working volume is what makes the residual mean anything."""
    A = np.asarray(cam_pts, dtype=float)
    B = np.asarray(base_pts, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("need matching (N,3) point sets")
    if len(A) < 3:
        raise ValueError("need at least 3 point pairs, got %d" % len(A))

    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    # Reflections are valid solutions of the SVD but not of physics - a
    # mirrored frame would fit the points and then send the arm to the
    # wrong side of everything. Force a proper rotation.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca

    pred = A @ R.T + t
    resid = np.linalg.norm(pred - B, axis=1)
    var_a = float((A0 ** 2).sum())
    best_scale = float((S[0] + S[1] + d * S[2]) / var_a) if var_a > 0 else 0.0

    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "residuals_mm": [round(float(r), 2) for r in resid],
        "rms_mm": round(float(np.sqrt((resid ** 2).mean())), 2),
        "max_mm": round(float(resid.max()), 2),
        "best_scale": round(best_scale, 5),
        "n": int(len(A)),
    }


def distance_ratio(cam_pts, base_pts, min_mm=100.0):
    """Median of (robot distance / camera distance) over all point pairs.

    A sharper scale diagnostic than fit_rigid's best_scale, for one
    reason: it never fits anything. A rigid transform preserves distance,
    so for every pair of samples the flange must have travelled exactly
    as far as the camera saw the target travel. The RATIO of those two
    distances is immune to wrist tilt, cup compression and every other
    robot-side error - those move the endpoints but average out over
    pairs - while a camera whose geometry is off by k% shifts EVERY
    ratio by k%, coherently. The median over all pairs is then a direct
    reading of the camera's metric scale, needing no transform at all.

    best_scale from the Umeyama fit cannot make that separation: at
    n=12 its own sampling noise under a few degrees of wrist tilt is
    about +-3% (one sd), so a 0.94 there is only suspicion. The same
    data's pairwise ratios pin it down.

    Pairs closer than min_mm are skipped - dividing two small noisy
    distances tells you about the noise, not the scale.

    Returns (median_ratio, n_pairs_used) or (None, 0)."""
    A = np.asarray(cam_pts, dtype=float)
    B = np.asarray(base_pts, dtype=float)
    ratios = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            dc = float(np.linalg.norm(A[i] - A[j]))
            db = float(np.linalg.norm(B[i] - B[j]))
            if dc >= min_mm and db >= min_mm:
                ratios.append(db / dc)
    if not ratios:
        return None, 0
    return float(np.median(ratios)), len(ratios)


# ------------------------- the support surface -------------------------
#
# Stored as a plane in the BASE frame: z = a*x + b*y + c. In base
# coordinates a level floor is simply a=b=0 and c = its height, and the
# cell's real ~39mm of tilt shows up as small a and b. This is the same
# job vision\'s floor_ref.json does, expressed in the frame that makes it
# trivial.

def plane_z(coef, x, y):
    """Height of the taught surface under (x, y)."""
    a, b, c = coef
    return a * np.asarray(x) + b * np.asarray(y) + c


def fit_plane(pts, iters=3, trim_mm=18.0):
    """Robust least-squares z = a*x + b*y + c over (N,3) base-frame points.

    Seeded flat at the median height, then refitted keeping only inliers
    within trim_mm. The trim is what stops a few cubes, a hand or a strip
    of through-glass background from tilting the whole plane."""
    P = np.asarray(pts, dtype=float)
    if len(P) < 50:
        return None
    coef = np.array([0.0, 0.0, float(np.median(P[:, 2]))])
    ok = None
    for _ in range(iters):
        keep = np.abs(P[:, 2] - plane_z(coef, P[:, 0], P[:, 1])) < trim_mm
        if keep.sum() < 50:
            break
        M = np.column_stack([P[keep, 0], P[keep, 1], np.ones(keep.sum())])
        sol, *_ = np.linalg.lstsq(M, P[keep, 2], rcond=None)
        coef, ok = sol, keep
    # Returning the flat seed would be a silent downgrade - a caller
    # cannot tell "fitted" from "gave up" - so say nothing rather than
    # something wrong.
    return coef.tolist() if ok is not None else None


def in_work_box(pts):
    """Boolean mask: which (...,3) base-frame points are inside the cell."""
    P = np.asarray(pts)
    return ((P[..., 0] >= WORK_X[0]) & (P[..., 0] <= WORK_X[1]) &
            (P[..., 1] >= WORK_Y[0]) & (P[..., 1] <= WORK_Y[1]))


# ------------------------- has the camera moved? -------------------------
#
# THE ONE FAILURE MODE A FIXED CAMERA HAS AND A WRIST CAMERA DOES NOT.
#
# A wrist camera cannot come loose without the arm noticing - its whole
# calibration is re-anchored on every move. A camera on a stand can be
# nudged by a sleeve at ten in the morning and every pick after that is
# wrong, in silence, with the detector still reporting confident cubes at
# confident millimetre positions. Nothing downstream can tell.
#
# So something must witness it. The witness cannot be the working surface
# itself, because on this cell's glass floor the surface is invisible to
# the depth camera - that is the whole reason it has to be taught. But
# whatever lies UNDER the glass is visible, is rigid, and is bolted to
# the same room as the robot. Its apparent height in the robot's frame is
# therefore a constant, and it stops being constant the moment the camera
# moves.
#
# Cheap, needs no extra hardware, and it fails in the right direction: a
# missing witness (an opaque floor, nothing below) skips the check
# instead of blocking the run.

DRIFT_WARN_MM = 15.0
DRIFT_FAIL_MM = 40.0


def background_signature(base_pts, floor_coef, below_mm=100.0, min_pts=500):
    """(median height, count) of whatever sits well below the surface.

    None when there is not enough of it to be a stable reference - which
    is the normal answer on an opaque floor."""
    P = np.asarray(base_pts)
    if len(P) < min_pts:
        return None
    h = P[:, 2] - plane_z(floor_coef, P[:, 0], P[:, 1])
    sel = h < -below_mm
    if int(sel.sum()) < min_pts:
        return None
    return float(np.median(P[sel, 2])), int(sel.sum())


# How far either side of the taught plane to look for the surface again.
SURFACE_BAND_MM = 60.0


def _drift_offset(base_pts, floor):
    """(offset_mm, what_was_measured) or (None, why_not).

    TWO WITNESSES, tried in order, because which one exists depends on
    what the floor is made of:

      the working surface itself   the obvious one, and the best one -
                                   but it only works on an OPAQUE floor,
                                   because a glass one is invisible to
                                   the depth camera, which is the whole
                                   reason it has to be taught

      whatever lies under the glass   rigid, bolted to the same room as
                                      the robot, and visible precisely
                                      when the floor is not

    Between them every floor this cell has had is covered."""
    P = np.asarray(base_pts)
    if len(P) < 200:
        return None, "too little of the scene is visible to judge"

    resid = P[:, 2] - plane_z(floor["coef"], P[:, 0], P[:, 1])
    near = P[np.abs(resid) < SURFACE_BAND_MM]
    centre = (float(np.mean(WORK_X)), float(np.mean(WORK_Y)))

    if len(near) >= 1000:
        coef = fit_plane(near)
        if coef is not None:
            return (float(plane_z(coef, *centre))
                    - float(plane_z(floor["coef"], *centre)),
                    "the working surface")

    # The surface is not where it was taught. If the teach saw plenty of
    # it and now almost none of it is in the band, that ABSENCE is the
    # measurement - and it must not be reported as "cannot tell", which
    # is how a guard fails open on exactly the large movement it exists
    # to catch.
    taught = floor.get("points") or 0
    if taught >= 1000 and len(near) < 0.2 * taught:
        return None, ("FAIL: the taught surface has almost vanished from "
                      "the scene (%d points where %d were measured). "
                      "Either the camera has moved a long way or the "
                      "table has." % (len(near), taught))

    stored = floor.get("background_z")
    if stored is not None:
        sig = background_signature(P, floor["coef"])
        if sig is not None:
            return sig[0] - stored, "the structure below the glass"
    return None, "no drift witness is visible right now"


def drift_check(base_pts, floor):
    """(status, message) where status is 'ok', 'skip', 'warn' or 'fail'."""
    off, witness = _drift_offset(base_pts, floor)
    if off is None:
        if witness.startswith("FAIL: "):
            return "fail", witness[6:] + \
                "\n   Re-run the calibration AND the floor teach, in that " \
                "order."
        return "skip", witness
    if abs(off) >= DRIFT_FAIL_MM:
        return "fail", (
            "%s reads %+.0fmm from where it was taught.\n"
            "   The camera has almost certainly been moved. Every position "
            "it reports is\n"
            "   wrong by about that much, so the arm would be aimed wrong "
            "by about that\n"
            "   much. Re-run the calibration AND the floor teach - in that "
            "order." % (witness, off))
    if abs(off) >= DRIFT_WARN_MM:
        return "warn", ("%s reads %+.0fmm from where it was taught - the "
                        "camera may have been nudged." % (witness, off))
    return "ok", "%s matches the calibration (%+.0fmm)" % (witness, off)


# ------------------------- the grasp frame -------------------------
#
# THE TOOL LENGTH IS NEVER MEASURED, AND THIS IS WHY.
#
# The calibration is done with a flat target STUCK TO THE CUP. Each pair
# it records is
#
#     p_cam  = where the camera sees the held target's top face
#     p_base = the FLANGE pose the controller reports at that moment
#
# and the target's top face IS the cup's contact plane, because that is
# the surface the cup is sealed against. So the transform it fits does
# not map "camera point -> where that point is". It maps
#
#     camera point  ->  the flange pose that puts the CUP ON that point
#
# The distance from the flange to the cup's face is inside the fitted
# translation, absorbed, never named. Nothing has to measure it, and
# nothing can get it wrong.
#
# WHAT SUCTION DELETES THAT FINGERS NEEDED
#
# Fingers grip the SIDES of a cube at some fraction of its height, so the
# grasp Z depended on how tall the cube was, and the calibration had to
# be told how tall ITS cube was to anchor that. A cup grips the TOP, and
# the top is exactly what the camera measures. So:
#
#     flange Z to grasp = top_z - CUP_PRESS_MM      any cube, any height
#
# No cube height, no calibration-cube height, no fraction. And a round
# cup on a flat face is rotationally symmetric, so the cube's yaw does
# not matter either - the whole yaw convention the finger version had to
# measure and match simply has nothing to line up.
#
# The price is the same as before: this frame's Z origin is not the
# robot's, being offset by the flange-to-cup distance. X and Y are true
# base millimetres (the offset is along the tool axis, which points
# straight down at every calibration pose, so it is purely vertical), and
# Z DIFFERENCES are exact, which is all anything uses:
#
#     height above the surface = top_z - plane_z(floor, x, y)
#     the cup touches the surface at exactly floor_z
#
# That second line is worth reading twice. Under fingers the surface's
# value in this frame was NOT where the fingers touched it, and getting
# that wrong permitted a pose 15mm into the table. Under suction the two
# coincide, because the calibration target's top face and a cube's top
# face are the same kind of thing. The clamp is simply floor_z + margin.
#
# Keep this straight when reading a number off the arm: the Z printed
# here is a FLANGE COMMAND, not a height above the table. "How high is
# it" is always height_mm.


def grasp_z(cube, floor_coef):
    """Flange Z that puts the cup on this cube's top face, clamped clear
    of the surface."""
    z = cube["top_z"] - CUP_PRESS_MM
    floor = float(plane_z(floor_coef, cube["center"][0], cube["center"][1]))
    return float(max(floor + MIN_ABOVE_FLOOR, z))


def cup_fits(cube):
    """(ok, reason) - is this top face big enough for the cup to seal?

    Measured on the SHORT side of the top face, not the long one: a cup
    landing on a 60x20mm face has nothing to seal against across the
    narrow direction, however generous the other one is."""
    need = CUP_DIA_MM + 2.0 * CUP_MARGIN_MM
    short = cube.get("short_mm", cube["width_mm"])
    if short < need:
        return False, ("top face is only %.0fmm across - an %.0fmm cup "
                       "needs %.0fmm to seal on" % (short, CUP_DIA_MM, need))
    return True, ""


# ------------------------- files -------------------------

def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        # A hand-edited or half-written json must not crash a launcher on
        # import - vision\ learned this from a corrupt places.json.
        print("   (could not read %s: %s)" % (os.path.basename(path), e))
        return None


def save_handeye(data):
    _save(HANDEYE_FILE, data)


def load_handeye():
    T = _load(HANDEYE_FILE)
    if T is None:
        return None
    if not all(k in T for k in ("R", "t")):
        print("   (handeye.json is missing R/t - re-run the calibration)")
        return None
    return T


def save_floor(data):
    _save(FLOOR_FILE, data)


def load_floor():
    f = _load(FLOOR_FILE)
    if f is None or "coef" not in f:
        return None
    return f


# ------------------------- the vacuum gripper -------------------------
#
# The xArm drives the cup through two tool-GPIO outputs and reads the
# vacuum switch back on a tool-GPIO input, and the SDK wraps all three:
#
#     set_vacuum_gripper(on, wait=True, timeout=)   turn it on/off, and
#                                                   optionally block until
#                                                   the switch says sealed
#     get_vacuum_gripper()  ->  -1 off
#                                0 on, nothing held
#                                1 on, OBJECT HELD
#
# That last one is the whole verdict. It is a pressure measurement at the
# cup, not an inference from anything the camera said - the same
# principle as the two-finger gripper's grasp-status register, and for
# the same reason: the only trustworthy answer to "did I pick it up"
# comes from the hardware that is holding it.
#
# Note the asymmetry that matters for safety. A seal that fails is
# obvious and harmless: state 0, nothing picked, retry. A seal that is
# LOST mid-carry drops the cube wherever the arm happens to be. So the
# state is re-read after the lift, not only at the pick.

HELD, NOT_HELD, VAC_OFF, VAC_UNKNOWN = "held", "not held", "off", "unknown"

# WHICH RETURN CODES MEAN "THIS FAILED", AND WHICH DO NOT.
#
# This cost a whole debugging session, so it is worth stating exactly.
# The SDK's codes split cleanly by SIGN:
#
#   negative   the wrapper refused the call and NEVER SENT IT
#              -1 NOT_CONNECTED, -2 NOT_READY
#
#   0, 1, 2    the controller RECEIVED the command, acted on it, and
#              replied. 1 = HAS_ERROR and 2 = HAS_WARN do not describe
#              this command at all - they say a fault or a warning is
#              LATCHED on the controller, which stays true for every
#              call until it is cleared.
#
# So treating any non-zero code as failure throws away results that
# actually happened. The symptom was unmistakable once seen: the log was
# full of 'set_tgpio_digital -> code=1' while the cup was audibly pulling
# air. The same lesson was already learned one level down for the
# two-finger gripper, where code 2 on every modbus call was advisory too.
#
# The verdict on a pick therefore never comes from a return code. It
# comes from the vacuum switch.
ADVISORY_CODES = (0, 1, 2)


# The SDK's own timeout when set_vacuum_gripper(wait=True) gives up
# waiting for a seal (APIState.SUCTION_CUP_TOUT, xarm/x3/code.py:48).
#
# THIS IS NOT AN ERROR. It is the normal way a missed pick reports
# itself: the cup was commanded, the controller waited, no seal
# appeared. Treating it as a refused command - which is what "any code
# outside ADVISORY_CODES means the call failed" does - turns every
# ordinary miss into an alarming message about the controller. The bug
# was dormant only because the readback faulted before this code could
# ever be reached.
SUCTION_TIMEOUT_CODE = 41


def vacuum_state(arm):
    """HELD / NOT_HELD / VAC_UNKNOWN.

    DELIBERATELY NOT arm.get_vacuum_gripper(). The SDK's public wrapper
    hard-codes check_on=True (xarm_api.py:2083), and that path starts by
    calling get_tgpio_output_digital(), which reads tool register 0x0A18
    (x3/gpio.py:125). That register is unanswered on this cell's end
    module firmware 1.2.0: it faults the controller with error 28 in
    2-3ms, 100% of the time, measured 30/30 and 8/8 on separate runs.

    Strip check_on and the SDK's own answer is get_tgpio_digital(ionum=0)
    - the vacuum gripper's object-detection input. That reads perfectly
    here: 15/15 with the pump running, controller at error 0 throughout.
    So the seal verdict was never actually lost. It was hidden behind one
    unnecessary register read on the way to it.

    Note VAC_OFF is no longer returned. It came from the check_on branch
    - "the outputs are not set the way an ON cup would set them" - and
    nothing in this project ever tested for it (fixed_pick and
    fixed_calibrate both ask only '!= HELD'). With the cup off the input
    reads 0, which is NOT_HELD, and that is true: nothing is held."""
    pin = 0 if VACUUM_HW == 1 else 3
    try:
        code, value = arm.get_tgpio_digital(ionum=pin)
    except Exception:
        return VAC_UNKNOWN
    if code not in ADVISORY_CODES:
        return VAC_UNKNOWN
    return {1: HELD, 0: NOT_HELD}.get(value, VAC_UNKNOWN)


def vacuum_on(arm, wait=True):
    """Turn the cup on. With wait=True the SDK blocks until the switch
    reports a seal, so the return value already IS the pick verdict.

    Returns (ok, state). ok is False on a timeout, which is the normal
    way a missed pick reports itself - not an error to raise on."""
    try:
        code = arm.set_vacuum_gripper(True, wait=wait,
                                      timeout=VACUUM_TIMEOUT_S,
                                      hardware_version=VACUUM_HW)
    except Exception as e:
        print("   (vacuum ON refused: %s)" % e)
        return False, VAC_UNKNOWN
    if code == SUCTION_TIMEOUT_CODE:
        # Waited for a seal and did not get one. An ordinary miss, not a
        # fault - see SUCTION_TIMEOUT_CODE. Fall through and let the
        # switch have the final word, exactly as a successful call does.
        pass
    elif code not in ADVISORY_CODES:
        print("   (the controller would not accept the vacuum command: "
              "code %s)" % code)
        return False, VAC_UNKNOWN
    # The switch decides, not the return code - see ADVISORY_CODES.
    state = vacuum_state(arm)
    return state == HELD, state


def vacuum_off(arm, settle=True):
    """Release. Deliberately waits after cutting the vacuum: a cup does
    not let go the instant the valve closes, and moving away early drags
    the cube off the drop spot."""
    try:
        arm.set_vacuum_gripper(False, wait=False,
                               hardware_version=VACUUM_HW)
    except Exception as e:
        print("   (vacuum OFF refused: %s)" % e)
        return False
    if settle:
        time.sleep(VACUUM_RELEASE_S)
    return True


def setup_arm(arm, tool_io=True):
    """Bring the arm up for THIS project.

    Deliberately NOT vision_common.setup_arm: that one enables the
    two-finger modbus gripper and RAISES if the enable is refused, which
    is exactly what happens when the tool on the wrist is a vacuum cup.
    Everything else it does is reproduced here, plus the tool load for
    the vacuum tool and a guarantee that the cup starts OFF - an arm that
    boots holding something from a previous run is a surprise nobody
    needs.

    tool_io=False brings the arm up WITHOUT TOUCHING THE TOOL IO AT ALL -
    no cup command, no switch read, no energise check.

    That exists for a real, measured condition on this cell: when the
    wrist's end-module link is failing, ANY tool-IO transaction - even a
    single read of the vacuum switch - latches error 28 within
    milliseconds, and a latched error halts motion. The arm itself is
    fine and drives perfectly, as long as the tool is left alone. So a
    job that needs the arm to MOVE but never to GRIP can still run: the
    taped-plate calibration is exactly that job.

    Nothing that has to pick something up may use this. There is no cup
    here, and no way to know whether anything is held."""
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_tcp_load(TOOL_MASS_KG, TOOL_COG)
    arm.set_state(0)
    code, ew = arm.get_err_warn_code()
    if code != 0 or ew[0] != 0:
        raise RuntimeError("arm still in error %s - run "
                           "'0 - RESET EVERYTHING'" % ew)

    if not tool_io:
        print("   tool IO deliberately untouched: no cup command, no")
        print("   switch read, no energise check. The arm can move; it")
        print("   cannot grip, and cannot tell whether it is holding")
        print("   anything.")
        return

    # THE TOOL GATE. A pick that faults mid-carry freezes the arm over
    # the floor with a cube attached, so the tool is proved BEFORE any
    # motion is allowed - while the arm is still parked and the cost of
    # finding out is nothing.
    #
    # WHY IT IS DONE ONE TRANSACTION AT A TIME, AND WHAT IT ALREADY
    # CAUGHT. The first version ran the ordinary bring-up - read the
    # switch, release the cup, read again - and only then "checked" for
    # a fault. Three tool transactions had already happened by that
    # point, so when it found the error it blamed the last thing on its
    # mind and announced "energising the cup faults the controller".
    #
    # That was wrong, and measurably so: set_vacuum_gripper faulted 0
    # times in 30, while the readback faulted 30 times in 30. Energising
    # was never the problem. The readback was - specifically the
    # 0x0A18 register read buried inside the SDK's get_vacuum_gripper
    # wrapper, which vacuum_state() no longer uses.
    #
    # Issuing each transaction alone, and checking the error register
    # immediately after, is what turned "the wrist is broken" into "one
    # register is unanswered". Keeping it that way is cheap, and if this
    # ever trips again the name it prints will be worth having.
    def tripped():
        try:
            code, ew = arm.get_err_warn_code()
        except Exception:
            return 0
        return ew[0] if code in ADVISORY_CODES else 0

    if tripped():                       # not ours to attribute
        raise RuntimeError(
            "the controller is already in error before the tool was "
            "touched - run '0 - RESET EVERYTHING'")

    print("   tool check: one transaction at a time ...")

    # 1. A READ. The gentlest thing that can be asked of the wrist.
    state = vacuum_state(arm)
    fault, blame = tripped(), "READING the tool (the vacuum switch)"

    # 2. A WRITE, which also guarantees the cup starts OFF - an arm that
    #    boots still holding something from a previous run is a surprise
    #    nobody needs.
    if not fault:
        if state == HELD:
            print("   the cup is ON and HOLDING something - releasing it.")
        vacuum_off(arm)
        fault, blame = tripped(), "WRITING to the tool (switching the cup off)"

    # 3. ENERGISING, where the pump actually draws current.
    if not fault:
        try:
            arm.set_vacuum_gripper(True, wait=False,
                                   hardware_version=VACUUM_HW)
        except Exception:
            pass
        for _ in range(12):                        # ~0.6s at 20Hz
            time.sleep(0.05)
            fault = tripped()
            if fault:
                break
        blame = "ENERGISING the cup (the pump drawing current)"
        vacuum_off(arm, settle=False)

    if fault:
        for clean in (arm.clean_error, arm.clean_warn):
            try:
                clean()
            except Exception:
                pass
        try:
            arm.set_state(0)
        except Exception:
            pass
        raise RuntimeError(
            "%s faults the controller (error %s), so nothing moves. This "
            "is NOT the known 0x0A18 problem - vacuum_state() no longer "
            "reads that register, and every transaction this gate makes "
            "was measured clean on this cell. Something new is wrong. "
            "Run 'python fixed_vacuum.py --watch'." % (blame, fault))

    print("   tool check passed: read, write and energise all survived.")
    if state == VAC_UNKNOWN:
        print("   WARNING: the vacuum switch cannot be read. Every pick "
              "will report\n"
              "            'unknown' and be treated as a miss. Check the "
              "wiring with\n"
              "            'python fixed_vacuum.py'.")
    else:
        print("   vacuum gripper ready (tool IO, plug-in wiring)"
              if VACUUM_HW == 1 else
              "   vacuum gripper ready (tool IO, contact wiring)")


# ------------------------- reach -------------------------

def in_reach(x, y):
    """Cheap annulus test, used to reject candidates before spending an
    IK round-trip on them. The real test is always vc.ik on the pose."""
    r = float(np.hypot(x, y))
    return REACH_MIN_MM <= r <= REACH_MAX_MM


def out_of_reach_hint(x, y):
    r = float(np.hypot(x, y))
    if r > REACH_MAX_MM:
        return ("that spot is about %.0fcm outside the arm's reach"
                % ((r - REACH_MAX_MM) / 10.0 + 0.5))
    if r < REACH_MIN_MM:
        return "that spot is in over the arm's own base"
    return "the wrist cannot face that spot pointing straight down"

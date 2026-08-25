#!/usr/bin/env python3
"""Finding cubes with a camera that does not move.

Nothing here resembles vision\\detect_cube. That detector works in the
IMAGE: it finds a YOLO box, pulls a top-face contour out of the depth
inside it, and applies squareness and fill gates to that contour. Every
one of those gates assumes the camera is looking straight DOWN at the
cube, because only then is a cube's top face a square in the image.

From a camera bolted at 45 degrees off to the side, a cube's top face is
a foreshortened parallelogram, its side faces are visible, and the
"floor" is a different depth in every row. Re-tuning image-space gates
for that is a losing game.

So this does the opposite. It deprojects the whole depth frame, puts it
through the hand-eye transform into the ROBOT'S frame, and only then
looks for cubes - in a frame where the floor is flat, up is +Z, and a
cube's top face is a square again. The oblique mount stops being a
geometry problem and becomes only a SAMPLING problem: fewer points per
cube, spread anisotropically. Gates are set with that in mind.

WHAT REPLACED YOLO, AND WHY

Nothing detects "cube-ness" from appearance here at all. In base-frame
coordinates the question "is there an object on the table" is answered
exactly by "are there points more than POP_MM above the taught surface",
which needs no model, no training set and no colour assumption. YOLO was
only ever standing in for that measurement, and it needed retraining
every time the scene changed. cube_model.pt is not loaded by this
project. That also means a moved camera invalidates the calibration but
never the "training", because there is none.

THE FAILURE MODE THIS HAS THAT THE OLD ONE DID NOT

A fixed camera watches the ARM as well as the cubes. Slice a vertical
object - a finger, a wrist, a person's arm - at cube height and the
slice is cube-sized and cube-shaped. Two guards close that:

  1. anything with points ABOVE the cube band, in the same footprint, is
     part of something taller and is refused outright (tall_mask);
  2. the caller passes the TCP's current XY, and clusters near it are
     ignored - the arm always knows where its own hand is.
"""
import cv2
import numpy as np

import fixed_common as fx
from fixed_common import (MAX_CUBE_MM, MIN_CUBE_MM, POP_MM, TOP_BAND_MM,
                          WORK_X, WORK_Y)


# Cell size of the top-down occupancy raster used for clustering, in mm.
# It has to be coarse enough that neighbouring samples of one cube's face
# land in adjacent cells - at ~1m and 45 degrees the depth samples are
# roughly 3-4mm apart on a horizontal surface - and fine enough that two
# cubes 20mm apart still separate.
GRID_MM = 4.0

# Ignore anything higher than a cube can be. The band's ceiling is also
# what creates the sliced-arm problem, which tall_mask exists to answer.
BAND_TOP_MM = MAX_CUBE_MM + 25.0

# Shape gates on the top face, in the base frame where a cube top really
# is a square. Looser than vision\'s (1.30 / 0.70) on purpose: the point
# set is sparser and anisotropic here, so a genuine face measures a
# little raggeder. They are a backstop against merged pairs, not the
# primary detector - separated cubes are separated in the raster.
MAX_LEN_RATIO = 1.35
FILL_MIN = 0.65

# Seal gates, for the suction cup. RMS deviation of the top face from its
# own best-fit plane, and how far that plane is off level. Both are
# starting points sized against this camera's depth noise (~1-2mm at 1m)
# rather than measured, and both are printed by the viewer for every
# cube so they can be tightened once there are real numbers.
FLAT_MAX_MM = 4.0
TILT_MAX_DEG = 20.0

# A cluster must have at least this many points before it is worth
# measuring. At ~4mm sampling a 15mm cube's top face is only ~16 points.
MIN_POINTS = 25

# How close to the arm's own TCP a cluster may be before it is assumed to
# BE the arm. Generous: the gripper body is wide and its shadow moves.
TCP_IGNORE_MM = 130.0


# ------------------------- colour -------------------------

def classify_color(bgr_pts):
    """'red', 'blue' or 'unknown' from the colour of a cluster's points.

    Counts pixels in two hue bands rather than taking a median hue: red
    straddles the 0/180 wrap, and a median across the wrap lands on cyan.
    Same two-band reasoning as vision\\detect_cube.classify_color, applied
    to a point list instead of an image crop."""
    if bgr_pts is None or len(bgr_pts) == 0:
        return "unknown"
    px = np.asarray(bgr_pts, dtype=np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lit = (s > 80) & (v > 50)
    if lit.sum() < 5:
        return "unknown"
    h = h[lit]
    red = int(((h <= 12) | (h >= 168)).sum())
    blue = int(((h >= 95) & (h <= 135)).sum())
    if max(red, blue) < 0.25 * len(h):
        return "unknown"
    # Deterministic tie-break, deliberately: a coin-flip here would make
    # the colour-priority order jitter between frames.
    return "red" if red >= blue else "blue"


# ------------------------- the raster -------------------------

def _raster_index(xy):
    """Grid indices for base-frame XY points, and the raster's shape."""
    nx = int(np.ceil((WORK_X[1] - WORK_X[0]) / GRID_MM))
    ny = int(np.ceil((WORK_Y[1] - WORK_Y[0]) / GRID_MM))
    ix = np.clip(((xy[:, 0] - WORK_X[0]) / GRID_MM).astype(np.int32),
                 0, nx - 1)
    iy = np.clip(((xy[:, 1] - WORK_Y[0]) / GRID_MM).astype(np.int32),
                 0, ny - 1)
    return ix, iy, (ny, nx)


def _measure(pts, cols):
    """Turn one cluster's base-frame points into a cube description, or
    return (None, reason).

    The top face is taken as a thin band below the cluster's top. The
    'top' is the 95th percentile of Z rather than the maximum: one noisy
    point 10mm high would otherwise lift the band clear of the real face
    and leave it measuring nothing."""
    if len(pts) < MIN_POINTS:
        return None, "too few points"

    top_z = float(np.percentile(pts[:, 2], 95))
    band = pts[:, 2] >= top_z - TOP_BAND_MM
    if band.sum() < MIN_POINTS // 2:
        return None, "no flat top face"

    face3 = pts[band]
    face = face3[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(face)
    long_mm, short_mm = max(w, h), min(w, h)
    if short_mm < 1.0:
        return None, "degenerate top face"

    if not (MIN_CUBE_MM <= long_mm <= MAX_CUBE_MM):
        return None, "%.0fmm across - not a cube" % long_mm
    ratio = long_mm / short_mm
    if ratio > MAX_LEN_RATIO:
        return None, "top face is %.2f:1 - two cubes touching?" % ratio

    # FLATNESS AND TILT. These gates exist only because the tool is a
    # SUCTION CUP. Fingers grip the sides and do not care what the top
    # looks like; a cup has to seal against it, and it can only seal
    # against something flat and roughly level. A domed lid, a crumpled
    # bag or a cube resting on another cube at an angle all pass every
    # other gate here and would simply hiss.
    M = np.column_stack([face3[:, 0], face3[:, 1],
                         np.ones(len(face3))])
    sol, *_ = np.linalg.lstsq(M, face3[:, 2], rcond=None)
    flat_mm = float(np.sqrt(np.mean((face3[:, 2] - M @ sol) ** 2)))
    tilt_deg = float(np.degrees(np.arctan(np.hypot(sol[0], sol[1]))))
    if flat_mm > FLAT_MAX_MM:
        return None, "top is not flat (%.1fmm) - a cup cannot seal" % flat_mm
    if tilt_deg > TILT_MAX_DEG:
        return None, "top is tilted %.0f deg - a cup cannot seal" % tilt_deg

    # Fill: how much of the fitted rectangle the face actually occupies.
    # Measured on a raster rather than from a contour, because the point
    # set is sparse - a contour through sparse points hugs them and
    # always reads full. Closing first bridges the sampling gaps that are
    # an artefact of the viewing angle, not of the cube's shape.
    ix, iy, _ = _raster_index(face)
    sub = np.zeros((iy.max() - iy.min() + 1, ix.max() - ix.min() + 1),
                   np.uint8)
    sub[iy - iy.min(), ix - ix.min()] = 255
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    rect_cells = max(1.0, (long_mm * short_mm) / (GRID_MM * GRID_MM))
    fill = float((sub > 0).sum()) / rect_cells
    if fill < FILL_MIN:
        return None, "top face only %.0f%% filled" % (fill * 100)

    cube = {
        "center": [float(cx), float(cy)],
        "top_z": top_z,
        "width_mm": float(long_mm),
        "short_mm": float(short_mm),
        "yaw_rect_deg": float(angle),
        "flat_mm": round(flat_mm, 2),
        "tilt_deg": round(tilt_deg, 1),
        "fill": round(fill, 2),
        "ratio": round(float(ratio), 2),
        "n_points": int(len(pts)),
        "color": classify_color(cols[band] if cols is not None else None),
    }
    ok, why = fx.cup_fits(cube)
    if not ok:
        return None, why
    return cube, None


# ------------------------- the detector -------------------------

def detect_all(depth_mm, color_bgr, rays, T, floor, tcp_xy=None):
    """Every cube on the taught surface, in base-frame millimetres.

    Returns (cubes, rejects). Each cube gains "height_mm" - its height
    above the taught surface - which is exact even though the frame's Z
    origin is not the robot's (see fixed_common on the grasp frame).

    rejects is a list of (center_xy, reason) so the viewer can show WHY
    something visible was not offered, which is the difference between a
    tool you can debug and one you can only stare at."""
    cloud = fx.cloud_cam(depth_mm, rays)
    base = fx.apply_transform(T, cloud)
    valid = depth_mm > 0

    x, y, z = base[..., 0], base[..., 1], base[..., 2]
    surface = fx.plane_z(floor["coef"], x, y)
    height = z - surface

    inside = valid & fx.in_work_box(base)
    band = inside & (height > POP_MM) & (height <= BAND_TOP_MM)
    tall = inside & (height > BAND_TOP_MM)

    pts = base[band]
    cols = color_bgr[band] if color_bgr is not None else None
    if len(pts) < MIN_POINTS:
        return [], []

    ix, iy, shape = _raster_index(pts[:, :2])
    occ = np.zeros(shape, np.uint8)
    occ[iy, ix] = 255
    # Close before labelling so one cube's sampling gaps do not split it
    # into three clusters. The kernel is ~12mm across, well under the
    # smallest gap between two cubes the protocol allows.
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Footprint of everything TALLER than a cube, dilated. A cluster that
    # touches it is a slice of something tall - the arm, a hand, a box -
    # and is not a free-standing cube however cube-shaped the slice is.
    tall_occ = np.zeros(shape, np.uint8)
    if tall.any():
        tx, ty, _ = _raster_index(base[tall][:, :2])
        tall_occ[ty, tx] = 255
        tall_occ = cv2.dilate(tall_occ, np.ones((5, 5), np.uint8))

    n_labels, labels = cv2.connectedComponents(occ, connectivity=8)
    owner = labels[iy, ix]          # cluster id of every kept point

    cubes, rejects = [], []
    for cid in range(1, n_labels):
        member = owner == cid
        if member.sum() < MIN_POINTS:
            continue
        cpts = pts[member]
        ccols = cols[member] if cols is not None else None
        here = (float(cpts[:, 0].mean()), float(cpts[:, 1].mean()))

        if tall_occ[iy[member], ix[member]].any():
            rejects.append((here, "part of something taller than a cube"))
            continue
        if tcp_xy is not None and \
                np.hypot(here[0] - tcp_xy[0], here[1] - tcp_xy[1]) \
                < TCP_IGNORE_MM:
            rejects.append((here, "this is the gripper"))
            continue

        cube, why = _measure(cpts, ccols)
        if cube is None:
            rejects.append((here, why))
            continue
        cube["height_mm"] = float(
            cube["top_z"] - fx.plane_z(floor["coef"], *cube["center"]))
        if not (MIN_CUBE_MM <= cube["height_mm"] <= MAX_CUBE_MM):
            rejects.append((here, "%.0fmm tall - not a cube"
                            % cube["height_mm"]))
            continue
        cubes.append(cube)

    cubes.sort(key=lambda c: -c["n_points"])       # best-measured first
    return cubes, rejects


def pick_one(cubes, want_color=None, prefer_near=None):
    """Choose a cube to go for.

    want_color filters BEFORE anything else, so that a colour phase can
    say "there is no red here" decisively and move on - the same reason
    vision_pick3 filters candidates rather than finds-then-filters.
    prefer_near breaks ties by base-frame distance, used to keep working
    the same corner rather than crossing the cell every cycle."""
    pool = [c for c in cubes
            if want_color is None or c["color"] == want_color]
    if not pool:
        return None
    if prefer_near is not None:
        pool.sort(key=lambda c: np.hypot(c["center"][0] - prefer_near[0],
                                         c["center"][1] - prefer_near[1]))
        return pool[0]
    return pool[0]


# ------------------------- the calibration target -------------------------

def find_held_target(depth_mm, color_bgr, rays, want_color="red",
                     near_mm=(300.0, 2000.0), T=None):
    """Locate the flat TARGET PLATE stuck to the cup, for the hand-eye
    calibration.

    WHY A PLATE AND NOT A CUBE. With fingers, the calibration target was
    a cube held between them and the camera could see its top face. A cup
    grips the top face, so on a held cube that face is underneath the cup
    and the tool - invisible, and it is the one surface that matters,
    because it is the plane the cup seals against.

    A flat plate solves it: the cup takes the middle, and the top face
    stays visible as a broad ring all around it. That ring's outline
    gives the plate's centre, and the plate's top surface IS the cup's
    contact plane, so the transform fitted from it maps a camera point
    to "the flange that puts the cup ON that point" - exactly what a pick
    needs.

    THE PLATE MUST BE WIDE COMPARED WITH THE CUP'S STANDOFF. Seen from
    the side, the cup and its fitting cast a shadow across the plate
    roughly as long as they are tall. If the plate is small the shadow
    eats its far edge and the measured outline shrinks toward the camera
    - the very bias this is trying to avoid. A plate three times the
    cup's standoff keeps the outer edges clear, and the outline is set by
    the edges, not by the shadowed middle. 100-150mm square is ample.

    Two modes, and the calibration uses both in sequence:

      T is None  the first pass. Returns the CENTROID of the coloured
                 surface in CAMERA coordinates. It is biased - toward the
                 camera, and away from the cup's shadow - and the bias
                 changes as the plate moves across the view, which is
                 exactly the error a single pass cannot see.

      T given    the second pass. With even a rough transform the points
                 can be put in the base frame, and then the plate's top
                 face is well defined and its outline's centre is
                 viewpoint-independent AND shadow-independent, because a
                 minimum-area rectangle is fixed by the extremes and a
                 notch out of the middle does not move them.

    Returns (point_xyz_camera_frame, n_points) or (None, reason)."""
    mask_col = _color_mask(color_bgr, want_color)
    ok = mask_col & (depth_mm > near_mm[0]) & (depth_mm < near_mm[1])
    if ok.sum() < 60:
        return None, "no %s target in view (%d coloured pixels with depth)" \
            % (want_color, int(ok.sum()))

    # Keep only the largest coloured blob: a red sleeve or a stray cube
    # on the bench must not be averaged into the answer.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ok.astype(np.uint8), connectivity=8)
    if n_labels < 2:
        return None, "no coherent coloured blob"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = labels == biggest
    if sel.sum() < 60:
        return None, "the coloured blob is too small (%d px)" % int(sel.sum())

    cloud = fx.cloud_cam(depth_mm, rays)
    pts_cam = cloud[sel]
    # Drop depth outliers: the blob's edge pixels straddle the cube's
    # silhouette and read the background behind it.
    med = np.median(pts_cam[:, 2])
    keep = np.abs(pts_cam[:, 2] - med) < 60.0
    pts_cam = pts_cam[keep]
    if len(pts_cam) < 40:
        return None, "not enough clean depth on the target"

    if T is None:
        return np.asarray(pts_cam.mean(axis=0), dtype=float), len(pts_cam)

    pts_base = fx.apply_transform(T, pts_cam)
    top_z = float(np.percentile(pts_base[:, 2], 95))
    # A wider band than a cube top gets: the plate is broad, so at a
    # steep viewing angle its far side reads a little deeper than its
    # near side, and a 6mm band would keep only the near half and pull
    # the centre with it.
    band = pts_base[:, 2] >= top_z - 2.0 * TOP_BAND_MM
    if band.sum() < 15:
        return None, "the target's top face is not resolved"
    face = pts_base[band][:, :2].astype(np.float32)
    (cx, cy), _, _ = cv2.minAreaRect(face)
    # Answer in CAMERA coordinates, because that is what the fit consumes:
    # invert the rough transform on the refined base-frame point.
    R = np.asarray(T["R"], dtype=float)
    t = np.asarray(T["t"], dtype=float)
    refined_base = np.array([cx, cy, top_z], dtype=float)
    return R.T @ (refined_base - t), int(band.sum())


# NOTE: there is deliberately no yaw measurement here any more. The
# finger version needed one - it had to know the constant between the
# base-frame top-face angle and the wrist's zero, so the fingers could be
# lined up with the cube's faces. A round cup on a flat face is
# rotationally symmetric, so there is nothing to line up: the wrist stays
# at one yaw for every pick, which also means one less joint to swing and
# one less way for a move to be refused.


# Hue bands, in OpenCV's 0-179 scale. Red straddles the wrap, so it is
# handled separately below.
#
# The cubes are only ever red or blue - see classify_color, which is a
# different job and stays as it is. These extra bands exist for the
# CALIBRATION PLATE, which just has to be findable, and which the
# operator has to make out of whatever is to hand. Insisting it be red
# when a green folder is on the desk is a pointless obstacle.
# Purple starts where blue stops, with no gap. It had one - blue ended at
# 135 and purple began at 140 - and a violet plate landing at 138 matched
# NOTHING, which reads as "the camera cannot see your plate" rather than
# as a bug in a lookup table.
HUE_BANDS = {
    "blue":   (95, 135),
    "purple": (136, 165),
    "green":  (40, 85),
    "yellow": (20, 35),
}


def _plane_through(pts, iters=3, trim_mm=15.0):
    """Robust plane through (N,3) points. Returns (point, unit normal).

    NOT fixed_common.fit_plane, which fits z = ax+by+c and so assumes z
    is "up" - true in the robot's frame, false in the camera's, where z
    is distance from the lens. This one is orientation-free: the normal
    is the smallest singular vector of the centred points, which is the
    same answer whatever direction the surface happens to face."""
    P = np.asarray(pts, dtype=float)
    if len(P) < 50:
        return None
    keep = np.ones(len(P), dtype=bool)
    centre, normal = None, None
    for _ in range(iters):
        Q = P[keep]
        if len(Q) < 50:
            break
        centre = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - centre, full_matrices=False)
        normal = vt[-1]
        nxt = np.abs((P - centre).dot(normal)) < trim_mm
        if nxt.sum() < 50:
            break
        keep = nxt
    if centre is None or normal is None:
        return None
    # Point the normal at the camera, which sits at the origin, so
    # "height above the surface" comes out positive for things standing
    # on it rather than sometimes negative depending on the fit.
    if float(centre.dot(normal)) > 0:
        normal = -normal
    return centre, normal


# A first, generous slice off the top of the cube - wide enough that
# depth noise cannot empty it, and not required to be clean.
CUBE_TOP_BAND_MM = 8.0

# How close to the FITTED top plane a point must then lie to count as
# the top face.
#
# The two stages exist because no single band works. Any band, however
# thin, also catches the top rim of the SIDE face - those points sit at
# the same height as the top face's edge, and being on a vertical
# surface they all lie at the silhouette nearest the camera, so they
# drag the centroid that way in proportion to the band's thickness.
# Measured: 10mm band gave 5.1mm error, 3mm gave 1.9mm, straight-line
# down. But shrinking the band is not the answer either, because depth
# noise then decides which points survive.
#
# Fitting a plane to the rough slice and keeping only what lies ON it
# separates them properly: the top face is flat and stays, while the
# side rim falls away from that plane immediately.
CUBE_FACE_SLAB_MM = 2.5


def find_cube_top(depth_mm, color_bgr, rays, want_color="red"):
    """The top face centre of a coloured cube standing on a surface, in
    CAMERA coordinates. For the touch calibration.

    WHY NOT find_held_target. That one is built for a flat PLATE, where
    the whole coloured blob IS the top face and its centroid is the
    answer. A cube is not flat: from 45 degrees the camera sees the top
    face and a side face, both the same colour and much the same area,
    so the blob's centroid sits about 8mm below the top and 7mm toward
    the camera - and the offset CHANGES with the cube's position in the
    view, because which sides are visible changes.

    Measured, not guessed: the first touch calibration fitted 9.2mm RMS
    with a scale of 0.973, and both are what that varying bias does to a
    rigid fit.

    So the top face is separated first, using the surface the cube is
    standing on. Fitting the dominant plane of the whole scene gives
    "up" in camera coordinates directly - no hand-eye transform needed,
    which is the point, since this runs before one exists."""
    mask_col = _color_mask(color_bgr, want_color)
    ok = mask_col & (depth_mm > 300.0) & (depth_mm < 2000.0)
    if ok.sum() < 60:
        return None, ("no %s cube in view (%d coloured pixels with depth)"
                      % (want_color, int(ok.sum())))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ok.astype(np.uint8), connectivity=8)
    if n_labels < 2:
        return None, "no coherent coloured blob"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = labels == biggest
    if sel.sum() < 60:
        return None, "the coloured blob is too small (%d px)" % int(sel.sum())

    cloud = fx.cloud_cam(depth_mm, rays)

    # The surface the cube stands on: everything with depth that is NOT
    # the cube. The cube is small next to the table, and the fit trims
    # outliers anyway, but excluding it costs nothing and stops a big
    # near cube from tilting the plane it is supposed to be measured
    # against.
    floor_sel = (depth_mm > 300.0) & (depth_mm < 2000.0) & (~sel)
    plane = _plane_through(cloud[floor_sel])
    if plane is None:
        return None, "the surface under the cube could not be measured"
    centre, normal = plane

    pts = cloud[sel]
    med = np.median(pts[:, 2])
    pts = pts[np.abs(pts[:, 2] - med) < 60.0]      # silhouette outliers
    if len(pts) < 40:
        return None, "not enough clean depth on the cube"

    height = (pts - centre).dot(normal)
    top = float(np.percentile(height, 95))
    rough = pts[height >= top - CUBE_TOP_BAND_MM]
    if len(rough) < 12:
        return None, "the cube's top face is not resolved"

    # Stage two: keep only what actually lies on the top face's own
    # plane - see CUBE_FACE_SLAB_MM. The plane is fitted to the rough
    # slice rather than assumed parallel to the floor, so a cube resting
    # on something slightly tilted is still measured correctly.
    if len(rough) >= 50:
        top_plane = _plane_through(rough, iters=2, trim_mm=4.0)
        if top_plane is not None:
            tc, tn = top_plane
            face = rough[np.abs((rough - tc).dot(tn)) <= CUBE_FACE_SLAB_MM]
            if len(face) >= 12:
                return _face_centre(face, tc, tn), len(face)
    return np.asarray(rough.mean(axis=0), dtype=float), len(rough)


def _face_centre(face, tc, tn):
    """The centre of a flat face - from its OUTLINE, not its points.

    Averaging the points is biased, and by a surprising amount: a
    surface seen at a slant is sampled unevenly, because the near half
    covers more pixels per millimetre than the far half. The mean is
    pulled toward the camera by a couple of millimetres, and the pull
    changes with where the cube sits in the view, so it does not cancel
    in the fit - it distorts it.

    A minimum-area rectangle is fixed by the extreme points, so it does
    not care how densely the middle was sampled. Measured on the
    synthetic bench: 2.5mm error from the mean, under 1mm from this."""
    u = np.array([1.0, 0.0, 0.0])
    if abs(float(tn.dot(u))) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - tn * float(u.dot(tn))
    u /= np.linalg.norm(u)
    v = np.cross(tn, u)

    rel = face - tc
    flat = np.column_stack([rel.dot(u), rel.dot(v)]).astype(np.float32)
    try:
        (ca, cb), _, _ = cv2.minAreaRect(flat)
    except Exception:
        return np.asarray(face.mean(axis=0), dtype=float)
    return np.asarray(tc + u * float(ca) + v * float(cb), dtype=float)


def _color_mask(color_bgr, want):
    """Pixels of a given colour, strongly enough coloured to be sure.

    The saturation floor is what actually matters, and it is why the
    plate cannot be white, grey or beige: the cell's floor is white and
    the arm holding the plate is white too, so an unsaturated target
    cannot be told from either. Black fails for a different reason - it
    passes this mask happily but returns no DEPTH, because matt black
    absorbs the projector's infrared."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 60)
    band = HUE_BANDS.get(want)
    if band is not None:
        return lit & (h >= band[0]) & (h <= band[1])
    return lit & ((h <= 12) | (h >= 168))          # red wraps around 0

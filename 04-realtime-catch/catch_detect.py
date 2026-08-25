#!/usr/bin/env python3
"""Fast, floor-free cube detection for a cube that is MOVING.

Built on the same two ideas the handoff detector uses, for the same
reason - the v3 detector is floor-gated, and a cube in motion may be
sliding on the floor, skidding across a bench or being carried through
the air, so there is no single surface to subtract:

    top  = a low percentile of the valid depths inside the YOLO box
    keep everything in a thin band below that
    minAreaRect on what is left -> centre, grasp angle, true size in mm

The SAME cube_model.pt is used, borrowed from detect_cube rather than
re-loaded, so only one copy sits in VRAM and this detector always looks
through exactly the net the floor picker was trained with. A moving cube
still looks like a cube; nothing here needs retraining.

WHAT IS DIFFERENT FROM THE HANDOFF DETECTOR, AND WHY

  * ONE FRAME, NEVER A MEDIAN. Every sibling in this cell stabilises a
    reading by taking several frames and taking the median. That is the
    right trade when the cube is not going anywhere and completely wrong
    here: six frames at 30fps is 200ms, and 200ms of a cube's motion is
    exactly the thing being measured. Averaging it away would smear the
    cube across its own path and report a position it was at some
    unknowable time in the middle. Stability is recovered downstream, in
    catch_track, by fitting a LINE through single-frame samples - which
    filters noise without pretending the cube stood still.

  * EVERY READING IS TIMESTAMPED, from the depth frame itself rather
    than from the clock when Python got round to looking at it. YOLO
    inference and this refinement together take tens of milliseconds and
    they vary; charging that jitter to the cube would put a wobble in
    the velocity fit that no amount of smoothing removes. The frame
    timestamp is when the photons arrived, which is the only time that
    means anything for velocity.

  * A LOWER STANDS-CLEAR STEP. See STEP_MM.

Two things it cannot do, both inherited and both accepted:

  * HEIGHT. With no floor there is nothing to subtract a top from, so it
    falls back to the same cube heuristic the v3 detector uses when its
    own floor estimate looks wrong - as tall as it is wide - and marks
    the reading height_est. The grab height only uses a fraction of it.
  * It cannot tell a cube from a cube-shaped thing. The caller does
    that, by refusing anything outside the catch band and anything whose
    motion does not fit a straight line.
"""
import time

import cv2
import numpy as np
import pyrealsense2 as rs

import catch_common as cc       # noqa: F401 - imported for its side
#                                 effect: it puts vision\ on sys.path,
#                                 which is what makes detect_cube below
#                                 resolvable from this folder
import detect_cube as dc

# A cube in motion is motion-blurred, often tilted, and photographed
# against whatever it happens to be crossing, so it scores lower than one
# lying still on a clean floor. Low on purpose: every box still has to
# survive the depth refinement below, which is the real filter, and a box
# YOLO never emits is a cube the arm can never catch.
CONF_MIN = 0.25

ROI_PAD = 10
MIN_DEPTH_MM = 150.0     # below the camera's range - not a measurement

# Depth band kept under the top face. Wider than the handoff detector's
# 10mm: there is no hand a few millimetres below the cube to be swallowed
# here, and a moving cube's depth reading is noisier, so a tight band
# starts dropping half the top face on the frames where it matters.
TOP_BAND_MM = 14.0
ABOVE_TOP_MM = 8.0
MIN_AREA_PX = 200

# A cube STANDS OUT: something in its neighbourhood must be measurably
# farther away than its top face. Without it, any flat surface filling a
# YOLO box - a bench, a wall, the side of a big box - reads as a perfect
# square top face at a plausible depth.
#
# 12mm, not the handoff project's 20mm. This project must accept a cube
# SLIDING ON THE FLOOR, and such a cube's only evidence of standing clear
# is its own height. At the gripper's smallest cube (15mm) a 20mm step
# demands more clearance than the cube physically has, and every
# floor-borne cube is refused. 12mm sits below the smallest real cube and
# still well above the depth noise at this range (a few mm), which is the
# only thing it has to beat.
CONTEXT_PAD = 30
STEP_MM = 12.0
STEP_FRAC = 0.02

# Percentile ladder for "the nearest surface". 10 is the cube top in the
# ordinary case; 25 and 40 step back to the cube when something nearer
# (a finger, a motion-blur artefact) captures the first band.
NEAR_PCTL = (10, 25, 40)

# Same shape gates as both siblings, for the same reason: a real cube top
# is near-square and fills its bounding rect, while two merged objects
# are neither, and the centre of that merge is the gap between them.
SQUARENESS = 0.62
FILL_MIN = 0.70
MAX_LEN_RATIO = 1.30

MIN_CUBE_MM, MAX_CUBE_MM = 12.0, 120.0   # sanity only; the picker is
                                         # stricter (cc.MIN/MAX_CUBE_MM)


def frame_time_s(depth_frame):
    """When the frame was CAPTURED, in seconds, monotonic-ish.

    RealSense reports milliseconds on its own clock. Falling back to the
    wall clock is fine - it only ever costs a constant offset, and every
    consumer of this differences two timestamps rather than reading one
    absolutely."""
    try:
        ts = float(depth_frame.get_timestamp())
        if ts > 0.0:
            return ts / 1000.0
    except Exception:
        pass
    return time.time()


def _stands_clear(dmm, box_xyxy, top):
    """Is there anything measurably BEHIND this surface?

    Looked for on a ring wider than the cube, because whatever is
    immediately beside a moving cube (a hand that just let go, the lip of
    a chute) sits at nearly the cube's own depth. The honest background
    is farther out. Unknown counts as clear: with no evidence either way
    the catch-band check downstream is what keeps the arm off the floor,
    and refusing on no evidence just makes the detector sulk."""
    x1 = max(0, int(box_xyxy[0]) - CONTEXT_PAD)
    y1 = max(0, int(box_xyxy[1]) - CONTEXT_PAD)
    x2 = min(dmm.shape[1], int(box_xyxy[2]) + CONTEXT_PAD)
    y2 = min(dmm.shape[0], int(box_xyxy[3]) + CONTEXT_PAD)
    ctx = dmm[y1:y2, x1:x2]
    valid = ctx[ctx > MIN_DEPTH_MM]
    if valid.size < 100:
        return True
    behind = valid[valid > top + STEP_MM]
    return behind.size >= STEP_FRAC * valid.size


def _best_contour(mask):
    """The largest blob in the mask that is actually cube-SHAPED.

    Every blob is tested and the largest one that PASSES wins, rather
    than testing only the biggest - the biggest is regularly something
    else in the frame, and then the only thing left to report is that
    other thing's outline. Returns (rect, reason_if_none)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_area, reason = None, 0.0, "nothing cube-shaped in the depth"
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        rect = cv2.minAreaRect(c)
        (_cx, _cy), (w, h), _ang = rect
        short, long_ = min(w, h), max(w, h)
        if short <= 0.0:
            continue
        if short / long_ < SQUARENESS or long_ > MAX_LEN_RATIO * short:
            reason = "the top face is not square - something is touching it"
            continue
        if area < FILL_MIN * w * h:
            reason = "the top face is L-shaped or partly missing"
            continue
        if area > best_area:
            best, best_area = rect, area
    return best, (None if best is not None else reason)


def _rect_local(dmm, box_xyxy):
    """Rotated rect of the cube top inside a YOLO box, with no floor.

    Returns (rect_in_full_image_coords, top_depth_mm, reason). On success
    reason is None; on failure it names the gate that refused."""
    x1 = max(0, int(box_xyxy[0]) - ROI_PAD)
    y1 = max(0, int(box_xyxy[1]) - ROI_PAD)
    x2 = min(dmm.shape[1], int(box_xyxy[2]) + ROI_PAD)
    y2 = min(dmm.shape[0], int(box_xyxy[3]) + ROI_PAD)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None, "too small to measure"
    roi = dmm[y1:y2, x1:x2]
    valid = roi[roi > MIN_DEPTH_MM]
    if valid.size < 60:
        return None, None, "no depth there - more light, or slow it down"

    reason = "nothing cube-shaped in the depth"
    for pctl in NEAR_PCTL:
        top = float(np.percentile(valid, pctl))
        if not _stands_clear(dmm, box_xyxy, top):
            reason = "that is flat against a surface, not a separate cube"
            continue
        m = ((roi > top - ABOVE_TOP_MM)
             & (roi < top + TOP_BAND_MM)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        rect, why = _best_contour(m)
        if rect is None:
            reason = why
            continue
        (cx, cy), (w, h), ang = rect
        return ((cx + x1, cy + y1), (w, h), ang), top, None
    return None, None, reason


def moving_candidates(color_img, depth_frame, intrinsics):
    """Every cube-shaped top face in view, floor-free, from ONE frame.

    Returns (candidates, n_yolo_boxes, reason). The box count and the
    reason exist so the caller can tell "no cube in view" from "I can see
    a cube but here is what is wrong with it" - only the second has a fix
    the operator can act on."""
    model = dc._get_model()         # deliberately the SAME loaded model
    res = model.predict(color_img, conf=CONF_MIN, verbose=False)[0]
    n_boxes = len(res.boxes)
    if n_boxes == 0:
        return [], 0, None

    dmm = dc.depth_mm_image(depth_frame)
    out, reasons = [], []
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        rect, top, why = _rect_local(dmm, xyxy)
        if rect is None:
            reasons.append(why)
            continue
        (cx, cy), (w_px, h_px), angle = rect
        # the refined rect has to belong to the box that produced it
        if not (xyxy[0] <= cx <= xyxy[2] and xyxy[1] <= cy <= xyxy[3]):
            reasons.append("the measured face is not on the cube")
            continue

        box = cv2.boxPoints(rect)
        depth_m = top / 1000.0
        pts3d = [rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(px), float(py)], depth_m) for px, py in box]
        side1 = np.linalg.norm(np.subtract(pts3d[0], pts3d[1])) * 1000.0
        side2 = np.linalg.norm(np.subtract(pts3d[1], pts3d[2])) * 1000.0
        width_mm = float(min(side1, side2))
        length_mm = float(max(side1, side2))
        if not (MIN_CUBE_MM <= width_mm <= MAX_CUBE_MM):
            reasons.append("that measures %.0fmm across" % width_mm)
            continue
        if length_mm > MAX_LEN_RATIO * width_mm:
            reasons.append("that is not square in millimetres")
            continue

        out.append({
            "pixel": [round(cx, 1), round(cy, 1)],
            "depth_mm": round(top, 1),
            "width_mm": round(width_mm, 1),
            "length_mm": round(length_mm, 1),
            "width_px": round(min(w_px, h_px), 1),
            "angle_deg": round(dc._fold_angle(w_px, h_px, angle), 1),
            "box": box.tolist(),
            "conf": round(float(b.conf[0]), 2),
            # no floor to measure against: a cube is as tall as it is
            # wide, the same fallback the v3 detector uses when its own
            # floor estimate looks wrong
            "height_mm": round(width_mm, 1),
            "height_est": True,
            "floor_mm": None,
            "color": dc._color_of_box(color_img, box),
        })
    reason = None
    if not out and reasons:
        # the commonest complaint, not a list - the operator can only act
        # on one thing at a time
        reason = max(set(reasons), key=reasons.count)
    return out, n_boxes, reason


def nearest_to(cands, pixel, radius=None):
    """The candidate nearest a pixel, or None.

    radius rejects everything beyond that many pixels and returns None -
    "the cube I was tracking is not here any more" is a fact the caller
    must be told, not papered over with a different cube. Silently
    swapping targets mid-track is how a velocity fit ends up describing
    a line between two different cubes."""
    if not cands:
        return None
    pool = cands
    if radius is not None:
        pool = [c for c in pool
                if np.hypot(c["pixel"][0] - pixel[0],
                            c["pixel"][1] - pixel[1]) <= radius]
        if not pool:
            return None
    return min(pool, key=lambda c: np.hypot(c["pixel"][0] - pixel[0],
                                            c["pixel"][1] - pixel[1]))


def detect_moving(color_img, depth_frame, intrinsics, prefer_near=None,
                  near_radius=None):
    """(best cube or None, all candidates, n_yolo_boxes, reason, t_frame).

    With no prefer_near the NEAREST cube to the camera wins, which
    looking straight down means the highest one - the same rule the
    handoff project uses, and the one that picks a cube being carried
    over a floor covered in others."""
    cands, n_boxes, reason = moving_candidates(color_img, depth_frame,
                                               intrinsics)
    t = frame_time_s(depth_frame)
    if prefer_near is not None:
        best = nearest_to(cands, prefer_near, near_radius)
    else:
        best = min(cands, key=lambda c: c["depth_mm"]) if cands else None
    return best, cands, n_boxes, reason, t

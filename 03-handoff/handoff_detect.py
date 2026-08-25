#!/usr/bin/env python3
"""Cube detection with NO floor - for a cube held in a hand.

The v3 detector finds a cube by subtracting the floor: it keeps depth
pixels that stand proud of a floor plane, and the picker then insists
the cube top sits 8..98mm above it. That is the right design for cubes
lying on a surface and completely wrong for one held in mid-air - the
plane is hundreds of millimetres away, the hand is in the same blob, and
the pop-out test rejects everything.

The fix is to stop referencing a global surface at all. YOLO already
puts a box around the cube (the SAME cube_model.pt - it was trained with
heavy colour/rotation augmentation on depth-derived labels, and a cube
in a hand still looks like a cube). Inside that box the cube's top face
is simply the NEAREST surface to the camera, whatever is behind it. So:

    top = a low percentile of the valid depths in the box
    keep everything within a thin band below that
    minAreaRect on what is left -> centre, grasp angle, true size in mm

That is floor-free, works over a bench, a hand, a glass panel or thin
air, and returns the same dict the rest of the pipeline already speaks.

Two things it cannot do that the floor version can:

  * HEIGHT. Height was floor-minus-top; with no floor there is nothing
    to subtract from. It falls back to the cube heuristic (as tall as it
    is wide) that the v3 detector already uses whenever its own floor
    estimate looks wrong, and marks the reading height_est. A cube is a
    cube, so the error here is small - and the grab height only uses
    half of it.
  * It cannot tell a cube from a cube-shaped thing at that distance.
    The caller does that, by refusing anything outside the hold band.

Known residue, accepted: two cubes touching DIAGONALLY, offset about
half a cube in both axes, merge into a square blob that fills ~0.78 of
its rect - square enough and full enough to pass both shape gates, so
the measured centre lands on the seam. No fill threshold separates that
case from a real top face on glass (real ones measure 0.74-0.79 there).
The consequence is bounded: the fingers close on the gap, the gripper's
grasp sensor reports nothing caught, and the attempt is retried. Do not
hand over two cubes stuck together.

Nothing in vision/ is modified. The model is borrowed rather than
re-loaded so only one copy sits in VRAM, and so this detector is always
looking through exactly the net the floor picker was trained with.
"""
import cv2
import numpy as np
import pyrealsense2 as rs

import handoff_common as hc     # noqa: F401 - imported for its side
#                                 effect: it puts vision\ on sys.path,
#                                 which is what makes detect_cube below
#                                 resolvable from this folder
import detect_cube as dc

# A cube gripped in a hand is partly occluded, tilted, and against a
# background the model never saw in training, so it scores far lower
# than one lying flat on a clean floor. Low on purpose: every box still
# has to survive the depth refinement below, which is the real filter,
# and a box YOLO never emits is a cube the arm can never see.
CONF_MIN = 0.25

# Enough padding that the region of interest always contains some of
# whatever is BEHIND the cube - that background is what proves the top
# face is a separate object and not part of a larger surface (see
# STEP_MM). The floor detector pads more, to catch floor context.
ROI_PAD = 10

MIN_DEPTH_MM = 150.0     # below the camera's range - not a measurement
# Depth band under the top face kept in the mask. TIGHT on purpose. The
# floor detector can afford 15mm because the only thing near a cube's
# top face is air; here the fingers HOLDING it are a few millimetres
# below, and every millimetre of band is another chance to swallow one
# and turn a square top face into an L.
TOP_BAND_MM = 10.0
ABOVE_TOP_MM = 8.0       # noise allowance above it
MIN_AREA_PX = 200

# A cube STANDS OUT: something in its neighbourhood must be measurably
# farther away than its top face. Without this, any flat surface filling
# a YOLO box - a table, a wall, the side of a big box - reads as a
# perfect square top face at a plausible depth, and the arm would dive at
# it. The floor detector gets this check for free (it only ever looks at
# pixels standing proud of a floor); with no floor it has to be explicit.
#
# Measured on a ring WIDER than the cube's own box. Measuring it inside
# the box asked the wrong question: a hand fills the few pixels around
# a held cube at nearly the cube's own depth, so a genuinely airborne
# cube kept being reported "flat against a surface". Widen the window
# and the real background - the floor, a metre below - is always there.
CONTEXT_PAD = 30         # how far outside the box to look
STEP_MM = 20.0           # how much farther "behind" has to be
STEP_FRAC = 0.02         # and how much of that window has to be behind

# Percentile ladder for "the nearest surface". 10 is the cube top in the
# ordinary case. When a finger curls over the top face it is nearer than
# the cube, so the first band locks onto skin and fails the shape gates;
# 25 and 40 then step back to the cube itself. Trying the ladder costs a
# few milliseconds and turns "hand in the way" from a miss into a hit.
NEAR_PCTL = (10, 25, 40)

# Same shape gates as the floor detector, and for the same reasons: a
# real cube top is near-square and fills its bounding rect, while two
# merged objects (or a cube plus the fingers holding it) are neither.
SQUARENESS = 0.62
FILL_MIN = 0.70
MAX_LEN_RATIO = 1.30

MIN_CUBE_MM, MAX_CUBE_MM = 12.0, 120.0   # sanity only; the picker is
                                         # stricter (hc.MIN/MAX_CUBE_MM)


def _stands_clear(dmm, box_xyxy, top):
    """Is there anything measurably BEHIND this surface?

    Looked for on a ring wider than the cube, because right next to a
    held cube is the hand holding it - at almost the cube's own depth.
    The honest background is farther out. Unknown counts as clear: with
    no evidence either way the hold-band check downstream is what keeps
    the arm off the floor, and refusing on no evidence just makes the
    detector sulk."""
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

    Testing only the biggest blob - what this did at first - throws the
    cube away whenever the hand holding it makes a bigger one, which is
    most of the time. Then the only thing left to report is the hand's
    own outline: "the top face is not square", "that measures 140mm
    across". Every blob gets tested, and the largest one that passes
    wins. Returns (rect, reason_if_none)."""
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
        # A real top face is near-square and fills its rect. Two touching
        # cubes - or a cube merged with the fingers holding it - are
        # neither, and the CENTRE of that merge is the gap between them,
        # exactly where the fingers would close on air.
        if short / long_ < SQUARENESS or long_ > MAX_LEN_RATIO * short:
            reason = "the top face is not square - something is touching it"
            continue
        if area < FILL_MIN * w * h:
            reason = "the top face is L-shaped - hold it by the bottom edges"
            continue
        if area > best_area:
            best, best_area = rect, area
    return best, (None if best is not None else reason)


def _rect_local(dmm, box_xyxy):
    """Rotated rect of the cube top inside a YOLO box, with no floor.

    Returns (rect_in_full_image_coords, top_depth_mm, reason). On success
    reason is None; on failure it names the gate that refused, in words
    the operator can act on - "I can see a cube but cannot measure it" is
    a useless thing to be told while holding one.
    """
    x1 = max(0, int(box_xyxy[0]) - ROI_PAD)
    y1 = max(0, int(box_xyxy[1]) - ROI_PAD)
    x2 = min(dmm.shape[1], int(box_xyxy[2]) + ROI_PAD)
    y2 = min(dmm.shape[0], int(box_xyxy[3]) + ROI_PAD)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None, "too small to measure"
    roi = dmm[y1:y2, x1:x2]
    valid = roi[roi > MIN_DEPTH_MM]
    if valid.size < 60:
        return None, None, "no depth there - move it into better light "\
                           "or away from glass"

    reason = "nothing cube-shaped in the depth"
    for pctl in NEAR_PCTL:
        top = float(np.percentile(valid, pctl))
        if not _stands_clear(dmm, box_xyxy, top):
            reason = "that is flat against a surface, not held clear of it"
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


def held_candidates(color_img, depth_frame, intrinsics):
    """Every cube-shaped top face in view, floor-free.

    Returns (candidates, n_yolo_boxes, reason). The box count and the
    reason exist so the caller can tell "no cube in view" from "I can see
    a cube but here is what is wrong with it" - only the second has a fix
    the person holding it can act on."""
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
            # wide, which is the same fallback the v3 detector uses when
            # its floor estimate looks wrong
            "height_mm": round(width_mm, 1),
            "height_est": True,
            "floor_mm": None,
            "color": dc._color_of_box(color_img, box),
        })
    # report the commonest complaint, not a list - the operator can only
    # act on one thing at a time
    reason = None
    if not out and reasons:
        reason = max(set(reasons), key=reasons.count)
    return out, n_boxes, reason


def pick_held(cands, prefer_near=None, near_radius=None):
    """Choose the cube being OFFERED, out of everything in view.

    Nearest to the camera wins. That single rule is what separates the
    cube in your hand from the ones on the floor below it: looking
    straight down, held higher means measured closer, every time. It
    needs no floor, no colour and no size assumption.

    prefer_near=(px, py) switches to "the one nearest this pixel", used
    once the loop has locked onto a cube so a second cube entering the
    view cannot steal the lock. near_radius rejects everything beyond
    that many pixels and returns None - "the cube I was following is not
    here any more" is a fact the caller must be told, not papered over
    with a different cube (the failure that caused the v3 retreat loop).
    """
    if not cands:
        return None
    if prefer_near is not None:
        pool = cands
        if near_radius is not None:
            pool = [c for c in pool
                    if np.hypot(c["pixel"][0] - prefer_near[0],
                                c["pixel"][1] - prefer_near[1]) <= near_radius]
            if not pool:
                return None
        return min(pool, key=lambda c: np.hypot(
            c["pixel"][0] - prefer_near[0], c["pixel"][1] - prefer_near[1]))
    return min(cands, key=lambda c: c["depth_mm"])


def detect_held(color_img, depth_frame, intrinsics, prefer_near=None,
                near_radius=None):
    """(best held cube or None, all candidates, n_yolo_boxes, reason)."""
    cands, n_boxes, reason = held_candidates(color_img, depth_frame,
                                             intrinsics)
    return (pick_held(cands, prefer_near, near_radius), cands, n_boxes,
            reason)

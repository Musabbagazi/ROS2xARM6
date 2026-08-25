#!/usr/bin/env python3
"""Cube detector v3: YOLO (any color) + depth refinement.

YOLO finds cube candidates in the color image (a model trained on YOUR
cubes by train_cubes.py - COCO has no cube class). A plain YOLO box is
axis-aligned and carries no rotation, so each box is then refined
against the depth image: the cube's top face pops out of the floor
plane, and a minAreaRect on that depth mask recovers the precise
center, grasp angle and true size in mm.

detect_cube(color_img, depth_frame, intrinsics) keeps the SAME return
contract as the old detect_red_cube() so all the v2 math carries over,
plus extra keys: "conf", "floor_mm", "height_mm".
"""
import os

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "cube_model.pt")

CONF_MIN = 0.40         # YOLO confidence gate
MIN_AREA_PX = 250       # ignore specks smaller than this (top-face mask)
# w/h of the refined rect must be at least this. Two identical touching
# cubes merge into a 2:1 blob (aspect 0.50) whose center is the SEAM
# between them - grasping there clamps both cubes. 0.62 rejects every
# same-height touching pair while keeping real (near-square) cube tops.
SQUARENESS = 0.62
POP_MM = 10.0           # an object must stick out of the floor this much
TOP_BAND_MM = 15.0      # depth band below the top face kept in the mask
ROI_PAD = 14            # px padding around the YOLO box before refining
MIN_CUBE_MM, MAX_CUBE_MM_DET = 8.0, 120.0
GRIPPABLE_MM = 78.0     # prefer cubes the gripper can actually take, so
                        # one oversized object can't shadow pickable ones.
                        # SINGLE SOURCE OF TRUTH: vision_pick3's width
                        # gate imports this - if the detector called
                        # anything wider "grippable", the caller would
                        # skip it every cycle and pickable cubes behind
                        # it would never be reached.
MERGED_BLOB_RATIO = 1.5  # refined rect this much bigger than its YOLO
                         # box = two touching cubes merged; reject
MAX_LEN_RATIO = 1.30     # a real cube top is near-square in mm; a more
                         # elongated rect is two STAGGERED touching
                         # cubes merged - its center is the seam
FILL_MIN = 0.70          # a real top face fills ~95% of its minAreaRect;
                         # a staggered pair forms an L filling only
                         # w/(w+o), so this and MAX_LEN_RATIO together
                         # must cover EVERY stagger offset: ratio kills
                         # offsets below 0.538w, fill kills those above
                         # (1-FILL)/FILL. At 0.70 that is 0.429w - still
                         # overlapping, with margin.
                         # Was 0.80 (cutoff 0.25w). Lowered after a
                         # transparent floor: the cube/glass boundary is
                         # a huge depth step, so top faces come out
                         # slightly ragged and filled only 0.74-0.79 -
                         # six of seven real cubes were being thrown away
                         # by this gate alone. Do NOT go below 0.65: at
                         # 0.60 a gap opens at offsets 0.54-0.67w and
                         # merged pairs get through (their centre is the
                         # seam, where the fingers close on nothing).

_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise RuntimeError(
                "cube_model.pt not found - run capture_dataset.bat and then "
                "train_cubes.bat once before using the v3 detector")
        import torch
        from ultralytics import YOLO
        _model = YOLO(MODEL_FILE)
        _model.to("cuda" if torch.cuda.is_available() else "cpu")
    return _model


def depth_mm_image(depth_frame):
    """The whole depth frame as a float32 mm image (0 = no data)."""
    scale = depth_frame.get_units() * 1000.0
    return np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale


def floor_depth_mm(dmm, step=6):
    """Estimate the floor depth: with a down-looking camera the floor is
    the deep bulk of the depth histogram (nothing real lies beyond it).
    Scalar; kept for simple previews. Prefer floor_plane() for detection
    - a single number is wrong when the floor is tilted w.r.t. the
    camera (the near side then reads as a huge raised object)."""
    s = dmm[::step, ::step]
    s = s[(s > 150.0) & (s < 1500.0)]
    if s.size < 200:
        return None
    return float(np.percentile(s, 80))


def floor_plane(dmm, step=8, iters=6, band=18.0):
    """Robust least-squares plane fit of the floor: depth ~ a*x+b*y+c,
    tolerant of a floor tilted relative to the camera. Objects (shallower
    than the floor) and far reflection outliers are trimmed by a
    symmetric inlier band around the evolving plane, seeded flat at the
    median depth. Returns coef [a, b, c] (mm) or None."""
    H, W = dmm.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    d = dmm[::step, ::step].astype(np.float64)
    # drop no-data and far reflection outliers. 1300mm clears the floor
    # even from the high search vantages (~740mm flange -> ~700mm floor
    # depth); the band trim below removes any remaining outliers.
    m = (d > 150.0) & (d < 1300.0)
    X, Y, D = xs[m].astype(np.float64), ys[m].astype(np.float64), d[m]
    if D.size < 200:
        return None
    A = np.column_stack([X, Y, np.ones(D.size)])
    coef = np.array([0.0, 0.0, float(np.median(D))])   # seed: flat @ median
    refined = False
    for _ in range(iters):
        keep = np.abs(D - A @ coef) < band
        if keep.sum() < 100:
            break
        coef, *_ = np.linalg.lstsq(A[keep], D[keep], rcond=None)
        refined = True
    return coef if refined else None    # never trust the un-refit flat seed


def floor_depth_map(coef, shape):
    """Per-pixel floor depth (mm) for the fitted plane, as an HxW array."""
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W]
    return (coef[0] * xs + coef[1] * ys + coef[2]).astype(np.float32)


def floor_depth_at(coef, x, y):
    """Floor depth (mm) under image pixel (x, y) for the fitted plane."""
    return float(coef[0] * x + coef[1] * y + coef[2])


def classify_color(color_img, mask):
    """Name the dominant colour of the masked (top-face) pixels: "red",
    "blue", or "unknown". Hue-based (OpenCV H in 0..180), using only
    reasonably saturated/bright pixels so shadows and the grey floor do
    not vote."""
    pix = color_img[mask > 0]
    if pix.size < 30 * 3:
        return "unknown"
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV) \
        .reshape(-1, 3)
    sel = hsv[(hsv[:, 1] > 60) & (hsv[:, 2] > 40)]
    if sel.shape[0] < 20:
        return "unknown"
    # count pixels per hue band rather than taking a median: red wraps
    # around 0/180, so a median of red pixels can land mid-range and
    # misread; counting both red sub-bands is wrap-safe
    h = sel[:, 0]
    n_red = int(np.count_nonzero((h <= 12.0) | (h >= 168.0)))
    n_blue = int(np.count_nonzero((h >= 95.0) & (h <= 135.0)))
    dominant = max(n_red, n_blue)
    if dominant < 0.30 * sel.shape[0]:
        return "unknown"                    # no clear colour
    return "red" if n_red >= n_blue else "blue"


def _color_of_box(color_img, box):
    """Colour of the top face outlined by `box` (its 4 corner points).

    Same test as classify_color, but on a crop instead of a full-frame
    mask, so EVERY candidate can be classified cheaply. The picker sorts
    by colour, and to prefer a red cube over a blue one it has to know
    the colour of each cube in view - not just of the one it would
    otherwise have chosen."""
    pts = np.array(box, dtype=np.int32)
    x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
    x2 = min(color_img.shape[1], int(pts[:, 0].max()) + 1)
    y2 = min(color_img.shape[0], int(pts[:, 1].max()) + 1)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return "unknown"
    m = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(m, [pts - np.array([x1, y1], dtype=np.int32)], 255)
    return classify_color(color_img[y1:y2, x1:x2], m)


def _fold_angle(w_px, h_px, angle):
    """minAreaRect angle -> angle of the SHORT side vs x-axis, in (-90,90]."""
    grip_angle = angle if w_px >= h_px else angle + 90.0
    while grip_angle > 90.0:
        grip_angle -= 180.0
    while grip_angle < -90.0:
        grip_angle += 180.0
    return grip_angle


def _rect_from_depth(dmm, box_xyxy, fmap):
    """Rotated rect of the cube TOP FACE inside a YOLO box, from depth.
    fmap is the per-pixel floor-plane depth (mm). Returns
    (rect_in_full_image_coords, top_depth_mm) or (None, None)."""
    x1 = max(0, int(box_xyxy[0]) - ROI_PAD)
    y1 = max(0, int(box_xyxy[1]) - ROI_PAD)
    x2 = min(dmm.shape[1], int(box_xyxy[2]) + ROI_PAD)
    y2 = min(dmm.shape[0], int(box_xyxy[3]) + ROI_PAD)
    roi = dmm[y1:y2, x1:x2]
    floor_roi = fmap[y1:y2, x1:x2]
    above = roi[(roi > 100.0) & (roi < floor_roi - POP_MM)]
    if above.size < 50:
        return None, None
    top = float(np.percentile(above, 25))       # depth of the top face
    m = ((roi > top - 8.0) & (roi < top + TOP_BAND_MM)).astype(np.uint8)
    m *= 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None, None
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if min(w, h) / max(w, h) < SQUARENESS:
        return None, None
    if cv2.contourArea(c) < FILL_MIN * w * h:
        return None, None                # L-shaped staggered-pair merge
    return ((cx + x1, cy + y1), (w, h), ang), top


def cube_candidates(color_img, depth_frame, intrinsics, floor_coef=None):
    """Every cube-shaped top face in view, as raw candidate records.

    Shared by detect_cube (which then selects ONE) and detect_all (which
    reports them all) so the picker and the viewer can never drift
    apart. Returns (candidates, floor_plane_coef); the list is empty
    when YOLO sees nothing, the floor cannot be fitted, or no box
    survives the shape/size gates.

    floor_coef: use this floor plane instead of fitting one to the
    depth image. For a floor the camera cannot see - transparent, so
    the infrared measures something far below it - a plane taught once
    through an opaque sheet is the only correct answer; fitting would
    either fail or, worse, lock confidently onto the wrong surface.
    """
    model = _get_model()
    res = model.predict(color_img, conf=CONF_MIN, verbose=False)[0]
    if len(res.boxes) == 0:
        return [], None

    dmm = depth_mm_image(depth_frame)
    coef = np.asarray(floor_coef, dtype=float) if floor_coef is not None \
        else floor_plane(dmm)
    if coef is None:
        return [], None
    fmap = floor_depth_map(coef, dmm.shape)

    cands = []
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        rect, top = _rect_from_depth(dmm, xyxy, fmap)
        if rect is None:
            continue
        (cx, cy), (w_px, h_px), angle = rect
        # the refined rect must BELONG to this YOLO box: its center
        # inside the (unpadded) box, and not a merged multi-cube blob -
        # a merged blob's center is the GAP between the cubes and the
        # arm would close its fingers on nothing there
        if not (xyxy[0] <= cx <= xyxy[2] and xyxy[1] <= cy <= xyxy[3]):
            continue
        box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if w_px * h_px > MERGED_BLOB_RATIO * box_area:
            continue

        depth_m = top / 1000.0
        box = cv2.boxPoints(rect)
        pts3d = [rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(px), float(py)], depth_m) for px, py in box]
        side1 = np.linalg.norm(np.subtract(pts3d[0], pts3d[1])) * 1000.0
        side2 = np.linalg.norm(np.subtract(pts3d[1], pts3d[2])) * 1000.0
        width_mm = float(min(side1, side2))
        if not (MIN_CUBE_MM <= width_mm <= MAX_CUBE_MM_DET):
            continue
        if float(max(side1, side2)) > MAX_LEN_RATIO * width_mm:
            continue                     # merged staggered pair
        cands.append({
            "rect": rect, "top": top, "box": box,
            "conf": float(b.conf[0]), "area_px": w_px * h_px,
            "width_mm": width_mm, "length_mm": float(max(side1, side2)),
            "color": _color_of_box(color_img, box),
        })
    return cands, coef


def _describe(cand, color_img, coef, intrinsics):
    """Turn one candidate into the public detection dict (+ its mask)."""
    (cx, cy), (w_px, h_px), angle = cand["rect"]
    top = cand["top"]
    box = cand["box"]
    mask_full = np.zeros(color_img.shape[:2], np.uint8)
    cv2.fillPoly(mask_full, [np.intp(box)], 255)
    color = cand["color"]
    center3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy],
                                               top / 1000.0)

    # local floor directly under this cube (tilt-correct), for height and
    # for the grab-Z clamp downstream
    floor_here = floor_depth_at(coef, cx, cy)

    # height above the floor from depth; fall back to the cube heuristic
    # (as tall as wide) if the floor estimate is off
    h = floor_here - top
    height_est = False
    if not (POP_MM <= h <= 110.0) or abs(h - cand["width_mm"]) > 35.0:
        h = cand["width_mm"]
        height_est = True

    out = {
        "pixel": [round(cx, 1), round(cy, 1)],
        "cam_xyz_mm": [round(v * 1000.0, 1) for v in center3d],
        "depth_mm": round(top, 1),
        "width_mm": round(cand["width_mm"], 1),
        "length_mm": round(cand["length_mm"], 1),
        "width_px": round(min(w_px, h_px), 1),
        "length_px": round(max(w_px, h_px), 1),
        "angle_deg": round(_fold_angle(w_px, h_px, angle), 1),
        "box": box.tolist(),
        "conf": round(cand["conf"], 2),
        "floor_mm": round(floor_here, 1),
        "height_mm": round(h, 1),
        "color": color,
    }
    if height_est:
        out["height_est"] = True
    return out, mask_full


def detect_cube(color_img, depth_frame, intrinsics, prefer_near=None,
                near_radius=None, want_color=None, floor_coef=None):
    """Return (dict about the best cube, full-size cube mask).

    dict keys match detect_red_cube(): pixel, cam_xyz_mm, depth_mm,
    width_mm, length_mm, width_px, length_px, angle_deg, box - plus
    conf, floor_mm, height_mm, color (and height_est when height came
    from the width heuristic instead of depth).

    prefer_near=(px, py): pick the candidate whose centre is nearest
    this pixel instead of the largest one. Callers that KNOW where
    their cube should appear (e.g. the mid-descent look, where it must
    sit near the calibrated reference pixel) use this so a bigger
    NEIGHBORING cube elsewhere in view cannot hijack the detection.
    near_radius: with prefer_near, reject candidates farther than this
    many pixels - returns None rather than a far-away DIFFERENT cube,
    so the caller learns "my cube is not here" instead of silently
    aiming at someone else's.
    want_color="red"/"blue": hunt that colour only - see below.
    """
    mask_full = np.zeros(color_img.shape[:2], np.uint8)
    cands, coef = cube_candidates(color_img, depth_frame, intrinsics,
                                  floor_coef)
    if not cands:
        return None, mask_full

    # Colour priority. When the caller is hunting one colour, a cube of
    # any other colour is not a candidate at all - and with none of the
    # wanted colour in view the answer is None, NOT the next best cube.
    # That "nothing red here" is exactly what lets the picker exhaust
    # every red cube (across all its vantages) before it starts on the
    # blues. Cubes the classifier cannot call ("unknown") never match a
    # colour hunt; the picker sweeps those up in a final any-colour pass
    # so they are never stranded.
    if want_color is not None:
        cands = [c for c in cands if c["color"] == want_color]
        if not cands:
            return None, mask_full

    # largest GRIPPABLE cube first; oversized objects only if nothing
    # grippable is in view (the caller's width gate then skips cleanly)
    grippable = [c for c in cands if c["width_mm"] <= GRIPPABLE_MM]
    pool = grippable if grippable else cands
    if prefer_near is not None:
        if near_radius is not None:
            pool = [c for c in pool
                    if np.hypot(c["rect"][0][0] - prefer_near[0],
                                c["rect"][0][1] - prefer_near[1])
                    <= near_radius]
            if not pool:
                return None, mask_full   # our cube is not where expected
        cand = min(pool, key=lambda c: np.hypot(
            c["rect"][0][0] - prefer_near[0],
            c["rect"][0][1] - prefer_near[1]))
    else:
        cand = max(pool, key=lambda c: c["area_px"])

    return _describe(cand, color_img, coef, intrinsics)


def detect_all(color_img, depth_frame, intrinsics, floor_coef=None):
    """Every cube in view, each described exactly like detect_cube's
    result. For the live viewer: seeing all the cubes and the colour
    read off each one - at once, with no arm moving - is how you confirm
    the classifier before trusting the arm to sort by it."""
    cands, coef = cube_candidates(color_img, depth_frame, intrinsics,
                                  floor_coef)
    return [_describe(c, color_img, coef, intrinsics)[0] for c in cands]

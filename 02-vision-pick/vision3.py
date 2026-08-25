#!/usr/bin/env python3
"""Shared helpers for the v3 (YOLO, dynamic) pipeline.

Additive on top of vision_common.py - the v2 red-cube scripts are left
untouched and keep working. New here:
  * Camera        - a persistent RealSense pipeline (fast repeated
                    captures for the tracking loop, one warmup total)
  * Camera.stable - multi-frame median detection via detect_cube (YOLO)
  * calib3.json   - Phase A results (Jacobian) for v3
  * grip_ref.json - the ONE-TIME grab reference: XY anchor (g0<->p0),
                    the vertical constant zC, and the grip stall anchor.
                    Valid until the camera mount / fingers / SCAN pose
                    change - NOT per cube position.
  * grab_z()      - grab height for any cube from depth alone
"""
import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs

import vision_common as vc
from camera_test import CAPT
from detect_cube import detect_cube

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB3_FILE = os.path.join(HERE, "calib3.json")
GRIP_REF_FILE = os.path.join(HERE, "grip_ref.json")
PLACES_FILE = os.path.join(HERE, "places.json")
FLOOR_REF_FILE = os.path.join(HERE, "floor_ref.json")

LOOK_H = 180.0              # half-way look height above the grab pose (mm)
MIN_MID_ABOVE_FLOOR = 6.0   # never aim the grab mid-point lower than this


# ------------------------- camera -------------------------

class Camera:
    """One RealSense pipeline kept open across many captures."""

    def __init__(self, warmup=25, floor_ref=None):
        # floor_ref: a floor plane taught once through an opaque sheet,
        # for a surface the depth camera cannot see. Held here so every
        # caller only has to say what height it is looking from.
        self.floor_ref = floor_ref
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.skip(warmup)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

    def skip(self, n):
        """Drop n frames (lets auto-exposure settle after an arm move)."""
        for _ in range(n):
            self.pipe.wait_for_frames()

    def frame(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        return np.asanyarray(color.get_data()), depth, intr

    def stable(self, tag="stable", n=6, min_found=3, settle=4,
               prefer_near=None, near_radius=None, want_color=None,
               flange_z=None):
        """Detect the cube over n frames and take medians for stability.

        Returns a dict with the detect_cube keys stabilized (pixel,
        depth_mm, width_mm, width_px, angle_deg, height_mm, floor_mm,
        frames_found), or None. Angles are median-folded so square
        symmetry can't split the votes. prefer_near=(px, py) makes both
        the per-frame detection and the cluster choice favour the cube
        nearest that pixel (see detect_cube). want_color restricts every
        frame to cubes of that colour, so None means "no cube of that
        colour is in view here" - the colour-priority search relies on
        that. Saves captures/<tag>_annotated.png.
        """
        self.skip(settle)
        # flange_z + a taught floor => use that plane instead of fitting
        # one to depth (a transparent floor has no depth of its own)
        fcoef = virtual_floor_coef(self.floor_ref, flange_z)
        hits = []
        last_img, last_found = None, None
        for _ in range(n):
            img, depth, intr = self.frame()
            found, _mask = detect_cube(img, depth, intr,
                                       prefer_near=prefer_near,
                                       near_radius=near_radius,
                                       want_color=want_color,
                                       floor_coef=fcoef)
            last_img = img
            if found:
                hits.append(found)
                last_found = found
        # keep only the largest pixel-proximity CLUSTER of hits: with
        # two similar cubes in view the per-frame best candidate can
        # alternate between them, and a median across both would aim
        # the arm at the empty space (or the seam) between them
        clusters = []
        for h in hits:
            p = h["pixel"]
            for cl in clusters:
                c0 = cl[0]["pixel"]
                if abs(p[0] - c0[0]) <= 25.0 and abs(p[1] - c0[1]) <= 25.0:
                    cl.append(h)
                    break
            else:
                clusters.append([h])
        if clusters:
            if prefer_near is not None:
                hits = min(clusters, key=lambda cl: np.hypot(
                    cl[0]["pixel"][0] - prefer_near[0],
                    cl[0]["pixel"][1] - prefer_near[1]))
            else:
                hits = max(clusters, key=len)
            last_found = hits[-1]
        if len(hits) < min_found:
            if last_img is not None:
                ann = last_img.copy()
                # name the colour being hunted: during the red pass a
                # frame full of blue cubes is a legitimate "none here",
                # not a detection failure, and the saved image should
                # say which of the two it was
                cv2.putText(ann, "NO STABLE %sCUBE (%d/%d frames)"
                            % ("" if want_color is None
                               else want_color.upper() + " ",
                               len(hits), n), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(CAPT, tag + "_annotated.png"), ann)
            return None

        a0 = hits[0]["angle_deg"]
        floors = [h["floor_mm"] for h in hits if h.get("floor_mm")]
        med = {
            "pixel": [float(np.median([h["pixel"][0] for h in hits])),
                      float(np.median([h["pixel"][1] for h in hits]))],
            "depth_mm": float(np.median([h["depth_mm"] for h in hits])),
            "width_mm": float(np.median([h["width_mm"] for h in hits])),
            "width_px": float(np.median([h["width_px"] for h in hits])),
            "angle_deg": a0 + float(np.median(
                [vc.wrap90(h["angle_deg"] - a0) for h in hits])),
            "height_mm": float(np.median([h["height_mm"] for h in hits])),
            "floor_mm": float(np.median(floors)) if floors else None,
            "conf": float(np.median([h.get("conf", 0.0) for h in hits])),
            "frames_found": len(hits),
        }
        # colour vote: count the two known colours, ignore "unknown"
        # unless nothing else was seen; deterministic tie-break to red
        # (never depends on set/hash iteration order)
        colors = [h.get("color", "unknown") for h in hits]
        n_red, n_blue = colors.count("red"), colors.count("blue")
        if n_red == 0 and n_blue == 0:
            med["color"] = "unknown"
        else:
            med["color"] = "red" if n_red >= n_blue else "blue"
        if all(h.get("height_est") for h in hits):
            med["height_est"] = True

        ann = last_img.copy()
        box = np.intp(last_found["box"])
        cv2.drawContours(ann, [box], 0, (0, 255, 0), 2)
        cv2.putText(ann, "%s %.0fmm h%.0f %.0fdeg c%.2f (%d/%d)"
                    % (med["color"], med["width_mm"], med["height_mm"],
                       med["angle_deg"], med["conf"], len(hits), n),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(CAPT, tag + "_annotated.png"), ann)
        return med


# ------------------------- calibration files -------------------------

def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_calib3(data):
    _save(CALIB3_FILE, data)


def load_calib3():
    return _load(CALIB3_FILE)


def save_places(data):
    _save(PLACES_FILE, data)


def load_places():
    """{'red': [x,y,z,r,p,yaw], 'blue': [...]} or None. Only entries that
    are full 6-number poses are returned; a corrupt or hand-edited file
    degrades to None (treated as 'not set') instead of crashing."""
    try:
        data = _load(PLACES_FILE)
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) == 6:
            try:
                out[k] = [float(x) for x in v]
            except (TypeError, ValueError):
                continue                    # skip a malformed entry
    return out or None


# ------------------- floor the camera cannot see -------------------
#
# A transparent floor (glass / acrylic) returns no depth of its own: the
# infrared goes through it and measures whatever is far below. The floor
# is still FLAT and at a FIXED height, though, so it can be measured
# ONCE - with an opaque sheet laid over it - and that stored plane used
# ever after in place of a per-frame fit.
#
# This deliberately trades a measurement for an assumption. Nothing can
# then notice if the surface moves or the camera shifts, which is why
# it is opt-in (the file simply does not exist on a normal floor) and
# why teach_floor.py must be re-run after any change to the cell.

def save_floor_ref(data):
    _save(FLOOR_REF_FILE, data)


def load_floor_ref():
    """Stored floor plane, or None. Refuses a reference taught under a
    different SCAN pose - the heights would no longer line up."""
    ref = _load(FLOOR_REF_FILE)
    if not ref or not all(k in ref for k in ("coef", "ref_z", "scan_pose")):
        return None
    if list(ref.get("scan_pose") or []) != list(vc.SCAN_POSE):
        print("(floor_ref.json was taught under a different SCAN pose - "
              "ignoring it; re-run teach_floor)")
        return None
    return ref


def virtual_floor_coef(floor_ref, flange_z):
    """Floor-plane coefficients as seen from a straight-down pose at
    flange_z. The camera looks straight down, so raising the arm by dz
    simply adds dz to every floor depth - the tilt is unchanged."""
    if not floor_ref or flange_z is None:
        return None
    a, b, c = floor_ref["coef"]
    return [a, b, c + (flange_z - floor_ref["ref_z"])]


def save_grip_ref(data):
    _save(GRIP_REF_FILE, data)


def load_grip_ref():
    ref = _load(GRIP_REF_FILE)
    if ref is None:
        return None
    need = ("g0", "p0", "d0", "h0", "w0_mm", "angle0", "stall0", "zC",
            "floor0", "look_h", "scan_pose")
    if not all(k in ref for k in need):
        return None
    return ref


# ------------------------- grab-height math -------------------------
#
# With the camera looking straight down from a pose whose flange Z is
# Zm, a cube whose TOP the camera sees at depth d and whose height is h
# is grabbed (fingers at the cube's mid-height) at flange
#     Zg = Zm - d - h/2 + zC
# zC lumps the camera-to-flange and flange-to-finger vertical geometry
# and is measured ONCE by the finger teach:
#     zC = g0_z - Zscan + d0 + h0/2

# How far the measured floor may disagree with the predicted one before
# the reading is refused. The floor's depth is PREDICTABLE: from a
# straight-down pose it must read floor0 + (flange_z - SCAN_z). A
# reading that disagrees means either the floor moved (recalibrate) or
# the camera is not seeing the floor at all - a transparent surface, for
# instance, where the infrared passes through and measures whatever lies
# underneath.
#
# Why 40mm: grab_z's clamp window only starts pulling a grab DOWNWARD
# once the floor reads roughly 57mm too deep (sooner for tall cubes,
# ~84mm for a 38mm one), so 40mm catches every harmful case with margin.
# Above it sit the honest sources of disagreement - floor tilt across
# the frame (~30mm measured in this cell) and floor0 having been taken
# at the reference cube rather than the image centre.
FLOOR_TOL_MM = 40.0


def expected_floor(ref, measured_from_z=None):
    """Depth the floor SHOULD read, looking straight down from flange Z."""
    zm = vc.SCAN_POSE[2] if measured_from_z is None else measured_from_z
    return ref["floor0"] + (zm - vc.SCAN_POSE[2])


def floor_error(ref, floor_mm, measured_from_z=None):
    """How far a floor measurement is from the predicted one (mm, signed:
    positive = reads DEEPER than it should). None if there is nothing to
    check."""
    if not floor_mm or not ref or "floor0" not in ref:
        return None
    return float(floor_mm - expected_floor(ref, measured_from_z))


def floor_is_trustworthy(ref, floor_mm, measured_from_z=None,
                         tol=FLOOR_TOL_MM):
    """Is this floor measurement believable?

    Guards the two places a bad floor does damage: grab_z's clamp (which
    can drive the arm far below the cube) and the dataset labeler (which
    would otherwise save a frame full of cubes as empty background). No
    measurement at all is fine - the callers already fall back to the
    predicted floor."""
    err = floor_error(ref, floor_mm, measured_from_z)
    return err is None or abs(err) <= tol


def compute_zc(g0_z, d0, h0, scan_z=None):
    scan_z = vc.SCAN_POSE[2] if scan_z is None else scan_z
    return float(g0_z - scan_z + d0 + h0 / 2.0)


def grab_z(ref, depth_mm, height_mm, floor_mm=None, measured_from_z=None):
    """Flange Z to grab a cube of height h whose top is at camera depth
    depth_mm, measured from the SCAN height (or measured_from_z).
    Clamped so the grab mid-point never goes below the floor."""
    zm = vc.SCAN_POSE[2] if measured_from_z is None else measured_from_z
    z = zm - depth_mm - height_mm / 2.0 + ref["zC"]
    # floor0 was measured at the SCAN height; when this observation is
    # from a different height (measured_from_z), the floor is that much
    # farther, so keep the fallback height-consistent (zm - floor must
    # stay constant) or the clamp would push an elevated grab too high.
    floor = floor_mm if floor_mm else \
        ref["floor0"] + (zm - vc.SCAN_POSE[2])
    z_min = zm - floor + ref["zC"] + MIN_MID_ABOVE_FLOOR
    z_max = z_min + 90.0
    return float(max(z_min, min(z_max, z)))


# ------------------- aim from an arbitrary hover pose -------------------
#
# The picker can observe a cube from poses OTHER than the SCAN pose (it
# rises for a wider view, or sweeps a grid, when the scan view is empty).
# The camera looks straight down at scan yaw, so its optical axis meets
# the floor at a FIXED base-frame offset from the flange - independent of
# height. That offset falls straight out of the existing SCAN anchor, so
# no extra calibration is needed:
#
#   cube_base_xy = flange_xy + camera_offset + (d_obs/d_ref) * J*(p-CENTER)
#
# where d_obs is the ACTUAL measured top depth (the pixel->mm scale grows
# with depth). At the SCAN pose this is algebraically identical to the
# validated SCAN-anchored aim.

CENTER = (320.0, 240.0)      # image principal point (approx)


def camera_offset(calib, ref):
    """Base-frame XY from the flange axis to the camera optical axis at
    scan yaw. Constant (camera looks straight down); derived from the
    SCAN anchor, so it costs no extra calibration."""
    J = np.asarray(calib["J"], dtype=float)
    d_ref = calib.get("d_ref") or ref["d0"]
    anchor = (ref["d0"] / d_ref) * (J @ (np.asarray(ref["p0"], dtype=float)
                                         - np.asarray(CENTER)))
    return np.asarray([ref["g0"][0] - vc.SCAN_POSE[0] - anchor[0],
                       ref["g0"][1] - vc.SCAN_POSE[1] - anchor[1]])


def aim_from_pose(calib, ref, obs_pose, seen):
    """Grasp target (gx, gy, gz, yaw) for a cube seen at pixel
    seen["pixel"] from a straight-down hover pose obs_pose=[x,y,z,...].

    At obs_pose == SCAN_POSE this reduces exactly to the validated
    SCAN-anchored aim (verified algebraically); it generalizes by using
    the actual measured top depth for the pixel->mm scale and adding the
    flange's displacement from the scan spot via the constant camera
    offset. Rotation is height-independent (yaw mirror, as at scan)."""
    J = np.asarray(calib["J"], dtype=float)
    d_ref = calib.get("d_ref") or ref["d0"]
    d_obs = seen["depth_mm"]
    off = (d_obs / d_ref) * (J @ (np.asarray(seen["pixel"], dtype=float)
                                  - np.asarray(CENTER)))
    co = camera_offset(calib, ref)
    gx = float(obs_pose[0] + co[0] + off[0])
    gy = float(obs_pose[1] + co[1] + off[1])
    gz = grab_z(ref, d_obs, seen["height_mm"], seen.get("floor_mm"),
                measured_from_z=obs_pose[2])
    yaw = ref["g0"][3] - vc.wrap90(seen["angle_deg"] - ref["angle0"])
    return gx, gy, gz, yaw

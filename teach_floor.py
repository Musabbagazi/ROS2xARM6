#!/usr/bin/env python3
"""Teach the floor plane for a surface the depth camera cannot see.

A transparent floor (glass / acrylic) returns no depth of its own: the
infrared passes through and measures whatever lies far below. Detection
and the grab height both need to know where the floor IS, so measure it
ONCE through an opaque sheet and store the plane.

You do NOT need to cover the whole floor. A single A4 sheet is enough
to fix the floor's HEIGHT. Measuring it in two or three different spots
also pins down its TILT, which is what makes grabs accurate at the edges
of the view - so it is worth the extra minute.

  1. Take the cubes OFF the floor (they would be measured as floor).
  2. Put a flat opaque sheet - paper, card, a thin mat - under the
     middle of the camera view.
  3. Run this. It measures, then offers to measure again.
  4. Slide the sheet to a different part of the floor and measure again.
     Two or three well-spread spots give a proper tilt.
  5. Take the sheet away. Done.

Re-run it whenever the cell changes: a different floor, a moved camera,
a new SCAN pose. Nothing can detect a stale reference for you, because
the whole point is that this surface cannot be re-measured in use.

Delete vision/floor_ref.json to go back to measuring the floor live
(which is correct on a normal, opaque floor).

Usage: python teach_floor.py [robot_ip]
"""
import sys

import numpy as np
from xarm.wrapper import XArmAPI

import vision3 as v3
import vision_common as vc
from detect_cube import depth_mm_image, floor_depth_at

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

FRAMES = 8              # frames averaged per sheet position
STEP = 8                # depth subsampling, as in floor_plane
SEED_BAND = 80.0        # mm around the seeded depth that counts as sheet
TRIM_BAND = 18.0        # inlier band once fitting, as in floor_plane
MIN_POINTS = 150        # per position - an A4 gives ~1000
# Fitting a tilt needs a WIDE baseline. A plane fitted across one small
# sheet and extrapolated over the whole frame amplifies any angular
# error, so below this span (fraction of the frame, in BOTH axes) the
# floor is stored flat: no tilt is better than a badly guessed one.
TILT_SPAN_MIN = 0.55
MAX_OFF_EXPECTED = 150.0   # refuse a surface this far from the expected
                           # floor - that is not the sheet, it is
                           # whatever lies below a transparent panel


def collect(cam, seed, band=SEED_BAND):
    """Points on the sheet, over several frames. `seed` is a scalar
    depth or a per-pixel map of where the floor is expected; only depths
    within `band` of it are taken, so the surface far below a
    transparent floor cannot pull the fit onto itself."""
    X, Y, D = [], [], []
    for _ in range(FRAMES):
        _img, depth, _intr = cam.frame()
        dmm = depth_mm_image(depth)
        H, W = dmm.shape
        ys, xs = np.mgrid[0:H:STEP, 0:W:STEP]
        d = dmm[::STEP, ::STEP].astype(np.float64)
        ref = np.full(d.shape, float(seed)) if np.isscalar(seed) \
            else seed[::STEP, ::STEP]
        m = (d > 150.0) & (d < 2000.0) & (np.abs(d - ref) < band)
        X.append(xs[m]); Y.append(ys[m]); D.append(d[m])
    return (np.concatenate(X).astype(np.float64),
            np.concatenate(Y).astype(np.float64),
            np.concatenate(D))


def centre_seed(cam):
    """Depth under the middle of the view - where the sheet is put for
    the first measurement. The nearest quartile, so a stray deep pixel
    cannot drag it toward the surface below a transparent floor."""
    vals = []
    for _ in range(FRAMES):
        _img, depth, _intr = cam.frame()
        dmm = depth_mm_image(depth)
        c = dmm[168:312, 224:416]           # central ~30% of the frame
        v = c[(c > 150.0) & (c < 2000.0)]
        if v.size > 200:
            vals.append(float(np.percentile(v, 25)))
    return float(np.median(vals)) if vals else None


def fit(X, Y, D, allow_tilt):
    """Robust plane (or flat offset) through the collected sheet points."""
    if not allow_tilt:
        coef = np.array([0.0, 0.0, float(np.median(D))])
        for _ in range(4):
            keep = np.abs(D - coef[2]) < TRIM_BAND
            if keep.sum() < MIN_POINTS:
                break
            coef[2] = float(np.median(D[keep]))
        return coef
    A = np.column_stack([X, Y, np.ones(D.size)])
    coef = np.array([0.0, 0.0, float(np.median(D))])
    for _ in range(6):
        keep = np.abs(D - A @ coef) < TRIM_BAND
        if keep.sum() < MIN_POINTS:
            break
        coef, *_ = np.linalg.lstsq(A[keep], D[keep], rcond=None)
    return coef


def span(X, Y):
    """How much of the frame the measured points cover, per axis."""
    if X.size == 0:
        return 0.0, 0.0
    return ((X.max() - X.min()) / 640.0, (Y.max() - Y.min()) / 480.0)


def main():
    print(__doc__.split("Usage:")[0].strip())
    print("\nConnecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
        scan_angles = vc.ik(arm, vc.SCAN_POSE)
        if scan_angles is None:
            raise RuntimeError("SCAN pose not reachable")
    except RuntimeError as e:
        print("ABORT (no motion):", e)
        arm.disconnect()
        return

    grip = v3.load_grip_ref()
    print("\nTake the CUBES OFF the floor, and put the sheet under the "
          "MIDDLE of the view.")
    if input("Ready? [y/N] ").strip().lower() != "y":
        print("Aborted."); arm.disconnect(); return

    allX = allY = allD = None
    coef = None
    ref_z = None
    try:
        vc.movej(arm, "SCAN pose", scan_angles)
        code, pose = arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError("could not read the arm position")
        ref_z = pose[2]
        expected = None if not grip else \
            grip["floor0"] + (ref_z - vc.SCAN_POSE[2])
        if expected is not None:
            print("(the calibrated floor would read %.0fmm from here)"
                  % expected)

        with v3.Camera() as cam:
            spot = 0
            while True:
                spot += 1
                if spot > 1:
                    print("\nSlide the sheet to a DIFFERENT part of the "
                          "floor (a corner of the view).")
                    if input("Measure another spot? [y/N] ").strip().lower() \
                            != "y":
                        break
                # seed: the middle of the view for the first sheet, the
                # plane we already have for later ones (the sheet is
                # off-centre by then, so the centre says nothing)
                if coef is None:
                    seed = centre_seed(cam)
                    if seed is None:
                        print("  no usable depth in the middle of the view "
                              "- is the sheet there?")
                        spot -= 1
                        continue
                    print("  sheet reads %.0fmm under the centre" % seed)
                else:
                    ys, xs = np.mgrid[0:480, 0:640]
                    seed = coef[0] * xs + coef[1] * ys + coef[2]

                X, Y, D = collect(cam, seed)
                if X.size < MIN_POINTS:
                    print("  only %d points found - the sheet is too small, "
                          "too shiny, or not in view." % X.size)
                    spot -= 1
                    continue
                allX = X if allX is None else np.concatenate([allX, X])
                allY = Y if allY is None else np.concatenate([allY, Y])
                allD = D if allD is None else np.concatenate([allD, D])
                sx, sy = span(allX, allY)
                tilt_ok = sx >= TILT_SPAN_MIN and sy >= TILT_SPAN_MIN
                coef = fit(allX, allY, allD, tilt_ok)
                print("  spot %d: %d points | coverage so far %.0f%% x "
                      "%.0f%% of the view | %s"
                      % (spot, X.size, 100 * sx, 100 * sy,
                         "tilt fitted" if tilt_ok
                         else "flat (too narrow for a tilt yet)"))
    except RuntimeError as e:
        print("\nSTOPPED:", e)
        arm.disconnect()
        return
    finally:
        arm.disconnect()

    if coef is None or allD is None:
        print("\nFAILED: nothing was measured.")
        return

    centre = floor_depth_at(coef, 320, 240)
    corners = [floor_depth_at(coef, x, y)
               for x, y in ((0, 0), (639, 0), (0, 479), (639, 479))]
    tilt = max(corners) - min(corners)
    sx, sy = span(allX, allY)
    tilted = sx >= TILT_SPAN_MIN and sy >= TILT_SPAN_MIN

    expected = None if not grip else \
        grip["floor0"] + (ref_z - vc.SCAN_POSE[2])
    if expected is not None and abs(centre - expected) > MAX_OFF_EXPECTED:
        print("\nREFUSED: the surface measured %.0fmm, but the floor should "
              "be near %.0fmm\n(%.0fmm out). That is almost certainly NOT "
              "the sheet - on a transparent\nfloor it is whatever lies "
              "below. Nothing was saved."
              % (centre, expected, centre - expected))
        print("Check the sheet really is under the camera, flat and opaque.")
        return

    v3.save_floor_ref({
        "note": "Floor plane measured through an opaque sheet, for a "
                "surface the depth camera cannot see. Re-run teach_floor "
                "if the floor, the camera or the SCAN pose changes. "
                "Delete this file to measure the floor live instead.",
        "coef": [float(c) for c in coef],
        "ref_z": float(ref_z),
        "scan_pose": list(vc.SCAN_POSE),
        "centre_depth_mm": round(float(centre), 1),
        "tilt_mm": round(float(tilt), 1),
        "tilt_fitted": bool(tilted),
        "coverage": [round(float(sx), 2), round(float(sy), 2)],
        "points": int(allD.size),
    })

    print("\n===== floor taught =====")
    print("floor at the centre of the view: %.1f mm (from flange Z %.0f)"
          % (centre, ref_z))
    if tilted:
        print("tilt across the view:           %.1f mm (fitted)" % tilt)
    else:
        print("tilt:                           assumed FLAT - the sheet "
              "covered only\n                                %.0f%% x %.0f%% "
              "of the view, too narrow to fit a tilt."
              % (100 * sx, 100 * sy))
        print("   Good enough to start. If grabs come out slightly high or "
              "low at the\n   EDGES of the view, re-run this and measure "
              "2-3 spread-out spots.")
    if expected is not None:
        print("against the calibrated floor (%.1f): %+.1f mm"
              % (expected, centre - expected))
    print("\nsaved: %s" % v3.FLOOR_REF_FILE)
    print("Now TAKE THE SHEET AWAY. Check with '0 - Live Camera View', "
          "then run a pick.")


if __name__ == "__main__":
    main()

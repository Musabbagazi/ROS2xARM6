#!/usr/bin/env python3
"""Teach the support surface, once, in the robot's frame.

CAMERA ONLY - THE ARM IS NEVER COMMANDED BY THIS SCRIPT.

The detector finds cubes by asking which points sit more than POP_MM
above the surface, so it has to know where the surface is. On an ordinary
opaque table it could fit that live, every frame. This cell's floor is
GLASS: its infrared passes straight through and the depth camera reports
whatever is underneath, roughly 660mm below, so a live fit either finds
nothing or locks onto the structure below and puts the "floor" far too
low. vision\\ hit this and solved it the same way - lay something opaque
down, measure once, store it.

Storing it is not a workaround here, it is the right answer: the camera
does not move, so a surface measured once stays measured. The file is
only stale if the camera is knocked or the table is moved - and both of
those invalidate handeye.json too, so both mean a full re-calibration.

WHAT TO DO
    1. take every cube OFF the floor,
    2. lay flat opaque paper or card over the working area - it does not
       have to cover all of it, see below,
    3. send the arm out of the camera's view (go_home.bat),
    4. run this.

PARTIAL COVERAGE IS FINE, AND TILT IS ONLY FITTED WHEN IT IS EARNED.
A single sheet in the middle of the view pins the surface's HEIGHT
perfectly well, but a tilt extrapolated from one small patch to the far
corners of the cell is worse than assuming the floor is level. So the
tilt terms are only kept when the measured points span enough of the
working area in BOTH axes; otherwise the surface is stored flat and said
so. Cover more of the floor, or run it again with the sheet in several
places, if the real tilt matters.
"""
import sys

import numpy as np

import fixed_common as fx


# How many frames to median together. The surface is not going anywhere,
# so there is no reason to trust one frame.
FRAMES = 10

# Points are collected within this much of the seed height. Wide enough
# for a real tilt across the cell (~40mm here), tight enough to exclude
# the through-glass background hundreds of millimetres below.
SEED_BAND_MM = 70.0

# Fraction of the working area the measurements must span, in EACH axis,
# before the tilt terms are believed.
TILT_SPAN_MIN = 0.55

# Plausibility guard on the surface's height. NOT a calibration.
#
# What it exists to catch is the failure that actually happened on this
# cell before: the plane locking onto the structure UNDER the glass, some
# 660mm below, and coming out confidently wrong while looking perfectly
# well fitted. A grab planned from that would be driven straight into the
# table.
#
# The anchor is the calibration's own `z_low` - the flange height the
# operator jogged to with the plate about JOG_CLEAR_MM above the floor
# before calibrating. In the grasp frame a point on the surface maps to
# the flange that puts the CUP on it, so the surface must sit that much
# below z_low. The operator's "about 20mm" is a rough number, hence a
# band far wider than any plausible misjudgement of it - but still an
# order of magnitude tighter than the failure being guarded against.
#
# There is deliberately no absolute fallback. The old one was arithmetic
# on the TWO-FINGER gripper's measured grasp heights, and a vacuum tool
# of unknown length makes any absolute band meaningless.
JOG_CLEAR_MM = 20.0
FLOOR_TOL_MM = 150.0


def measure(cam, frames=FRAMES):
    """(coef, info) for the surface, or (None, reason)."""
    T = fx.load_handeye()
    if T is None:
        return None, ("handeye.json is missing - calibrate the camera "
                      "BEFORE teaching the floor.\n"
                      "   The surface is stored in the robot's frame, so "
                      "there is no frame to store it in yet.")

    rays = cam.rays()
    fits, spans, counts, backs = [], [], [], []
    for i in range(frames):
        depth_mm, _, _ = cam.frame()
        base = fx.apply_transform(T, fx.cloud_cam(depth_mm, rays))
        keep = (depth_mm > 0) & fx.in_work_box(base)
        pts = base[keep]
        if len(pts) < 400:
            continue

        # Seed high, not at the median: with a glass floor most of what
        # is in the box is the background far below, so the median lands
        # under the table. The surface is the highest broad thing in the
        # box once the cubes are off it.
        seed = float(np.percentile(pts[:, 2], 90))
        near = pts[np.abs(pts[:, 2] - seed) < SEED_BAND_MM]
        if len(near) < 300:
            continue
        coef = fx.fit_plane(near)
        if coef is None:
            continue
        fits.append(coef)
        counts.append(len(near))
        spans.append(((near[:, 0].max() - near[:, 0].min()),
                      (near[:, 1].max() - near[:, 1].min())))
        sig = fx.background_signature(pts, coef)
        if sig is not None:
            backs.append(sig[0])
        print("   frame %2d/%d: %6d points, height %.1fmm at the centre"
              % (i + 1, frames, len(near),
                 fx.plane_z(coef, np.mean(fx.WORK_X), np.mean(fx.WORK_Y))))

    if len(fits) < max(3, frames // 2):
        return None, ("only %d of %d frames produced a surface.\n"
                      "   Is the opaque sheet down, and are the cubes off "
                      "the floor?" % (len(fits), frames))

    F = np.asarray(fits)
    centre = [float(np.mean(fx.WORK_X)), float(np.mean(fx.WORK_Y))]
    heights = [fx.plane_z(c, *centre) for c in F]
    spread = float(np.max(heights) - np.min(heights))
    if spread > 12.0:
        return None, ("the surface moved %.0fmm between frames - the depth "
                      "is not stable enough to trust.\n"
                      "   Try again with the lights off (this camera's "
                      "depth is better in the dark) or a matt sheet."
                      % spread)

    coef = [float(np.median(F[:, 0])), float(np.median(F[:, 1])),
            float(np.median(F[:, 2]))]

    span_x = float(np.median([s[0] for s in spans]))
    span_y = float(np.median([s[1] for s in spans]))
    cover_x = span_x / (fx.WORK_X[1] - fx.WORK_X[0])
    cover_y = span_y / (fx.WORK_Y[1] - fx.WORK_Y[0])
    tilt_fitted = cover_x >= TILT_SPAN_MIN and cover_y >= TILT_SPAN_MIN
    if not tilt_fitted:
        # Keep the height at the centre of the measured patch, where it
        # was actually observed, and drop the slope.
        mid = [float(np.mean(fx.WORK_X)), float(np.mean(fx.WORK_Y))]
        coef = [0.0, 0.0, float(fx.plane_z(coef, *mid))]

    height = fx.plane_z(coef, *centre)
    corners = [fx.plane_z(coef, x, y)
               for x in fx.WORK_X for y in fx.WORK_Y]
    info = {
        "coef": coef,
        "height_at_centre": round(float(height), 1),
        "tilt_mm": round(float(np.max(corners) - np.min(corners)), 1),
        "tilt_fitted": bool(tilt_fitted),
        "coverage": [round(cover_x, 2), round(cover_y, 2)],
        "frames_used": len(fits),
        "points": int(np.median(counts)),
        "frame_spread_mm": round(spread, 1),
        # The drift witness - see fixed_common.background_signature. Only
        # present when there IS something visible below the surface, i.e.
        # on the glass floor. On an opaque floor it is None and the check
        # simply does not run.
        "background_z": (round(float(np.median(backs)), 1)
                         if len(backs) >= 3 else None),
    }

    z_low = T.get("z_low")
    if z_low is None:
        print("   (handeye.json has no z_low, so the surface height cannot")
        print("    be sanity-checked. Check it in '3 - Camera View'.)")
    else:
        expect = float(z_low) - JOG_CLEAR_MM
        if abs(height - expect) > FLOOR_TOL_MM:
            return None, (
                "the surface came out at %.0f, but the calibration was "
                "started with the\n"
                "   cup about %.0fmm above the floor at flange %.0f, which "
                "puts the surface\n"
                "   near %.0f. Being %.0fmm out is the signature of the "
                "plane locking onto\n"
                "   something else - the structure under the glass, a wall, "
                "or the arm.\n"
                "   Nothing was saved."
                % (height, JOG_CLEAR_MM, z_low, expect, height - expect))
        print("   sanity: surface %.1f vs %.1f expected from the "
              "calibration (%+.0fmm)" % (height, expect, height - expect))
    return coef, info


def main():
    fx.start_log("floor")
    print("=" * 62)
    print("   FIXED CAMERA  -  teach the support surface")
    print("=" * 62)
    print("   The arm is NOT moved by this script.")
    print("")
    print("   Before you continue:")
    print("     - every cube OFF the floor,")
    print("     - flat opaque paper or card laid over the working area,")
    print("     - the arm parked out of the camera's view (go_home.bat).")
    print("")
    try:
        if input("   Ready? [y/N] ").strip().lower() != "y":
            print("   nothing done.")
            return 1
    except (EOFError, KeyboardInterrupt):
        return 1

    print("")
    cam = fx.Camera()
    try:
        cam.start()
        coef, info = measure(cam)
    finally:
        cam.close()

    if coef is None:
        print("")
        print("   REFUSED: %s" % info)
        return 1

    fx.save_floor(info)
    print("")
    print("   surface height at the cell centre : %.1f (grasp frame)"
          % info["height_at_centre"])
    print("   tilt across the working area      : %.1fmm%s"
          % (info["tilt_mm"],
             "" if info["tilt_fitted"] else "  (stored FLAT - see below)"))
    print("   measured over                     : %.0f%% x %.0f%% of the "
          "working area" % (info["coverage"][0] * 100,
                            info["coverage"][1] * 100))
    if not info["tilt_fitted"]:
        print("")
        print("   The sheet covered too little of the floor to measure its")
        print("   slope honestly, so the surface is stored LEVEL at the")
        print("   height above. That is the safe choice, and good enough")
        print("   unless this cell's floor really is tilted - it was ~39mm")
        print("   across the frame when the wrist camera measured it. To")
        print("   capture that, cover more of the floor and run this again.")
    print("")
    print("   saved %s" % fx.FLOOR_FILE)
    print("   Next: '3 - Camera View' to see what the detector makes of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

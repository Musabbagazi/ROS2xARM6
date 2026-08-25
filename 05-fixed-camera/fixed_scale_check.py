#!/usr/bin/env python3
"""Is the CAMERA's metric scale right? No arm involved.

THE ARM DOES NOT MOVE. It is not even connected.

WHY THIS EXISTS

The touch calibration of 2026-08-12 fitted 16.7mm RMS with a best-fit
scale of 0.9375, and that number cannot say which side it comes from: a
tilted wrist blows up the RMS without touching the true scale, while a
camera whose geometry is off shifts the scale coherently. The two need
different repairs - one is a procedure fix, the other is the camera's
own calibration - so guessing is expensive in whole runs.

This separates them with a tape measure. A rigid transform preserves
distance, so if the camera is metrically right, the distance IT measures
between two cube positions must equal the distance YOU measure with the
tape - no hand-eye transform, no arm, no fit. If the camera says 320mm
where the tape says 300, every calibration pair it feeds is 6.7% long
and no touch procedure, however careful, can fit them rigidly.

HOW TO MEASURE THE TRUE DISTANCE WITHOUT FUSS

Centre-to-centre between two cube positions is awkward with a ruler.
Edge-to-same-edge is not, and it IS the centre distance: slide the tape
from the LEFT face of the cube at A to the LEFT face of it at B (same
face, same cube). Lay the tape flat on the floor along the line between
the two spots. A couple of millimetres of tape error on a 300mm move is
under 1% - fine for catching a 6% fault.

Use moves of 250mm or more, in different directions across the view.
Three or four are plenty.

THE VERDICT

  all ratios within ~2% of 1.00   the camera is fine. A bad calibration
                                  fit is robot-side or detection-side;
                                  do not chase the camera.

  ratios consistently off, same   the camera's own geometry is off.
  direction, more than 2%         Run the on-chip self-calibration:
                                  RealSense Viewer > More (top bar) >
                                  On-Chip Calibration, cube ~40cm away,
                                  then run this again to confirm.

Usage:  python fixed_scale_check.py [--colour red|blue|green|...]
"""
import sys

import numpy as np

import fixed_common as fx
import fixed_detect as fd

FRAMES_PER_SPOT = 10

# Below this the tape error and the depth noise are too big a slice of
# the answer for the ratio to mean much.
MIN_MOVE_MM = 150.0


def measure_spot(cam, colour, label):
    """Median top-face centre over several frames, plus its frame-to-
    frame scatter - the scatter is reported because it is the noise
    floor of everything else this cell does."""
    rays = cam.rays()
    pts = []
    why = None
    for _ in range(FRAMES_PER_SPOT):
        depth_mm, color_bgr, _ = cam.frame()
        got, info = fd.find_cube_top(depth_mm, color_bgr, rays, colour)
        if got is not None:
            pts.append(got)
        else:
            why = info
    if len(pts) < FRAMES_PER_SPOT // 2:
        print("      %s: seen in only %d/%d frames (%s)"
              % (label, len(pts), FRAMES_PER_SPOT, why or "no reason"))
        return None
    pts = np.asarray(pts)
    centre = np.median(pts, axis=0)
    scatter = float(np.linalg.norm(pts - centre, axis=1).std())
    print("      %s measured (%d frames, +-%.1fmm frame noise)"
          % (label, len(pts), scatter))
    return centre


def main():
    args = sys.argv[1:]
    colour = "red"
    if "--colour" in args:
        i = args.index("--colour")
        if i + 1 < len(args):
            colour = args[i + 1].lower()

    fx.start_log("scalecheck")
    print("=" * 62)
    print("   FIXED CAMERA  -  metric scale check (tape measure vs camera)")
    print("=" * 62)
    print("   THE ARM DOES NOT MOVE and is not needed.")
    print("")
    print("   You need: one %s cube, a tape measure, a clear floor." % colour)
    print("")
    print("   For each check:")
    print("     1. put the cube down, press ENTER - the camera measures it;")
    print("     2. slide it 250mm+ away along the floor, MEASURE the move")
    print("        with the tape (same face to same face), press ENTER;")
    print("     3. type the tape reading in mm.")
    print("")
    print("   Do 3-4 moves in different directions across the view.")
    print("")

    cam = fx.Camera()
    ratios = []
    try:
        cam.start()
        n = 0
        while True:
            n += 1
            print("")
            print("   --- move %d ---" % n)
            try:
                ans = input("      cube at the START spot. ENTER to measure"
                            " (or 'done'): ")
                if ans.strip().lower() in ("done", "d", "q"):
                    break
                a = measure_spot(cam, colour, "start")
                if a is None:
                    continue
                ans = input("      now slide it to the END spot. ENTER when"
                            " set (or 'done'): ")
                if ans.strip().lower() in ("done", "d", "q"):
                    break
                b = measure_spot(cam, colour, "end")
                if b is None:
                    continue
                tape = input("      tape reading for that move, in mm: ")
                true_mm = float(tape.strip())
            except (EOFError, KeyboardInterrupt):
                break
            except ValueError:
                print("      that was not a number - move skipped.")
                continue
            if true_mm < MIN_MOVE_MM:
                print("      under %.0fmm the ratio is mostly noise -"
                      % MIN_MOVE_MM)
                print("      use a longer move.")
                continue

            cam_mm = float(np.linalg.norm(b - a))
            ratio = cam_mm / true_mm
            ratios.append(ratio)
            print("      camera: %.1fmm   tape: %.1fmm   ratio %.4f"
                  % (cam_mm, true_mm, ratio))

        print("")
        if len(ratios) < 2:
            print("   Fewer than 2 usable moves - no verdict. Run it again.")
            return 1

        med = float(np.median(ratios))
        spread = float(max(ratios) - min(ratios))
        print("   %d moves   ratios: %s" % (len(ratios),
                                            [round(r, 4) for r in ratios]))
        print("   median %.4f, spread %.4f" % (med, spread))
        print("")
        if spread > 0.03:
            print("   The moves DISAGREE with each other by more than the")
            print("   method's own noise. That is not a scale error - it is")
            print("   a measurement problem: tape not along the move, cube")
            print("   detected badly at one spot, or someone walked through")
            print("   the view. Run it again before believing anything.")
            return 1
        if abs(med - 1.0) <= 0.02:
            print("   CAMERA SCALE IS FINE (within 2%%). If a calibration")
            print("   still fits badly, the error is robot-side - wrist")
            print("   orientation, cup compression, a nudged cube - or in")
            print("   the detection, NOT in the camera. Do not recalibrate")
            print("   the camera; it has nothing to confess.")
            return 0
        print("   THE CAMERA'S METRIC SCALE IS OFF by %.1f%%, consistently."
              % ((med - 1.0) * 100.0))
        print("   Every calibration pair it produces is stretched by that")
        print("   much, and no rigid fit can absorb it. A calibration run")
        print("   against this camera would report a best-fit scale of")
        print("   about %.3f - the fit shrinking to meet points the camera"
              % (1.0 / med))
        print("   spread too far apart.")
        print("   Fix: RealSense Viewer > More > On-Chip Calibration (a")
        print("   flat wall or the floor ~40cm away works), then run THIS")
        print("   again to confirm before recalibrating the cell.")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n   FAILED: %s" % e)
        return 1
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())

"""Can this camera measure a 30mm cube well enough to calibrate with?

NOTHING MOVES. This tool never commands the arm to a pose. It talks to
the controller only to switch the vacuum on and to read the seal switch.

WHY THIS EXISTS

The plan is to calibrate by sticking a cube to the cup and driving the
arm through a pose grid, so the grab offset is rigid and absorbs into the
transform instead of having to be measured. That removes every error the
last attempt suffered from - hand placement, press depth, wrist tilt.

But it rests on one thing that arithmetic cannot settle: whether the
camera can measure a HELD 30mm cube well. The visible ring around an
18mm cup on a 30mm cube is (30-18)/2 = 3mm wide, which at this range is
about three pixels, and both its edges are depth discontinuities - which
is exactly where flying pixels live.

So measure it before writing several hundred lines on top of it.

    PHASE A   cube on the table, arm parked out of view.
              The baseline. Also worth having on its own: it says
              whether the PICKER can see a 30mm cube well enough to
              aim at 3mm.

    PHASE B   same cube stuck to the cup, arm held still by hand-jog
              at a pose typical of the grid.

The difference between A and B is the cost of the cup and its shadow,
measured rather than guessed.

WHAT THE NUMBER MEANS

The spread reported is repeatability, not accuracy. A biased estimator
can be perfectly repeatable and perfectly wrong. Repeatability is
nonetheless the right first question: if a stationary target does not
repeat, no fit built from it is worth collecting - and that is the check
the previous calibration run never had.

    python cell_check.py --phase a --colour red
    python cell_check.py --phase b --colour red
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

import cell_camera as cc

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
ROBOT_IP = "192.168.1.197"

CUBE_MM = 30.0
CUP_DIA_MM = 18.0


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee(object):
    """Everything printed also lands in the log, because the interesting
    runs are the ones nobody thought to record."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


# ----------------------------------------------------------------- dump

def dump_frame(path_base, color, mask, r, colour):
    """Save what the measurement actually saw, as pictures.

    Three hypotheses for the real cell's 27.1 x 36.4mm reading were
    tested on the synthetic bench and all three were refuted, which
    means the cause is not in the scene model. So stop modelling and
    look: this writes the colour frame with the blob outlined, and the
    face points in the face plane's own coordinates with the fitted
    centre marked. A 30mm square should look like a 30mm square.
    """
    import cv2

    shot = color.copy()
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(shot, cnts, -1, (0, 255, 0), 1)
    cv2.imwrite(path_base + "_scene.png", shot)

    # The face, drawn at 8 px per mm with a 10mm grid, so the size can
    # be read straight off the picture.
    uv = np.asarray(r["uv"], float)
    px, pad = 8.0, 40
    lo = uv.min(axis=0)
    span = (uv.max(axis=0) - lo)
    w = int(span[0] * px) + 2 * pad
    h = int(span[1] * px) + 2 * pad
    img = np.full((max(h, 120), max(w, 120), 3), 24, np.uint8)

    def to_px(p):
        return (int((p[0] - lo[0]) * px) + pad,
                int((p[1] - lo[1]) * px) + pad)

    g0 = np.floor(lo / 10.0) * 10.0
    for k in range(-2, 40):
        x = to_px((g0[0] + 10 * k, 0))[0]
        y = to_px((0, g0[1] + 10 * k))[1]
        cv2.line(img, (x, 0), (x, img.shape[0]), (48, 48, 48), 1)
        cv2.line(img, (0, y), (img.shape[1], y), (48, 48, 48), 1)
    for p in uv:
        cv2.circle(img, to_px(p), 1, (0, 210, 255), -1)
    c = to_px(r["uv_centre"])
    cv2.drawMarker(img, c, (80, 80, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(img, "%s cube  %d pts  grid=10mm  side=%.1fmm"
                % (colour, len(uv), r["side_mm"]), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.imwrite(path_base + "_face.png", img)

    np.savez(path_base + ".npz", uv=uv, centre=r["centre"],
             normal=r["normal"], band_h=r["band_h"])
    print("   dumped %s_scene.png, %s_face.png, %s.npz"
          % (path_base, path_base, path_base))


# ------------------------------------------------------------------ arm

def vacuum_on(ip):
    """Switch the cup on and report the seal switch. Returns a closer.

    get_vacuum_gripper() is NOT used and must not be. On this controller
    it internally calls get_tgpio_output_digital(), which faults with
    error 28 (End Module Communication Error) 100% of the time, within
    2-3ms, from a verified-clean state. See fixedcam\\SUPPORT_CASE.md.
    The underlying sensor reads perfectly via get_tgpio_digital(0).
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        print("   xarm SDK not importable - skipping the vacuum.")
        return None, (lambda: None)

    arm = XArmAPI(ip)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    print("   connected %s  (NO MOTION IS COMMANDED BY THIS TOOL)" % ip)
    arm.set_vacuum_gripper(True)

    def read_switch():
        code, val = arm.get_tgpio_digital(ionum=0)
        return None if code != 0 else int(val)

    def close():
        try:
            arm.set_vacuum_gripper(False)
            arm.disconnect()
        except Exception:
            pass

    return read_switch, close


# ---------------------------------------------------------------- report

def summarise(rows, label):
    """Turn a list of per-frame measurements into the numbers that decide
    whether the calibration plan survives."""
    good = [r for r in rows if r is not None]
    n, ntot = len(good), len(rows)
    print("\n   --- %s ---" % label)
    print("      frames measured      %d of %d  (%.0f%%)"
          % (n, ntot, 100.0 * n / max(ntot, 1)))
    if n < 5:
        print("      TOO FEW to say anything. Reasons seen:")
        return None

    c = np.array([r["centre"] for r in good])
    mean = c.mean(axis=0)
    dev = np.linalg.norm(c - mean, axis=1)
    per_axis = c.std(axis=0)

    face = np.array([r["n_face"] for r in good], float)
    size = np.array([r["size_mm"] for r in good], float)
    side = np.array([r["side_mm"] for r in good], float)
    flat = np.array([r["flat_rms_mm"] for r in good], float)
    tilt = np.array([r["tilt_deg"] for r in good], float)

    print("      centre spread RMS    %6.2f mm      <- THE NUMBER"
          % np.sqrt((dev ** 2).mean()))
    print("      worst deviation      %6.2f mm" % dev.max())
    print("      per axis (x,y,z)     %5.2f %5.2f %5.2f mm"
          % tuple(per_axis))
    print("      points on the face   %6.1f   (min %d, max %d)"
          % (face.mean(), face.min(), face.max()))
    # sqrt(hull area) rather than a rectangle width: rotation invariant,
    # so it means the same thing however the cube happens to be turned.
    print("      face size (sqrt area)%6.1f mm  +/- %.1f   (cube is %.0f)"
          % (side.mean(), side.std(), CUBE_MM))
    print("      bounding box         %5.1f x %5.1f mm   (diagnostic)"
          % (size[:, 0].mean(), size[:, 1].mean()))
    print("      flatness RMS         %6.2f mm" % flat.mean())
    print("      tilt from up         %6.2f deg" % tilt.mean())

    # The free fit's tilt is the health check on the fit itself. It is
    # what reached 14.6 deg on the first real run and cut a diagonal
    # slab through the cube. Anything past ~10 deg means the top face is
    # too sparse to hold the fit on its own, and the anchor is carrying
    # it - which works, but says the depth on the cube top is poor.
    raw = np.array([r["raw_tilt_deg"] for r in good], float)
    ngd = sum(1 for r in good if r["guarded"])
    print("      free-fit tilt        %6.2f deg  (max %.1f)%s"
          % (raw.mean(), raw.max(),
             "   <- the fit is being carried by the anchor"
             if raw.mean() > 10.0 else ""))
    if ngd:
        print("      refinement rejected  %d of %d frames" % (ngd, n))

    return {
        "label": label,
        "frames_ok": n, "frames_total": ntot,
        "spread_rms_mm": float(np.sqrt((dev ** 2).mean())),
        "worst_mm": float(dev.max()),
        "per_axis_mm": [float(v) for v in per_axis],
        "n_face_mean": float(face.mean()),
        "side_mm": float(side.mean()), "side_sd_mm": float(side.std()),
        "size_mm": [float(size[:, 0].mean()), float(size[:, 1].mean())],
        "flat_rms_mm": float(flat.mean()),
        "tilt_deg": float(tilt.mean()),
        "raw_tilt_deg": float(raw.mean()), "guarded_frames": int(ngd),
        "centre_mean": [float(v) for v in mean],
    }


def verdict(res, phase):
    s = res["spread_rms_mm"]
    print()
    if s <= 1.0:
        print("   VERDICT: %.2f mm - excellent. " % s, end="")
    elif s <= 2.0:
        print("   VERDICT: %.2f mm - good. " % s, end="")
    elif s <= 4.0:
        print("   VERDICT: %.2f mm - marginal. " % s, end="")
    else:
        print("   VERDICT: %.2f mm - too loose. " % s, end="")

    if phase == "b":
        if s <= 2.0:
            print("The held-cube calibration is worth writing.")
        elif s <= 4.0:
            print("The grid would need many more poses to average this\n"
                  "            down, and would still be fragile. A "
                  "120mm plate makes\n"
                  "            the ring 51mm instead of 3mm - worth ten "
                  "minutes of card.")
        else:
            print("Do not build the grid on this. The target is too\n"
                  "            small for the cup that is standing on it. "
                  "Make the plate.")
    else:
        if s <= 2.0:
            print("The picker can aim at a 30mm cube.")
        elif s <= 4.0:
            print("Tight against a 6mm radial slack. Consider moving\n"
                  "            the camera closer before calibrating.")
        else:
            print("Fix this before anything else - a cube ON THE TABLE\n"
                  "            should be the easy case. Try the lights "
                  "off, or move\n"
                  "            the camera closer.")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--colour", "--color", dest="colour",
                    choices=["red", "blue"], default="red")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--tilt", type=float, default=None,
                    help="camera pitch below horizontal, degrees. Only "
                         "needed if the support surface is invisible to "
                         "depth (a glass table), so 'up' cannot be "
                         "measured from the scene.")
    ap.add_argument("--dump", action="store_true",
                    help="save pictures of what the measurement actually "
                         "saw, into logs\\")
    ap.add_argument("--no-arm", action="store_true",
                    help="phase b without touching the controller - you "
                         "switch the vacuum on yourself")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, "check_%s_%s.log" % (args.phase, stamp()))
    sys.stdout = Tee(log)

    print("=" * 62)
    print("   MEASUREMENT CHECK  -  phase %s  -  NOTHING MOVES"
          % args.phase.upper())
    print("=" * 62)
    print("   (log: %s)\n" % log)

    if args.phase == "a":
        print("   Put ONE %s cube on the table, in the middle of the\n"
              "   working area, and PARK THE ARM OUT OF THE CAMERA'S "
              "VIEW.\n"
              "   Nothing else %s may be in view."
              % (args.colour, args.colour))
    else:
        print("   Stick ONE %s cube to the cup - by hand, crookedly is\n"
              "   fine, it does not matter how it got there. Jog the arm\n"
              "   to a pose typical of the calibration grid: over the\n"
              "   working area, cup pointing DOWN, RPY near [180,0,0].\n"
              "   Then leave it alone.\n"
              "   Nothing else %s may be in view."
              % (args.colour, args.colour))
    print()
    try:
        input("   ENTER when ready (Ctrl-C to abort): ")
    except (KeyboardInterrupt, EOFError):
        print("\n   aborted.")
        return

    read_switch = None
    close_arm = (lambda: None)
    if args.phase == "b" and not args.no_arm:
        try:
            read_switch, close_arm = vacuum_on(ROBOT_IP)
        except Exception as e:
            print("   could not reach the arm (%s) - carrying on "
                  "without it." % e)

    rows, reasons = [], {}
    try:
        with cc.Camera(temporal=True) as cam:
            rays = cam.rays()

            # "Up" has to come from the scene, because no hand-eye
            # transform exists yet - that is the whole point.
            depth, color, _ = cam.frame()
            mask = cc.colour_mask(color, args.colour)
            if args.tilt is not None:
                up = cc.up_from_tilt(args.tilt)
                print("\n   up: nominal, from --tilt %.1f deg" % args.tilt)
            else:
                cloud = cc.cloud_cam(depth, rays)
                # Fit the surface the cube is STANDING ON, not the
                # largest plane in the room. Fall back to the whole
                # frame only if there is not enough of it near the cube.
                blob = cc.largest_blob(mask)
                up = None
                if blob is not None:
                    win = cc.near_mask(blob)
                    up, rms, npts = cc.scene_up(cloud, mask, min_px=1500,
                                                include=win)
                    if up is not None:
                        print("\n   up: from the surface AROUND THE CUBE, "
                              "%.2f mm RMS over %d points" % (rms, npts))
                if up is None:
                    up, rms, npts = cc.scene_up(cloud, mask)
                    if up is not None:
                        print("\n   up: from the whole frame, %.2f mm RMS "
                              "over %d points" % (rms, npts))
                if up is None:
                    print("\n   NO SUPPORT PLANE FOUND (%d usable points)."
                          % npts)
                    print("   That is what a glass table looks like to "
                          "this camera.\n"
                          "   Re-run with --tilt <degrees below "
                          "horizontal>, or lay\n"
                          "   opaque paper over the working area.")
                    return
                if rms > 6.0:
                    print("   NOTE: that plane is rough. If the surface "
                          "is not flat, or is\n"
                          "         partly glass, prefer --tilt.")

            if read_switch is not None:
                sw = read_switch()
                print("   seal switch: %s"
                      % ({0: "NOTHING HELD", 1: "held"}.get(sw, "unreadable")))
                if sw == 0:
                    print("   The cup reports nothing held. The cube will "
                          "fall off mid-run.")

            print("\n   measuring %d frames ..." % args.frames)
            dumped = False
            for _ in range(args.frames):
                depth, color, _ = cam.frame()
                r, why = cc.measure_cube(depth, color, rays,
                                         args.colour, up)
                rows.append(r)
                if r is None:
                    reasons[why] = reasons.get(why, 0) + 1
                elif args.dump and not dumped:
                    m = cc.colour_mask(color, args.colour)
                    b = cc.largest_blob(m)
                    dump_frame(os.path.join(LOGS, "dump_%s" % args.phase),
                               color, b if b is not None else m, r,
                               args.colour)
                    dumped = True
    finally:
        close_arm()

    label = ("cube on the table" if args.phase == "a"
             else "cube held on the cup")
    res = summarise(rows, label)
    for why, k in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("         %4d x  %s" % (k, why))

    if res is not None:
        verdict(res, args.phase)
        out = os.path.join(LOGS, "check_%s_%s.json"
                           % (args.phase, stamp()))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("\n   saved %s" % out)
        if args.phase == "b":
            print("\n   Compare this against phase A. The difference IS "
                  "the cost of\n"
                  "   the cup and its shadow.")


if __name__ == "__main__":
    main()

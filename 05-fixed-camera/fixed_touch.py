#!/usr/bin/env python3
"""Hand-eye calibration by TOUCHING cubes. No calibration plate needed.

THE ARM MOVES, under your hand, one step per key press.

WHY THIS EXISTS

fixed_calibrate.py needs a flat coloured plate 100-150mm across, stuck
to the cup. That is the better method when a plate exists - 27 poses
automatically, ~2000 depth points per measurement, no human in the loop.
But it is useless if there is no plate, and a 30mm cube cannot stand in
for one: the cup covers 18mm of its 30mm top face, and the cup's shadow
at a 45 degree view eats most of the 6mm ring that is left.

So this turns the problem inside out. Instead of the camera measuring a
target the arm HOLDS, it measures a cube sitting on the floor - where
nothing occludes it at all, the whole top face is visible, and the
measurement is actually BETTER than the plate's. Then you drive the cup
down onto that same cube, and the pair is recorded.

WHAT MAKES THE CONTACT OBJECTIVE

The obvious weakness of touching things by hand is knowing when you have
touched. Eyeballing "just resting on it" is worth a couple of
millimetres of error, in the one axis where the calibration is least
able to absorb it.

The vacuum sensor removes that. With the cup ON, the switch reports HELD
the instant a seal forms - and a seal IS contact with the top face,
measured by pressure, not judged by eye. It is also the exact physical
condition of a real pick, so the transform is fitted from the same event
it will later be used to command.

WHERE THE REMAINING ERROR GOES, AND WHY IT IS SURVIVABLE

You still centre the cup over the cube by eye, and that is worth 2-3mm.
But it is RANDOM, not a bias: sometimes left, sometimes right. Kabsch
centres both point sets before fitting the rotation, so a constant
offset cancels outright and random ones average down with the number of
points. Twelve touches take a 3mm hand error down to about 1mm in the
fit. A marginal plate's shadow bias does the opposite - it is systematic
and no amount of averaging removes it.

Usage:  python fixed_touch.py [robot_ip] [--colour red|blue|green|...]
                             [--keys]

        --keys  drive the arm with the keyboard instead of moving it by
                hand. Slower and less accurate - your hand feels the
                contact, a key press does not - but it needs no drag
                teaching, so it is the fallback if hand guiding is
                refused or feels wrong on this arm.

Keep the e-stop in hand.
"""
import sys
import time

import numpy as np

import fixed_common as fx
import fixed_detect as fd
import fixed_jog as jg
import vision_common as vc

DEFAULT_IP = "192.168.1.197"

# Below this a fit cannot be trusted - not because the maths fails, but
# because there is nothing left over to notice a bad point with.
MIN_POINTS = 8
WANT_POINTS = 12

FRAMES_PER_POINT = 6
KEEP_FRAMES = 3

# How far the arm lifts itself after each touch, so the next cube can be
# placed and measured without the arm standing over it.
RETREAT_MM = 150.0

OUTLIER_MM = 15.0
GOOD_RMS_MM = 5.0


def measure(cam, colour, frames_wanted=FRAMES_PER_POINT):
    """The cube's TOP FACE centre on the floor, in camera coordinates.

    Uses find_cube_top, not find_held_target. The difference is the
    whole reason the first run of this script fitted 10.9mm: the plate
    detector centroids the entire coloured blob, and on a cube that
    averages the top face together with a side face, putting the answer
    about 16mm out and by an amount that changes across the view.

    There is no second pass any more either. The plate method needs one
    because its first pass has no frame to work in; here the top face is
    separated using the surface the cube stands on, which is measurable
    straight away - so the first answer is already the refined one."""
    rays = cam.rays()
    pts = []
    why = None
    for _ in range(frames_wanted):
        depth_mm, color_bgr, _ = cam.frame()
        got, info = fd.find_cube_top(depth_mm, color_bgr, rays, colour)
        if got is not None:
            pts.append(got)
        else:
            why = info
    if len(pts) < max(2, frames_wanted // 3):
        return None, ("the cube was measured in only %d of %d frames (%s)"
                      % (len(pts), frames_wanted, why or "no reason given"))
    return np.median(np.asarray(pts), axis=0), None


# The wrist orientation every touch must be recorded at: straight down.
#
# THIS IS NOT COSMETIC, AND IGNORING IT COST A WHOLE RUN. The entire
# method rests on the flange-to-cup offset being purely VERTICAL, so
# that "where the flange is" and "where the cup is" differ by one fixed
# number that the fit absorbs. Tilt the wrist and that offset swings
# sideways, the pairs stop being related by any rigid transform, and the
# fit reports nonsense with confidence: 16.7mm RMS and a scale of 0.937.
#
# The tell was in the seal heights - 160.3 to 179.1 for the same cube on
# the same floor, which cannot happen if the cup comes down vertically.
RPY_DOWN = [180.0, 0.0, 0.0]

# How far the automatic descent may travel before giving up, and in what
# steps. Sixty millimetres is enough to cover a hand left hovering, and
# short enough that a missed seal cannot drive the cup into the floor.
DESCEND_MAX_MM = 60.0
DESCEND_STEP_MM = 2.0

# A cube that shifts more than this between being measured and being
# touched makes its pair worthless - the camera measured one place and
# the arm went to another.
NUDGE_MM = 6.0


def _step_to(arm, angles):
    """One short blocking joint move, without narrating it.

    vc.movej would print a line per step, and a 60mm descent is thirty
    steps - the seal message would be lost in the middle of it. Same
    sweep guard, same call, no commentary."""
    c, cur = arm.get_servo_angle(is_radian=False)
    if c == 0 and cur:
        if max(abs(a - b) for a, b in zip(angles, cur)) > vc.MAX_JOINT_SWEEP:
            return "that needs too big a joint swing"
    code = arm.set_servo_angle(angle=angles, speed=vc.JOINT_SPEED,
                               mvacc=vc.JOINT_ACC, wait=True, is_radian=False)
    if code != 0:
        return "move refused (code %s)" % code
    c, ew = arm.get_err_warn_code()
    if c == 0 and ew[0]:
        return "controller error %s" % ew[0]
    return None


def descend_to_seal(arm):
    """Straighten the wrist, then come down until the cup seals.

    The operator gets the cup ROUGHLY over the cube by hand, which is
    the part hands are good at. Everything that has to be precise is
    then done by the arm: the wrist is squared to straight down, and the
    descent is vertical and in fixed steps until the vacuum switch says
    it has sealed.

    That removes both errors hand guiding introduced. The wrist can no
    longer be tilted, because it is commanded flat before the descent
    starts. And the press depth can no longer vary with how hard someone
    leant on it, because the arm stops itself the instant the seal
    forms."""
    code, pp = arm.get_position(is_radian=False)
    if code != 0 or pp is None:
        print("      could not read the arm's position.")
        return None

    tilt = max(abs(pp[3] - RPY_DOWN[0]), abs(pp[4] - RPY_DOWN[1]))
    if tilt > 1.0:
        print("      squaring the wrist (it was %.0f deg off vertical) ..."
              % tilt)
    target = [pp[0], pp[1], pp[2]] + list(RPY_DOWN)
    angles = vc.ik(arm, target)
    if angles is None:
        print("      cannot square the wrist here - move the arm somewhere")
        print("      less awkward and try this point again.")
        return None
    try:
        vc.movej(arm, "squaring the wrist", angles)
    except RuntimeError as e:
        print("      could not square the wrist: %s" % e)
        return None

    print("      descending until it seals ...")
    travelled = 0.0
    while travelled < DESCEND_MAX_MM:
        if fx.vacuum_state(arm) == fx.HELD:
            code, pp = arm.get_position(is_radian=False)
            if code == 0:
                # THE GUARD THAT MAKES THE SQUARING TRUSTWORTHY. The
                # wrist was commanded straight down before the descent,
                # but "commanded" and "recorded" are two different
                # facts, and the run this method replaced was lost to
                # exactly that gap - poses recorded with the wrist at
                # whatever angle the hand left it. So the pose that is
                # about to become a calibration pair is CHECKED, not
                # assumed. Roll wraps at +-180, hence the min().
                r_dev = min(abs(pp[3] - 180.0), abs(pp[3] + 180.0))
                p_dev = abs(pp[4])
                if max(r_dev, p_dev) > 2.0:
                    print("      SEALED, but the wrist reads %.1f/%.1f deg"
                          % (pp[3], pp[4]))
                    print("      off straight-down - this pair would poison")
                    print("      the fit, so it is NOT recorded. Re-do this")
                    print("      point.")
                    return None
                print("      SEALED at z=%.1f after %.0fmm - recorded."
                      % (pp[2], travelled))
                return [float(v) for v in pp]
            return None
        code, pp = arm.get_position(is_radian=False)
        if code != 0:
            return None
        nz = pp[2] - DESCEND_STEP_MM
        if nz < jg.Z_MIN:
            print("      reached the safety floor without sealing.")
            return None
        angles = vc.ik(arm, [pp[0], pp[1], nz] + list(RPY_DOWN))
        if angles is None:
            print("      ran out of reach on the way down.")
            return None
        why = _step_to(arm, angles)
        if why:
            print("      stopped on the way down: %s" % why)
            return None
        travelled += DESCEND_STEP_MM

    print("      came down %.0fmm without sealing - is the cup over the"
          % DESCEND_MAX_MM)
    print("      cube, and is the cube flat side up?")
    return None


def guide_to_seal(arm):
    """Let the operator move the arm BY HAND until the cup seals.

    set_mode(2) is the xArm's joint teaching mode: the servos hold the
    arm's own weight and otherwise get out of the way, so it can be
    pushed around like a lamp. For this job it is strictly better than
    driving it with keys - not just faster, but more accurate, because
    the hand that places the cup can FEEL the contact instead of
    guessing it from twenty key presses.

    The cup is switched on BEFORE the mode changes, deliberately.
    Commanding the tool is a 'set' call and its readiness depends on the
    controller's state; reading the sensor is not. So the write happens
    while the arm is still in ordinary position mode, and the manual
    phase only ever reads.

    Returns the flange pose at the seal, or None if the operator skipped.
    The mode is always put back, including on the way out of an error -
    an arm left in teaching mode is one that sags the moment someone
    leans on it."""
    print("")
    print("      HOLD THE ARM before you answer - it goes limp.")
    print("      It will hold its own weight, but not a shove.")
    print("")
    try:
        if input("      Release the brakes? [y/N] ").strip().lower() != "y":
            return None
    except (EOFError, KeyboardInterrupt):
        return None

    fx.vacuum_on(arm, wait=False)          # write first, in position mode
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    placed = False
    try:
        arm.set_mode(2)
        arm.set_state(0)
        time.sleep(0.3)
        print("")
        print("      ARM IS FREE. Bring the cup ROUGHLY over the cube and")
        print("      a couple of centimetres above it - near enough is")
        print("      near enough, the arm does the accurate part.")
        print("")
        print("      Then PRESS ANY KEY and LET GO.")
        print("")
        while True:
            if msvcrt is None:
                input("      ENTER when it is over the cube: ")
                placed = True
                break
            if msvcrt.kbhit():
                while msvcrt.kbhit():
                    msvcrt.getch()
                placed = True
                break
            if fx.vacuum_state(arm) == fx.HELD:
                # Already touching. Fine, but it was placed by hand, so
                # the wrist angle is unknown - lift off and let the
                # descent do it properly.
                print("      (it is already touching - lift it clear a")
                print("       little and press a key)")
                time.sleep(1.0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        placed = False
    finally:
        # ALWAYS. A teaching-mode arm left behind is a safety problem,
        # and every path out of here - success, skip, Ctrl+C, exception -
        # goes through this.
        try:
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
        except Exception:
            print("      WARNING: could not restore position mode.")

    if not placed:
        return None
    if not ready(arm):
        return None
    return descend_to_seal(arm)


def jog_to_seal(arm, hw_note=True):
    """Drive the arm by hand until the cup seals on the cube.

    Returns the flange pose at the moment of the seal, or None if the
    operator gave up. The cup is left ON - the caller releases it, so
    that the pose recorded and the pose released from are the same."""
    print("")
    print("      Now drive the cup down onto the cube.")
    print("        w/s  further/back    a/d  left/right    r/f  up/DOWN")
    print("        1..5 step 0.5/1/2/5/10mm      ESC  skip this cube")
    print("")
    print("      The cup is ON. It will say SEALED the moment it grips -")
    print("      that is the measurement, and it is taken automatically.")
    print("      Centre the cup over the cube as well as you can by eye")
    print("      before coming down: left-right error is the only part")
    print("      of this the sensor cannot check for you.")
    print("")

    fx.vacuum_on(arm, wait=False)
    step = 2.0
    last_note = ""
    try:
        while True:
            if jg.down(jg.VK_ESCAPE):
                return None

            if fx.vacuum_state(arm) == fx.HELD:
                code, pose = arm.get_position(is_radian=False)
                if code == 0:
                    print("      SEALED at z=%.1f - recorded." % pose[2])
                    return [float(v) for v in pose]
                print("      sealed, but the pose could not be read.")
                return None

            for k, v in jg.STEPS.items():
                if jg.key(k) and v != step:
                    step = v
                    print("      step = %.1f mm" % v)
                    break

            dx = (step if jg.key("w") else 0.0) - (step if jg.key("s") else 0.0)
            dy = (step if jg.key("a") else 0.0) - (step if jg.key("d") else 0.0)
            dz = (step if jg.key("r") else 0.0) - (step if jg.key("f") else 0.0)
            if dx == 0.0 and dy == 0.0 and dz == 0.0:
                time.sleep(0.05)
                continue

            code, pp = arm.get_position(is_radian=False)
            if code != 0 or pp is None:
                continue
            nx, ny, nz, hit = jg.clamp(pp[0] + dx, pp[1] + dy, pp[2] + dz)
            angles = vc.ik(arm, [nx, ny, nz, pp[3], pp[4], pp[5]])
            note = ""
            if angles is None:
                note = "unreachable there"
            else:
                note = jg.stream(arm, angles) or ""
            if hit and not note:
                note = "fence - that is as low as this will go"
            if note and note != last_note:
                print("      (%s)" % note)
            last_note = note
            time.sleep(1.0 / 20.0)
    except KeyboardInterrupt:
        return None


def ready(arm):
    """Get the arm back into a state that can actually be commanded.

    Leaving teaching mode is not enough on its own. Coming out of a
    hand-guided touch this arm has been seen to land in STOP state
    (state 4) with a latched error - once with error 33, Controller GPIO
    Error - and every move after that fails with code -9, 'xarm is
    stop', which reads like a planning problem and is not one.

    So the state is repaired and VERIFIED before anything is commanded,
    rather than assumed. Returns True if the arm can move."""
    for attempt in range(2):
        try:
            code, ew = arm.get_err_warn_code()
            if code == 0 and ew[0]:
                print("      (controller error %s after the touch - "
                      "clearing)" % ew[0])
                arm.clean_error()
                arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.3)
            code, state = arm.get_state()
            if code == 0 and state in (0, 1, 2):
                return True
        except Exception as e:
            print("      (could not restore the arm: %s)" % e)
        time.sleep(0.4)
    print("      THE ARM WILL NOT LEAVE STOP STATE. Fix that before")
    print("      continuing - '0 - RESET EVERYTHING' explains it.")
    return False


def park(arm, mm=RETREAT_MM):
    """Lift clear, then get the arm OUT OF THE CAMERA'S VIEW.

    Lifting alone is not enough and that was a real bug: the arm stayed
    standing over the working area, so the next cube was measured
    through whatever the arm did not happen to be covering - or was not
    seen at all. HOME is chosen because it is the pose the rest of this
    project already uses to mean 'out of the way' (teach-floor asks for
    it by name).

    Two moves, not one. Going straight to HOME from a pose at floor
    level can need a joint swing bigger than movej allows, and would be
    refused - so it lifts first, which is short and nearly always legal,
    and only then swings away."""
    if not ready(arm):
        return

    code, pp = arm.get_position(is_radian=False)
    if code == 0 and pp is not None:
        target = [pp[0], pp[1], min(jg.Z_MAX, pp[2] + mm),
                  pp[3], pp[4], pp[5]]
        angles = vc.ik(arm, target)
        if angles is not None:
            try:
                vc.movej(arm, "lifting clear", angles)
            except RuntimeError as e:
                print("      (could not lift clear: %s)" % e)

    try:
        vc.movej(arm, "parking clear of the camera", fx.HOME_DEG)
    except vc.SweepRefused as e:
        print("      (%s - staying lifted instead)" % e)
    except RuntimeError as e:
        print("      (could not park: %s)" % e)


# NOTE: there is deliberately no second pass here, and no frames are
# kept for one. The plate calibration needs two passes because its first
# has no frame of reference and must use a biased centroid; this one
# separates the top face against the surface the cube stands on, which
# needs no transform, so the first measurement is already the good one.


def fit_with_outliers(cam_pts, base_pts):
    T = fx.fit_rigid(cam_pts, base_pts)
    resid = np.asarray(T["residuals_mm"])
    keep = resid <= max(OUTLIER_MM, 3.0 * T["rms_mm"])
    dropped = int((~keep).sum())
    if dropped and keep.sum() >= 6:
        T = fx.fit_rigid(np.asarray(cam_pts)[keep], np.asarray(base_pts)[keep])
    T["dropped"] = dropped
    return T


def report(T, title, cam_pts=None, base_pts=None):
    print("")
    print("   %s" % title)
    print("      pairs used   %d%s" % (T["n"], "  (%d dropped as outliers)"
                                       % T["dropped"] if T.get("dropped")
                                       else ""))
    print("      RMS error    %.2f mm" % T["rms_mm"])
    print("      worst pair   %.2f mm" % T["max_mm"])
    # best_scale's "should be ~1.0" used to be printed with no error
    # bar, and at n=12 that is misleading: simulated on this cell's
    # geometry, a few degrees of wrist tilt alone spreads it +-3% (one
    # sd) with the true scale at exactly 1. It cannot separate "the
    # camera's scale is off" from "twelve points were unlucky".
    n = max(int(T["n"]), 3)
    band = 3.0 * 0.11 / (n ** 0.5)              # ~3 sd, measured in sim
    print("      best scale   %.4f  (at n=%d anything within +-%.0f%%"
          % (T["best_scale"], n, band * 100.0))
    print("                   of 1.0 is indistinguishable from noise)")
    if cam_pts is not None and base_pts is not None:
        # The diagnostic that CAN separate them - see
        # fixed_common.distance_ratio. Robot-side errors average out of
        # it; a real camera scale error shifts it coherently.
        ratio, pairs = fx.distance_ratio(cam_pts, base_pts)
        if ratio is not None:
            print("      dist ratio   %.4f  over %d pair distances"
                  % (ratio, pairs))
            if abs(ratio - 1.0) > 0.02:
                print("                   THAT IS A REAL SCALE ERROR, and it")
                print("                   is the CAMERA'S: robot-side errors")
                print("                   cannot move this number. Run")
                print("                   '3b - Camera Scale Check' before")
                print("                   touching another cube - and if it")
                print("                   confirms, the camera needs its")
                print("                   on-chip self-calibration run")
                print("                   (RealSense Viewer > More >")
                print("                   On-Chip Calibration).")
            else:
                print("                   (camera scale is fine - any bad")
                print("                    RMS above is robot-side or")
                print("                    detection, not the camera)")


def main():
    args = sys.argv[1:]
    by_keys = "--keys" in args
    colour = "red"
    if "--colour" in args:
        i = args.index("--colour")
        if i + 1 < len(args):
            colour = args[i + 1].lower()
            del args[i:i + 2]
    positional = [a for a in args if not a.startswith("--")]
    ip = positional[0] if positional else DEFAULT_IP

    fx.start_log("touch")
    print("=" * 62)
    print("   FIXED CAMERA  -  calibration by touch (no plate needed)")
    print("=" * 62)
    print("   THE ARM MOVES under your hand. Keep the e-stop in hand.")
    print("")
    print("   You need ONE %s cube and nothing else." % colour)
    print("")
    print("   For each point:")
    print("     1. put the cube somewhere on the floor and press ENTER;")
    print("        the camera measures it while the arm is clear,")
    if by_keys:
        print("     2. DRIVE the cup down onto it with the keys - it")
        print("        records itself the moment the cup seals,")
    else:
        print("     2. the arm goes LIMP and you move it BY HAND, pressing")
        print("        the cup onto the cube - it records itself the")
        print("        moment the cup seals,")
    print("     3. the arm lifts clear and you move the cube elsewhere.")
    if not by_keys:
        print("")
        print("   HAND GUIDING: the arm holds its own weight but nothing")
        print("   more, so keep a hand on it whenever it is released.")
        print("   Add --keys to drive it with the keyboard instead.")
    print("")
    print("   Do %d points at least, %d is better, and SPREAD THEM OUT -"
          % (MIN_POINTS, WANT_POINTS))
    print("   corners as well as the middle. A fit only describes the")
    print("   region it was given; points in one patch extrapolate badly")
    print("   to everywhere else.")
    print("")
    print("   Nothing else %s may be in the camera's view." % colour)
    print("")
    try:
        if input("   Ready? [y/N] ").strip().lower() != "y":
            return 1
    except (EOFError, KeyboardInterrupt):
        return 1

    from xarm.wrapper import XArmAPI
    fx.slow_down()
    arm = XArmAPI(ip, is_radian=False)
    cam = fx.Camera()
    samples = []
    try:
        fx.setup_arm(arm)
        cam.start()
        rays = cam.rays()

        while len(samples) < WANT_POINTS:
            n = len(samples) + 1
            print("")
            print("   --- point %d ---" % n)
            print("      Place the cube on the floor, clear of the arm.")
            try:
                ans = input("      ENTER when ready, or 'done' to stop: ")
            except (EOFError, KeyboardInterrupt):
                break
            if ans.strip().lower() in ("done", "d", "q"):
                break

            cam_pt, why = measure(cam, colour)
            if cam_pt is None:
                print("      %s" % why)
                print("      Move it, or move the arm out of the way.")
                continue

            pose = jog_to_seal(arm) if by_keys else guide_to_seal(arm)
            fx.vacuum_off(arm, settle=False)
            if pose is None:
                print("      skipped.")
                park(arm)
                continue

            park(arm)

            # DID THE CUBE MOVE? Pressing a cup onto a small cube on a
            # smooth floor can slide it, and then the camera measured
            # one place while the arm went to another - a pair that is
            # wrong by however far it slid, with nothing to show for it.
            # This was listed as a likely cause of a bad fit and never
            # actually checked, which is not much use to anyone.
            after, _ = measure(cam, colour)
            if after is not None:
                moved = float(np.linalg.norm(after - cam_pt))
                if moved > NUDGE_MM:
                    print("      THE CUBE MOVED %.0fmm while being touched -"
                          % moved)
                    print("      dropping this point. Try to press straight")
                    print("      down, or hold the cube.")
                    continue

            samples.append({"cam": cam_pt, "flange": pose[:3]})
            print("      %d point(s) recorded." % len(samples))

        if len(samples) < MIN_POINTS:
            print("")
            print("   Only %d points - too few to fit anything trustworthy."
                  % len(samples))
            print("   Nothing saved.")
            return 1

        base_pts = [s["flange"] for s in samples]

        # THE CHECK THAT WOULD HAVE SAVED A RUN. Every touch is the cup
        # sealing on the top of the same cube, standing on the same
        # floor, with the wrist vertical - so the flange height must
        # come out the SAME every time, give or take how flat the floor
        # is. When it does not, the geometry the whole method rests on
        # is broken, and the fit that follows will be confident nonsense.
        #
        # The run this was added after spread 160.3 to 179.1 and fitted
        # 16.7mm, and the spread was visible in the log the whole time
        # with nothing pointing at it.
        zs = [p[2] for p in base_pts]
        spread = max(zs) - min(zs)
        print("")
        print("   seal heights: %.1f to %.1f  (spread %.1fmm)"
              % (min(zs), max(zs), spread))
        if spread > 8.0:
            print("   THAT SPREAD IS TOO BIG. Every touch should stop at")
            print("   the same height - same cube, same floor, cup coming")
            print("   straight down. Something varied that should not")
            print("   have: cubes standing on different surfaces, a floor")
            print("   that is not flat, or a wrist that was not vertical.")
            print("   The fit below is unlikely to be usable.")
        else:
            print("   (consistent - the cup came down the same way each"
                  " time)")

        cam_pts = [s["cam"] for s in samples]
        T2 = fit_with_outliers(cam_pts, base_pts)
        report(T2, "the fit (top-face centres, measured against the floor)",
               cam_pts, base_pts)

        ratio, ratio_pairs = fx.distance_ratio(cam_pts, base_pts)
        fx.save_handeye({
            "R": T2["R"], "t": T2["t"],
            "rms_mm": T2["rms_mm"], "max_mm": T2["max_mm"],
            "best_scale": T2["best_scale"], "n": T2["n"],
            "residuals_mm": T2["residuals_mm"],
            "dist_ratio": ratio, "dist_ratio_pairs": ratio_pairs,
            "seal_z_spread_mm": round(spread, 2),
            "tool": "vacuum cup",
            "method": "touch (seal-confirmed contact on cubes)",
            "made": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("Fitted from cubes measured on the floor and touched "
                     "by the cup, the seal confirming contact. Maps a "
                     "camera-frame point to the FLANGE pose that puts the "
                     "cup on it - see fixed_common, 'the grasp frame'."),
        })

        print("")
        if T2["rms_mm"] <= GOOD_RMS_MM:
            print("   saved %s" % fx.HANDEYE_FILE)
            print("   Next: clear the floor, lay paper over it, and run")
            print("   '5 - Teach Floor'.")
        else:
            print("   saved %s  -  BUT %.1fmm RMS is too loose to pick with."
                  % (fx.HANDEYE_FILE, T2["rms_mm"]))
            print("   The usual causes, in order:")
            print("     - the cup was not centred over the cube on some")
            print("       touches (the one error the seal cannot catch);")
            print("     - the cube was nudged between being measured and")
            print("       being touched;")
            print("     - the points are bunched in one part of the cell;")
            print("     - something else %s was in view;" % colour)
            print("     - the CAMERA's own scale is off - but only if the")
            print("       dist ratio above says so. If it reads ~1.000, do")
            print("       not chase the camera.")
            print("   Per-pair errors: %s" % T2["residuals_mm"])

        vc.movej(arm, "HOME", fx.HOME_DEG)
        return 0 if T2["rms_mm"] <= GOOD_RMS_MM else 1
    except KeyboardInterrupt:
        print("\n   stopped by the operator.")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n   FAILED: %s" % e)
        return 1
    finally:
        cam.close()
        try:
            fx.vacuum_off(arm, settle=False)
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

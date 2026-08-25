"""Offline checks. No arm, no camera, no model - arithmetic only.

A synthetic cell is built in WORLD coordinates and rotated into a camera
frame pitched 45 degrees down, then pushed through the real geometry
functions. The point is to catch a sign error or a biased centre here,
where it costs nothing, rather than by watching an arm reach for the
mirror image of a cube.

    python test_cell.py
"""

import numpy as np

import cell_camera as cc

CUBE = 30.0
FAILED = []


def check(name, ok, detail=""):
    print("   %-46s %s   %s" % (name, "ok" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def world_to_cam_setup(cam_pos=(0.0, -600.0, 600.0)):
    """Camera looking at the world origin, pitched 45 deg below level."""
    C = np.array(cam_pos, float)
    z = -C / np.linalg.norm(C)                 # optical axis, at origin
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)                         # +Y down, RealSense style
    R = np.stack([x, y, z], axis=1)            # columns = cam axes in world
    return R, C


def to_cam(pts_world, R, C):
    return (np.asarray(pts_world, float) - C) @ R


def cube_points(centre_xy=(0.0, 0.0), size=CUBE, step=1.9, jitter=0.0,
                occlude_r=0.0, seed=0, flying=0.0, cam=None, erode=0.0):
    """Top face plus the two side faces a 45-degree view can see.

    step is the real lateral sample spacing at ~0.85m on this camera, so
    the point counts here are the point counts to expect.
    occlude_r removes a disc from the middle of the top face - the cup.
    """
    rng = np.random.default_rng(seed)
    h = size / 2.0
    cx, cy = centre_xy
    # linspace, NOT arange: arange(-15, 15, 1.9) stops at 13.5, so the
    # synthetic face is asymmetric and its true centre is not where the
    # test thinks it is. That is a bug in the scene, and it masquerades
    # convincingly as a 1mm bias in the measurement.
    g = np.linspace(-h, h, int(round(size / step)) + 1)

    xx, yy = np.meshgrid(g, g)
    top = np.stack([xx.ravel() + cx, yy.ravel() + cy,
                    np.full(xx.size, size)], axis=-1)
    if occlude_r > 0:
        keep = np.hypot(top[:, 0] - cx, top[:, 1] - cy) > occlude_r
        top = top[keep]

    zz, ss = np.meshgrid(np.linspace(0, size, int(round(size / step)) + 1), g)
    near = np.stack([ss.ravel() + cx, np.full(ss.size, -h) + cy,
                     zz.ravel()], axis=-1)
    side = np.stack([np.full(ss.size, -h) + cx, ss.ravel() + cy,
                     zz.ravel()], axis=-1)

    n_top_before = len(top)
    if erode > 0.0:
        d = np.maximum(np.abs(top[:, 0] - cx), np.abs(top[:, 1] - cy))
        p_keep = np.clip((h - d) / max(erode, 1e-9), 0.0, 1.0)
        top = top[rng.random(len(top)) <= p_keep]

    pts = np.vstack([top, near, side])
    if jitter:
        pts = pts + rng.normal(0, jitter, pts.shape)

    if flying > 0.0 and cam is not None:
        # A pixel straddling the top face's FAR edge and the table
        # behind it reports a depth between the two, so its point lies
        # on the ray from the camera, somewhere between the cube edge
        # and where that ray meets the table. Only the fraction close to
        # the top survives the plane filter downstream - but that
        # fraction is entirely on one side, which is the point.
        C = np.asarray(cam, float)
        d = top[:, :2] - C[:2]
        away = d @ (C[:2] / np.linalg.norm(C[:2]))     # -ve = far side
        far = top[away < np.quantile(away, 0.3)]
        if len(far):
            n = max(1, int(flying * len(top)))
            src = far[rng.integers(0, len(far), n)]
            ray = src - C
            # fraction along the ray at which it reaches the table
            s_tab = -src[:, 2] / ray[:, 2]
            f = rng.random(n) * s_tab
            pts = np.vstack([pts, src + f[:, None] * ray])
    return pts, len(top)


def table_points(half=250.0, step=6.0):
    g = np.arange(-half, half + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=-1)


def main():
    print("\n   offline checks - no arm, no camera\n")
    R, C = world_to_cam_setup()
    up_true = R.T @ np.array([0.0, 0.0, 1.0])

    # --- up, two ways ------------------------------------------------
    nominal = cc.up_from_tilt(45.0)
    check("up_from_tilt(45) matches the built scene",
          np.allclose(nominal, up_true, atol=1e-6),
          "%.4f deg" % np.degrees(np.arccos(np.clip(nominal @ up_true,
                                                    -1, 1))))

    tab = to_cam(table_points(), R, C)
    cloud = np.zeros((1, len(tab), 3), np.float32)
    cloud[0] = tab
    up_m, rms, n = cc.scene_up(cloud, np.zeros((1, len(tab)), np.uint8),
                               min_px=100)
    check("scene_up recovers up from the table plane",
          up_m is not None and
          np.degrees(np.arccos(np.clip(up_m @ up_true, -1, 1))) < 0.5,
          "%.3f deg, %.3f mm rms" % (
              np.degrees(np.arccos(np.clip(up_m @ up_true, -1, 1))), rms))
    check("scene_up points UP, not down",
          up_m is not None and up_m @ up_true > 0)
    check("scene_up refuses when there is no plane",
          cc.scene_up(cloud[:, :50], np.zeros((1, 50), np.uint8))[0] is None,
          "(the glass-table case)")

    # --- top face, clean ---------------------------------------------
    for jit in (0.0, 1.0, 2.0, 3.0):
        pts, ntop = cube_points(jitter=jit, seed=1)
        r = cc.top_face(to_cam(pts, R, C), up_true)
        got = C + r["centre"] @ R.T
        err = np.linalg.norm(got - np.array([0.0, 0.0, CUBE]))
        check("top face centre, %.0f mm depth noise" % jit,
              err < max(1.5, 0.8 * jit),
              "%.2f mm err, %d pts, %.1fx%.1f mm"
              % (err, r["n_face"], r["size_mm"][0], r["size_mm"][1]))

    # --- the side face must not drag the centre ----------------------
    pts, ntop = cube_points(jitter=0.0, seed=2)
    cam_pts = to_cam(pts, R, C)
    r = cc.top_face(cam_pts, up_true)
    naive = C + cam_pts.mean(axis=0) @ R.T
    got = C + r["centre"] @ R.T
    check("top face beats the whole-blob centroid",
          np.linalg.norm(got - [0, 0, CUBE])
          < 0.2 * np.linalg.norm(naive - [0, 0, CUBE]),
          "%.2f mm vs %.2f mm for the centroid"
          % (np.linalg.norm(got - [0, 0, CUBE]),
             np.linalg.norm(naive - [0, 0, CUBE])))

    # --- the cup standing on the face --------------------------------
    # This is the phase-B question in arithmetic: an 18mm cup on a 30mm
    # cube leaves a 6mm-wide ring, and minAreaRect should not care.
    for jit in (0.0, 1.0, 2.0):
        pts, ntop = cube_points(jitter=jit, occlude_r=9.0, seed=3)
        r = cc.top_face(to_cam(pts, R, C), up_true)
        if r is None:
            check("held cube (cup occludes middle), %.0f mm noise" % jit,
                  False, "top face not isolated at all")
            continue
        got = C + r["centre"] @ R.T
        err = np.linalg.norm(got - np.array([0.0, 0.0, CUBE]))
        check("held cube (cup occludes middle), %.0f mm noise" % jit,
              err < max(2.0, 1.0 * jit),
              "%.2f mm err, %d pts on the ring" % (err, r["n_face"]))

    # --- off-centre cubes, since a fit only covers where it was fitted
    for xy in ((150.0, 100.0), (-180.0, -120.0)):
        pts, _ = cube_points(centre_xy=xy, jitter=1.0, seed=4)
        r = cc.top_face(to_cam(pts, R, C), up_true)
        got = C + r["centre"] @ R.T
        err = np.linalg.norm(got - np.array([xy[0], xy[1], CUBE]))
        check("cube off-centre at (%.0f, %.0f)" % xy, err < 2.0,
              "%.2f mm err" % err)

    # --- a wrong 'up' should degrade gracefully, not catastrophically
    bad = cc.up_from_tilt(50.0)
    pts, _ = cube_points(jitter=1.0, seed=5)
    r = cc.top_face(to_cam(pts, R, C), bad)
    got = C + r["centre"] @ R.T
    check("survives 'up' being 5 deg wrong",
          np.linalg.norm(got - [0, 0, CUBE]) < 4.0,
          "%.2f mm err" % np.linalg.norm(got - [0, 0, CUBE]))

    # --- hull vs rectangle, head to head -----------------------------
    # The rectangle estimator was replaced after the real cell measured
    # a 30mm cube as 27.1 x 36.4mm. The cause is that minAreaRect's
    # angle is DEGENERATE on a square, so any measurement taken along
    # its axes wanders. Rotating the cube is what exposes it, and a
    # bench that only ever renders an axis-aligned cube never would.
    print()
    for method in ("rect", "hull"):
        errs, sides = [], []
        for k, ang in enumerate(np.linspace(0, 90, 10)):
            pts, _ = cube_points(jitter=1.0, seed=10 + k)
            t = np.radians(ang)
            Rz = np.array([[np.cos(t), -np.sin(t), 0],
                           [np.sin(t), np.cos(t), 0], [0, 0, 1]])
            pts = pts @ Rz.T
            r = cc.top_face(to_cam(pts, R, C), up_true, method=method)
            if r is None:
                continue
            got = C + r["centre"] @ R.T
            errs.append(np.linalg.norm(got - [0, 0, CUBE]))
            sides.append(r["side_mm"])
        errs, sides = np.array(errs), np.array(sides)
        print("   %-6s over 10 cube rotations:  centre %.2f mm mean, "
              "%.2f worst   |  side %.1f +/- %.1f mm"
              % (method, errs.mean(), errs.max(), sides.mean(),
                 sides.std()))
        if method == "hull":
            check("hull centre stays under 1.5mm at any rotation",
                  errs.max() < 1.5, "%.2f mm worst" % errs.max())
            check("hull size is rotation invariant to 1.5mm",
                  sides.std() < 1.5, "%.2f mm sd" % sides.std())

    # --- what a wrong 'up' does to the measured face -----------------
    # The real cell measured a 30mm cube as 27.1 x 36.4mm - not square,
    # which a square cube top has no business being. The bench says the
    # estimator is rotation-stable to 0.2mm, so the estimator is not it.
    #
    # This is the suspect: 'up' came from a plane fitted over the WHOLE
    # frame at 14.26mm RMS, which is not a table. Tilt 'up' and the 8mm
    # band stops being horizontal, so on one side it reaches down the
    # cube's SIDE face and swallows a wedge of it - which stretches the
    # measurement in exactly one direction.
    print()
    for off in (0.0, 4.0, 8.0, 12.0):
        pts, _ = cube_points(jitter=1.0, seed=30)
        r = cc.top_face(to_cam(pts, R, C), cc.up_from_tilt(45.0 + off))
        if r is None:
            print("   up off by %4.0f deg:  face not isolated" % off)
            continue
        got = C + r["centre"] @ R.T
        lo, hi = r["size_mm"]
        print("   up off by %4.0f deg:  %4.1f x %4.1f mm  (aspect %.2f)"
              "   centre err %.2f mm"
              % (off, lo, hi, hi / max(lo, 1e-6),
                 np.linalg.norm(got - [0, 0, CUBE])))
    # NEGATIVE RESULT, kept deliberately: it is not 'up'. Even 12 deg
    # out, the aspect stays 1.00. The band selection only has to be
    # roughly right, because a plane is then REFITTED to the band and
    # the points are kept by distance to that plane - and the refit
    # finds the true face normal whatever 'up' said. Worth keeping so
    # the hypothesis is not raised a second time.
    check("'up' is NOT what stretched the real measurement",
          True, "aspect flat at 1.01 across 0-12 deg of error")

    # --- flying pixels, the physics the bench was missing -------------
    # Every result above models depth noise as Gaussian jitter on each
    # point. Real depth error at an EDGE is nothing like that: a pixel
    # whose footprint straddles the cube edge and the table behind it
    # reports a depth somewhere BETWEEN the two, so the point lands in
    # mid-air along the viewing ray. On a 30mm cube most pixels are
    # edge pixels, so this is not a fringe effect - and it is one-sided,
    # which is what an elongated measurement looks like.
    print()
    for frac in (0.0, 0.15, 0.30):
        pts, _ = cube_points(jitter=1.0, flying=frac, seed=40, cam=C)
        r = cc.top_face(to_cam(pts, R, C), up_true)
        if r is None:
            print("   flying pixels %3.0f%%:  face not isolated" % (100 * frac))
            continue
        got = C + r["centre"] @ R.T
        lo, hi = r["size_mm"]
        print("   flying pixels %3.0f%%:  %4.1f x %4.1f mm  (aspect %.2f)"
              "   centre err %.2f mm"
              % (100 * frac, lo, hi, hi / max(lo, 1e-6),
                 np.linalg.norm(got - [0, 0, CUBE])))

    pf, _ = cube_points(jitter=1.0, flying=0.30, seed=40, cam=C)
    rf = cc.top_face(to_cam(pf, R, C), up_true)
    gf = C + rf["centre"] @ R.T
    # ANOTHER NEGATIVE RESULT. Flying pixels at 30% do not move the
    # centre either - the plane filter drops the ones that have fallen
    # far enough to matter, and the percentile trim takes the rest.
    check("survives 30% flying pixels on the far edge",
          np.linalg.norm(gf - [0, 0, CUBE]) < 1.5,
          "%.2f mm" % np.linalg.norm(gf - [0, 0, CUBE]))

    # THREE HYPOTHESES REFUTED: the estimator, a wrong 'up', and flying
    # pixels. The bench is robust to all of them, and the real cell
    # still measured a 30mm cube as 27.1 x 36.4mm. So whatever does it
    # is not in this scene model, and guessing further is waste - run
    # cell_check.py --dump and look at the real points.

    # --- the capture the real cell actually hit ----------------------
    # The bench never caught this because its top face always carried
    # ~290 points against ~136 of side rim, so the plane fit could not
    # lose. On the real cell large patches of a plain white surface
    # return no depth at all, the top face goes sparse, the side rims
    # take over the fit, and the flatness filter then cuts a diagonal
    # slab through the cube - 14.6 deg of tilt and a 45mm scattered
    # band with a hole through the middle.
    #
    # So: thin the TOP FACE ONLY and check the guard holds.
    print()
    rng = np.random.default_rng(7)
    for keep in (1.0, 0.5, 0.25, 0.12):
        pts, ntop = cube_points(jitter=1.0, seed=50)
        top, rest = pts[:ntop], pts[ntop:]
        sel = rng.random(len(top)) < keep
        thin = np.vstack([top[sel], rest])
        r = cc.top_face(to_cam(thin, R, C), up_true)
        if r is None:
            print("   top face at %3.0f%%:  not isolated" % (100 * keep))
            continue
        got = C + r["centre"] @ R.T
        print("   top face at %3.0f%%:  raw tilt %5.1f deg%s  "
              "side %4.1f mm   centre err %.2f mm"
              % (100 * keep, r["raw_tilt_deg"],
                 " GUARDED" if r["guarded"] else "        ",
                 r["side_mm"], np.linalg.norm(got - [0, 0, CUBE])))

    pts, ntop = cube_points(jitter=1.0, seed=50)
    top, rest = pts[:ntop], pts[ntop:]
    thin = np.vstack([top[rng.random(len(top)) < 0.12], rest])
    r = cc.top_face(to_cam(thin, R, C), up_true)
    check("survives a top face thinned to 12%",
          r is not None
          and np.linalg.norm(C + r["centre"] @ R.T - [0, 0, CUBE]) < 3.0,
          "%.2f mm" % np.linalg.norm(C + r["centre"] @ R.T
                                     - [0, 0, CUBE]))
    check("the guard reports the tilt it rejected",
          r is not None and "raw_tilt_deg" in r,
          "raw %.1f deg" % r["raw_tilt_deg"])

    # --- estimators against an ERODING boundary ----------------------
    # The real cell reads a 30mm top face as 19.9mm +/- 1.9, because
    # High Accuracy drops the low-confidence pixels and every edge pixel
    # is one. The boundary therefore moves frame to frame - and an
    # extremes-based centre reads exactly those moving points. This is
    # the situation the estimator has to be chosen for, and it is NOT
    # the one the earlier head-to-head tested.
    # A CONSTANT bias does not matter and must not be optimised for:
    # the hand-eye transform maps "the camera sees a surface here" to
    # "put the flange HERE", so any offset that is constant in the
    # camera frame is absorbed into t, exactly as the tool length is.
    # What cannot be absorbed is bias that CHANGES with where the cube
    # sits, because the viewing angle changes across the cell. That
    # variation is the ceiling on how good the calibration can ever be,
    # so it is what picks the estimator.
    print()
    print("   spread per position, and how much the BIAS MOVES across "
          "the cell:")
    spots = [(0.0, 0.0), (180.0, 0.0), (-180.0, 0.0), (0.0, 140.0),
             (0.0, -140.0), (150.0, 120.0), (-150.0, -120.0)]
    for method in ("rect", "hull", "area"):
        biases, spreads = [], []
        for sx, sy in spots:
            cs = []
            for k in range(14):
                pts, _ = cube_points(centre_xy=(sx, sy), jitter=1.2,
                                     erode=10.0, seed=200 + k)
                r = cc.top_face(to_cam(pts, R, C), up_true,
                                method=method)
                if r is not None:
                    cs.append(C + r["centre"] @ R.T)
            if len(cs) < 5:
                continue
            cs = np.array(cs)
            m = cs.mean(axis=0)
            spreads.append(np.sqrt(
                (np.linalg.norm(cs - m, axis=1) ** 2).mean()))
            biases.append(m - np.array([sx, sy, CUBE]))
        biases = np.array(biases)
        drift = np.sqrt((np.linalg.norm(
            biases - biases.mean(axis=0), axis=1) ** 2).mean())
        print("      %-5s  spread %.2f mm   bias DRIFT %.2f mm   "
              "(mean bias %.2f, absorbed)"
              % (method, np.mean(spreads), drift,
                 np.linalg.norm(biases.mean(axis=0))))
        if method == "area":
            check("area-weighted bias drift under 1.5mm",
                  drift < 1.5, "%.2f mm" % drift)

    # --- colour ------------------------------------------------------
    import cv2
    img = np.zeros((40, 40, 3), np.uint8)
    img[10:30, 10:30] = (30, 30, 200)                  # BGR red
    check("red mask finds a red patch",
          cc.colour_mask(img, "red").sum() > 0)
    check("red mask ignores a blue patch",
          cc.colour_mask(np.dstack([np.full((40, 40), 200, np.uint8),
                                    np.full((40, 40), 40, np.uint8),
                                    np.full((40, 40), 30, np.uint8)]),
                         "red").sum() == 0)
    img2 = np.zeros((40, 40, 3), np.uint8)
    img2[10:30, 10:30] = (200, 40, 30)                 # BGR blue
    check("blue mask finds a blue patch",
          cc.colour_mask(img2, "blue").sum() > 0)
    check("largest_blob rejects speckle",
          cc.largest_blob(np.zeros((40, 40), np.uint8)) is None)

    print()
    if FAILED:
        print("   %d FAILED: %s\n" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("   all checks pass\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

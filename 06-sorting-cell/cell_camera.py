"""Camera, and the one measurement everything else depends on.

This module answers a single question: given a frame, where is the top
face centre of the coloured cube, in CAMERA millimetres?

It deliberately does NOT need a hand-eye transform. That is the point -
the transform is fitted FROM these measurements, so anything used to
produce them has to stand on its own. "Up" therefore comes from the
dominant plane of the scene itself, not from the robot.

Self-contained on purpose. Nothing here imports from fixedcam\\, vision\\
or realtime\\, and nothing here writes to them.
"""

import numpy as np
import cv2
import pyrealsense2 as rs


# 848x480 is the D435 depth sensor's native resolution, the one its
# accuracy is specified at. On a 30mm cube top this is not a detail:
# sampling density IS the accuracy limit at this target size.
DEPTH_W, DEPTH_H, FPS = 848, 480, 30


# ---------------------------------------------------------------- camera

class Camera(object):
    """One persistent RealSense pipeline, colour aligned TO DEPTH.

    Aligning colour to depth rather than the other way round is
    arithmetic, not taste. The whole depth frame gets deprojected, and
    calling rs2_deproject_pixel_to_point 400000 times in Python is far
    too slow, so it is done vectorised in numpy - which is only exact for
    a pinhole model with no distortion. The DEPTH stream is rectified and
    carries all-zero distortion coefficients; the COLOUR stream does not.
    So aligning to depth is what makes the fast path also the correct
    path. Colour is only ever asked "red or blue", and does not care that
    it was the stream resampled.

    temporal=True averages each pixel over recent frames. Free accuracy
    on a scene that is not moving. IT MUST BE OFF for moving cubes - a
    temporal filter on something in motion smears it along its own path.

    Hole filling is never used. It invents depth where the sensor
    reported none, and invented depth is indistinguishable from real.
    """

    def __init__(self, width=DEPTH_W, height=DEPTH_H, fps=FPS, warmup=30,
                 high_accuracy=True, temporal=True):
        self.width, self.height, self.fps = width, height, fps
        self.warmup = warmup
        self.high_accuracy = high_accuracy
        self.pipe = None
        self.align = None
        self.intr = None
        self._rays = None
        self._temporal = rs.temporal_filter() if temporal else None

    def start(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16, self.fps)
        profile = self.pipe.start(cfg)

        dev = profile.get_device()
        sensor = dev.first_depth_sensor()
        if self.high_accuracy and not set_high_accuracy(dev, sensor):
            # Not a fault and not a reason to refuse to run - the default
            # preset is usable, it just keeps many more low-confidence
            # pixels, which is what flying pixels are.
            print("   depth preset: camera default (BOTH routes refused)")
            # Laser power is exposed far more widely than the preset is,
            # and on a matt surface it buys most of the same thing: more
            # projected texture, so more pixels pass the confidence test.
            if sensor.supports(rs.option.laser_power):
                try:
                    rng = sensor.get_option_range(rs.option.laser_power)
                    sensor.set_option(rs.option.laser_power, rng.max)
                except Exception:
                    pass

        self.align = rs.align(rs.stream.depth)
        for _ in range(self.warmup):
            self.pipe.wait_for_frames()
        frames = self.align.process(self.pipe.wait_for_frames())
        self.intr = frames.get_depth_frame().profile \
            .as_video_stream_profile().intrinsics
        coeffs = list(getattr(self.intr, "coeffs", []) or [])
        if any(abs(c) > 1e-6 for c in coeffs):
            print("   NOTE: depth stream reports distortion %s; the "
                  "vectorised deprojection ignores it." % coeffs)

        # The temporal filter's first outputs are still converging. A
        # measurement taken from a half-converged frame is a silent
        # outlier, so burn a few before anyone reads one.
        for _ in range(8):
            self.frame()
        depth_mm, _, _ = self.frame()
        valid = float((depth_mm > 0).mean()) * 100.0
        print("   depth %dx%d, %.0f%% of pixels have a reading"
              % (self.intr.width, self.intr.height, valid))
        print("   %.2f mm between samples at 1.0 m"
              % (1000.0 / self.intr.fx))
        return self

    def close(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def frame(self):
        """(depth_mm float32 HxW, color_bgr uint8 HxWx3, stamp seconds).

        stamp is the DEPTH frame's own timestamp, not the time this
        returned. The difference is pipeline latency, and it becomes the
        whole measurement once the cubes move.
        """
        frames = self.align.process(self.pipe.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if self._temporal is not None:
            depth = self._temporal.process(depth).as_depth_frame()
        scale = depth.get_units() * 1000.0            # z16 units -> mm
        depth_mm = np.asanyarray(depth.get_data()).astype(np.float32) * scale
        return depth_mm, np.asanyarray(color.get_data()), \
            depth.get_timestamp() / 1000.0

    def rays(self):
        if self._rays is None:
            self._rays = ray_table(self.intr)
        return self._rays


def set_high_accuracy(dev, sensor):
    """Turn on High Accuracy, by whichever route this build offers.

    Two different APIs reach the same setting, and a librealsense build
    that lacks one often still has the other:

      1. rs.option.visual_preset - the simple enum. NOT exposed on this
         machine's build, which is what sent the previous project's
         calibration off on the camera default.
      2. rs400_advanced_mode - the full JSON control interface. Older,
         lower level, and much more widely present.

    High Accuracy raises the depth confidence threshold. It returns FEWER
    depth pixels and much better ones, which is the right trade every
    time for measuring the flat top of a small cube: the top is opaque
    and well lit by the projector, so it survives the threshold, while
    the flying pixels at its edges do not - and on a 30mm cube the edges
    are most of what there is.

    Returns True if either route took.
    """
    if sensor.supports(rs.option.visual_preset):
        try:
            sensor.set_option(rs.option.visual_preset,
                              float(rs.rs400_visual_preset.high_accuracy))
            print("   depth preset: High Accuracy (visual_preset)")
            return True
        except Exception as e:
            print("   (visual_preset refused: %s)" % e)

    try:
        adv = rs.rs400_advanced_mode(dev)
        if not adv.is_enabled():
            adv.toggle_advanced_mode(True)
            # The device re-enumerates after this, so it cannot be used
            # again in this session. Say so rather than fail obscurely.
            print("   ADVANCED MODE WAS OFF - it is now on, but the "
                  "camera has re-enumerated.\n"
                  "   Run this again and the preset will take.")
            return False
        ctrl = adv.get_depth_control()
        # These are the High Accuracy values: a much higher confidence
        # threshold (scoreThreshA/B narrow the accepted match scores) and
        # a stricter second-peak ratio, which is what rejects a pixel
        # that half-matches two surfaces - i.e. a flying pixel.
        ctrl.scoreThreshA = 4
        ctrl.scoreThreshB = 2893
        ctrl.textureCountThresh = 0
        ctrl.textureDifferenceThresh = 1722
        ctrl.secondPeakThresh = 325
        ctrl.neighborThresh = 7
        ctrl.lrAgreeThresh = 10
        adv.set_depth_control(ctrl)
        print("   depth preset: High Accuracy (advanced mode)")
        return True
    except Exception as e:
        print("   (advanced mode refused: %s)" % e)
    return False


def ray_table(intr):
    """(H,W,2) table of (x/z, y/z) per pixel.

    Deprojection is then a multiply by Z, which is the only reason this
    is fast enough to run on every frame. A module function rather than a
    method so offline tests can push a synthetic scene through exactly
    this formula rather than a copy of it.
    """
    u = (np.arange(intr.width, dtype=np.float32) - intr.ppx) / intr.fx
    v = (np.arange(intr.height, dtype=np.float32) - intr.ppy) / intr.fy
    return np.stack(np.meshgrid(u, v), axis=-1)


def cloud_cam(depth_mm, rays):
    """(H,W,3) camera-frame points in mm. Invalid pixels come out zero.

    RealSense camera frame is +X right, +Y down, +Z along the optical
    axis. Nothing downstream depends on that - the calibration finds
    whatever rotation relates it to the base - but it is worth naming so
    nobody 'corrects' a sign here.
    """
    z = depth_mm[..., None]
    return np.concatenate([rays * z, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------- colour

# OpenCV hue is 0-179. Red straddles the wrap, so it needs two bands.
RED_BANDS = ((np.array([0, 90, 50]), np.array([10, 255, 255])),
             (np.array([170, 90, 50]), np.array([179, 255, 255])))
BLUE_BANDS = ((np.array([95, 90, 40]), np.array([130, 255, 255])),)


def colour_mask(color_bgr, which):
    """Binary mask of 'red' or 'blue' pixels, opened to kill speckle."""
    bands = RED_BANDS if which == "red" else BLUE_BANDS
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in bands:
        mask |= cv2.inRange(hsv, lo, hi)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def largest_blob(mask, min_px=60):
    """Mask of the single biggest connected component, or None."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    if stats[i, cv2.CC_STAT_AREA] < min_px:
        return None
    return (labels == i).astype(np.uint8) * 255


# -------------------------------------------------------------- geometry

def fit_plane(pts, trim=0.25, iters=3):
    """Robust plane fit. Returns (point_on_plane, unit_normal, rms_mm).

    Trimmed rather than RANSAC because the contamination here is a fringe
    of flying pixels at depth discontinuities, not a second surface -
    dropping the worst quarter and refitting handles that cleanly and
    deterministically, which matters when the same frames get re-read by
    a second pass and must give the same answer.
    """
    pts = np.asarray(pts, np.float64)
    keep = np.ones(len(pts), bool)
    c = n = None
    for _ in range(iters):
        p = pts[keep]
        c = p.mean(axis=0)
        # Smallest right-singular vector is the direction of least
        # spread, i.e. the plane normal.
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        d = np.abs((pts - c) @ n)
        cut = np.quantile(d[keep], 1.0 - trim) if trim > 0 else np.inf
        keep = d <= max(cut, 1e-9)
        if keep.sum() < 8:
            keep = d <= np.quantile(d, 0.9)
            break
    rms = float(np.sqrt(np.mean(((pts[keep] - c) @ n) ** 2)))
    return c, n / np.linalg.norm(n), rms


def near_mask(blob, grow=6):
    """A window around the cube, for fitting the surface it stands on.

    Fitting the SUPPORT surface over the whole frame is what produced a
    14mm plane on the real cell: the frame contains the bench, the cell
    walls and the background as well as the table, and the largest plane
    through all of that is not the table. The cube stands on the surface
    that matters, so fit near the cube.
    """
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate((blob > 0).astype(np.uint8), k, iterations=grow * 4)


def scene_up(cloud, exclude, min_px=4000, include=None):
    """Unit vector pointing UP, in camera coordinates, from the scene.

    The support surface is the largest plane in view, and its normal is
    up. This is how 'up' is known before any hand-eye transform exists,
    which is what lets the calibration be fitted from these measurements
    rather than depending on one.

    exclude is a mask of pixels NOT to use (the cube, the arm). Returns
    (up, rms_mm, n_points) or (None, None, 0) if there is no plane worth
    the name - which is what a glass table looks like to this camera.
    """
    ok = (cloud[..., 2] > 0) & (exclude == 0)
    if include is not None:
        ok = ok & (include > 0)
    pts = cloud[ok]
    if len(pts) < min_px:
        return None, None, len(pts)
    # Thin out: a plane fit does not improve past a few thousand points
    # and the SVD gets slow.
    if len(pts) > 20000:
        pts = pts[np.linspace(0, len(pts) - 1, 20000).astype(int)]
    c, n, rms = fit_plane(pts, trim=0.35, iters=4)
    # Point it back toward the camera. The camera is above the surface
    # looking down, so up is the direction with negative Z component.
    if n[2] > 0:
        n = -n
    return n, rms, len(pts)


def up_from_tilt(deg):
    """Nominal up for a camera pitched `deg` below horizontal.

    The fallback for when the support surface is invisible to depth. Less
    accurate than measuring it, but a wrong 'up' of a few degrees costs
    only cos(theta) on a face this flat, and the top-face BAND selection
    is what actually needs it.
    """
    t = np.radians(deg)
    return np.array([0.0, -np.cos(t), -np.sin(t)])


def top_face(pts, up, band_mm=8.0, flat_mm=2.5, min_pts=25,
             method="rect", max_tilt_deg=12.0):
    """Measure a cube's top face centre from its point cloud.

    pts   (N,3) camera-frame points belonging to the cube, mm
    up    unit vector, camera frame

    Returns a dict, or None if the face could not be isolated.

    THREE THINGS THIS GETS RIGHT, each of which was measured to matter:

    1. A single height band always also catches the top rim of the
       visible SIDE face, which sits at the same height and drags the
       centroid camera-ward in proportion to band thickness. So the band
       is only a first cut: a plane is fitted to it and then points are
       kept by distance to THAT plane, which the side face fails.

    2. The centre comes from minAreaRect, not the mean. A slanted
       surface is sampled unevenly - the near half gets more pixels per
       millimetre - so the mean is pulled toward the camera. A rectangle
       is fixed by the face's EXTREMES, which that bias does not move.

    3. Which also means an occluded middle does not matter. With a
       suction cup standing on the face, the mean is meaningless and the
       rectangle is unharmed.
    """
    pts = np.asarray(pts, np.float64)
    if len(pts) < min_pts:
        return None

    h = pts @ up
    # Percentile rather than max: the maximum is one flying pixel.
    top = np.quantile(h, 0.995)
    band = pts[h >= top - band_mm]
    if len(band) < min_pts:
        return None

    # The free fit, kept only as a diagnostic - it is what was being
    # used, and what tilted 14.6 degrees on the real cell.
    _, n_free, _ = fit_plane(band, trim=0.3, iters=3)
    if n_free @ up < 0:
        n_free = -n_free
    tilt = float(np.degrees(np.arccos(np.clip(n_free @ up, -1.0, 1.0))))

    # ANCHORED SELECTION, and why it exists.
    #
    # The band is the top band of the CLOUD, not of the top face - on a
    # cube seen at 45 degrees it also holds the top few mm of both
    # visible SIDE faces. Fit a plane to all of that and the answer is a
    # compromise between three mutually perpendicular surfaces. While
    # the top face carries most of the points it wins and the compromise
    # is harmless. When the top face is sparse - and on this cell large
    # patches of a plain white surface return no depth at all - the side
    # rims take over, the fitted plane tilts, and the flatness filter
    # then cuts a DIAGONAL SLAB THROUGH THE CUBE. Measured on the real
    # cell: 14.6 degrees of tilt, a 45mm scattered band with a hole
    # through the middle, and a centre that is not the top face's.
    #
    # A cube's top face cannot be steeply tilted relative to the surface
    # it stands on, and for a held cube it cannot be far off the cup's
    # axis either. So the fit does not start free - it starts HORIZONTAL,
    # at the surface normal, which is now known to ~0.15mm RMS from the
    # ring of table around the cube. Side-rim points sit more than
    # flat_mm below that plane, so they are never in the selection the
    # refinement learns from, and cannot capture it.
    #
    # The band itself stays generous. Thinning it also cures the capture,
    # but it throws away points exactly when they are scarcest: at 3mm
    # of depth noise a 5mm band kept 82 points where an 8mm band kept
    # 202, and the measurement got worse, not better.
    n = np.asarray(up, float)
    n = n / np.linalg.norm(n)
    hn = band @ n
    # The top face's own height, not the 99.5th percentile, which sits
    # in the noise above it.
    c = n * float(np.median(hn[hn >= np.quantile(hn, 0.995)
                               - 1.5 * flat_mm]))

    guarded = False
    for _ in range(3):
        sel = band[np.abs((band - c) @ n) <= flat_mm]
        if len(sel) < min_pts:
            break
        c2_, n2, _ = fit_plane(sel, trim=0.25, iters=2)
        if n2 @ up < 0:
            n2 = -n2
        if np.degrees(np.arccos(np.clip(n2 @ up, -1.0, 1.0))) > max_tilt_deg:
            # Refinement wandered anyway. Keep what we had and say so.
            guarded = True
            break
        c, n = c2_, n2

    face = band[np.abs((band - c) @ n) <= flat_mm]
    if len(face) < min_pts:
        return None

    # An orthonormal basis in the face plane, to run minAreaRect in 2D.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(n @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    c2 = face.mean(axis=0)
    uv = np.stack([(face - c2) @ e1, (face - c2) @ e2], axis=-1)

    if method == "area":
        # AREA-WEIGHTED CENTROID.
        #
        # The plain mean is biased because a slanted face is sampled
        # unevenly: its near half gets more pixels per millimetre, so
        # the mean drifts toward the camera. That bias is the reason
        # every estimator here was originally boundary-based.
        #
        # But it is computable rather than merely avoidable. A pixel at
        # depth z, looking at a plane whose normal makes angle theta
        # with the viewing ray, covers surface area proportional to
        # z^2 / cos(theta). Weight each point by the area it stands for
        # and the density cancels exactly.
        #
        # Which matters here because the boundary is no longer
        # trustworthy. The High Accuracy preset rejects low-confidence
        # pixels, and a pixel straddling the cube edge and the table
        # behind it is exactly that - so the face erodes by several mm
        # all round, and the amount it erodes CHANGES frame to frame.
        # An extremes-based centre is then reading the noisiest points
        # in the set. This one reads all of them.
        rng_ = np.linalg.norm(face, axis=1)
        cos_t = np.abs((face / np.maximum(rng_, 1e-9)[:, None]) @ n)
        w = (face[:, 2] ** 2) / np.maximum(cos_t, 1e-3)
        w = w / w.sum()
        centre3 = (face * w[:, None]).sum(axis=0)
        cx = float((centre3 - c2) @ e1)
        cy = float((centre3 - c2) @ e2)
        (_, _), (w_, hh), _ = cv2.minAreaRect(uv.astype(np.float32))
        w = w_
        area = float(w * hh)
        side = float(np.sqrt(max(area, 0.0)))
    elif method == "hull":
        # THE AREA CENTROID OF THE CONVEX HULL.
        #
        # It has every property this measurement needs, and unlike the
        # alternatives it has them all at once:
        #
        #   uneven sampling   a slanted face gets more pixels per mm on
        #                     its near half, which drags a point MEAN
        #                     toward the camera. A hull is a polygon;
        #                     its area centroid does not count points.
        #   occluded middle   a hull spans a hole, so a suction cup
        #                     standing on the face changes nothing.
        #   square faces      minAreaRect's angle is DEGENERATE on a
        #                     square - four-fold symmetry means it
        #                     wanders with noise, and any measurement
        #                     taken along those axes wanders with it.
        #                     A hull has no orientation to get wrong.
        #
        # What it is sensitive to is a point outside the true face,
        # since extremes are the whole definition - hence the trim.
        pts2 = uv.astype(np.float32)
        for _ in range(2):
            d = np.linalg.norm(pts2 - np.median(pts2, axis=0), axis=1)
            keep = d <= np.quantile(d, 0.995)
            if keep.sum() >= min_pts:
                pts2 = pts2[keep]
        hull = cv2.convexHull(pts2)
        M = cv2.moments(hull)
        if M["m00"] <= 1e-9:
            return None
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        area = float(M["m00"])
        # sqrt(area) is rotation invariant, so unlike a rectangle's
        # width it means the same thing however the cube is turned.
        side = float(np.sqrt(max(area, 0.0)))
        (_, _), (w, hh), _ = cv2.minAreaRect(hull)
    else:
        # Kept so the two can be compared on the bench rather than
        # swapped on faith. Midpoint of robust extremes along the
        # rectangle's own axes.
        (_, _), (_, _), ang = cv2.minAreaRect(uv.astype(np.float32))
        th = np.radians(ang)
        a1 = np.array([np.cos(th), np.sin(th)])
        a2 = np.array([-np.sin(th), np.cos(th)])
        mids, spans = [], []
        for a in (a1, a2):
            s = uv @ a
            lo, hi = np.percentile(s, [1.5, 98.5])
            mids.append(0.5 * (lo + hi))
            spans.append(hi - lo)
        cx = mids[0] * a1[0] + mids[1] * a2[0]
        cy = mids[0] * a1[1] + mids[1] * a2[1]
        w, hh = spans
        area = float(w * hh)
        side = float(np.sqrt(max(area, 0.0)))

    centre = c2 + cx * e1 + cy * e2

    resid = float(np.sqrt(np.mean(((face - c) @ n) ** 2)))
    return {
        "centre": centre,                       # (3,) camera mm
        "n_face": int(len(face)),
        "n_band": int(len(band)),
        "side_mm": side,                        # rotation invariant
        "size_mm": (float(min(w, hh)), float(max(w, hh))),
        "flat_rms_mm": resid,
        "tilt_deg": float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))),
        "raw_tilt_deg": tilt,       # before the guard - the diagnostic
        "guarded": bool(guarded),
        "normal": n,
        # kept for --dump: the face as it was actually measured, in the
        # face plane's own 2D coordinates
        "uv": uv, "uv_centre": (float(cx), float(cy)),
        "band_h": (band @ up) - float(np.quantile(pts @ up, 0.995)),
    }


def measure_cube(depth_mm, color_bgr, rays, which, up,
                 exclude=None, **kw):
    """Whole pipeline for one frame: colour -> blob -> cloud -> top face.

    Returns (result_dict_or_None, reason_string).
    """
    mask = colour_mask(color_bgr, which)
    if exclude is not None:
        mask[exclude > 0] = 0
    blob = largest_blob(mask)
    if blob is None:
        return None, "no %s blob" % which
    cloud = cloud_cam(depth_mm, rays)
    ok = (blob > 0) & (depth_mm > 0)
    n_ok = int(ok.sum())
    if n_ok < 25:
        return None, "blob has only %d valid depth pixels" % n_ok
    r = top_face(cloud[ok], up, **kw)
    if r is None:
        return None, "top face not isolated (%d cube points)" % n_ok
    r["n_blob"] = n_ok
    return r, "ok"

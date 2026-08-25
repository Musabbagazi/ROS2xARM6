#!/usr/bin/env python3
"""Offline checks for the fixed-camera picker.

NO ARM, NO CAMERA, NO MODEL. Safe to run anywhere, any time.

Most of it is arithmetic on synthetic data, but the two that matter most
are end-to-end: a whole scene is RENDERED through the same ray table the
real camera uses - a floor, a cube, a viewpoint 45 degrees off to the
side - and then pushed through the real detector and the real
calibration. Those two tests are the ones that would catch a sign error
in the transform, and a sign error in the transform is the failure that
sends the arm to the mirror image of the cube.

Usage:  python test_fixed.py
"""
import math
import sys

import numpy as np

import fixed_common as fx
import fixed_detect as fd

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print("   %s %s%s" % ("ok  " if condition else "FAIL", name,
                          "" if condition or not detail else
                          "\n        %s" % detail))


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ------------------------- a synthetic camera -------------------------

class FakeIntr(object):
    """Just enough of an rs intrinsics for ray_table."""

    def __init__(self, width=640, height=480, f=385.0):
        self.width, self.height = width, height
        self.fx = self.fy = float(f)
        self.ppx, self.ppy = width / 2.0, height / 2.0
        self.coeffs = [0.0] * 5


def look_at(cam_xyz, target_xyz):
    """A camera->base transform for a camera at cam_xyz looking at
    target_xyz, in the RealSense convention (+X right, +Y down, +Z along
    the optical axis)."""
    o = np.asarray(cam_xyz, dtype=float)
    z = np.asarray(target_xyz, dtype=float) - o
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0:                    # keep it a rotation
        R = np.column_stack([-x, y, z])
    return {"R": R.tolist(), "t": o.tolist()}


def render(T, intr, floor_z=None, boxes=(), noise_mm=0.0, seed=1):
    """Depth (mm, camera Z) and colour for a scene of axis-aligned boxes
    on a level floor, seen through T.

    boxes are (lo_xyz, hi_xyz, bgr). Deliberately built on ray_table and
    on T the same way the real pipeline consumes them, so a convention
    error cannot cancel itself out between the renderer and the code
    under test."""
    rays = fx.ray_table(intr)
    u = np.concatenate([rays, np.ones(rays.shape[:2] + (1,), np.float32)],
                       axis=-1)
    R = np.asarray(T["R"], dtype=float)
    o = np.asarray(T["t"], dtype=float)
    d = u @ R.T                                  # ray directions in base

    best = np.full(rays.shape[:2], np.inf)
    color = np.full(rays.shape[:2] + (3,), 70, np.uint8)

    with np.errstate(divide="ignore", invalid="ignore"):
        if floor_z is not None:
            s = (floor_z - o[2]) / d[..., 2]
            s = np.where(np.isfinite(s) & (s > 0), s, np.inf)
            best = np.minimum(best, s)

        for lo, hi, bgr in boxes:
            s_in = np.full(rays.shape[:2], -np.inf)
            s_out = np.full(rays.shape[:2], np.inf)
            for ax in range(3):
                t1 = (lo[ax] - o[ax]) / d[..., ax]
                t2 = (hi[ax] - o[ax]) / d[..., ax]
                lo_t, hi_t = np.minimum(t1, t2), np.maximum(t1, t2)
                parallel = np.abs(d[..., ax]) < 1e-9
                inside = lo[ax] <= o[ax] <= hi[ax]
                lo_t = np.where(parallel, -np.inf if inside else np.inf, lo_t)
                hi_t = np.where(parallel, np.inf if inside else -np.inf, hi_t)
                s_in, s_out = np.maximum(s_in, lo_t), np.minimum(s_out, hi_t)
            hit = (s_in <= s_out) & (s_in > 0) & (s_in < best)
            best = np.where(hit, s_in, best)
            color[hit] = bgr

    depth = np.where(np.isfinite(best), best, 0.0).astype(np.float32)
    if noise_mm:
        rng = np.random.RandomState(seed)
        depth = np.where(depth > 0,
                         depth + rng.normal(0, noise_mm, depth.shape), 0.0)
    return depth.astype(np.float32), color


# ------------------------- the fit -------------------------

def test_fit():
    print("\n-- the rigid fit --")
    rng = np.random.RandomState(7)
    ang = np.radians(37.0)
    R_true = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                       [np.sin(ang), np.cos(ang), 0.0],
                       [0.0, 0.0, 1.0]])
    t_true = np.array([120.0, -45.0, 610.0])
    cam = rng.uniform(-300, 300, (14, 3))
    base = cam @ R_true.T + t_true

    T = fx.fit_rigid(cam, base)
    check("recovers a known transform exactly", T["rms_mm"] < 1e-6,
          "rms %.4g" % T["rms_mm"])
    check("recovers the rotation", np.allclose(T["R"], R_true, atol=1e-9))
    check("recovers the translation", np.allclose(T["t"], t_true, atol=1e-6))
    check("reports unit scale on exact data", close(T["best_scale"], 1.0,
                                                    1e-6))

    scaled = fx.fit_rigid(cam, (cam * 1.05) @ R_true.T + t_true)
    check("best_scale exposes a scale error the residual hides",
          close(scaled["best_scale"], 1.05, 0.01),
          "got %.4f" % scaled["best_scale"])

    mirrored = cam.copy()
    mirrored[:, 0] *= -1.0
    Tm = fx.fit_rigid(mirrored, base)
    check("never returns a reflection",
          np.linalg.det(np.asarray(Tm["R"])) > 0.99,
          "det %.4f" % np.linalg.det(np.asarray(Tm["R"])))

    for bad, why in (((np.zeros((2, 3)), np.zeros((2, 3))), "2 points"),
                     ((np.zeros((5, 3)), np.zeros((4, 3))), "mismatched")):
        try:
            fx.fit_rigid(*bad)
            check("refuses %s" % why, False)
        except ValueError:
            check("refuses %s" % why, True)

    p = np.array([10.0, 20.0, 30.0])
    check("apply_transform matches R.p+t",
          np.allclose(fx.apply_transform(T, p), R_true @ p + t_true,
                      atol=1e-3))


# ------------------------- the plane -------------------------

def test_plane():
    print("\n-- the support surface --")
    rng = np.random.RandomState(3)
    xy = rng.uniform([250, -400], [780, 400], (3000, 2))
    coef_true = [0.02, -0.015, 160.0]
    z = fx.plane_z(coef_true, xy[:, 0], xy[:, 1]) + rng.normal(0, 1.5, 3000)
    pts = np.column_stack([xy, z])
    # A fifth of the points are 300mm below - the through-glass background
    # that broke the naive fit on real hardware.
    junk = pts[:600].copy()
    junk[:, 2] -= 300.0

    coef = fx.fit_plane(np.vstack([pts, junk]))
    check("fits a tilted plane through 17% gross outliers",
          coef is not None and
          close(fx.plane_z(coef, 500, 0), fx.plane_z(coef_true, 500, 0), 3.0),
          "" if coef is None else "centre %.1f vs %.1f"
          % (fx.plane_z(coef, 500, 0), fx.plane_z(coef_true, 500, 0)))
    check("says nothing rather than guessing when it cannot fit",
          fx.fit_plane(np.zeros((10, 3))) is None)

    inside = fx.in_work_box(np.array([[500.0, 0.0, 0.0],
                                      [900.0, 0.0, 0.0],
                                      [500.0, 900.0, 0.0]]))
    check("the work box rejects points outside the cell",
          list(inside) == [True, False, False])


# ------------------------- the grasp frame -------------------------

def test_grasp():
    print("\n-- the grasp frame --")
    coef = [0.0, 0.0, 155.0]

    # Suction grips the TOP, and the top is what the camera measures, so
    # the grasp height does not depend on how tall the cube is at all.
    for h in (18.0, 30.0, 45.0, 70.0):
        cube = {"center": [500.0, 0.0], "top_z": 155.0 + h, "height_mm": h}
        want = 155.0 + h - fx.CUP_PRESS_MM
        check("the cup lands on the top face of a %.0fmm cube" % h,
              close(fx.grasp_z(cube, coef), want, 1e-9),
              "got %.2f want %.2f" % (fx.grasp_z(cube, coef), want))

    # Under fingers the surface's value in this frame was NOT where the
    # fingers touched it. Under suction the two coincide, so the clamp is
    # simply the surface plus a margin - and it has to hold even when the
    # measured height is nonsense.
    sunk = {"center": [500.0, 0.0], "top_z": 40.0, "height_mm": 30.0}
    got = fx.grasp_z(sunk, coef)
    check("a nonsense height cannot drive the cup into the table",
          got >= 155.0 + fx.MIN_ABOVE_FLOOR - 1e-9,
          "clamped to %.2f, the table is at 155.0" % got)

    tilted = [0.02, -0.01, 155.0]
    surface = fx.plane_z(tilted, 700.0, -300.0)
    off = {"center": [700.0, -300.0], "top_z": surface + 30.0,
           "height_mm": 30.0}
    check("the grasp follows the surface's tilt",
          close(fx.grasp_z(off, tilted), surface + 30.0 - fx.CUP_PRESS_MM,
                1e-9))

    check("pressing harder never reaches the table",
          fx.grasp_z({"center": [500.0, 0.0], "top_z": 155.0 + 16.0,
                      "height_mm": 16.0}, coef) >= 155.0 + fx.MIN_ABOVE_FLOOR)


def test_cup():
    print("\n-- what the cup can seal on --")
    need = fx.CUP_DIA_MM + 2 * fx.CUP_MARGIN_MM
    check("an %.0fmm cup needs %.0fmm of face" % (fx.CUP_DIA_MM, need),
          need <= 30.0,
          "a 30mm cube could not be picked at all if this exceeded 30")

    ok, _ = fx.cup_fits({"width_mm": 30.0, "short_mm": 30.0})
    check("a 30mm cube fits", ok)
    ok, why = fx.cup_fits({"width_mm": 30.0, "short_mm": 18.0})
    check("a face narrower than the cup is refused", not ok, why)
    check("and the refusal says why", "seal" in why, why)

    # The SHORT side decides. A long thin face has nothing to seal
    # against across its narrow direction, however generous the other is.
    ok, _ = fx.cup_fits({"width_mm": 70.0, "short_mm": 20.0})
    check("a long thin face is judged on its narrow side", not ok)


# ------------------------- the drift witness -------------------------

def test_drift():
    print("\n-- has the camera moved --")
    coef = [0.0, 0.0, 155.0]
    rng = np.random.RandomState(11)
    xy = rng.uniform([250, -400], [780, 400], (4000, 2))

    def glass(shift):
        """Half the points on the surface, half on the structure below it
        - the glass floor. `shift` is the camera moving."""
        z = np.where(np.arange(4000) % 2 == 0, 155.0, -500.0) + shift
        return np.column_stack([xy, z])

    def opaque(shift):
        """An ordinary table: the surface is visible, nothing below it."""
        return np.column_stack([xy, np.full(4000, 155.0) + shift])

    # Witness 1: the surface itself, which is what an opaque floor gives.
    bare = {"coef": coef, "points": 4000}
    check("an opaque floor witnesses itself",
          fx.drift_check(opaque(0.0), bare)[0] == "ok")
    check("and catches a knock", fx.drift_check(opaque(70.0), bare)[0]
          == "fail")
    check("the witness is named in the message",
          "surface" in fx.drift_check(opaque(0.0), bare)[1])

    # The fail-open hole: after a big move the surface leaves the search
    # band entirely, and "I cannot see it" must not read as "fine".
    gone = np.column_stack([xy, np.full(4000, -900.0)])
    check("a surface that has vanished fails rather than skipping",
          fx.drift_check(gone, bare)[0] == "fail")
    check("but with no baseline count it only skips",
          fx.drift_check(gone, {"coef": coef})[0] == "skip")

    # Witness 2: the glass case, where the surface is invisible.
    floor = {"coef": coef, "background_z": -500.0}
    check("a glass floor witnesses what is under it",
          fx.drift_check(glass(0.0), floor)[0] == "ok")
    check("a nudge warns", fx.drift_check(glass(22.0), floor)[0] == "warn")
    check("a knock stops the run",
          fx.drift_check(glass(70.0), floor)[0] == "fail")


# ------------------------- end to end -------------------------

def test_detect_scene():
    print("\n-- a rendered scene, 45 degrees off to the side --")
    intr = FakeIntr()
    T = look_at([1000.0, -700.0, 800.0], [500.0, 0.0, 155.0])
    floor_z = 155.0
    floor = {"coef": [0.0, 0.0, floor_z], "height_at_centre": floor_z}

    cx, cy, h, w = 500.0, 40.0, 40.0, 40.0
    boxes = [((cx - w / 2, cy - w / 2, floor_z),
              (cx + w / 2, cy + w / 2, floor_z + h), (40, 40, 200))]
    depth, color = render(T, intr, floor_z=floor_z, boxes=boxes,
                          noise_mm=1.0)
    rays = fx.ray_table(intr)

    cubes, rejects = fd.detect_all(depth, color, rays, T, floor)
    check("finds exactly one cube", len(cubes) == 1,
          "found %d, refused %d %s" % (len(cubes), len(rejects),
                                       [r[1] for r in rejects]))
    if cubes:
        c = cubes[0]
        check("its position is right to 5mm",
              close(c["center"][0], cx, 5.0) and close(c["center"][1], cy,
                                                       5.0),
              "got [%.1f, %.1f] want [%.1f, %.1f]"
              % (c["center"][0], c["center"][1], cx, cy))
        check("its height is right to 5mm", close(c["height_mm"], h, 5.0),
              "got %.1f want %.1f" % (c["height_mm"], h))
        check("its width is right to 6mm", close(c["width_mm"], w, 6.0),
              "got %.1f want %.1f" % (c["width_mm"], w))
        check("it is called red", c["color"] == "red",
              "got %s" % c["color"])

    # The failure mode a fixed camera has and a wrist camera does not: a
    # tall object sliced at cube height looks exactly like a cube.
    post = [((640.0, -140.0, floor_z), (680.0, -100.0, floor_z + 400.0),
             (60, 200, 60))]
    depth2, color2 = render(T, intr, floor_z=floor_z, boxes=boxes + post,
                            noise_mm=1.0)
    cubes2, rejects2 = fd.detect_all(depth2, color2, rays, T, floor)
    check("a tall post is not mistaken for a cube",
          len(cubes2) == 1 and any("taller" in r[1] for r in rejects2),
          "found %d, refused %s" % (len(cubes2), [r[1] for r in rejects2]))

    cubes3, _ = fd.detect_all(depth, color, rays, T, floor,
                              tcp_xy=(cx + 40.0, cy))
    check("a cluster under the tool is ignored", len(cubes3) == 0)

    # A cube the fingers would have taken happily, that the cup cannot
    # seal on. It has to be refused, and the reason has to say so.
    small = 20.0
    tiny = [((cx - small / 2, cy - small / 2, floor_z),
             (cx + small / 2, cy + small / 2, floor_z + small),
             (40, 40, 200))]
    depth4, color4 = render(T, intr, floor_z=floor_z, boxes=tiny,
                            noise_mm=1.0, seed=5)
    cubes4, rejects4 = fd.detect_all(depth4, color4, rays, T, floor)
    check("a cube too small for the cup is refused, not attempted",
          len(cubes4) == 0 and any("seal" in r[1] for r in rejects4),
          "found %d, refused %s" % (len(cubes4), [r[1] for r in rejects4]))


def plate_scene(x, y, z, cup_h=40.0, plate=120.0, cup=18.0):
    """The calibration target as it really is: a flat red plate with the
    suction cup standing on the middle of it.

    The cup is included ON PURPOSE. It is opaque and it casts a shadow
    across the plate from a side view, which is the exact bias pass 2 has
    to survive - a test with a bare plate would prove nothing."""
    return [((x - plate / 2, y - plate / 2, z - 3.0),
             (x + plate / 2, y + plate / 2, z), (40, 40, 200)),
            ((x - cup / 2, y - cup / 2, z),
             (x + cup / 2, y + cup / 2, z + cup_h), (50, 50, 50))]


def test_calibration_roundtrip():
    print("\n-- the calibration, end to end --")
    intr = FakeIntr()
    T_true = look_at([1000.0, -700.0, 800.0], [500.0, 0.0, 300.0])
    rays = fx.ray_table(intr)

    # The plate carried through the working volume, exactly as the real
    # calibration carries it. TOOL is the flange-to-cup-face distance -
    # the number the whole design exists to avoid measuring.
    TOOL = 172.0
    nodes = [(x, y, z) for x in (400.0, 500.0, 600.0)
             for y in (-150.0, 0.0, 150.0)
             for z in (250.0, 380.0, 500.0)]        # plate TOP height

    cam_pts, base_pts = [], []
    for x, y, z in nodes:
        depth, color = render(T_true, intr, floor_z=None,
                              boxes=plate_scene(x, y, z), noise_mm=1.0,
                              seed=int(x + y + z))
        got, _ = fd.find_held_target(depth, color, rays, "red")
        if got is None:
            continue
        cam_pts.append(got)
        # What the controller reports: the flange, TOOL above the cup
        # face, and the cup face is sitting on the plate's top.
        base_pts.append([x, y, z + TOOL])

    check("the held plate is found at most poses", len(cam_pts) >= 20,
          "found at %d of 27" % len(cam_pts))
    if len(cam_pts) < 8:
        return

    T1 = fx.fit_rigid(cam_pts, base_pts)
    refined = []
    for x, y, z in nodes:
        depth, color = render(T_true, intr, floor_z=None,
                              boxes=plate_scene(x, y, z), noise_mm=1.0,
                              seed=int(x + y + z))
        got, _ = fd.find_held_target(depth, color, rays, "red", T=T1)
        if got is not None:
            refined.append((got, [x, y, z + TOOL]))
    T2 = fx.fit_rigid([r[0] for r in refined], [r[1] for r in refined])

    print("      pass 1 rms %.2fmm   pass 2 rms %.2fmm"
          % (T1["rms_mm"], T2["rms_mm"]))
    check("pass 2 fits better than pass 1", T2["rms_mm"] <= T1["rms_mm"],
          "%.2f vs %.2f" % (T2["rms_mm"], T1["rms_mm"]))
    check("the calibration is good enough to pick with",
          T2["rms_mm"] < 5.0, "rms %.2fmm" % T2["rms_mm"])
    check("scale comes out at 1", close(T2["best_scale"], 1.0, 0.02),
          "got %.4f" % T2["best_scale"])

    # The real claim: a cube on the floor, seen later, produces a flange
    # pose that lands the CUP on that cube's top face - without anything
    # ever having measured TOOL.
    floor_z_true = 155.0
    boxes = [((520.0 - 20, -60.0 - 20, floor_z_true),
              (520.0 + 20, -60.0 + 20, floor_z_true + 40.0), (40, 40, 200))]
    depth, color = render(T_true, intr, floor_z=floor_z_true, boxes=boxes,
                          noise_mm=1.0, seed=99)
    # The floor as it lands in the GRASP frame - what teach_floor stores.
    base_cloud = fx.apply_transform(T2, fx.cloud_cam(depth, rays))
    keep = (depth > 0) & fx.in_work_box(base_cloud)
    pts = base_cloud[keep]
    seed = float(np.percentile(pts[:, 2], 90))
    coef = fx.fit_plane(pts[np.abs(pts[:, 2] - seed) < 70.0])
    check("the floor is measurable in the grasp frame", coef is not None)
    if coef is None:
        return
    floor = {"coef": coef}
    cubes, _ = fd.detect_all(depth, color, rays, T2, floor)
    check("the floor cube is found", len(cubes) == 1,
          "found %d" % len(cubes))
    if not cubes:
        return
    gz = fx.grasp_z(cubes[0], coef)
    # Truth: the flange that puts the cup face on the cube's top face,
    # pressed in by CUP_PRESS_MM.
    want = floor_z_true + 40.0 + TOOL - fx.CUP_PRESS_MM
    check("the commanded flange lands the cup on the cube's top face, "
          "with the tool length never measured",
          close(gz, want, 6.0), "commanded %.1f, correct %.1f" % (gz, want))
    check("and its XY is right to 5mm",
          close(cubes[0]["center"][0], 520.0, 5.0) and
          close(cubes[0]["center"][1], -60.0, 5.0),
          "got [%.1f, %.1f]" % tuple(cubes[0]["center"]))


# ---------------- the taped-plate calibration (no tool IO) ----------------
#
# Both of these exist because of a live hardware fault: the wrist's
# end-module link latches error 28 on ANY tool-IO transaction, which halts
# motion, while the arm drives perfectly if the tool is left alone. So
# fixed_calibrate --taped must (a) never touch the tool IO, or the run
# dies half way round the grid on the real arm, and (b) still catch a
# plate that falls off, which is normally the vacuum switch's job.

class ToolIOTouched(AssertionError):
    """The fake arm raises this the moment anything touches the tool."""


class FakeArm(object):
    """Enough xArm to drive collect(), and a tripwire on the tool IO.

    Every tool call raises rather than returning a value, because a
    silent return would let a bug through: the point is that these are
    never called at all."""

    # Joints are not millimetres. The fake still has to respect that,
    # because movej refuses a move needing more than MAX_JOINT_SWEEP -
    # so a "joint angle" of 600 makes every move illegal. A tenth of a
    # millimetre per degree keeps the whole grid inside one branch.
    MM_PER_DEG = 10.0

    def __init__(self, plate_offset=172.0, detach_after=None):
        self.pose = [400.0, 0.0, 300.0, 180.0, 0.0, 0.0]
        self.plate_offset = plate_offset
        self.detach_after = detach_after      # pose number the tape lets go
        self.moves = 0
        self.dropped_at = None                # where the plate was left

    # --- motion: allowed ---
    def _joints(self, pose):
        return [p / self.MM_PER_DEG for p in pose[:3]] + [0.0, 0.0, 0.0]

    def get_inverse_kinematics(self, pose, **kw):
        return 0, self._joints(pose)

    def get_servo_angle(self, **kw):
        return 0, self._joints(self.pose)

    def set_servo_angle(self, angle=None, **kw):
        self.moves += 1
        if (self.detach_after is not None
                and self.dropped_at is None
                and self.moves > self.detach_after):
            # The plate stays where it was when the tape gave way.
            self.dropped_at = list(self.pose[:3])
        self.pose = [a * self.MM_PER_DEG for a in angle[:3]] + [180.0, 0.0, 0.0]
        return 0

    def get_position(self, **kw):
        return 0, list(self.pose)

    def get_err_warn_code(self):
        return 0, [0, 0]

    def clean_warn(self):
        return 0

    # --- the tool: forbidden ---
    def get_vacuum_gripper(self, **kw):
        raise ToolIOTouched("get_vacuum_gripper")

    def set_vacuum_gripper(self, *a, **kw):
        raise ToolIOTouched("set_vacuum_gripper")

    def get_tgpio_output_digital(self, **kw):
        raise ToolIOTouched("get_tgpio_output_digital")

    def set_tgpio_digital(self, *a, **kw):
        raise ToolIOTouched("set_tgpio_digital")

    def get_tgpio_digital(self, ionum=None):
        raise ToolIOTouched("get_tgpio_digital")


class SensorArm(object):
    """An arm whose ONLY working tool call is get_tgpio_digital.

    Models this cell exactly: register 0x0A18 is unanswered, so
    get_tgpio_output_digital - and therefore the SDK's
    get_vacuum_gripper, which reads it on the way past - faults. Anything
    that reaches for those here fails the test loudly rather than
    silently working in the fake and faulting on the real arm."""

    def __init__(self, di0=0):
        self.di0 = di0
        self.reads = 0

    def get_tgpio_digital(self, ionum=None):
        self.reads += 1
        return 0, self.di0 if ionum is not None else [self.di0, 0, 0, 0]

    def get_vacuum_gripper(self, **kw):
        raise AssertionError("get_vacuum_gripper reads 0x0A18 - it faults "
                             "this cell. Use get_tgpio_digital.")

    def get_tgpio_output_digital(self, **kw):
        raise AssertionError("get_tgpio_output_digital IS the 0x0A18 read "
                             "that faults this cell.")


def test_vacuum_read_path():
    """The seal verdict must never route through the faulting register.

    Two days were lost to this: the SDK's get_vacuum_gripper() hard-codes
    check_on=True, which reads 0x0A18 before it reads the sensor, and
    this end module answers that register with error 28 in 2-3ms. The
    sensor itself was fine the whole time."""
    print("\n-- reading the cup without the broken register --")

    arm = SensorArm(di0=1)
    check("a sealed cup reads HELD", fx.vacuum_state(arm) == fx.HELD,
          "got %s" % fx.vacuum_state(arm))
    check("and it got there via get_tgpio_digital", arm.reads > 0)

    arm = SensorArm(di0=0)
    check("an unsealed cup reads NOT_HELD",
          fx.vacuum_state(arm) == fx.NOT_HELD,
          "got %s" % fx.vacuum_state(arm))

    class Refusing(SensorArm):
        def get_tgpio_digital(self, ionum=None):
            return -1, None                 # a code outside ADVISORY_CODES

    check("a refused read is UNKNOWN, never a false NOT_HELD",
          fx.vacuum_state(Refusing()) == fx.VAC_UNKNOWN)

    # The SDK's seal timeout is a MISS, not a broken controller. It only
    # became reachable once the reads started working.
    class TimingOut(SensorArm):
        def set_vacuum_gripper(self, *a, **kw):
            return fx.SUCTION_TIMEOUT_CODE

    ok, state = fx.vacuum_on(TimingOut(di0=0))
    check("a seal timeout reports a miss, not a controller failure",
          ok is False and state == fx.NOT_HELD,
          "got ok=%s state=%s" % (ok, state))

    class Sealed(SensorArm):
        def set_vacuum_gripper(self, *a, **kw):
            return 0

    ok, state = fx.vacuum_on(Sealed(di0=1))
    check("a successful grasp reports HELD", ok is True and state == fx.HELD,
          "got ok=%s state=%s" % (ok, state))


def test_wiring_probe():
    """The wiring probe must key off the SEAL, not off a state change.

    It used to watch the switch go 'off' -> 'on, nothing held' when the
    cup was commanded. That transition came from the SDK checking the
    0x0A18 output register, which this end module will not answer.
    Reading the sensor pin directly there is no such transition, so with
    nothing held both wirings read 0 throughout - and the probe declared
    'check the tool cable' about a cup that worked. The seal is the only
    honest signal."""
    print("\n-- identifying the wiring by the seal --")
    import fixed_vacuum as fv

    class Cup(object):
        """A cup wired to ONE of the two pin pairs, with something held
        against it. It seals only when the right pins are driven."""

        def __init__(self, wired_hw, held=True):
            self.wired_hw = wired_hw
            self.held = held
            self.on = False

        def set_vacuum_gripper(self, on, wait=False, hardware_version=1,
                               **kw):
            if hardware_version == self.wired_hw:
                self.on = bool(on)
            return 0

        def get_tgpio_digital(self, ionum=None):
            return 0, 1 if (self.on and self.held) else 0

        def get_err_warn_code(self):
            return 0, [0, 0]

        def clean_error(self):
            return 0

        def clean_warn(self):
            return 0

        def set_state(self, s):
            return 0

    sealed1, trip1 = fv.probe(Cup(wired_hw=1), 1)
    sealed2, _ = fv.probe(Cup(wired_hw=1), 2)
    check("the wired pins seal and are identified", sealed1 is True)
    check("the other wiring does not", sealed2 is False)
    check("and no fault is blamed on a healthy cup", trip1 is None)

    # The failure that started this: nothing held against the cup.
    nothing, _ = fv.probe(Cup(wired_hw=1, held=False), 1)
    check("with nothing held, it reports no seal rather than a fault",
          nothing is False)


class FakeCam(object):
    """Renders whatever the plate is currently attached to."""

    def __init__(self, arm, T_true, intr):
        self.arm, self.T_true, self.intr = arm, T_true, intr
        self._rays = fx.ray_table(intr)

    def rays(self):
        return self._rays

    def frame(self):
        # The plate sits plate_offset BELOW the flange while it is held,
        # or wherever it fell.
        if self.arm.dropped_at is not None:
            x, y, z = self.arm.dropped_at
        else:
            x, y, z = self.arm.pose[:3]
        depth, color = render(self.T_true, self.intr, floor_z=None,
                              boxes=plate_scene(x, y,
                                                z - self.arm.plate_offset),
                              noise_mm=1.0, seed=int(abs(x) + abs(y) + abs(z)))
        return depth, color, None


def test_taped_calibration():
    print("\n-- the taped-plate calibration (wrist fault workaround) --")
    import fixed_calibrate as fc

    intr = FakeIntr()
    T_true = look_at([1000.0, -700.0, 800.0], [500.0, 0.0, 300.0])
    poses = [[x, y, z, 180.0, 0.0, 0.0]
             for z in (300.0, 440.0)
             for y in (-150.0, 0.0, 150.0)
             for x in (400.0, 500.0, 600.0)]

    arm = FakeArm()
    cam = FakeCam(arm, T_true, intr)
    try:
        samples, skipped, lost = fc.collect(arm, cam, poses, "red",
                                            use_vacuum=False)
        touched = None
    except ToolIOTouched as e:
        samples, skipped, lost, touched = [], [], False, str(e)

    check("taped mode never touches the tool IO", touched is None,
          "it called %s" % touched)
    check("it still collects the poses", len(samples) >= 12,
          "got %d of %d" % (len(samples), len(poses)))
    check("and does not cry 'plate came off' when it did not", not lost)

    # The same run with the tape letting go a third of the way round. The
    # plate stays on the floor, still red, still perfectly visible - the
    # exact failure the vacuum switch used to catch.
    arm2 = FakeArm(detach_after=6)
    cam2 = FakeCam(arm2, T_true, intr)
    samples2, _, lost2 = fc.collect(arm2, cam2, poses, "red",
                                    use_vacuum=False)
    check("a plate that falls off IS caught, with no switch to ask", lost2)
    check("and it is caught quickly, before the run is wasted",
          lost2 and len(samples2) <= 10, "kept %d poses" % len(samples2))


def test_cube_top_face():
    """Measuring a cube standing on the floor, for the touch calibration.

    This is the failure that cost a whole calibration run: the first
    version reused find_held_target, which centroids the entire coloured
    blob. On a flat plate that is the top face. On a 30mm CUBE seen from
    45 degrees it is the top face AND a side face averaged together, so
    the answer sits well below and in front of the truth - 9.2mm RMS and
    a scale of 0.973 in the real run."""
    print("\n-- measuring a cube on the floor (touch calibration) --")
    intr = FakeIntr()
    T_true = look_at([1000.0, -700.0, 800.0], [500.0, 0.0, 300.0])
    rays = fx.ray_table(intr)
    floor_z = 155.0
    size = 40.0

    old_err, new_err = [], []
    for (x, y) in [(420.0, -150.0), (520.0, 0.0), (620.0, 150.0),
                   (450.0, 120.0), (600.0, -120.0)]:
        boxes = [((x - size / 2, y - size / 2, floor_z),
                  (x + size / 2, y + size / 2, floor_z + size),
                  (40, 40, 200))]
        depth, color = render(T_true, intr, floor_z=floor_z, boxes=boxes,
                              noise_mm=1.0, seed=int(x + y))
        truth = np.array([x, y, floor_z + size])   # top face centre

        got, _ = fd.find_cube_top(depth, color, rays, "red")
        if got is not None:
            base = fx.apply_transform(T_true, np.array([got]))[0]
            new_err.append(float(np.linalg.norm(base - truth)))

        old, _ = fd.find_held_target(depth, color, rays, "red")
        if old is not None:
            base = fx.apply_transform(T_true, np.array([old]))[0]
            old_err.append(float(np.linalg.norm(base - truth)))

    check("the cube's top face is found at every position",
          len(new_err) == 5, "found %d of 5" % len(new_err))
    if new_err:
        worst = max(new_err)
        mean_new = sum(new_err) / len(new_err)
        mean_old = sum(old_err) / len(old_err) if old_err else 0.0
        print("      top-face method  : %.1fmm average, %.1fmm worst"
              % (mean_new, worst))
        print("      whole-blob centroid: %.1fmm average  (the old bug)"
              % mean_old)
        check("it lands within 2mm of the true top-face centre",
              worst < 2.0, "worst %.1fmm" % worst)
        check("and it beats the whole-blob centroid by a wide margin",
              mean_old > mean_new * 5.0,
              "old %.1fmm vs new %.1fmm" % (mean_old, mean_new))


def test_fault_counting():
    """The bug that made 'it worked!!!' unreadable.

    The first keep-alive counted every 100ms SAMPLE taken while an
    error was latched, so one stuck fault and a hundred brief ones
    printed the same number - and since it also polled the vacuum
    switch (a tool-IO read, the very thing that trips the fault), the
    total was nothing but its own poll rate. These pin the fix."""
    print("\n-- counting faults, not samples --")
    from vacuum_hold import FaultLog

    # One fault, latched for a second at 10Hz. That is ONE fault.
    log = FaultLog()
    for i in range(10):
        log.sample(28, i * 0.1)
    check("one latched fault counts once, not once per poll",
          log.edges == 1, "counted %d" % log.edges)

    # Ten separate faults, each cleared. That is TEN.
    log = FaultLog()
    t = 0.0
    for _ in range(10):
        log.sample(28, t)
        log.clear_result(True)      # observed clean again
        t += 0.1
        log.sample(0, t)
        t += 0.1
    check("ten separate faults count ten", log.edges == 10,
          "counted %d" % log.edges)

    # A clear that did NOT work must not re-arm the edge counter, or one
    # unclearable fault inflates to the poll rate all over again.
    log = FaultLog()
    for i in range(10):
        log.sample(28, i * 0.1)
        log.clear_result(False)
    check("a fault that will not clear still counts once",
          log.edges == 1, "counted %d" % log.edges)
    check("and the refused clears are reported separately",
          log.refused == 10 and log.cleared == 0,
          "%d refused, %d took" % (log.refused, log.cleared))

    # The headline number must be per-transaction, not per-second.
    log = FaultLog()
    log.sample(28, 0.0)
    log.touch(4)
    check("tool transactions are counted", log.touches == 4,
          "counted %d" % log.touches)

    log = FaultLog()
    for i, t in enumerate([0.0, 1.0, 3.0]):
        log.sample(28, t)
        log.clear_result(True)
        log.sample(0, t + 0.05)
    check("gaps between faults are measured", log.gaps() == [1.0, 2.0],
          "got %s" % log.gaps())


def _cube(x, y, color="blue", top_z=200.0):
    return {"center": [x, y], "top_z": top_z, "color": color}


def test_tracking():
    """Following cubes that move, on the turntable and on the belt.

    The turntable cases matter most, because a straight-line prediction
    does not look wrong - it looks slightly late, and misses by v^2t^2/2r
    while every plot of it appears sensible."""
    print("\n-- tracking moving cubes --")
    import fixed_track as ft

    # --- a turntable: three cubes at different radii, one centre ---
    CX, CY, W = 500.0, 0.0, 0.6          # rad/s
    radii = [150.0, 250.0, 320.0]
    phases = [0.0, 2.0, 4.0]
    trk = ft.Tracker()
    dt = 1.0 / 30.0
    for k in range(20):
        t = k * dt
        frame = [_cube(CX + r * math.cos(W * t + p),
                       CY + r * math.sin(W * t + p))
                 for r, p in zip(radii, phases)]
        tracks = trk.update(frame, t)

    check("it keeps one track per cube, not one per frame",
          len(tracks) == 3, "got %d" % len(tracks))
    check("it recognises the surface is rotating",
          trk.model == "rotating", "called it %s" % trk.model)
    if trk.centre:
        err = math.hypot(trk.centre[0] - CX, trk.centre[1] - CY)
        check("and finds the axis to within 15mm", err < 15.0,
              "out by %.1fmm at [%.0f, %.0f]"
              % (err, trk.centre[0], trk.centre[1]))

    # The claim the whole design rests on: predicting round the arc beats
    # predicting along the tangent, by enough to matter to a 4mm margin.
    tr = trk.pickable()[0]
    horizon = 0.30
    t_future = tr.last_t + horizon
    r_true = math.hypot(tr.position[0] - CX, tr.position[1] - CY)
    a_true = math.atan2(tr.position[1] - CY, tr.position[0] - CX) + W * horizon
    truth = (CX + r_true * math.cos(a_true), CY + r_true * math.sin(a_true))

    arc = tr.predict(t_future, trk.centre)
    tangent = tr.predict(t_future, None)
    e_arc = math.hypot(arc[0] - truth[0], arc[1] - truth[1])
    e_tan = math.hypot(tangent[0] - truth[0], tangent[1] - truth[1])
    print("      over %.2fs: arc off by %.1fmm, straight line off by %.1fmm"
          % (horizon, e_arc, e_tan))
    check("the arc prediction lands inside the cup's margin",
          e_arc < 4.0, "off by %.1fmm" % e_arc)
    check("and a straight line would have missed",
          e_tan > e_arc * 2.0,
          "straight %.1fmm vs arc %.1fmm" % (e_tan, e_arc))

    # --- a conveyor: same code, no centre should be invented ---
    ft.Track._next_id = 1
    belt = ft.Tracker()
    for k in range(20):
        t = k * dt
        frame = [_cube(300.0 + 200.0 * t, -100.0),
                 _cube(300.0 + 200.0 * t, 60.0)]
        belt.update(frame, t)
    check("a belt is not mistaken for a turntable",
          belt.model == "straight", "called it %s" % belt.model)
    tr = belt.pickable()[0]
    v = tr.velocity()
    check("and its speed is measured", v is not None and
          close(math.hypot(v[0], v[1]), 200.0, 12.0),
          "got %s" % (None if v is None else round(math.hypot(*v), 1)))
    ahead = tr.predict(tr.last_t + 0.3, belt.centre)
    check("its straight-line prediction is right",
          close(ahead[0], tr.position[0] + 60.0, 4.0),
          "predicted x %.1f, expected %.1f" % (ahead[0],
                                               tr.position[0] + 60.0))

    # --- identity: two cubes passing close must not swap ---
    ft.Track._next_id = 1
    cross = ft.Tracker()
    ids = []
    for k in range(16):
        t = k * dt
        # one going right, one going left, meeting in the middle
        a = _cube(300.0 + 150.0 * t, 0.0, color="blue")
        b = _cube(400.0 - 150.0 * t, 40.0, color="red")
        live = cross.update([a, b], t)
        ids.append({tr.color: tr.id for tr in live if tr.n > 2})
    settled = [d for d in ids if len(d) == 2]
    check("two cubes passing close keep their identities",
          len(settled) > 4 and
          all(d == settled[0] for d in settled),
          "ids drifted: %s" % settled[:4])

    # --- the same turntable, but through noisy depth ---
    #
    # The clean case proves the maths; this proves it is usable. A real
    # centre estimate is fitted to positions that jitter by a millimetre
    # or two, and an axis found by crossing near-parallel lines is
    # exactly the sort of thing that noise destroys.
    ft.Track._next_id = 1
    rng = np.random.RandomState(7)
    noisy = ft.Tracker()
    for k in range(24):
        t = k * dt
        frame = []
        for r, p in zip(radii, phases):
            frame.append(_cube(CX + r * math.cos(W * t + p) +
                               rng.normal(0, 1.5),
                               CY + r * math.sin(W * t + p) +
                               rng.normal(0, 1.5)))
        noisy.update(frame, t)
    check("noise does not turn the turntable into a belt",
          noisy.model == "rotating", "called it %s" % noisy.model)
    if noisy.centre:
        err = math.hypot(noisy.centre[0] - CX, noisy.centre[1] - CY)
        print("      axis found %.1fmm from truth through 1.5mm noise" % err)
        check("the axis survives 1.5mm of measurement noise", err < 40.0,
              "out by %.1fmm" % err)
        tr = noisy.pickable()[0]
        r_true = math.hypot(tr.position[0] - CX, tr.position[1] - CY)
        a_true = (math.atan2(tr.position[1] - CY, tr.position[0] - CX) +
                  W * 0.30)
        truth = (CX + r_true * math.cos(a_true), CY + r_true * math.sin(a_true))
        arc = tr.predict(tr.last_t + 0.30, noisy.centre)
        e = math.hypot(arc[0] - truth[0], arc[1] - truth[1])
        print("      noisy 0.30s prediction off by %.1fmm" % e)
        check("and the prediction is still good enough to grasp on",
              e < 6.0, "off by %.1fmm" % e)

    # --- a track that stops being seen is dropped ---
    ft.Track._next_id = 1
    gone = ft.Tracker()
    for k in range(8):
        gone.update([_cube(400.0, 0.0)], k * dt)
    gone.update([], 5.0)
    check("a cube that vanishes is forgotten, not aimed at",
          len(gone.tracks) == 0, "%d left" % len(gone.tracks))

    # --- a brand-new track is not offered as a target ---
    ft.Track._next_id = 1
    fresh = ft.Tracker()
    fresh.update([_cube(400.0, 0.0)], 0.0)
    fresh.update([_cube(404.0, 0.0)], dt)
    check("a two-frame track is not yet pickable",
          len(fresh.pickable()) == 0,
          "offered %d" % len(fresh.pickable()))


def main():
    print("=" * 62)
    print("   FIXED CAMERA  -  offline checks (no arm, no camera)")
    print("=" * 62)
    test_fit()
    test_plane()
    test_grasp()
    test_cup()
    test_drift()
    test_detect_scene()
    test_calibration_roundtrip()
    test_taped_calibration()
    test_cube_top_face()
    test_fault_counting()
    test_vacuum_read_path()
    test_wiring_probe()
    test_tracking()
    print("")
    print("=" * 62)
    print("   %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name in FAIL:
        print("      FAILED: %s" % name)
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

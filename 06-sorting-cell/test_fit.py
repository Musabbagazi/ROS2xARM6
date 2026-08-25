"""Offline checks for the fit. No arm, no camera, no target.

A known transform is invented, points are pushed through it, noise and
outliers are added, and the fit has to recover what was invented. So a
sign error, a reflection, or an outlier pass that quietly eats the data
is caught here rather than by an arm reaching for the mirror image of a
cube.

    python test_fit.py
"""

import os
import tempfile

import numpy as np

import cell_fit as cf

FAILED = []


def check(name, ok, detail=""):
    print("   %-52s %s   %s" % (name, "ok" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def rot(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make(n=27, noise=0.0, seed=0, heights=3, span=(240, 200, 100),
         planar=False):
    """A grid of flange poses, and where a camera would have seen them."""
    rng = np.random.default_rng(seed)
    R_true = rot(-135.0, 3.0, 20.0)
    t_true = np.array([384.0, -469.0, 695.0])

    per = int(round((n / max(heights, 1)) ** 0.5))
    xs = np.linspace(-span[0] / 2, span[0] / 2, per)
    ys = np.linspace(-span[1] / 2, span[1] / 2, per)
    zs = np.array([0.0]) if planar else np.linspace(0, span[2], heights)
    g = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    p_base = g + np.array([300.0, 0.0, 200.0])

    # invert: p_base = R p_cam + t  ->  p_cam = R^T (p_base - t)
    p_cam = (p_base - t_true) @ R_true
    if noise:
        p_cam = p_cam + rng.normal(0, noise, p_cam.shape)
    return p_cam, p_base, R_true, t_true


def main():
    print("\n   offline checks for the fit - no arm, no camera\n")

    # --- exact recovery ----------------------------------------------
    p_cam, p_base, R_true, t_true = make(noise=0.0)
    f = cf.fit_rigid(p_cam, p_base)
    check("recovers a known transform exactly",
          f["rms_mm"] < 1e-6
          and np.allclose(f["R"], R_true, atol=1e-8)
          and np.allclose(f["t"], t_true, atol=1e-6),
          "rms %.2e mm" % f["rms_mm"])
    check("R is a proper rotation, not a reflection",
          abs(np.linalg.det(f["R"]) - 1.0) < 1e-9
          and np.allclose(f["R"] @ f["R"].T, np.eye(3), atol=1e-9),
          "det %.9f" % np.linalg.det(f["R"]))
    check("best_scale is 1 on clean data",
          abs(f["best_scale"] - 1.0) < 1e-9,
          "%.9f" % f["best_scale"])

    # --- noise, and what RMS it should produce -----------------------
    # With isotropic noise s on the camera points, the residual RMS
    # tends to s*sqrt(3) per point in 3D, less the 6 dof absorbed.
    for s in (0.3, 1.0, 2.0):
        p_cam, p_base, _, _ = make(noise=s, seed=3)
        f = cf.fit_rigid(p_cam, p_base)
        ideal = s * np.sqrt(3)
        check("noise %.1fmm -> RMS near %.2fmm" % (s, ideal),
              0.6 * ideal < f["rms_mm"] < 1.5 * ideal,
              "%.2f mm" % f["rms_mm"])

    # --- the scale red herring, reproduced ---------------------------
    # The previous project read best_scale 0.9375 as a camera fault. At
    # n=12 with that much noise it is under 2 sigma from 1.0.
    p_cam, p_base, _, _ = make(n=12, noise=9.6, seed=11, planar=True)
    f = cf.fit_rigid(p_cam, p_base)
    check("reports the standard error on best_scale",
          f["scale_se"] > 0.01,
          "scale %.4f +/- %.4f = %.1f sigma"
          % (f["best_scale"], f["scale_se"], f["scale_sigmas"]))
    check("a noisy best_scale is NOT flagged as significant",
          f["scale_sigmas"] < 3.0,
          "%.1f sigma from 1.0" % f["scale_sigmas"])

    # --- outliers ----------------------------------------------------
    p_cam, p_base, R_true, t_true = make(noise=0.4, seed=5)
    p_cam[4] += np.array([25.0, -18.0, 12.0])
    p_cam[17] += np.array([-30.0, 22.0, -9.0])
    plain = cf.fit_rigid(p_cam, p_base)
    f = cf.fit_with_outliers(p_cam, p_base)
    check("outlier pass finds both bad pairs",
          f["dropped"] == 2 and 4 not in f["kept_index"]
          and 17 not in f["kept_index"],
          "dropped %d, rms %.2f -> %.2f"
          % (f["dropped"], plain["rms_mm"], f["rms_mm"]))
    check("and recovers the true transform after dropping them",
          np.allclose(f["R"], R_true, atol=0.01)
          and np.linalg.norm(f["t"] - t_true) < 1.5,
          "t off by %.2f mm" % np.linalg.norm(f["t"] - t_true))

    # --- the outlier pass must not eat the data ----------------------
    p_cam, p_base, _, _ = make(noise=6.0, seed=9)
    f = cf.fit_with_outliers(p_cam, p_base)
    check("outlier pass never drops more than a quarter",
          f["dropped"] <= int(0.25 * 27) + 1,
          "dropped %d of 27" % f["dropped"])

    # --- coplanar samples are caught ---------------------------------
    p_cam, p_base, _, _ = make(noise=0.3, seed=7, planar=True)
    f = cf.fit_rigid(p_cam, p_base)
    ok, bad = cf.verdict(f, p_base)
    check("a single-height grid is REFUSED however good its RMS",
          not ok and any("flat" in b for b in bad),
          "rms was only %.2f mm" % f["rms_mm"])

    ext = cf.coverage(p_base)
    check("coverage reports the flat axis as ~0",
          ext[2] < 1.0, "extents %.0f / %.0f / %.1f mm" % tuple(ext))

    # --- the thresholds ----------------------------------------------
    p_cam, p_base, _, _ = make(noise=0.3, seed=13)
    good = cf.fit_rigid(p_cam, p_base)
    ok, bad = cf.verdict(good, p_base)
    check("a good fit passes", ok, "rms %.2f mm" % good["rms_mm"])

    p_cam, p_base, _, _ = make(noise=4.0, seed=15)
    poor = cf.fit_rigid(p_cam, p_base)
    ok, bad = cf.verdict(poor, p_base)
    check("a loose fit is rejected", not ok,
          "rms %.2f mm - %s" % (poor["rms_mm"], bad[0] if bad else ""))

    # --- refusing to save --------------------------------------------
    d = tempfile.mkdtemp()
    path = os.path.join(d, "handeye.json")
    try:
        cf.save_fit(path, poor)
        check("save_fit REFUSES a fit that failed", False, "it saved it")
    except ValueError:
        check("save_fit REFUSES a fit that failed", True,
              "and nothing was written" if not os.path.exists(path)
              else "BUT A FILE APPEARED")
    check("nothing was written by the refused save",
          not os.path.exists(path))

    cf.save_fit(path, good, extra={"target": "test"})
    R2, t2, doc = cf.load_fit(path)
    check("a good fit saves and loads back identically",
          np.allclose(R2, good["R"], atol=1e-9)
          and np.allclose(t2, good["t"], atol=1e-6)
          and doc["passed"] is True)

    # --- the transform is actually usable ----------------------------
    p_cam, p_base, R_true, t_true = make(noise=0.3, seed=21)
    f = cf.fit_rigid(p_cam, p_base)
    err = np.linalg.norm(cf.to_base(p_cam, f["R"], f["t"]) - p_base,
                         axis=1)
    check("to_base reproduces the flange poses",
          err.max() < 2.0, "worst %.2f mm" % err.max())

    print()
    if FAILED:
        print("   %d FAILED: %s\n" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("   all checks pass\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

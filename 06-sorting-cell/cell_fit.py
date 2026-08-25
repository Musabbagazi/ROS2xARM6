"""The fit: camera points in, a transform out - or a refusal.

This is deliberately separate from anything that talks to hardware, so
it can be tested exhaustively without an arm, a camera or a target.

WHAT IT FITS

Pairs of (p_cam, p_flange): where the camera saw the target, and where
the controller said the flange was at that moment. A rigid transform

    p_flange = R . p_cam + t

maps "the camera sees a surface here" to "put the flange HERE to put the
cup on it". The flange-to-cup distance is INSIDE the answer, so it never
has to be measured and can never be measured wrong. So is the offset of
however the target happens to be stuck to the cup - crooked is fine -
PROVIDED that offset is the same for every pair, which is why yaw is
held fixed across the grid.

WHY IT REFUSES

The previous calibration in this cell wrote a 16.7mm RMS transform to
disk with a warning, and every tool downstream then trusted it. A number
that is only advisory is a number that gets ignored at 11pm. So a fit
that fails its thresholds is NOT saved, and save_fit will not write one.
"""

import json
import os

import numpy as np


# An 18mm cup on a 30mm cube has (30-18)/2 = 6mm of radial slack, and a
# seal wants better than that. These are what "good enough to pick with"
# means, expressed as arithmetic rather than opinion.
MAX_RMS_MM = 3.0
MAX_WORST_MM = 8.0
MIN_PAIRS = 12


def fit_rigid(p_cam, p_base):
    """Kabsch/Umeyama with scale FORCED to 1. Returns a dict.

    Both frames are metric millimetres, so a scale is not a free
    parameter - fitting one would silently absorb a real error into a
    plausible-looking number. It is computed anyway and returned as
    best_scale, purely as a diagnostic.

    READ best_scale CAREFULLY. Its standard error is roughly
    sigma_noise / (sqrt(n) . r_rms), where r_rms is the spread of the
    points about their centroid. With 12 points over 400x300mm and 16mm
    of noise that is 0.034, so a best_scale of 0.94 is under 2 sigma from
    1.0 and means nothing at all. The previous project treated exactly
    that number as evidence of a camera fault and went looking for one.
    So the standard error is returned alongside it.
    """
    p_cam = np.asarray(p_cam, float)
    p_base = np.asarray(p_base, float)
    n = len(p_cam)
    if n < 3:
        raise ValueError("need at least 3 pairs, got %d" % n)

    ca, cb = p_cam.mean(axis=0), p_base.mean(axis=0)
    A, B = p_cam - ca, p_base - cb

    H = (A.T @ B) / n
    U, S, Vt = np.linalg.svd(H)
    # The reflection guard. Without it a noisy or near-degenerate set can
    # fit an IMPROPER rotation - a mirror image - which looks numerically
    # fine and sends the arm to the reflection of every cube.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    t = cb - R @ ca
    resid = np.linalg.norm(p_base - (p_cam @ R.T + t), axis=1)
    rms = float(np.sqrt((resid ** 2).mean()))

    var_a = float((A ** 2).sum() / n)
    best_scale = float((S * np.array([1.0, 1.0, d])).sum() / max(var_a, 1e-12))
    r_rms = float(np.sqrt(var_a))
    scale_se = float(rms / max(np.sqrt(n) * r_rms, 1e-9))

    return {
        "R": R, "t": t,
        "rms_mm": rms,
        "max_mm": float(resid.max()),
        "residuals_mm": [float(v) for v in resid],
        "n": int(n),
        "best_scale": best_scale,
        "scale_se": scale_se,
        "scale_sigmas": float(abs(best_scale - 1.0) / max(scale_se, 1e-9)),
        "spread_mm": r_rms,
    }


def fit_with_outliers(p_cam, p_base, drop_mm=8.0, max_drop=0.25):
    """Fit, drop the pairs that clearly do not belong, refit.

    A pair goes bad for physical reasons - the target shifted on the cup,
    the arm was still settling, the camera caught a frame mid-move - and
    one such pair drags the whole transform. But an outlier pass that is
    allowed to drop everything will happily 'improve' a fit by discarding
    the disagreement, so max_drop caps it at a quarter of the data and
    the count is reported.
    """
    p_cam = np.asarray(p_cam, float)
    p_base = np.asarray(p_base, float)
    keep = np.ones(len(p_cam), bool)

    first = fit_rigid(p_cam, p_base)
    for _ in range(3):
        f = fit_rigid(p_cam[keep], p_base[keep])
        resid = np.linalg.norm(
            p_base - (p_cam @ f["R"].T + f["t"]), axis=1)
        cut = max(drop_mm, 3.0 * f["rms_mm"])
        new = resid <= cut
        if new.sum() < len(p_cam) * (1.0 - max_drop) or new.sum() < 3:
            break
        if (new == keep).all():
            break
        keep = new

    out = fit_rigid(p_cam[keep], p_base[keep])
    out["dropped"] = int((~keep).sum())
    out["kept_index"] = [int(i) for i in np.flatnonzero(keep)]
    out["rms_before_drop_mm"] = first["rms_mm"]
    return out


def coverage(p_base):
    """How well the pairs span a volume - and whether they span one.

    A transform fitted over a thin slab is EXTRAPOLATING everywhere else,
    and its residual will not say so. The previous calibration used 12
    points at a single height and reported a residual that described only
    that plane. Returns the extent along each principal axis, largest
    first; a third value near zero means the samples are coplanar.
    """
    p = np.asarray(p_base, float)
    if len(p) < 3:
        return [0.0, 0.0, 0.0]
    d = p - p.mean(axis=0)
    sv = np.linalg.svd(d, full_matrices=False)[1]
    return [float(v / np.sqrt(len(p))) for v in sv]


def verdict(f, p_base=None, min_pairs=MIN_PAIRS,
            max_rms=MAX_RMS_MM, max_worst=MAX_WORST_MM,
            min_third_axis_mm=15.0):
    """Is this transform fit to pick with? Returns (ok, list_of_reasons)."""
    bad = []
    if f["n"] < min_pairs:
        bad.append("only %d pairs kept, want at least %d"
                   % (f["n"], min_pairs))
    if f["rms_mm"] > max_rms:
        bad.append("RMS %.2fmm is over the %.1fmm limit"
                   % (f["rms_mm"], max_rms))
    if f["max_mm"] > max_worst:
        bad.append("worst pair %.2fmm is over the %.1fmm limit"
                   % (f["max_mm"], max_worst))
    if p_base is not None:
        ext = coverage(p_base)
        if ext[2] < min_third_axis_mm:
            bad.append("the samples are nearly flat (thinnest axis "
                       "%.0fmm) - this transform is extrapolating "
                       "wherever it is actually used" % ext[2])
    return (len(bad) == 0), bad


def save_fit(path, f, extra=None, force=False):
    """Write the transform - but ONLY if it passed.

    The previous project saved a 16.7mm fit with a printed warning and
    everything downstream trusted the file. Refusing is the difference.
    """
    ok, bad = verdict(f)
    if not ok and not force:
        raise ValueError("refusing to save: " + "; ".join(bad))
    doc = {
        "R": [[float(v) for v in row] for row in f["R"]],
        "t": [float(v) for v in f["t"]],
        "rms_mm": round(f["rms_mm"], 3),
        "max_mm": round(f["max_mm"], 3),
        "n": f["n"],
        "dropped": f.get("dropped", 0),
        "best_scale": round(f["best_scale"], 5),
        "scale_se": round(f["scale_se"], 5),
        "scale_sigmas": round(f["scale_sigmas"], 2),
        "residuals_mm": [round(v, 2) for v in f["residuals_mm"]],
        "passed": bool(ok),
    }
    if extra:
        doc.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    return doc


def load_fit(path):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return np.array(d["R"], float), np.array(d["t"], float), d


def to_base(p_cam, R, t):
    """Camera millimetres -> the flange pose that puts the cup there."""
    p = np.asarray(p_cam, float)
    return p @ np.asarray(R, float).T + np.asarray(t, float)

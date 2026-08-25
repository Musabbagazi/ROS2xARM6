#!/usr/bin/env python3
"""Where the cube is going, and whether the arm can get there first.

This is the part neither sibling project has. vision_pick3 waits for
motion to stop and handoff_pick waits for it to pause; both therefore
only ever need the cube's POSITION. To catch a cube you need its
position, its velocity, an honest measure of how much to trust that
velocity, and a decision about where to be.

WHY A LINE FIT AND NOT A FILTER

A Kalman filter is the reflexive answer and it is the wrong one here.
A filter is for continuously correcting an estimate you keep using; this
project makes ONE decision from the track ("ambush there, close then")
and is blind from that moment on. What matters is not the smoothest
running estimate but an honest answer to "is this cube's motion actually
a straight line at constant speed, or am I about to bet on noise?" - and
the residual of a plain least-squares fit answers exactly that, in
millimetres, with nothing tuned.

It also fails LOUDLY in the cases that must fail. A cube that is
tumbling, bouncing, decelerating under friction, or being carried by
hand round a curve does not fit a line, the residual climbs, and
confident() refuses. A filter would have quietly tracked all of them and
handed back a confident-looking prediction.

WHY POSITIONS ARE IN THE ARM'S FRAME, NOT PIXELS

Every sample is converted through catch_common.aim_moving before it gets
here, so the history is base-frame millimetres. That matters because the
ARM MOVES while it tracks. In pixels, the arm's own motion and the
cube's motion are added together and cannot be separated; in the base
frame the arm's motion cancels out exactly, and what is left is the cube.
"""
import numpy as np

import catch_common as cc


# How much history the fit uses. Long enough that a velocity means
# something, short enough that a cube which changes speed is described by
# what it is doing NOW rather than by an average over its whole journey.
WINDOW_S = 1.20

# Drop the whole history if the cube was not seen for this long. A gap
# means an occlusion, a detection dropout, or a different cube - and a
# line fitted across a gap describes none of them.
MAX_GAP_S = 0.35

# What "enough evidence" means. Samples alone are not enough: ten samples
# taken in 60ms have no baseline and will fit a confident line to pure
# noise, so the time SPAN is gated separately.
MIN_SAMPLES = 5
MIN_SPAN_S = 0.35

# How far the samples may sit from the fitted line before the data is
# too scattered to mean anything.
#
# This is a DATA-QUALITY gate and nothing more: it catches a detector
# flickering between two objects, a half-occluded cube whose measured
# centre jumps, and association mistakes. It is deliberately NOT the
# curvature gate, which was the first thing tried here and does not
# work - see drift_at.
MAX_RESID_MM = 6.0

# How many sigma of measured turn have to survive before it counts as a
# real bend rather than noise. 2 is the usual "probably not chance".
TURN_SIGMA_K = 2.0

# The share of the fingers' clearance a BEND may consume. The gap has to
# cover the curve, the timing jitter and the tracking error together, so
# no one of them gets all of it.
DRIFT_BUDGET_FRAC = 0.5

# Vertical motion. A cube sliding or being carried level fits a flat line
# in Z; one that is falling does not, and a falling cube is not catchable
# by an arm that has to be parked half a second early. Refused explicitly
# rather than left to the residual, because Z error and XY error mean
# different things to a grab.
MAX_VZ_MM_S = 60.0


def _fit_axis(ts, vs):
    """Least-squares (value_at_t0, rate) for one axis.

    Written out rather than calling polyfit so a degenerate span - every
    sample at the same instant, which happens if the camera repeats a
    frame - returns a flat line instead of a warning and a NaN."""
    n = len(ts)
    tm = float(np.mean(ts))
    vm = float(np.mean(vs))
    den = float(np.sum((ts - tm) ** 2))
    if n < 2 or den < 1e-9:
        return vm, 0.0
    rate = float(np.sum((ts - tm) * (vs - vm)) / den)
    return vm - rate * tm, rate


class CubeTrack:
    """One cube's recent path in the arm's frame, and the line through it.

    Times are the DEPTH FRAME's timestamps, not the wall clock - see
    catch_detect.frame_time_s. Everything here differences two of them,
    so a constant offset in that clock is harmless."""

    def __init__(self, window_s=WINDOW_S, max_gap_s=MAX_GAP_S):
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self.t = []
        self.x = []
        self.y = []
        self.z = []
        self.last_seen = None       # the most recent detection dict
        self.last_t = None

    # ------------------------- history -------------------------

    def reset(self):
        self.t, self.x, self.y, self.z = [], [], [], []
        self.last_seen = None
        self.last_t = None

    def add(self, t, x, y, z, seen=None):
        """Record one base-frame sample. Drops the history across a gap."""
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.reset()
        self.t.append(float(t))
        self.x.append(float(x))
        self.y.append(float(y))
        self.z.append(float(z))
        self.last_t = float(t)
        if seen is not None:
            self.last_seen = seen
        self._trim()

    def _trim(self):
        if not self.t:
            return
        cutoff = self.t[-1] - self.window_s
        keep = 0
        while keep < len(self.t) - 2 and self.t[keep] < cutoff:
            keep += 1
        if keep:
            self.t = self.t[keep:]
            self.x = self.x[keep:]
            self.y = self.y[keep:]
            self.z = self.z[keep:]

    @property
    def n(self):
        return len(self.t)

    @property
    def span_s(self):
        return 0.0 if self.n < 2 else self.t[-1] - self.t[0]

    # ------------------------- the fit -------------------------

    def fit(self):
        """The straight line through the history, or None.

        Returns a dict: x0/y0/z0 (position AT TIME t0=0 on the frame
        clock), vx/vy/vz (mm/s), speed, heading_deg, resid_mm, n, span_s.
        Absolute positions are only ever produced through predict()."""
        if self.n < 2:
            return None
        ts = np.asarray(self.t, dtype=float)
        x0, vx = _fit_axis(ts, np.asarray(self.x, dtype=float))
        y0, vy = _fit_axis(ts, np.asarray(self.y, dtype=float))
        z0, vz = _fit_axis(ts, np.asarray(self.z, dtype=float))
        px = x0 + vx * ts
        py = y0 + vy * ts
        resid = float(np.sqrt(np.mean((px - np.asarray(self.x)) ** 2
                                      + (py - np.asarray(self.y)) ** 2)))
        speed = float(np.hypot(vx, vy))
        return {
            "x0": x0, "y0": y0, "z0": z0,
            "vx": vx, "vy": vy, "vz": vz,
            "speed": speed,
            "heading_deg": float(np.degrees(np.arctan2(vy, vx))),
            "resid_mm": resid,
            "n": self.n,
            "span_s": self.span_s,
        }

    def predict(self, t, f=None):
        """(x, y, z) on the fitted line at frame-clock time t."""
        f = self.fit() if f is None else f
        if f is None:
            return None
        return (f["x0"] + f["vx"] * t,
                f["y0"] + f["vy"] * t,
                f["z0"] + f["vz"] * t)

    # ------------------------- curvature -------------------------
    #
    # The fit residual cannot police curvature, and finding that out the
    # hard way is worth recording. The residual measures how far the
    # samples sit from their own best-fit line ACROSS THE OBSERVED
    # WINDOW; what actually matters is how wrong that line is a couple of
    # seconds LATER, at the ambush point. Those two numbers are related
    # by the square of the horizon ratio, so they are nothing like each
    # other: a turn slow enough to leave under 1mm of residual over one
    # second puts the prediction tens of millimetres out over three. To
    # refuse such a turn by residual alone the threshold would have to be
    # a fraction of a millimetre - far below the detector's own scatter -
    # and it would then refuse every real cube instead.
    #
    # So curvature is measured directly, as the rate at which the
    # velocity VECTOR is rotating, and converted into the only units that
    # matter: millimetres of predicted error at the moment of the catch.

    def turn_rate(self, f=None):
        """(omega, sigma) rad/s the velocity direction is rotating.

        Measured by fitting the two halves of the history separately and
        taking the signed angle between the resulting velocity vectors.
        sigma is how much of that angle plain noise would explain, from
        the standard error of a slope (sd/(sd_t*sqrt(n))) with the fit
        residual standing in for the per-sample scatter. Returns None
        when there is not enough history to split."""
        n = self.n
        if n < 6:
            return None
        f = self.fit() if f is None else f
        if f is None or f["speed"] < 1e-6:
            return None
        k = n // 2
        ts = np.asarray(self.t, dtype=float)
        xs = np.asarray(self.x, dtype=float)
        ys = np.asarray(self.y, dtype=float)
        _a, vx1 = _fit_axis(ts[:k], xs[:k])
        _b, vy1 = _fit_axis(ts[:k], ys[:k])
        _c, vx2 = _fit_axis(ts[k:], xs[k:])
        _d, vy2 = _fit_axis(ts[k:], ys[k:])
        if np.hypot(vx1, vy1) < 1e-6 or np.hypot(vx2, vy2) < 1e-6:
            return None
        dt = float(np.mean(ts[k:]) - np.mean(ts[:k]))
        if dt <= 1e-6:
            return None
        dtheta = float(np.arctan2(vx1 * vy2 - vy1 * vx2,
                                  vx1 * vx2 + vy1 * vy2))
        omega = dtheta / dt

        half_span = max(1e-6, self.span_s / 2.0)
        half_n = max(2, k)
        sd_v = (f["resid_mm"] * np.sqrt(12.0)
                / (half_span * np.sqrt(half_n)))
        # two independent velocity estimates, each contributing an
        # angular error of sd_v/speed
        sd_theta = np.sqrt(2.0) * sd_v / f["speed"]
        return float(omega), float(sd_theta / dt)

    def drift_at(self, lead_s, f=None):
        """Millimetres a straight-line prediction will be out after
        lead_s, from the SIGNIFICANT part of the measured turn.

        A velocity rotating at omega is a lateral acceleration of
        speed*omega, so the straight-line prediction falls behind by
        (1/2)(speed*omega)T^2. Only the part of omega that survives
        TURN_SIGMA_K sigma is charged, so ordinary detector noise on a
        genuinely straight track reads as zero drift rather than as a
        slow bend."""
        tr = self.turn_rate(f)
        if tr is None:
            return 0.0
        omega, sigma = tr
        eff = max(0.0, abs(omega) - TURN_SIGMA_K * sigma)
        f = self.fit() if f is None else f
        return float(0.5 * f["speed"] * eff * lead_s ** 2)

    # ------------------------- is it catchable -------------------------

    def confident(self, f=None):
        """(ok, reason). reason is None when ok.

        Every gate here is a reason to REFUSE a catch, phrased so the
        operator is told what to change. Refusing costs one cube; a
        confident fit to bad data costs a collision with whatever is
        pushing it."""
        f = self.fit() if f is None else f
        if f is None or f["n"] < MIN_SAMPLES:
            return False, "still watching it"
        if f["span_s"] < MIN_SPAN_S:
            return False, "still watching it"
        if f["resid_mm"] > MAX_RESID_MM:
            # scatter, not curvature - curvature is the planner's job
            return False, ("I cannot get a clean read on it (%.0fmm of "
                           "scatter) - more light, or slow it down"
                           % f["resid_mm"])
        if abs(f["vz"]) > MAX_VZ_MM_S:
            return False, "it is rising or falling too fast to meet"
        if f["speed"] < cc.MIN_CATCH_SPEED:
            return False, ("that is barely moving - the floor picker "
                           "handles a still cube better than I can")
        if f["speed"] > cc.MAX_CATCH_SPEED:
            return False, ("too fast for me (%.0fmm/s, I top out near "
                           "%.0f) - send it slower"
                           % (f["speed"], cc.MAX_CATCH_SPEED))
        return True, None


# ------------------------- how long the arm takes -------------------------

class ArmTiming:
    """How long a move of a given distance actually takes, learned.

    The planner has to answer "can I get there before the cube does?",
    and that needs a number this codebase does not have: vision_common
    commands moves in JOINT degrees per second, and the relationship
    between that and millimetres of tool travel depends on where in the
    workspace the arm is. Deriving it analytically would mean duplicating
    the kinematics; measuring it takes one move.

    So it starts deliberately PESSIMISTIC and learns from every move the
    project makes. Pessimistic is the safe direction: over-estimating the
    arm's time makes the planner pick a point further along the path,
    which costs reach, while under-estimating it puts the fingers in the
    cube's way while the arm is still moving.

    The estimate uses the SLOWEST rate seen, not the mean, for the same
    reason."""

    # Until something has been measured. A shoulder turning at
    # cc.CATCH_SPEED (25 deg/s) with the tool ~600mm out sweeps
    # 600 * 25 * pi/180 = ~260mm/s, so 180 is that with a third taken off
    # for the joints that do not contribute and for the accel ramp.
    #
    # This number was 110 at first, on the theory that pessimism is free.
    # It is not: over-estimating the arm's time pushes the intercept
    # further along the cube's path, and past a point the whole path is
    # outside the cell - so an over-cautious rate does not produce a
    # careful catch, it produces "I cannot reach that" for every cube
    # that is actually catchable. 180 is conservative against the
    # geometry above while still leaving a workable band; the real value
    # is measured on the first move either way.
    DEFAULT_RATE_MM_S = 180.0
    FIXED_S = 0.45

    # Ignore very short moves when learning - they are all ramp and no
    # travel, and would teach an absurdly low rate.
    MIN_LEARN_MM = 40.0

    def __init__(self):
        self.rate = self.DEFAULT_RATE_MM_S
        self.samples = 0

    def estimate(self, dist_mm):
        """Seconds to move this far, erring long."""
        return self.FIXED_S + float(dist_mm) / max(1.0, self.rate)

    def record(self, dist_mm, secs):
        """Learn from a move that actually happened."""
        if dist_mm < self.MIN_LEARN_MM or secs <= self.FIXED_S:
            return
        rate = float(dist_mm) / (secs - self.FIXED_S)
        if rate <= 1.0:
            return
        self.rate = rate if self.samples == 0 else min(self.rate, rate)
        self.samples += 1


# ------------------------- where to ambush -------------------------

def plan_intercept(track, now, arm_xyz, timing, reach_test=None, f=None):
    """Pick the spot to wait at. Returns a plan dict, or (None, reason).

    Walks forward along the fitted path in LEAD_STEP_S steps and takes
    the FIRST point that satisfies all of:

      * it is inside the cell's usable annulus (cheap, no controller)
      * reach_test says the arm can actually hold that pose (IK)
      * the arm gets there at least ARRIVE_MARGIN_S before the cube

    First, not best, and deliberately: the earliest workable point is the
    one that leaves the most of the cube's path in reserve if the first
    attempt is refused, and every step further along is another
    LEAD_STEP_S of extrapolation to be wrong about.

    Returns (plan, None) or (None, reason). The plan carries t_arrive on
    the FRAME clock, which is what catch_pick times the close from."""
    f = track.fit() if f is None else f
    ok, why = track.confident(f)
    if not ok:
        return None, why

    # Counted separately so the refusal names the RIGHT cause. Reporting
    # "out of reach" because some candidate along the way was
    # unreachable - when the ones that mattered were reachable and merely
    # too late - sends the operator to move the cube sideways when the
    # actual fix is to slow it down.
    n_reachable = 0
    n_curved = 0
    # How much of the fingers' clearance the CURVE is allowed to spend.
    # The rest is left for timing jitter and for the tracking error that
    # is already in the fit - the catch has one gap to spend and three
    # things wanting to spend it.
    drift_budget = DRIFT_BUDGET_FRAC * cc.half_gap_mm()
    lead = cc.MIN_LEAD_S
    while lead <= cc.MAX_LEAD_S + 1e-9:
        t_arrive = now + lead
        px, py, pz = track.predict(t_arrive, f)
        if not cc.in_reach(px, py):
            lead += cc.LEAD_STEP_S
            continue
        if reach_test is not None and not reach_test(px, py, pz):
            lead += cc.LEAD_STEP_S
            continue
        n_reachable += 1

        # Will the straight-line prediction still be true when the cube
        # gets here? Checked PER CANDIDATE rather than once, because the
        # error grows with the square of the lead: a bend that makes a
        # 4-second intercept hopeless may leave a 1-second one perfectly
        # sound, and refusing the whole cube for the far point would
        # throw away the near one that would have worked.
        if track.drift_at(lead, f) > drift_budget:
            n_curved += 1
            lead += cc.LEAD_STEP_S
            continue

        # Distance to the HOVER pose above the intercept, in three
        # dimensions. The arm watches from 680mm and the hover sits a
        # couple of hundred below that, so an XY-only distance would
        # under-count the move by more than the margin it is checked
        # against - and under-counting the arm's time is the one
        # direction that puts the fingers in the cube's way while the
        # arm is still moving.
        dist = float(np.linalg.norm([px - arm_xyz[0], py - arm_xyz[1],
                                     (pz + cc.HOVER_H) - arm_xyz[2]]))
        arm_eta = timing.estimate(dist)
        if lead >= arm_eta + cc.ARRIVE_MARGIN_S:
            return {
                "x": px, "y": py, "z": pz,
                "t_arrive": t_arrive,
                "lead_s": lead,
                "arm_eta_s": arm_eta,
                "slack_s": lead - arm_eta,
                "travel_mm": dist,
                "speed": f["speed"],
                "heading_deg": f["heading_deg"],
                "resid_mm": f["resid_mm"],
            }, None
        lead += cc.LEAD_STEP_S

    if n_curved and n_curved >= n_reachable:
        # Every point the arm could have reached was too far into a bend
        # to predict. Named before the reach and timing causes because it
        # is the one the operator would never guess.
        return None, ("it is curving too much to predict - send it in a "
                      "straight line")
    if n_reachable == 0:
        # Nothing on the path was ever reachable - the cube is not
        # crossing the cell at all, and no amount of timing fixes that.
        px, py, _pz = track.predict(now + cc.MIN_LEAD_S, f)
        return None, cc.out_of_reach_hint(px, py)
    # Points on the path WERE reachable and none was far enough ahead:
    # the cube is outrunning the arm along its own path. Saying so beats
    # "no intercept found", which is true of both causes.
    return None, ("it will be past me before I can get there - send it "
                  "across the cell rather than away from me, or slower")

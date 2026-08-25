#!/usr/bin/env python3
"""Track moving cubes in the ROBOT's frame, and say where they will be.

NO ARM, NO CAMERA. Pure arithmetic on positions, so it is fully testable
offline - which matters, because a prediction bug does not announce
itself. It sends the arm confidently to the wrong place.

WHAT THIS IS FOR

The stationary picker takes five frames, keeps the cubes that appear in
at least three, medians them into one position, and drives there. Every
part of that assumes the cube is still there when the arm arrives. Once
the cubes move, the arm has to go where the cube WILL BE, which means
knowing which blob in this frame is the same cube as that blob in the
last one, how fast it is going, and along what path.

ONE MODEL FOR THE TURNTABLE AND THE CONVEYOR

The cell runs on a turntable now and a belt later, and those are not two
problems. A straight line is a circle of infinite radius, so the only
question is whether the tracks share a centre of rotation and where it
is. Fit that, and prediction is "rotate about the centre"; find no
sensible centre, and it degenerates to "carry on in a straight line",
which is exactly the belt.

Doing it the other way round - straight-line prediction, patched later
for the turntable - fails quietly on the arc. Over a horizon t the
tangent departs from the circle by about

    v^2 t^2 / 2r

which at 200mm/s, a 0.3s horizon and a 200mm radius is 9mm. The cup has
4mm of margin. That is a miss, from a model that looks fine in a plot.

HOW THE CENTRE IS FOUND, AND ONE WAY THAT DOES NOT WORK

A single cube watched for half a second at 200mm/s traces a 100mm arc.
On a 300mm turntable that is 19 degrees, and fitting a circle to 19
degrees of arc is badly conditioned - the centre slides along the
perpendicular almost freely. Every cube shares the same centre, so the
obvious fix is to pool all their points into one circle fit.

That is wrong, and the tests caught it. Cubes at different radii lie on
CONCENTRIC circles, not on one circle, so a single fit through all of
them describes nothing: the best-fit ring splits the difference between
the radii, fits badly everywhere, and gets rejected as "not rotating"
on a turntable that plainly is.

What works is the instantaneous centre of rotation. On a rotating
surface every cube's velocity is perpendicular to its own radius, so the
centre lies somewhere on the line through the cube, at right angles to
its travel. Two cubes at different angles give two such lines and they
cross at the axis; more cubes, or one cube watched at several points
along its arc, give a least-squares crossing.

The degeneracy is the useful part. On a belt every cube travels the same
way, so all those perpendiculars are PARALLEL and never cross - the
solve is singular, and that singularity is exactly the statement "this
is not rotating". One test distinguishes the turntable from the
conveyor, and it is the conditioning of a 2x2 matrix.

A rigid table then has one more thing to say: every cube must share the
same angular speed, whatever its radius. Checking that the fitted
omegas agree is what stops three cubes drifting independently from
being read as a turntable.
"""
import math

import numpy as np

# How far a cube may be from where it was predicted and still be counted
# as the same cube. At 250mm/s and 30fps a cube moves ~8mm between
# frames, so this is generous - but it is applied to the PREDICTED
# position once a track has a velocity, not to the last one, which is
# what stops a fast cube being abandoned every frame.
GATE_MM = 45.0

# A track older than this without an update is dropped.
STALE_S = 0.5

# Positions kept per track. Long enough to fit a velocity through noise,
# short enough that a cube changing speed is not remembered as its old
# self for long.
HISTORY = 12

# Fewer points than this and a track has no opinion about where it is
# going.
MIN_FOR_VELOCITY = 3
MIN_FOR_ROTATION = 5

# A fitted rotation centre further away than this is not a turntable,
# it is a straight line with rounding error. Beyond it, predict on the
# tangent - which is the conveyor case, and the right answer for it.
MAX_RADIUS_MM = 3000.0

# How far from parallel the tracks' directions must be before a centre
# is believed. It is the smallest eigenvalue of the summed projection
# matrix, divided by the number of samples: 0 means every cube travels
# the same way (a belt, and no centre exists), 0.5 means they are spread
# right round the table. Straight lines are the safe default, so this
# sits low - it only has to exclude the genuinely parallel case.
MIN_DIRECTION_SPREAD = 0.06

# A rigid surface turns as one piece, so every cube on it must share an
# angular speed. This is the allowed scatter, as a fraction of the mean.
# Without it, several cubes drifting independently would be fitted a
# centre and called a turntable.
OMEGA_AGREEMENT = 0.25


def _fit_line(ts, xs):
    """Least-squares x = x0 + v*t. Returns (x0, v, rms)."""
    n = len(ts)
    t = np.asarray(ts, dtype=float)
    x = np.asarray(xs, dtype=float)
    tm, xm = t.mean(), x.mean()
    denom = float(((t - tm) ** 2).sum())
    v = float(((t - tm) * (x - xm)).sum() / denom) if denom > 1e-12 else 0.0
    x0 = float(xm - v * tm)
    rms = float(np.sqrt(np.mean((x - (x0 + v * t)) ** 2))) if n else 0.0
    return x0, v, rms


def centre_from_velocities(pairs):
    """Where do the perpendiculars to these velocities cross?

    pairs is [((x, y), (vx, vy)), ...]. Returns (cx, cy) or None when the
    directions are too nearly parallel to cross anywhere - which is not a
    failure, it is a belt.

    Each sample says "the centre is somewhere on my perpendicular", which
    written out is d . (c - p) = 0: the centre, seen from the cube, has
    no component ALONG the direction of travel. Least squares over all of
    them minimises the sum of those components, and the matrix that picks
    out the component along d is the projection d d^T. Summing gives one
    2x2 solve.

    Note it is d d^T and NOT (I - d d^T). The first version used the
    latter - the projection perpendicular to d - which measures the
    distance to the line ALONG the velocity rather than across it. It
    solved cleanly and returned a confident answer 100mm from the true
    axis on exact, noise-free input, which is the kind of wrong that only
    a test with a known answer catches."""
    M = np.zeros((2, 2))
    b = np.zeros(2)
    n = 0
    for p, v in pairs:
        speed = math.hypot(v[0], v[1])
        if speed < 1e-6:
            continue
        d = np.array([v[0] / speed, v[1] / speed])
        P = np.outer(d, d)
        M += P
        b += P.dot(np.array([p[0], p[1]], dtype=float))
        n += 1
    if n < 2:
        return None
    # The smallest eigenvalue measures how far from parallel the
    # directions are. Parallel directions leave the matrix singular, and
    # solving it anyway would invent an axis out of rounding error.
    try:
        smallest = float(np.linalg.eigvalsh(M)[0])
    except np.linalg.LinAlgError:
        return None
    if smallest / n < MIN_DIRECTION_SPREAD:
        return None
    try:
        c = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(c)):
        return None
    return float(c[0]), float(c[1])


def fit_circle(pts):
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r, rms) or None.

    Solves x^2 + y^2 = a*x + b*y + c, which is linear in (a, b, c) - so
    it is one least-squares solve with no starting guess to get wrong.
    It biases the radius slightly on short arcs, which does not matter
    here: the centre is what is wanted, and the angular speed is fitted
    separately against it."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return None
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones(len(p))])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    inside = sol[2] + cx * cx + cy * cy
    if inside <= 0:
        return None
    r = math.sqrt(inside)
    if not np.isfinite(r):
        return None
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(resid ** 2)))


def _unwrap(angles):
    """Make a sequence of angles continuous, so a track crossing pi does
    not read as a sudden reversal at full speed."""
    out = [float(angles[0])]
    for a in angles[1:]:
        prev = out[-1]
        d = float(a) - prev
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        out.append(prev + d)
    return out


class Track(object):
    """One cube, seen over time."""

    _next_id = 1

    def __init__(self, cube, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.color = cube.get("color", "unknown")
        self.samples = [(t, float(cube["center"][0]), float(cube["center"][1]),
                         float(cube.get("top_z", 0.0)))]
        self.cube = cube            # the most recent full detection
        self.misses = 0

    # ---- what it has seen ----

    @property
    def last_t(self):
        return self.samples[-1][0]

    @property
    def position(self):
        _, x, y, z = self.samples[-1]
        return x, y, z

    @property
    def n(self):
        return len(self.samples)

    @property
    def age_s(self):
        return self.samples[-1][0] - self.samples[0][0]

    def add(self, cube, t):
        self.samples.append((t, float(cube["center"][0]),
                             float(cube["center"][1]),
                             float(cube.get("top_z", 0.0))))
        if len(self.samples) > HISTORY:
            self.samples.pop(0)
        self.cube = cube
        self.misses = 0
        # A cube's colour can flicker frame to frame at the edge of a
        # hue band. Once it has been called something definite, keep
        # that rather than letting one bad frame rename it.
        c = cube.get("color", "unknown")
        if self.color == "unknown" and c != "unknown":
            self.color = c

    # ---- what it thinks it is doing ----

    def velocity(self):
        """(vx, vy) mm/s, or None if it has not seen enough."""
        if self.n < MIN_FOR_VELOCITY:
            return None
        ts = [s[0] for s in self.samples]
        _, vx, _ = _fit_line(ts, [s[1] for s in self.samples])
        _, vy, _ = _fit_line(ts, [s[2] for s in self.samples])
        return vx, vy

    def speed(self):
        v = self.velocity()
        return None if v is None else math.hypot(v[0], v[1])

    def pv_samples(self):
        """[(position, velocity), ...] from along this track's history.

        One pair per track would be enough if several cubes were always
        visible, but a single cube on a turntable has to supply the
        spread of directions by itself - and it does, because its heading
        rotates as it goes round. Splitting the history in half gives two
        headings from one cube, which is what lets the axis be found when
        only one cube is on the table."""
        if self.n < MIN_FOR_VELOCITY:
            return []
        halves = []
        if self.n >= 2 * MIN_FOR_VELOCITY:
            mid = self.n // 2
            halves = [self.samples[:mid], self.samples[mid:]]
        else:
            halves = [self.samples]
        out = []
        for part in halves:
            if len(part) < MIN_FOR_VELOCITY:
                continue
            ts = [s[0] for s in part]
            x0, vx, _ = _fit_line(ts, [s[1] for s in part])
            y0, vy, _ = _fit_line(ts, [s[2] for s in part])
            tmid = sum(ts) / len(ts)
            out.append(((x0 + vx * tmid, y0 + vy * tmid), (vx, vy)))
        return out

    def straight_residual(self):
        """How well a straight line explains this track, in mm."""
        if self.n < MIN_FOR_VELOCITY:
            return None
        ts = [s[0] for s in self.samples]
        _, _, rx = _fit_line(ts, [s[1] for s in self.samples])
        _, _, ry = _fit_line(ts, [s[2] for s in self.samples])
        return math.hypot(rx, ry)

    def omega_about(self, cx, cy):
        """Angular speed in rad/s about a given centre, or None."""
        if self.n < MIN_FOR_VELOCITY:
            return None
        ts = [s[0] for s in self.samples]
        angs = _unwrap([math.atan2(s[2] - cy, s[1] - cx)
                        for s in self.samples])
        _, w, _ = _fit_line(ts, angs)
        return w

    def predict(self, t, centre=None):
        """Where the cube will be at absolute time t. (x, y, z) or None.

        With a centre, the cube is carried round the arc; without one it
        carries on in a straight line. Z is held at the last measured
        value - the cubes ride on a surface, so their height is not
        something to extrapolate."""
        if self.n < MIN_FOR_VELOCITY:
            return None
        dt = t - self.last_t
        x, y, z = self.position
        if centre is not None:
            cx, cy = centre
            w = self.omega_about(cx, cy)
            if w is not None:
                a = math.atan2(y - cy, x - cx) + w * dt
                r = math.hypot(x - cx, y - cy)
                return cx + r * math.cos(a), cy + r * math.sin(a), z
        v = self.velocity()
        if v is None:
            return None
        return x + v[0] * dt, y + v[1] * dt, z


class Tracker(object):
    """Follows every cube, and works out how the surface is moving."""

    def __init__(self, gate_mm=GATE_MM, stale_s=STALE_S):
        self.gate_mm = gate_mm
        self.stale_s = stale_s
        self.tracks = []
        self.centre = None          # (cx, cy) of the turntable, or None
        self.centre_r = None
        self.model = "unknown"      # "rotating" | "straight" | "unknown"

    def update(self, cubes, t):
        """Take one frame's detections. Returns the live tracks."""
        unmatched = list(cubes)
        for tr in self.tracks:
            want = tr.predict(t, self.centre) or tr.position
            best, best_d = None, self.gate_mm
            for c in unmatched:
                d = math.hypot(float(c["center"][0]) - want[0],
                               float(c["center"][1]) - want[1])
                if d < best_d:
                    best, best_d = c, d
            if best is not None:
                tr.add(best, t)
                unmatched.remove(best)
            else:
                tr.misses += 1

        for c in unmatched:
            self.tracks.append(Track(c, t))

        self.tracks = [tr for tr in self.tracks
                       if t - tr.last_t <= self.stale_s]
        self._fit_surface()
        return self.tracks

    def _fit_surface(self):
        """Is this a turntable, and if so where is its axis?

        Crosses the perpendiculars to every observed velocity - see
        centre_from_velocities, and the module note on why fitting one
        circle through cubes at different radii cannot work. A straight
        line wins every tie, because curving a prediction that should
        have been straight is the more dangerous error: it aims the arm
        off the side of a belt."""
        moving = [tr for tr in self.tracks
                  if tr.n >= MIN_FOR_ROTATION and (tr.speed() or 0.0) > 5.0]
        if not moving:
            self.model, self.centre, self.centre_r = "unknown", None, None
            return

        pairs = [pv for tr in moving for pv in tr.pv_samples()]
        centre = centre_from_velocities(pairs)
        if centre is None:
            # The perpendiculars never cross: every cube is travelling
            # the same way. That is a belt, and it is the answer, not a
            # failure to find one.
            self.model, self.centre, self.centre_r = "straight", None, None
            return

        radii = [math.hypot(tr.position[0] - centre[0],
                            tr.position[1] - centre[1]) for tr in moving]
        if max(radii) > MAX_RADIUS_MM:
            self.model, self.centre, self.centre_r = "straight", None, None
            return

        # A rigid surface turns as one piece. If these cubes are really
        # riding on one table they must share an angular speed, however
        # different their radii - and if they do not, they are moving
        # independently and no single centre describes them.
        omegas = [w for w in (tr.omega_about(*centre) for tr in moving)
                  if w is not None and abs(w) > 1e-6]
        if len(omegas) >= 2:
            mean = sum(omegas) / len(omegas)
            spread = max(abs(w - mean) for w in omegas)
            if abs(mean) < 1e-6 or spread / abs(mean) > OMEGA_AGREEMENT:
                self.model, self.centre, self.centre_r = ("straight", None,
                                                          None)
                return

        self.model = "rotating"
        self.centre = centre
        self.centre_r = sum(radii) / len(radii)

    # ---- what the picker asks ----

    def pickable(self, color=None, min_age_s=0.15):
        """Tracks steady enough to aim at, best-established first.

        A track two frames old has a velocity fitted through noise and
        nothing else. Aiming at it is how an arm ends up lunging at a
        measurement error, so a track has to have been watched for a
        moment before it is offered."""
        out = [tr for tr in self.tracks
               if tr.age_s >= min_age_s and tr.n >= MIN_FOR_VELOCITY
               and (color is None or tr.color == color)]
        out.sort(key=lambda tr: (-tr.n, tr.id))
        return out

    def describe(self):
        if self.model == "rotating" and self.centre:
            return ("rotating about [%.0f, %.0f], radius %.0fmm"
                    % (self.centre[0], self.centre[1], self.centre_r or 0.0))
        if self.model == "straight":
            speeds = [tr.speed() or 0.0 for tr in self.tracks
                      if tr.speed() is not None]
            if speeds:
                return "travelling straight, %.0fmm/s" % (sum(speeds) /
                                                          len(speeds))
            return "travelling straight"
        return "motion not established yet"

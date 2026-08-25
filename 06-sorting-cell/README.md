# Vision-guided sorting cell — xArm 6, fixed D435, suction cup

Pick the blue cubes, leave the red ones. Later, do it while they move.

> **Status: design record only. No code yet. Nothing has run.**

This is a fresh tree. It does not import from `05-fixed-camera\`, `04-realtime-catch\` or
`02-vision-pick\`, and it does not modify them. It exists because those projects
are blocked on one thing, and that thing needs a different approach
rather than another pass over the same code.

---

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.


## Why this tree exists

`05-fixed-camera\` already implements the design this project wants: fixed
camera, no trained model, depth transformed into the base frame before
any geometry, height threshold for "is something there", colour read from
the same points. 44 offline checks pass. Its README argues the case well
and the argument is right.

It has never completed a pick, because the hand-eye calibration it is
using is wrong by more than the cup can tolerate.

```
handeye.json:  rms 16.70 mm   worst 32.48 mm   n 12
               method "touch (seal-confirmed contact on cubes)"
```

An 18 mm cup on a 30 mm cube top has `(30 − 18) / 2 = 6 mm` of radial
slack. The transform in use misses by 16.7 mm RMS. Nothing downstream can
compensate for that, and no amount of detector work changes it.

## What actually went wrong

From `05-fixed-camera\logs\touch_20260812_120408.log`, the run that produced the
transform now in use:

> *"the arm goes **LIMP** and you move it **BY HAND**, pressing the cup
> onto the cube — it records itself the moment the cup seals"*

A human hand placed the cup for all 12 samples. That one fact produces
three independent errors simultaneously.

### 1. Lateral centring — the dominant term

An 18 mm cup seals correctly anywhere within `(40 − 18) / 2 = 11 mm` of a
40 mm cube's centre. **The vacuum switch confirms contact and is blind to
centring.** A placement scattered uniformly within radius `a` has RMS
`a/√2`, so ~7.8 mm of purely random lateral error per sample.

The touch tool's own diagnostic names this first: *"the cup was not
centred over the cube on some touches (the one error the seal cannot
catch)"*. It was correct.

### 2. Press depth

One cube, one flat floor, so the flange height at seal should be
constant. It was not:

```
165.5 169.8 179.1 171.7 172.2 177.8 172.3 172.1 168.8 177.3 164.2 160.3
                                                       spread 18.8 mm
```

A bellows cup has real stroke, and a hand presses it to a different depth
every time. Note this is *not* wrist tilt: tilt moves the cup laterally
by `L·sin θ` and vertically by only `L·(1 − cos θ)`, which is second
order. Getting 18.8 mm of vertical spread from tilt alone at `L ≈ 100 mm`
needs `θ ≈ 36°`, which would simultaneously throw ~59 mm sideways — far
more than the observed 16.7 mm residual. Tilt is real here, but it is not
what moved Z.

### 3. Wrist tilt, unrecorded

Hand guiding in `set_mode(2)` leaves orientation free, and the tool
records `pose[:3]` only. The fit assumes the flange→cup offset is purely
vertical, which requires RPY pinned at `[180, 0, 0]`. It was not pinned
and not checked.

In quadrature these three reach 16.7 mm comfortably.

### The scale figure is a red herring

`best_scale = 0.9375` looks alarming against the tool's stated "should be
within ~1% of 1.0", and it invites a hunt for a depth-scale or intrinsics
fault. It should not.

The standard error on a fitted uniform scale is
`σ_noise / (√n · r_rms)`, where `r_rms` is the point spread about the
centroid. For 12 points over roughly 400×300 mm at a single height,
`r_rms = √((400/√12)² + (300/√12)²) ≈ 144 mm`, giving

```
σ_scale ≈ 16.7 / (√12 × 144) = 0.034
```

So 0.9375 is **1.9σ from 1.0** — ordinary noise at n=12, not a separate
fault. The "~1% of 1.0" tolerance only holds for clean data.

The camera path was checked as well and is self-consistent: `fixed_common`
aligns colour **to depth** (`rs.align(rs.stream.depth)`) and deprojects
with the **depth** stream's intrinsics, whose distortion coefficients are
legitimately zero because the D435's depth stream is already rectified.
There is no intrinsics mixup to find.

**Conclusion: the camera is not implicated. The procedure is.**

---

## The fix: the arm holds the target

The plate method in `fixed_calibrate.py` was always the better design,
and it was abandoned only because a 100–150 mm matt red plate was not
available. But the plate was never the point. **The point is that the
arm holds the target, so the target-to-flange offset is rigid and never
has to be measured.**

A cube can do that job.

> Stick one cube to the cup — by hand, crookedly, it does not matter.
> Drive the arm through the pose grid. At each pose, pair the camera's
> measurement of the held cube's top face against the flange pose the
> controller reports.

| error source | hand touch | arm holds the cube |
|---|---|---|
| lateral centring | ±11 mm random | **absorbed into `t`** — constant |
| press depth | 18.8 mm spread | **no contact event at all** |
| wrist tilt | free, unrecorded | arm-commanded, pinned |
| cube nudged between measure and touch | possible | **cube is held** |
| height coverage | one plane | **full 3D grid** |
| samples | 12, each a manual operation | 27, unattended |

The grab offset is absorbed exactly the way the tool length is: it is a
constant vector in the flange frame, so with yaw held fixed across the
grid it merges into `t` and vanishes. This is the same argument the
`fixedcam` README makes for the plate, and it does not care how large the
target is or how it came to be attached.

**Yaw must be held fixed during the grid.** If yaw varies, `R_i · w` is
no longer constant and the offset stops being absorbable. The picker
holds yaw fixed too — a round cup is rotationally symmetric, so there is
no reason for it not to.

### What it costs

The cube is 40 mm where the plate was 120 mm, so each sample is noisier.
Three things offset that:

- **27 poses instead of 12**, and unattended, so more is cheap.
- **Pass 2 takes the centre from `minAreaRect` of the top face**, which
  is fixed by the face's *extremes*. The cup's shadow sits in the middle
  and does not move an extreme. A cube's outer edge is a crisper feature
  than a card's cut edge.
- The bias that wrecked the first touch attempt — a 45° view averaging
  the top face together with a same-coloured side face, measured at
  16.2 mm on the synthetic bench — is a *camera-frame* problem. Pass 2
  works in the base frame, where the top face can be isolated by height.

### Why two passes, still

Pass 1 has no transform yet, so it cannot say which way is up and has to
use the coloured blob's centroid. That is biased toward the camera, and
the bias *changes* with position, so a single pass fits the bias and
reports a small residual while being wrong. Pass 2 re-reads the same
recorded frames using pass 1's rough transform, isolates the top face by
height in the base frame, and takes `minAreaRect`. It costs no extra
motion.

---

## Non-negotiables for the first run

These exist because the last calibration failed silently and was trusted.

1. **Repeatability gate before anything else.** Measure one static cube
   N times and report the spread. If a stationary target does not repeat
   under ~2 mm, no fit is worth collecting. The previous run had no such
   check, which is why 12 bad samples produced a confident-looking file.
2. **Refuse to save a bad fit.** The old tool wrote `handeye.json` at
   16.7 mm RMS with a warning, and everything downstream then trusted it.
   A fit above threshold must not become the active transform.
3. **Independent verification, not the residual.** A good RMS over the
   volume the target was *carried* through says nothing about the volume
   cubes actually *lie* in. Cube in each corner, tape measure, top-down
   plan in robot millimetres.
4. **Record RPY on every sample**, and assert it is what was commanded.
   The previous run stored `pose[:3]` and could not detect its own
   violated assumption.

## Carried over from `fixedcam`, as design not as import

- transform the whole depth frame to base coordinates *before* any
  geometry; a 45° mount then becomes a sparse overhead view
- "is there an object" = "are there points more than 8 mm above the
  taught surface" — geometry, not appearance, so no model and no GPU
- the **vacuum switch**, never the camera, decides whether a cube was
  picked, and it is asked again after the lift
- the fixed camera can be knocked and nothing downstream would know, so
  something rigid and always-visible is stored as a drift witness and
  checked before the first move
- `get_vacuum_gripper()` is unusable on this controller — it internally
  calls `get_tgpio_output_digital()`, which faults with error 28 100% of
  the time (see `05-fixed-camera\SUPPORT_CASE.md`). Read `get_tgpio_digital(0)`
  directly instead.

## Runtime

Windows-native Python 3.12, no ROS in the loop. `numpy`, `opencv-python`,
`pyrealsense2`, `xarm-python-sdk` — all already installed. Arm at
`192.168.1.197`, controller v1.6.9, SDK 1.18.4.

## Running it

Launchers live in two places and are kept identical:

    launchers\                                     (in this tree)
    Desktop\XArm6\Sorting Cell (blue cubes)\       (alongside the others)

```
1 - Offline Checks.bat          arithmetic only, safe anywhere
2 - Measure Check A             cube on the table   - NOTHING MOVES
3 - Measure Check B             cube on the cup     - NOTHING MOVES
```

Check B switches the vacuum on and reads the seal switch. It never
commands a pose. Run A before B: the difference between them is the cost
of the cup and its shadow, and it is what decides whether the held-cube
calibration is worth writing.

## Order of work

1. calibration by held cube, with the repeatability gate — **the blocker**
2. verification against a tape measure in the corners
3. teach the support surface
4. detect: cloud → base frame → height threshold → clusters → colour
5. dry run, then the stationary pick
6. only then, the turntable

# xArm 6 — Real-Time Catch

Picking a cube **while it is still moving**. No pause, no waiting for it
to settle, no open-loop dive onto a target that has stopped: the arm
watches the cube travel, works out where it is going, goes to a point it
has not reached yet, and closes as it arrives.

> **Status: built, checked offline, never run on hardware.** Every
> number that only a robot can measure is marked below under
> [Before the first run](#before-the-first-run). Read that section first.

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.

---

## Why this is its own project

Two projects in this cell already look like this one, and neither can be
extended into it. Both are correct for their own job, and both end in
*the target is stationary by the time the fingers arrive*:

| | control law | ends with |
|---|---|---|
| `02-vision-pick\vision_pick3.py` | scan → **wait for the cube to stop** → aim once → descend open-loop | a still cube |
| `03-handoff\handoff_pick.py` | watch → **follow** the cube → dive while the hand is **momentarily steady** | a still hand |
| **this one** | watch → **fit a velocity** → go where it *will be* → **close on the clock** | nothing ever stops |

Nothing in either sibling is modified. What is reused, read-only, is the
v3 calibration and the v3 model — see [What is reused](#what-is-reused).

## The constraint that shapes everything

The D435 measures nothing closer than **~190 mm**, and the camera sits
**~142 mm** (`zC`) above the fingers. At the grasp pose the camera is
therefore about **129 mm** from the cube top — *inside its own blind
zone*. The arm cannot watch the cube arrive. That is geometry, not
tuning, and `test_catch.py` asserts it as the project's premise.

So this project does not **chase**. Chasing spends the entire time budget
closing a gap that keeps moving, and then arrives blind at a position
guess carrying every tracking error accumulated on the way.

It **ambushes**:

```
WATCH from 680mm, straight down
   │
   ├─ fit a straight line to the cube's path in the ARM's frame
   │     (base-frame mm, so the arm's own motion cancels out exactly)
   │
   ▼
PLAN — walk forward along that line and take the first point that is
       reachable, not too far into a bend, and that the arm can get to
       at least 0.6s before the cube does
   │
   ▼
GO THERE FIRST ── hover above it ── one last look to re-time the arrival
   │
   ▼
DROP into the path, fingers open WIDE, and sit still
   │
   ▼
CLOSE on the clock ──► the gripper's grasp sensor gives the verdict
```

The arm is parked and settled when the cube arrives, so the only thing
left uncertain is **timing, not position** — and a timing error of `dt`
costs just `dt × speed` of displacement. At the speeds this accepts that
is a few millimetres. A position error would have cost the whole catch.

## What it refuses, and why that is the interesting part

Refusing costs one cube. A confident prediction built on bad data costs a
collision with whatever is pushing it. Every refusal names what to change.

**Curvature is judged by extrapolated drift, not by the fit residual.**
This was the one real design error found while building it, and it is
worth stating plainly because the wrong answer is so much more obvious
than the right one.

The natural instinct is to reject a bending path by its **fit residual** —
how far the samples sit from their own best-fit line. That does not work.
The residual measures scatter *across the observed window*; what matters
is how wrong that line is *a couple of seconds later*, at the ambush
point. The two are related by the **square** of the horizon ratio, so
they are nothing like each other. The test suite pins the exact case:

> an arc of radius 60 mm at 80 mm/s fits its own window to **5.5 mm** —
> comfortably inside any sane residual gate — and is **100.7 mm** out
> when extrapolated 2 s ahead.

To catch that by residual the threshold would have to be a fraction of a
millimetre, far below the detector's own noise, and it would then refuse
every real cube instead. So curvature is measured directly, as the rate
at which the velocity **vector** rotates, and converted into the only
units that matter — millimetres of predicted error at the moment of the
catch. Only the part of the turn that survives 2σ of measurement noise is
charged, so a straight track carrying ordinary scatter reads as **zero**
drift rather than as a slow bend.

The rest of the refusals:

| refused | because |
|---|---|
| still watching it | fewer than 5 samples, or under 0.35 s of baseline. Ten samples in 60 ms fit a confident line to pure noise |
| too much scatter | the detector is flickering between objects, or the cube is half-occluded |
| rising or falling | a falling cube cannot be met by an arm that must be parked half a second early |
| barely moving | under 15 mm/s the floor picker does this better — it has a closed-loop mid-descent correction this cannot use |
| too fast | see [the speed limit](#the-speed-limit) |
| out of reach | the path never crosses the cell. Sending it *across* rather than *away* is usually the fix |
| past me before I get there | points on the path were reachable, but none far enough ahead |
| curving too much | the drift above |

Note that the last two are distinguished deliberately. Reporting "out of
reach" when the real blocker was the clock sends you to move the cube
sideways when the actual fix is to slow it down.

### The speed limit

`MAX_CATCH_SPEED` is not a free parameter. The fingers wait
`CATCH_OPEN_EXTRA_MM/2` clear of the cube on each side, and displacement
error is `timing error × speed`, so

```
v_max = ½ · (CATCH_OPEN_EXTRA_MM / 2) / CLOSE_TIMING_SD_S
```

— the gap divided by the timing jitter, halved for safety. It currently
works out at **~172 mm/s**. Widen the fingers and the catchable speed
rises with them; the test suite asserts that relationship holds so the
two can never drift apart.

### Which way the fingers close

A square cube offers two equally valid grips, 90° apart. Given that free
choice, the closing axis is put as near **perpendicular to the travel**
as possible: closing *across* the direction of travel makes the open
fingers a **gate** the cube passes through, so an early or late close
still finds it somewhere in the channel. Closing *along* the travel makes
them a **wall** the cube runs into. This is an optimisation on top of a
grip that is correct either way, which is exactly why the finger-axis
convention it depends on can be left unverified.

## What is reused

Read-only, unchanged, from `02-vision-pick\`. All of it describes the **camera
mount**, the **fingers**, and **what a cube looks like** — none of it says
anything about the cube standing still, so all of it carries over:

- `calib3.json` — the pixel→mm Jacobian
- `grip_ref.json` — the camera-to-finger constant `zC`, and the p0↔g0 anchor
- `cube_model.pt` — **no retraining.** It was trained with heavy colour
  and rotation augmentation, and a moving cube still looks like a cube.
  The model is *borrowed* from `detect_cube` rather than re-loaded, so
  only one copy sits in VRAM.
- `places.json` — the drop spots, when they have been taught

**No new calibration.** `test_catch.py` asserts that `aim_moving` is
bit-identical to `v3.aim_from_pose` in X, Y and yaw; only the height
differs, and only because v3 clamps its answer into a window around the
floor, which is meaningless for a cube in motion.

Detection is floor-free, following the same reasoning as the handoff
project — a moving cube may be sliding on the floor, skidding across a
bench or being carried through the air, so there is no single surface to
subtract. One difference from that sibling matters: **never a multi-frame
median.** Six frames at 30 fps is 200 ms, and 200 ms of a cube's motion is
precisely the thing being measured. Stability is recovered instead by
fitting a line through *single-frame* samples, which filters noise without
pretending the cube stood still.

## Files

| file | what it is |
|---|---|
| `catch_pick.py` | the application — watch, plan, ambush, close, verify, deliver |
| `catch_track.py` | the velocity fit, the drift measurement, the intercept planner, and `ArmTiming` |
| `catch_detect.py` | fast single-frame floor-free detection, timestamped from the depth frame |
| `catch_common.py` | constants, geometry, the aim math, gripper widths |
| `catch_view.py` | **camera only, nothing moves** — watch the tracker work |
| `test_catch.py` | 51 offline checks: no arm, no camera, no model |

## Running it

```bat
launchers\1 - Catch View (no arm).bat
```

Start here, every time the cell or the lighting changes. It moves
nothing and needs no arm. Slide a cube across the view and watch the
fitted arrow, the residual, and the live verdict. **A cube that does not
read CATCHABLE here will be refused by the picker too** — this way you
find out without an arm in the room.

```bat
launchers\2 - Real-Time Catch.bat
```

The real thing. Needs `calib3.json`, `grip_ref.json` and `cube_model.pt`
in `02-vision-pick\`. Send the cube in a straight line, at a steady speed, across
the middle of the cell — and **let go of it**; a cube still in your hand
is the handoff project's job, and that one is built to keep your fingers
out of the grab.

```bat
launchers\3 - Offline Checks.bat
```

Arithmetic only. Safe anywhere, any time.

## Before the first run

Four numbers are derived from geometry rather than measured, and the
robot is the only thing that can settle them. None of them is dangerous
if wrong — each fails toward a refused catch — but each is worth an hour.

1. **`ArmTiming.DEFAULT_RATE_MM_S`** (180 mm/s). How fast the tool
   actually travels at `CATCH_SPEED`. `catch_pick` measures this on every
   move and **prints the measurement when it finishes** — run it once and
   copy the number in. It was 110 at first, and that taught a lesson
   worth keeping: over-estimating the arm's time pushes the intercept
   further along the cube's path until the whole path is outside the
   cell, so an over-cautious rate does not produce a careful catch — it
   produces *"I cannot reach that"* for every cube that was catchable.
2. **`GRIPPER_CLOSE_S`** (0.35 s). How long the fingers take to travel
   from the waiting opening to closed, at `CATCH_GRIPPER_SPEED`. Time it
   once. The close is scheduled this far *before* the predicted arrival,
   so an error here is a systematic early or late catch.
3. **`CLOSE_TIMING_SD_S`** (0.08 s). The jitter in the above. It sets the
   speed limit outright.
4. **`MIN_CAM_MM`** (190 mm). Inherited from the handoff project's
   measurement, not from the datasheet. Worth confirming on this camera.

Safety, unchanged from the handoff project and for the same reason — this
arm moves to a spot a cube is travelling toward, which may be a spot a
**hand** is travelling toward:

- arm at **25 deg/s** (the floor picker runs 60), fingers at **1500** (the
  floor picker runs 5000, the SDK maximum)
- the arm **stops between every step** — it never streams continuous motion
- ESC or `q` in the console stops it between steps
- **the e-stop is the only thing that stops it during a move.** Keep it in
  hand. Between committing and closing the arm is *deliberately sitting
  still in the cube's path* with the fingers open; that is the design, not
  a fault.

## What the tests do not cover

`test_catch.py` is arithmetic on synthetic tracks. It cannot tell you:

- whether a real cube sliding across *this* cell fits a line well enough
  to bet an arm on — run `catch_view.py`, it shows the residual live
- whether the four numbers above are right
- whether a cube arriving at an open gripper deflects off a fingertip
  instead of entering. That is the failure mode the wide opening and the
  perpendicular closing axis exist to prevent, and only hardware can say
  whether they are enough. The gripper's grasp sensor — never the camera —
  is what reports the truth.

## Repository convention

`.gitignore` is an **allow-list**: everything is ignored, and tracked
files are opted back in explicitly. Capture images, run logs and
annotated frames accumulate in this folder and must not enter the repo.
When you add a source file, add a `!` line for it.

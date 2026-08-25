# Handoff — taking a cube out of your hand

A **separate project** from `..\02-vision-pick\`. Nothing in the v3 floor picker
was changed: `vision_pick3.py` and everything it uses still behave
exactly as they did. This project sits alongside it, imports its
calibration read-only, and adds the one thing v3 structurally cannot do —
follow a **moving** cube that is **not on the floor**.

---

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.


## Why it couldn't just be a flag on v3

Two things in the floor pipeline block it, and both are load-bearing
there — they aren't oversights to be switched off.

**1. Detection is floor-referenced.** `detect_cube.py` keeps only depth
pixels standing ≥10 mm proud of a floor plane, and `vision_pick3` then
demands the cube top sit 8–98 mm above that floor. A cube held 300 mm up
fails both, and your hand joins the same blob. `handoff_detect.py`
replaces the global floor with a **local** reference — the nearest
surface inside each YOLO box — so it works at any height over anything.

**2. The control law waits for stillness.** `track_until_still` exists to
make sure the cube has *stopped* before an open-loop descent. A hand
never stops. `handoff_pick.py` servos instead: it follows the cube
continuously and commits only during a brief steady window.

What *is* reused, unchanged and read-only: `calib3.json` (the pixel→mm
Jacobian) and `grip_ref.json` (the camera-to-finger constant `zC` and the
p0↔g0 anchor). Those describe the **camera mount and the fingers**, not
the floor, so they are just as valid for a cube in mid-air. `aim_hand()`
is verified in the test suite to produce **bit-identical X, Y and yaw** to
the proven `v3.aim_from_pose`; only the height differs, and only because
v3 clamps its answer into a window around the floor.

No retraining. The same `cube_model.pt` is used — it was trained with
heavy colour and rotation augmentation, and a cube in a hand still looks
like a cube.

---

## The one hard limit worth understanding

The D435 cannot measure closer than ~190 mm, and the camera sits ~142 mm
(`zC`) above the fingers. So the cube leaves the camera's range about
**~60 mm before the fingers reach it**. Every straight-down pipeline in
this cell has that constraint — it is exactly why v3's look pose is
180 mm up.

Consequence: the arm tracks your hand down to 180 mm above the cube, and
**the last stretch is open-loop**. That is why it waits for your hand to
be still for about a second before diving, and why the **gripper's own
grasp sensor**, not the camera, decides whether it actually got it.

It is not "blind grabbing" — it is closed-loop right up to the last
180 mm, then a short committed dive with a hardware check at the end.

---

## Using it

Order matters. Never run step 2 before step 1 looks right.

| Step | Launcher | Arm moves? |
|---|---|---|
| 0 | `0 - Self Test (offline)` | no — no camera either |
| 1 | `1 - Hand View (no motion)` | **no** |
| 2 | `2 - Hand Pick (arm moves)` | yes |

**Step 1** is the gate. It shows every cube found, marks the one it would
go for, prints where that is in the arm's own coordinates, and says GREEN
or RED with the *same words the picker uses*. If it doesn't read your
hand-held cube reliably here, nothing should move.

**Step 2** follows and takes it. Press **ESC** or **q** with the console
focused to stop. Keep the e-stop in hand.

### How to hold the cube

* **By the bottom edges**, fingertips clear of the top face. The camera
  measures the *nearest surface* inside the cube's box — a finger over
  the top is what it will measure instead of the cube.
* Roughly **8–43 cm above the floor** (the exact band is printed at
  startup, and comes from the camera's near limit and the wait height).
* Near the middle of the cell.
* Drifting is fine — it follows. The dive needs about a second of calm.

### What it does with the cube

If `..\02-vision-pick\places.json` exists (taught with the v3 *3b — Teach Drop
Spots*), the cube is delivered to the spot for its colour — hand it a red
one, it goes to the red spot. Otherwise the arm lifts it and waits for
you to press Enter before releasing it back into your hand.

---

## Safety, and where it comes from

| Measure | Value | Why |
|---|---|---|
| Joint speed / accel | 25 °/s, 400 °/s² | v3 runs 60/1000; this works next to a person |
| Gripper speed | 1500 of 5000 | v3 runs the fingers at the SDK maximum. Stall detection is a *reaction* — slower fingers travel less before it triggers |
| Motion style | stop between every step | discrete `wait=True` moves, never streamed continuous motion |
| Max step per iteration | 60 mm XY, 50 mm Z | bounds how fast a wrong detection can move the arm |
| Minimum hold height | 80 mm above the floor | a cube on the floor can never be mistaken for one in a hand |
| Grab bite point | 38 % down from the top | ~4 mm more clearance from fingertips than v3's mid-height |
| Opening width | cube + 30 mm | narrower fingers are less likely to meet the hand |
| Commit gate | 3 consecutive steady samples | the hand must actually be still |
| Abort | ESC / q, any time | plus the e-stop, which is the real one |

The gripper stalls on contact rather than crushing, and the grasp-status
register (fw 3.6.0) reports what it caught. Still: **hold the cube by the
bottom edges** so the fingers close well above your fingertips.

`slow_down()` must run **before** `setup_arm()` — the finger speed is
applied by `enable_gripper()`, which `setup_arm` calls, so setting it
afterwards would leave the fingers at full speed for the whole run. The
test suite asserts the values actually land on the module the moves use.

One honest gap: ESC is checked **between** steps. During a single move
the script is blocked waiting for it to finish, so ESC won't land until
it completes. Steps are short (≤60 mm; the final dive is 180 mm, about a
second), but inside that window **the e-stop is the only stop**.

---

## Files

| File | What it is |
|---|---|
| `handoff_common.py` | geometry, safety limits, calibration loading, the arm-motion → pixel-motion prediction |
| `handoff_detect.py` | floor-free cube detection (YOLO box + local nearest-surface depth) |
| `handoff_view.py` | live view, **no motion** — the gate before anything moves |
| `handoff_pick.py` | the follow-and-take loop |
| `test_handoff.py` | 44 offline checks — no arm, no camera, no model |

---

## What is proven, and what is not

**Proven offline** (`test_handoff.py`, all 44 passing): the aim matches v3
exactly in X/Y/yaw; the anchor maps back onto itself; the height band
lands exactly on the camera's near limit; the pixel prediction holds a
stationary cube stationary across arm moves to 0.01 px; step capping;
nearest-cube selection and lock behaviour; and the detector's gates —
including the two that matter most, that a **flat surface is not a cube**
and that a **merged touching pair is refused rather than grabbed at the
seam**.

**Not yet tested on hardware** — nothing here has run against a real
hand:

* whether detection is *reliable enough* frame-to-frame on a real hand
  holding a real cube (step 1 answers this, with no motion);
* the loop rate in practice. Each step is a `wait=True` move, so the loop
  runs at roughly 2–3 Hz. Fast enough to follow a drifting hand, not fast
  enough for a fast one;
* whether the blind final 180 mm lands where the follow step said.

**Known residue, accepted:** two cubes touching *diagonally* at about
half-cube offset merge into a square blob that passes both shape gates,
so the aim lands on the seam. No fill threshold separates that from a
real top face on glass. The consequence is bounded — the fingers close on
nothing, the grasp sensor says so, and it retries. Don't hand over two
cubes stuck together.

If the loop rate turns out to be the limit, the upgrade is xArm mode 1
(`set_servo_cartesian`) for continuous streamed motion instead of
discrete `wait=True` steps. That trades away the per-move error check and
joint-sweep guard, so it is deliberately *not* what this first version
does.

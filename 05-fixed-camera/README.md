# xArm 6 â€” Fixed-Camera Picker (vacuum cup)

The camera comes off the wrist and goes on a stand, and the two-finger
gripper is replaced by a suction cup. One frame sees the whole cell, all
the time, so there is no scan pose, no search pattern, and no waiting for
anything to settle.

> **Status: built, 44 offline checks pass, never run on hardware.**
> Read [Before the first run](#before-the-first-run).

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.

---

## The question this project exists to answer

> The camera is stationary, and it is not where the arm is. How does the
> arm know where the object is?

With the camera on the wrist, the answer was long: the mapping from a
pixel to a millimetre depends on how high the wrist is, which way it is
pointing, and how far away the cube is. That is what `calib3.json`'s
Jacobian, `grip_ref.json`'s `zC`, `camera_offset()` and `aim_from_pose()`
compute, and it has to be re-evaluated on every move.

With the camera bolted down, the answer is one line:

```
p_base = R Â· p_cam + t
```

The camera does not move relative to the robot, so there is exactly one
rigid transform between them and it never changes. Twelve numbers,
measured once. `p_cam` comes free from the depth camera â€” deprojecting a
pixel and its depth gives a point in the camera's own millimetres.

## The idea everything else is built on

**Transform the whole depth frame into the robot's coordinates before
doing any geometry at all.**

After that step, "up" is base +Z, and it makes no difference where the
camera is bolted or at what angle it looks. A 45Â° side view becomes a
**virtual overhead view**. Then:

| question | answer in the base frame |
|---|---|
| is there an object here? | are there points more than 8 mm above the taught surface |
| where is it? | the centre of its top face, in millimetres the arm already speaks |
| how tall is it? | top minus surface |
| can the cup seal on it? | is the top face flat, level, and wider than the cup |

This is why the oblique mount costs far less than it looks like it
should. Working in *camera* coordinates, a 45Â° view wrecks every gate the
old detector has â€” a cube's top face is a foreshortened parallelogram,
the squareness test fails, the "floor" is a different depth in every row.
Working in *base* coordinates, the oblique view is just a **sparser
sampling of the same overhead picture**.

It also deletes the bugs that came with the old frame. The mirrored yaw,
the parallax depth scale, `wrap90` anchoring, the camera-offset term, the
floor-coefficient plumbing â€” every one of those existed only because the
camera moved.

### And YOLO is gone

Not replaced â€” **gone**. `cube_model.pt` is never loaded. In an image,
"is that a cube" is a question about appearance, which needs a trained
net. In base-frame coordinates it is a question about geometry, which
needs none. That removes ~2.5 GB of dependency, the GPU, and the retrain
that used to be needed every time the floor or the lighting changed.

## What the suction cup changes

Suction simplifies the geometry rather than complicating it. Fingers grip
the **sides** of a cube at some fraction of its height, so the grasp
height depended on how tall the cube was. A cup grips the **top** â€” which
is exactly the surface the camera measures.

| | two fingers | suction cup |
|---|---|---|
| grasp point | the sides, at mid-height | the top face centre |
| grasp Z | `top_z âˆ’ (h âˆ’ h_cal)Â·frac` | `top_z âˆ’ CUP_PRESS_MM`, any height |
| calibration cube height | had to be typed in | **not needed** |
| yaw | measured, and matched to the cube | **irrelevant** â€” a round cup is symmetric |
| width | set the finger opening | only has to be wide enough to seal on |
| "did I get it?" | the grasp-status register | the **vacuum switch** |

Two error sources, one typed number and a whole calibration step
disappear. What replaces them are two gates that fingers never needed â€”
the top face has to be **flat** and **roughly level**, or a cup just
hisses â€” and one new hazard: **a cup can lose its seal while carrying**,
which fingers do not do. So the vacuum switch is re-read after the lift,
not only at the pick.

`get_vacuum_gripper()` returns `-1` off / `0` nothing held / `1` **held**.
That is a pressure measurement at the cup, not an inference from anything
the camera said â€” the only trustworthy answer to "did I pick it up" comes
from the hardware that is holding it.

## The trick that makes the tool length disappear

The calibration target is **a flat plate stuck to the cup**. Each pair
recorded is

```
p_cam  = the camera's measurement of the held plate's top face
p_base = the FLANGE pose the controller reports at that instant
```

and the plate's top face **is** the cup's contact plane. So the fitted
transform does not map "camera point â†’ where that point is". It maps
**"camera sees a surface here" â†’ "put the flange HERE to put the cup on
it"**. The flange-to-cup distance is *inside* the answer. It is never
measured, so it can never be measured wrong.

**Why a plate and not a cube:** the cup covers the top of whatever it
holds, and the top is the one surface that matters. On a plate wide
enough, the top face stays visible as a ring all round the cup. It has to
be wide compared with the cup's standoff â€” seen from the side, the cup
casts a shadow across the plate roughly as long as it is tall, and on a
small plate that shadow eats the far edge and drags the measured centre
toward the camera. 100â€“150 mm square is ample.

The project works in what the code calls the **grasp frame**: X and Y are
true base millimetres, Z is offset by the flange-to-cup distance. Z
*differences* are exact, which is all anything uses â€” and unlike the
finger version, **the cup touches the surface at exactly `floor_z`**, so
the floor clamp is simply the surface plus a margin.

### Why the calibration runs twice over the same frames

Pass 1 has no frame to work in yet, so it uses the **centroid of the
visible coloured surface**. That is biased â€” toward the camera, and away
from the cup's shadow â€” and the bias *changes* as the plate moves across
the view. A single pass cannot see this: it fits the bias and reports a
small residual.

Pass 2 re-reads the same frames with pass 1's rough transform, which is
enough to say which way is up. In the base frame a minimum-area rectangle
is fixed by the plate's *extremes*, so neither the viewing angle nor the
shadow in the middle moves its centre. It costs no extra motion. On the
synthetic bench â€” which renders the cup and its shadow on purpose â€” it
takes the fit from **1.06 mm to 0.55 mm RMS**.

## Where accuracy actually comes from

In rough order of size:

1. **How close the camera is.** Depth noise and sampling density both get
   worse with range. At ~1.2 m a 30 mm cube's top face is ~90 points at
   848Ã—480; at 700â€“900 mm it is far better. This dominates everything
   below.
2. **848Ã—480 depth** â€” the D435 depth sensor's native resolution, the one
   its accuracy is specified at. Running 640Ã—480 throws away a quarter of
   the lateral samples for nothing.
3. **The High Accuracy preset** â€” fewer depth pixels, much better ones.
   The right trade for measuring a small flat top.
4. **Temporal filtering** â€” free noise reduction on a static scene.
   (It must be **off** for the moving-cube phase; averaging a moving
   object smears it along its own path.)
5. **Calibrating where you pick.** The grid's lowest level is wherever
   the arm is standing when you start it â€” you jog the plate to just
   above the floor first. A transform fitted only in the upper half of
   the cell is *extrapolating* down to where it is actually used.
6. **Lights off.** This camera's depth is measurably better in the dark
   on the glass floor (12% no-data vs 20%).

## What a fixed camera costs

**It can be knocked, and nothing downstream would ever know.** A wrist
camera cannot come loose without the arm noticing â€” its calibration is
re-anchored on every move. A camera on a stand nudged by a sleeve at ten
in the morning makes every pick after that wrong, in silence, with the
detector still reporting confident cubes at confident millimetre
positions.

So something has to witness it. It cannot be the working surface, because
this cell's glass floor is invisible to the depth camera. But whatever
lies **under** the glass is visible, rigid, and bolted to the same room
as the robot, so its apparent height in the robot's frame is a constant â€”
and stops being one the moment the camera moves. `floor_base.json` stores
it; the picker checks it before its first move and refuses to run if the
scene has shifted by more than 40 mm.

**The arm is in the picture.** Slice a vertical object at cube height and
the slice is cube-sized and cube-shaped. Two guards close it: anything
with points *above* the cube band in the same footprint is refused as
"part of something taller", and clusters near the TCP are ignored
outright â€” the arm always knows where its own hand is.

**The old projects stop working.** With one camera, moving it off the
wrist means `vision_pick3`, `handoff_pick` and `catch_pick` cannot run
until it is remounted â€” and they need the two-finger gripper back as
well.

## Files

| file | what it is |
|---|---|
| `fixed_vacuum.py` | which wiring, and does the seal work â€” **no motion** |
| `fixed_calibrate.py` | the hand-eye calibration â€” **the arm moves** |
| `fixed_floor.py` | teach the support surface â€” **camera only** |
| `fixed_detect.py` | cloud â†’ base frame â†’ clusters â†’ cubes |
| `fixed_common.py` | the transform, the grasp frame, the cup, the drift witness |
| `fixed_view.py` | camera view + top-down plan â€” **nothing moves** |
| `fixed_pick.py` | the picker, with a `--dry` mode that moves nothing |
| `test_fixed.py` | 44 offline checks: no arm, no camera, no model |

What is reused from `02-vision-pick\`, read-only: `vision_common`'s `movej` (with
its joint-sweep guard, per-move error check and latched-warning clear),
`ik`, `moveto` and the run log. Nothing else â€” and deliberately **not**
`vision_common.setup_arm`, which raises when the two-finger modbus
gripper does not answer, which is what a vacuum tool does.
`calib3.json` and `grip_ref.json` are both dead here.

## Running it, in order

```bat
launchers\1 - Offline Checks.bat
```
Arithmetic only. Safe anywhere.

```bat
launchers\2 - Vacuum Check.bat
```
Once, when the cup is first fitted. Tells you which wiring it is on and
lets you watch the switch change as you press something against the cup.
Nothing moves.

```bat
launchers\3 - Camera View (no arm).bat
```
**Before you commit to a camera position.** It runs uncalibrated and just
shows the camera. Check every corner of the working area is in frame and
that cube *tops* are visible everywhere, not just cube sides. Then don't
touch the stand again.

```bat
launchers\4 - Calibrate Camera.bat
```
**The arm moves.** You need the red plate on the cup, no other red object
in view, and the arm already jogged so the plate sits ~20 mm above the
floor. Under 5 mm RMS is good.

```bat
launchers\5 - Teach Floor.bat
```
Plate off, cubes off the floor, opaque paper over the working area, arm
parked out of view. Nothing moves.

```bat
launchers\3 - Camera View (no arm).bat
```
Again â€” now the right-hand panel shows the top-down plan **in robot
millimetres**. Put a cube in a corner and check it against a tape
measure. A fit can have a small residual and still be wrong out where the
cubes actually lie, because it was fitted over the volume the plate was
*carried* through and it is being used over the volume cubes *lie in*.

```bat
launchers\6 - Pick DRY RUN (no motion).bat
```
Walks the whole floor and prints every pose it would command, and the
reason for every cube it would refuse. Nothing moves.

```bat
launchers\7 - Pick.bat
```
The real thing.

## Before the first run

Numbers only hardware can settle. None is dangerous if wrong â€” each fails
toward a refusal â€” but each is worth getting right.

1. **`TOOL_MASS_KG` in `fixed_common.py`.** The vacuum tool's real mass,
   for `set_tcp_load`. Declaring it wrong does not break a pick but it
   degrades the controller's collision detection, which is a safety
   feature and not one to leave guessed.
2. **`CUP_DIA_MM`** (18) and **`CUP_MARGIN_MM`** (4). A cube's top face
   must be at least `CUP_DIA + 2Â·MARGIN` = 26 mm across, so a 30 mm cube
   has 2 mm to spare. If the real cup is bigger, small cubes stop being
   pickable â€” that is arithmetic, not a bug.
3. **`CUP_PRESS_MM`** (2). How far below the measured top face the flange
   is driven. The calibration is taken with the plate already stuck to
   the cup, so in principle no press is needed; it is not zero because a
   bellows cup hangs longer under gravity than it sits when pressed.
   **This is the first number to tune if picks fail to seal** â€” raise it
   in 1 mm steps. The floor clamp means it can never reach the table.
4. **`PLACES` in `fixed_pick.py`** (or `places_xy.json`). Drop spots as
   base-frame **XY only** â€” the release height is derived from the taught
   floor plus the cube's own height, so it adapts per cube and can never
   drive into the table. Keep both spots **outside** `WORK_X`/`WORK_Y`
   so delivered cubes are not seen again and re-picked.
5. **`WORK_X` / `WORK_Y` in `fixed_common.py`.** The crop that keeps the
   cell walls, the bench and the operator out of the candidate list.
6. **`FLAT_MAX_MM` / `TILT_MAX_DEG` in `fixed_detect.py`.** The seal
   gates. Sized against this camera's depth noise rather than measured;
   the viewer prints both for every cube so they can be tightened once
   there are real numbers.

Safety, and why:

- arm at **40 deg/s** â€” between the floor picker's full speed and the
  handoff project's caution, because this arm is aimed by a camera on a
  stand and the stand has not yet been trusted
- the arm **stops between every step**; nothing streams continuous motion
- every pre-grasp pose is IK-checked before anything is committed, and
  the delivery swing is checked **before the cube is picked up** â€”
  finding out mid-delivery means faulting while carrying
- the vacuum switch, never the camera, gives the verdict on whether a
  cube was picked, **and it is asked again after the lift**
- the cup is forced OFF at startup, so an arm that boots still holding
  something from a previous run is not a surprise
- **the e-stop is the only thing that stops it during a move.**

## What the tests do not cover

`test_fixed.py` renders a synthetic cell â€” floor, cube, camera 45Â° off to
the side, and for the calibration a plate with the cup standing on it and
casting its shadow â€” then pushes all of it through the real detector and
the real calibration. So a sign error in the transform is caught before
an arm reaches for the mirror image of a cube. It cannot tell you:

- whether real depth on **this** camera at **this** angle resolves a cube
  top well enough. Run the viewer; it shows the refusal reason
- whether the cup actually seals on a cube's top face at this aim
  accuracy â€” that is what `CUP_PRESS_MM` and the margin exist for, and
  only hardware can say
- whether the stand is rigid enough that the drift check stays quiet
- whether the calibration holds up in the corners it was not fitted in.
  That is what the tape measure in the viewer is for

## Next: moving cubes

This is the stationary half. The moving half is a much smaller step from
here than it was from the wrist camera, and for one reason: the
`04-realtime-catch\` project had to **ambush** â€” predict, park, close on the clock
â€” because at the grasp pose the wrist camera is inside its own blind zone
and the arm goes blind exactly when it matters. A fixed camera has no
blind zone. It watches the cube *and the tool* the whole way in, so
genuine closed-loop chasing becomes possible, and the arm's own motion no
longer contaminates the measurement.

One change will be needed there: **turn the temporal filter off**
(`Camera(temporal=False)`). It is free accuracy on a scene that is not
moving and quiet corruption on one that is.

## Repository convention

`.gitignore` is an **allow-list**: everything is ignored and tracked
files are opted back in by name. Adding a source file means adding a `!`
line for it.

# xArm 6 — Dynamic Vision Pick & Place with Colour Priority

*[النسخة العربية](README.ar.md)*

An autonomous pick-and-place application for a **UFACTORY xArm 6** with an
**Intel RealSense D435** mounted on the wrist (eye-in-hand). The arm finds
cubes on the floor by itself, picks them up, and sorts them by colour:

> **every RED cube first, then every BLUE one, then anything left over.**

The arm only moves on to blue once no red cube is visible from *any* of its
vantage points — so watching the order it works in is the plainest proof
that the colour recognition is actually working.

Nothing about the cubes is taught in advance. Cube size, grab height, grasp
angle and gripper width are all measured live from the depth image, so the
same program handles cubes of different sizes anywhere in the cell, and it
waits for a cube that is still being moved to hold still before committing.

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.

---

## How a pick works

```
SCAN pose (camera straight down)
   │
   ├─ nothing seen? ──► RISE for a wider view, then sweep a 13-vantage GRID
   │                     (centre-outward; stops at the first cube found)
   │
   ▼
TRACK the cube until it has been still for ~1 s   (it may still be moving)
   │
   ▼
AIM  — pixel→mm mapping measured during calibration, generalized to
       whatever vantage the cube was found from
   │
   ▼
DESCEND half-way over the cube ──► LOOK AGAIN and correct the aim
   │
   ▼
ROTATE to the grasp angle ──► DESCEND to a grab height computed from DEPTH
   │
   ▼
GRAB (width from the measured cube size) ──► VERIFY with the gripper's
       grasp sensor ──► DELIVER to the drop spot for that colour ──► repeat
```

If the cube shifts mid-descent, or the fingers close on nothing, the arm
retreats, reopens and retries rather than aborting the run.

### Detection

A custom-trained YOLOv8n locates cubes of **any** colour (COCO has no cube
class, so the model is trained from the arm's own captures). YOLO boxes carry
no rotation, so a depth-based `minAreaRect` inside each box recovers the
centre, the grasp angle and the true size in millimetres. Colour is then
classified from the hue of the top-face pixels.

### Safety

Every arm movement in this project is an IK-guarded joint move:

- every target pose is IK-preflighted before anything commits,
- a **120° joint-sweep guard** refuses any move that would swing the arm
  wildly to reach a Cartesian neighbour,
- controller errors are fatal, latched warnings are cleared and named,
- the payload is declared, gripper return codes are checked,
- a floor-plausibility guard refuses a cube whose measured floor depth
  disagrees with the expected one (this is what stops the arm being driven
  into the floor if the depth reading is bad),
- interrupted runs that left a cube in the fingers are recovered on restart.

**Keep the e-stop in your hand.** This program moves a real robot.

---

## Requirements

**Hardware**
- UFACTORY xArm 6 + UFACTORY 2-finger gripper (firmware 1.6.9 here)
- Intel RealSense D435 on the wrist, straight down at the scan pose
- An NVIDIA GPU for training (an RTX 4060 Laptop 8 GB was used)

**Software** — Windows-native Python 3.12 (**not** WSL: the camera is on a
Windows USB port; the xArm SDK is plain TCP either way).

```bash
python -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install --user -r requirements.txt
```

Install torch **first**, from the CUDA index. Installing `ultralytics` on its
own will silently replace a CUDA torch with the CPU-only build.

---

## Setting it up on your own arm

Run these once, in order. Each has a launcher in `launchers/`, or run the
Python file directly — every script takes the robot IP as an optional first
argument (`python vision_pick3.py 192.168.1.10`), defaulting to
`192.168.1.197`.

| # | Step | Script | What it produces |
|---|------|--------|------------------|
| 1 | Capture training data | `capture_dataset.py` | `dataset/` — the arm hovers at several heights and labels cubes automatically from depth, no hand-labelling |
| 2 | Train the detector | `train_cubes.py` | `cube_model.pt` |
| 3 | *See-through floor only* | `teach_floor.py` | `floor_ref.json` |
| 4 | Calibrate the camera | `auto_calibrate3.py` | `calib3.json`, `grip_ref.json` |
| 5 | Teach the drop spots | `teach_place.py` | `places.json` |
| 6 | **Run it** | `vision_pick3.py` | — |

Steps 1–2 can be skipped if the bundled `cube_model.pt` already detects your
cubes — check with `audit_dataset.py`, or just watch a run. Steps 3–5 are
specific to your cell and cannot be skipped.

**Step 3** is only needed if your floor is glass or clear acrylic. The D435's
infrared passes straight through such a surface, so it is measured once
through an opaque sheet and stored. Cover the floor with paper or card, take
the cubes off, run it, then remove the sheet. On a normal opaque floor, delete
`floor_ref.json` and the floor is fitted live every frame.

**Step 5 matters for the colour sorting to be visible at all.** Until blue has
its own taught spot, every cube — red and blue — is delivered to the red spot.
Put the two spots somewhere central; a drop spot far out on one side makes the
arm refuse cubes on the opposite side of the cell under the joint-sweep guard.

---

## Configuration

The values worth knowing, all near the top of `vision_pick3.py`:

```python
ROBOT_IP     = "192.168.1.197"          # or pass as argv[1]
PLACE_RED    = [436.5, -498.4, 407.6, -171.1, -9.8, -93.5]
PLACE_BLUE   = None                     # taught via teach_place.py
PICK_ORDER   = [("red", "RED"), ("blue", "BLUE"), (None, "any remaining")]
MAX_CYCLES   = 14                       # runaway guard
```

The third `None` phase is required, not optional: a colour hunt never matches
a cube the classifier could not call, so without it those cubes would be left
on the floor forever.

`SCAN_POSE` and the arm speeds live in `vision_common.py`. If the arm ever
trips collision detection, halve `JOINT_ACC` before touching `JOINT_SPEED`.

---

## Files

| File | Role |
|------|------|
| `vision_pick3.py` | The application — search, track, aim, grab, colour-sorted delivery |
| `vision3.py` | Camera pipeline, grab-height maths, floor reference, drop-spot config |
| `vision_common.py` | Arm setup, IK-guarded motion, sweep guard, gripper control, logging |
| `detect_cube.py` | YOLO + depth refinement + colour classification |
| `camera_test.py` | Capture constants shared by the pipeline |
| `auto_calibrate3.py` | Measures the pixel→mm mapping and the camera/finger anchor |
| `teach_floor.py` | Stores a floor plane the camera cannot see |
| `teach_place.py` | WASD-jog teaching of the per-colour drop spots |
| `gripper_reset.py` | Recovery when a run stops with the fingers closed |
| `capture_dataset.py` | Auto-labelled dataset capture |
| `train_cubes.py` | Trains / warm-starts `cube_model.pt` |
| `audit_dataset.py` | Flags frames where the model and the depth labeller disagree |

### The bundled `.json` files

`calib3.json`, `grip_ref.json` and `floor_ref.json` are the working reference
for **this** cell — this arm, this camera mount, this floor. They are included
so the setup is reproducible and reviewable, not because they will fit another
robot. Regenerate them with steps 3–4 above; keeping someone else's
calibration will simply aim the arm at the wrong place.

`places.json` is **not** included — teach your own drop spots.

### Retraining

`train_cubes.py` warm-starts from the existing `cube_model.pt` and backs the
old one up to `cube_model_prev.pt` first, so new scenes can be added without
forgetting the old ones. `python train_cubes.py [epochs] [fresh]` — pass
`fresh` to train from scratch instead. Roll back by copying
`cube_model_prev.pt` over `cube_model.pt`.

`capture_dataset.py` appends to an existing dataset rather than overwriting
it. Keep cubes **separated** while capturing — touching or stacked cubes make
a frame ambiguous and it gets discarded.

---

## Notes from the build

A few things that cost real time and are worth knowing if you extend this:

- **Never reuse a stored IK joint vector across a run.** Firmware 1.6.9 seeds
  IK from the arm's *current* joints, so a solution computed at startup
  belongs to a different configuration branch later on, and the sweep guard
  then measures branch-to-branch distance instead of the actual move.
- **A transparent floor is not a training problem.** YOLO found every cube on
  glass with the white-floor model. What breaks is the *depth* gate that the
  detector, the labeller and the grab height all rely on — hence `teach_floor`.
- **The gripper has a grasp sensor** (firmware ≥ 3.4.3, `get_gripper_status()`
  bits 0–1: 2 = object caught). The grab verdict comes from that sensor, not
  from comparing finger position, and not from vision.
- **`set_gripper_position(wait=True)` is not a success signal.** It returns 0
  after polling a gripper that never moved at all.
- With several similar cubes in view, "pick the largest" flips between them
  frame to frame. The tracker locks onto one cube by proximity instead.

# xArm 6 — robot cell projects

Six projects built on one UFACTORY xArm 6 between **2026-07-19** and
**2026-08-16**, from a picker that repeats two memorized poses up to a
vision-guided cell that finds cubes, sorts them by colour and tries to
take them out of a moving hand.

Each project is a self-contained folder with its own README, its own
`requirements.txt` where it needs one, and its own `launchers\` folder of
double-clickable `.bat` files. They share the arm and the calibration
conventions, not code — a later project does not import an earlier one
unless its README says so.

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.

---

## The cell

| | |
|---|---|
| **Arm** | UFACTORY xArm 6, controller firmware v1.6.9, wired at `192.168.1.197` |
| **Tools** | UFACTORY 2-finger parallel gripper; later a vacuum suction cup |
| **Camera** | Intel RealSense D435 — wrist-mounted first, later moved to a stationary side stand |
| **Workspace** | Enclosed, with a labelled floor grid |
| **Computers** | Windows 11 PC — vision and arm control run natively on Windows; ROS 2 Jazzy + MoveIt live in a WSL Ubuntu 24.04 distro. RTX 4060 laptop GPU for training. |

---

## The projects

| # | Folder | What it does | Status |
|---|---|---|---|
| 1 | [`01-teach-and-repeat/`](01-teach-and-repeat/) | Repeats one taught pick spot and one taught place spot. No camera. | **Working on hardware** |
| 2 | [`02-vision-pick/`](02-vision-pick/) | Wrist camera finds cubes anywhere on the floor, picks red first then blue | **Working on hardware** — best run 6 cubes |
| 3 | [`03-handoff/`](03-handoff/) | Takes a cube out of a moving human hand | Ran on hardware, never completed a take |
| 4 | [`04-realtime-catch/`](04-realtime-catch/) | Catches a moving cube by ambush rather than by chasing | Built and tested offline only |
| 5 | [`05-fixed-camera/`](05-fixed-camera/) | Stationary camera + suction cup, no trained model | Hardware proven, calibration not yet good enough |
| 6 | [`06-sorting-cell/`](06-sorting-cell/) | Rebuild of the measurement and calibration foundation | Design record — started 2026-08-16 |

### 1 — Teach and repeat

The simplest possible job: show the arm one place to pick from and one
place to put down, and let it repeat. Positions are memorized, not seen.
These scripts run **inside WSL** against the xArm Python SDK; the `.bat`
files in `launchers\` call into the Ubuntu-24.04 distro.

### 2 — Vision pick

The D435 on the wrist, a YOLO cube detector trained on this cell
(`cube_model.pt`, tracked here), depth for the grab height, and a colour
priority — red cubes before blue. Includes the dataset capture and
training scripts, the auto-calibration routine that measures the
pixel→millimetre Jacobian, and the floor-teaching tool.
Also documented in Arabic: [`README.ar.md`](02-vision-pick/README.ar.md).

**The trained detector ships with this repo**, along with the data behind it:

| Path | What it is |
|---|---|
| [`02-vision-pick/cube_model.pt`](02-vision-pick/cube_model.pt) | The model the application loads. YOLO26n fine-tuned on this cell, 80 epochs, **mAP50 0.979** |
| [`02-vision-pick/models/`](02-vision-pick/models/) | The earlier YOLOv8n-trained model, kept for comparison, plus notes on both |
| [`02-vision-pick/dataset/`](02-vision-pick/dataset/) | The 233 labelled frames it was trained on — one class, `cube` |
| [`02-vision-pick/training/`](02-vision-pick/training/) | Per-epoch metrics, PR curves and confusion matrices for both runs |

So the detector can be re-trained from scratch here: `python train_cubes.py`.

### 3 — Handoff

A separate tree from project 2, and deliberately so. Floor-referenced
detection cannot see a cube held 300 mm in the air, and a control law
that waits for stillness will wait forever on a hand. This project
replaces the global floor plane with a local one inside each detection
box, and servos continuously instead of descending open-loop. It reuses
project 2's `calib3.json` and `grip_ref.json` read-only.

### 4 — Real-time catch

Catching a cube that is already moving. Chasing loses; the arm predicts
where the cube will be and waits there. Fully built and covered by
offline checks, but it has never been run on the hardware.

### 5 — Fixed camera + vacuum

The camera comes off the wrist and onto a side stand, so the scene no
longer moves when the arm does. Depth is transformed into the base frame
*before* any geometry, height above the floor decides whether something
is there, and colour is read from the same points — no trained model at
all. 44 offline checks pass. It has never completed a pick, because the
hand-eye calibration in `handeye.json` is wrong by more than the suction
cup can tolerate. `CALIBRATION_QUESTION.md` and `SUPPORT_CASE.md` record
that investigation.

> **Live hardware note:** this cell has an open wrist fault — error 28 on
> any tool-IO access. Read the project README before touching it.

### 6 — Sorting cell

Not a fourth attempt at the same code. A rebuild of the measurement and
calibration foundation the other projects are blocked on, so that "pick
the blue cubes, leave the red ones" has something trustworthy to stand
on. Design record and measurement tooling; nothing has picked yet.

---

## Shared tooling and documents

| Path | What it is |
|---|---|
| [`tools/`](tools/) | `reset_all.py` — clears faults, re-enables motion and returns the arm to a known state. Every project's `0 - RESET EVERYTHING.bat` calls it. |
| [`docs/project-report.md`](docs/project-report.md) | Full written report on all six projects: what each one does, what was learned, where each stands |
| [`docs/reports/`](docs/reports/) | Weekly progress reports as PDF |

---

## Running any of this

Everything is Windows-native Python except project 1, which runs in WSL.

```
cd <project folder>
pip install -r requirements.txt
```

Then use the numbered `.bat` files in that project's `launchers\` folder,
in order. They are numbered because the order matters: reset the arm,
look without moving, calibrate, dry-run, then move.

**Keep the E-stop in your hand.** Every script that moves the arm asks
for confirmation first and runs slowly by default. That is on purpose.

---

## Running it on a different arm

This code drives a real robot. Read this before pointing it at yours.

The numbers in `calib3.json`, `grip_ref.json`, `floor_ref.json` and
`handeye.json` describe **this** cell — this camera mount, these fingers,
this floor. They are tracked deliberately, as the provenance of the runs
described in the READMEs, and they are **not portable**. On any other
setup they must be re-measured with the teaching and calibration tools in
each project (`teach_floor`, `auto_calibrate3`, `fixed_calibrate`), or the
arm will confidently drive to the wrong place.

The controller IP is hardcoded per project and will need changing. The
xArm SDK, a RealSense camera and a CUDA-capable GPU (for project 2 only)
are assumed.

Every project ships offline checks that need neither the arm nor the
camera. Run those first — they are the honest way to see what the code
does before it can move anything:

```
cd 05-fixed-camera
python test_fixed.py
```

## Status, honestly

Projects 1 and 2 have done real work on hardware. Project 3 ran but never
completed a take. Project 4 has never been run on the arm at all.
Project 5 is blocked on a calibration error larger than the suction cup
can tolerate. Project 6 is a design record with measurement tooling and
has not picked anything. Each README says so in its own opening lines.

## License

[MIT](LICENSE) — © 2026 Musab Bagazi and Yazan Bal'fakeeh. Use it, change
it, build on it; keep the copyright notice.

The trained model and the 233-frame dataset are covered by the same
licence. If the work is useful in something you publish, a citation of
the two authors above is appreciated.

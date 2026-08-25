# xArm 6 — Project Report

**Cell:** UFACTORY xArm 6 (controller firmware v1.6.9) on a wired link at 192.168.1.197, inside an enclosed workspace with a labelled floor grid.
**Tools used so far:** UFACTORY 2-finger parallel gripper, and later a vacuum suction cup.
**Camera:** Intel RealSense D435 — first mounted on the wrist, later moved to a stationary side stand.
**Computers:** Windows 11 PC (vision and arm control run natively on Windows; ROS 2 Jazzy + MoveIt live in a WSL Ubuntu 24.04 distro), RTX 4060 laptop GPU for training.

**Period covered:** 2026-07-19 → 2026-08-16.

---

## Authors

Designed and built by **Musab Bagazi** and **Yazan Bal'fakeeh**.


## Summary

Six projects were built on the same arm. Two of them have completed real work on hardware; the other four run, measure and diagnose correctly but have not yet completed a pick.

| # | Project | What it does | Status |
|---|---|---|---|
| 1 | Cube picker by memorized fixed place | Repeats one taught pick spot and one taught place spot | **Working on hardware** |
| 2 | Dynamic cube picker | Wrist camera finds cubes anywhere, picks red first then blue | **Working on hardware — best run 6 cubes** |
| 3 | Hand-off picker | Takes a cube out of a moving human hand | Ran on hardware, never completed a take |
| 4 | Catch cubes in motion | Catches a moving cube by ambush | Built and tested offline only |
| 5 | Fixed camera + vacuum picker | Stationary camera, suction cup, no trained model | Hardware proven, calibration not yet good enough |
| 6 | New sorting cell | Rebuild of the measurement and calibration foundation | Started 2026-08-16, measuring now |

---

## 1. Cube picker by memorized fixed place

**The idea.** The simplest possible job: show the arm one place to pick from and one place to put down, and let it repeat. No camera, no intelligence — the positions are memorized.

**How it was done.** The arm was hand-guided to the pick spot and the drop spot, and both poses were recorded. A separate teaching step measured how far the gripper must close on the actual box, so the grip is firm without crushing.

**What went wrong and how it was fixed.** The first version drove the tool in straight lines between points. Near the far edge of the workspace this forced the wrist to spin very fast, the controller read that as over-speed and stopped mid-job. A second, subtler fault came from the controller not knowing the gripper was attached — it read normal acceleration as a collision and triggered emergency stops. The fixes were to declare the tool weight at startup, and to replace every straight-line move with a joint move. A pre-flight check now proves every waypoint is reachable before the arm moves at all, and a recovery routine handles the case where a run stops while a box is still held.

**Status.** Working. This project also produced the drop location that the vision picker still uses today.

---

## 2. Dynamic cube picker — the main achievement

**The idea.** Put the camera on the wrist and let the arm find the cubes itself: any colour, any size, anywhere on the floor, and sort them by colour into two drop spots.

**How it works.** The arm moves to a scanning pose looking straight down. A trained cube detector finds candidate objects; depth measurement then recovers each cube's true size, angle and height above the floor. The arm converts what the camera sees into where the hand must go, descends, grips, and delivers. If no cube is visible from the scan pose, it rises and then sweeps a grid of vantage points until it finds one, or proves the floor is empty.

**What had to be built along the way.**

- **A training set of its own.** No off-the-shelf model knows what a cube is, so a capture tool was written that labels its own training images automatically, using depth to separate objects from the floor. The operator only rearranges cubes between rounds. Roughly 234 images were collected and a model trained from them; an audit tool later re-checked every image and found exactly one bad frame.
- **Calibration.** Rather than measuring the camera mount by hand, the arm moves a known distance and watches how far the image shifts, which gives the relationship between pixels and millimetres directly. A one-time teach step then fixes the height at which the fingers actually close on a cube.
- **A tilted floor.** The floor was found to slope by about 30 mm across the frame, which made the near floor look like a large raised object. Every floor measurement was changed from a single number to a fitted plane.
- **A transparent floor.** When the work surface was changed to glass, the camera could no longer see it at all — it looked straight through and read a surface far below. Detection itself was never the problem (the model still found every cube). The solution was to lay opaque paper over the glass once, teach the floor's true height, and store it. The arm now works on glass with no retraining and no recalibration.
- **Reliable target tracking.** Several runs looped because the detector kept switching between two similar cubes between frames, which looked like a cube moving. The tracker now locks onto one cube and ignores its neighbours, and the descent step re-checks it is still looking at the same cube by position and colour.
- **A real grasp verdict.** The gripper turned out to have a grasp sensor that reports whether an object is actually held. The pick no longer decides success from vision — it asks the hardware.
- **A search pattern derived from measurement.** Rather than guessing where to look, a planning tool asked the controller which poses are actually reachable and mapped what each vantage point can see. This revealed that some previously used heights were never reachable at all, and that an even grid of arm positions is not an even grid of camera views, because the camera sits off to one side of the flange. The final pattern is symmetric about the cell and covers essentially everything the arm can reach.
- **Speed.** The project had been running at roughly 7% of the controller's speed limit. Speeds and accelerations were raised substantially and two redundant moves per cycle were removed, without touching any safety check.

**Results.** 38 recorded runs. The best run picked and delivered six cubes in a single session, with the gripper's own sensor confirming each grasp and reporting the true cube width. Colour priority works: all red cubes are cleared before any blue one is touched.

**Open item.** The two drop spots have never been taught, so blue cubes are still delivered to the red spot with a warning. This is one short teaching session away from being fixed.

**Publication.** This project was published to a private GitHub repository on 2026-08-08, deliberately containing only the picker itself — the large capture, dataset and log folders stay local.

---

## 3. Hand-off picker — taking a cube from a human hand

**The idea.** Instead of picking from the floor, take the cube directly out of a person's hand while the hand is free to move.

**Why it needed a separate project.** The floor picker's logic is built around a floor: it keeps only what stands proud of the ground, and it waits for the cube to stop moving. A cube held in mid-air fails both conditions, and a hand never stops. So detection had to be rebuilt without a floor, and the waiting logic replaced with continuous following.

**The physical limit that shapes the design.** The depth camera cannot measure closer than about 19 cm, and it sits above the fingers. That means the cube disappears from view roughly 16 cm before the fingers reach it — the last part of every approach is necessarily blind. The design answer is to follow the hand closely, commit only during a steady moment, and let the gripper's grasp sensor give the verdict.

**Safety.** Because a human hand is inside the workspace, the arm runs at less than half the speed of the floor picker, the fingers close more slowly than the maximum, each step is limited in size, and the operator can abort between steps.

**Status.** It ran on the arm on 2026-08-03 — four sessions. The arm followed a hand and repositioned itself for focus, but no session ended in a completed take; the runs were mostly spent telling the operator to hold the cube lower or by its bottom edges. The offline test suite still passes today.

---

## 4. Catch cubes in motion

**The idea.** Pick a cube while it is still moving, without waiting for it to stop.

**The design insight.** The problem is not detection but a sensing blackout: at the moment of grasp the camera is far too close to see anything. So chasing the cube is impossible. Instead the arm should ambush it — measure the cube's velocity, move to a point further along its path, wait there with fingers open, and close on a timer. That converts an unsolvable position problem into a timing problem.

**A lesson worth keeping.** How well a motion fit matches the data it was fitted to says almost nothing about how well it will predict the future. A curved path can match its own observation window to within a few millimetres and still be 10 cm wrong two seconds later. The prediction must be judged by how fast the direction of travel is turning, not by the quality of the fit.

**Status.** Built and tested offline in a single day (2026-08-08). It has never been run on the arm, and several constants still need one hardware session to measure.

---

## 5. Fixed camera + vacuum cup picker

**The idea.** Take the camera off the wrist and put it on a stand at the side, looking down at an angle. Change the tool from fingers to a suction cup.

**Why this is simpler, not harder.** The whole depth image is converted into the robot's own frame before any measurement is made. Once that is done, the camera's mounting angle stops mattering — a 45° side view becomes a virtual overhead view. Every awkward correction the wrist camera needed existed only because the camera moved with the arm, and none of them survive. The trained model is also unnecessary: "is there an object" becomes "are there points standing above the taught surface", which needs no GPU and no retraining when the lighting or floor changes.

**What the suction cup changed.** Fingers grip the sides at a fraction of the cube's height, so the grasp height depended on knowing the height of each cube. A cup grips the top face, which is exactly what the camera measures directly. Rotation stops mattering as well, since a round cup is symmetric. New checks were needed instead: the top face must be flat, level and large enough for the cup to seal.

**The hardware fault, and its resolution.** The first real session hit a controller fault that appeared every time anything touched the tool connector. Days of measurement narrowed it down: the fault was not the cup, the cable, the pump or the wiring, but a single software call in the vendor's library that reads one particular register. That call was hidden inside every question asked about the cup's state. Replacing it with a different read of the same sensor removed the fault entirely — a stress test went from 148 faults in 296 operations to zero in 297. The vacuum chain was then confirmed end to end: command, pump, suction, sensor and verdict.

Two measurement mistakes were found and corrected during this work, both of which had made the diagnostic tools report confidently wrong conclusions: one tool was reading the sensor on every pass and therefore causing the faults it was counting, and both fault probes issued several operations before checking for an error, so they blamed the wrong one. Distinguishing "faults when reading" from "faults when energising" turned out to be the single most important fact for the vendor's support case.

**Where it stands.** The camera works, the cup works, the sensor works and the arm moves. What is missing is calibration — the transform that tells the arm where the camera's view actually is in the robot's frame. Two attempts finished but were rejected, at about 11 mm and 17 mm of error. The cup can tolerate roughly 6 mm on a 30 mm cube, so neither is usable, and no detector work can compensate for it. The project has never completed a pick.

**Why the calibration failed.** The method asked a person to guide the limp arm by hand and press the cup onto a cube; the vacuum seal signalled contact. The seal proves contact but is blind to centring, so every sample carried several millimetres of random hand error. Worse, a hand can tilt the wrist, and the whole method assumes the tool points straight down — the recorded contact heights varied by nearly 19 mm on a flat floor, which is the signature of that error. A separate measurement bug was also found: the tool built for a flat plate was reused on a cube, and it averaged the cube's top face together with a visible side face, putting the answer about 16 mm out. That was corrected by isolating the top face properly, which brought bench error down to under 1 mm.

**Requirements already agreed for the next phase.** All cubes will be moving — a turntable first, a conveyor later, at 100–250 mm/s — and only the blue ones are to be picked. The tracking and motion-prediction layer for this is already built and tested offline, including the case of cubes riding a turntable at different radii, which a naive circle fit describes incorrectly. It cannot be validated until calibration is fixed.

---

## 6. New sorting cell (current work)

**Why a fresh start.** The fixed-camera project's design is right and its argument holds. It is blocked on one number: the accuracy of the hand-eye calibration. Repeating the same calibration method would produce the same result, so a new tree was started to rebuild the foundation — measure first, calibrate second, pick third.

**What has run so far (2026-08-16).** A measurement check that moves nothing: one cube on the table, the arm parked out of view, fifty frames measured. The result is a single number — how much the measured cube centre wanders between frames. The current reading is 2.31 mm, described by the tool itself as marginal against the 6 mm the cup allows, with a recommendation to move the camera closer before calibrating. A hand-jog tool with a wrist-verticality check has also been run, and both offline test suites pass.

This is the correct order of work: prove the measurement is good enough *before* spending a session collecting calibration points with it.

---

## Cross-project: jogging and teaching tools

Several tools exist for moving the arm by hand or by keyboard. It is worth recording which were actually used, because the answer is not the obvious one.

| Tool | Type | Used? |
|---|---|---|
| Keyboard jog (vision project) | WASD keys | Never — it saves nothing and left no record |
| Mouse teleop (vision project) | Tool follows the mouse, button as deadman | Never run on the arm |
| Calibration teach step (v2) | Typed jog commands | Used once, 2026-07-21 |
| Calibration teach step (v3) | WASD keys | Used, 2026-07-22 — the working picker rests on it |
| Drop-spot teaching | WASD keys | Never completed — this is the outstanding gap in the vision picker |
| Fixed-camera jog | Keyboard, avoids the tool connector | Never run |
| Touch calibration | Hand guiding, arm goes limp | Used heavily, 2026-08-11 to 08-12 |
| New cell hand jog | Hand guiding with verticality check | Used today |

The purpose-built keyboard and mouse jog tools were never used on hardware. Every jog that mattered was one embedded inside a calibration or teaching step. Hand guiding, which replaced them, turned out to be both the fastest method and the source of the calibration error described above — which is why the newest tools guide roughly by hand and then let the arm perform the precise final motion itself.

---

## Safety practices used throughout

- The assistant never commands real arm motion; the operator runs every script with the emergency stop in hand.
- Every waypoint is checked for reachability before any motion begins, so an impossible job stops with the arm still parked.
- Every move is a joint move, checked against the controller's error state immediately afterwards.
- Any move requiring an unusually large joint rotation is refused, because the controller can otherwise return a flipped arm configuration that swings through the cell.
- The tool's weight is declared at startup, or ordinary acceleration is misread as a collision.
- Every run writes a log to disk. This was added after a failure whose only record was a console window that had been closed.

---

## Where things stand

**Working today:** the memorized-place picker, and the dynamic vision picker with colour priority, floor teaching, search pattern and grasp verification.

**One short session away:** teaching the two drop spots, which would make colour sorting visible instead of implied.

**Blocked:** the fixed-camera vacuum picker, and with it the moving-cube work, until the hand-eye calibration is redone with the corrected method.

**Untested on hardware:** the catch-in-motion project in its entirety.

**Currently in progress:** measurement quality for the new sorting cell — the camera may need to be moved closer before calibration is attempted.

---

*Report generated 2026-08-16 from run logs, saved calibration files and project notes.*

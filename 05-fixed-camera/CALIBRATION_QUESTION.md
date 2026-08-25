# Eye-to-hand calibration keeps failing (16.7mm RMS, scale 0.94) — what am I getting wrong?

I am calibrating a **fixed** RealSense D435 to an **xArm 6** with a suction
cup, and two attempts have come out far too loose. I think I have found
the latest cause, but I would like a sanity check on the whole approach
before I burn another run — and on two numbers I cannot fully explain.

## The cell

| | |
|---|---|
| Arm | UFACTORY xArm 6, controller v1.6.9, xArm-Python-SDK 1.18.4 |
| Camera | Intel RealSense D435, **fixed on a stand** off to one side, angled down roughly 45°, ~0.7–1.0 m from the work area. It does not move. |
| Depth | 848×480, ~90% valid pixels. **This librealsense build does not expose the High Accuracy visual preset**, so it runs the camera default. |
| Tool | UFACTORY xArm Vacuum Gripper, cup ~18 mm diameter, with a working vacuum switch (object-detect on tool DI0) |
| Scene | White table surface; targets are matt red/blue cubes 30–40 mm |
| Host | Windows 11, Python 3.12 |

## What I am fitting, and why it is not a checkerboard

Rather than a checkerboard and AX=XB, every sample pairs:

- **p_cam** — the camera's measurement of the **top face centre** of a cube
  standing on the table, in camera coordinates
- **p_base** — the **flange pose the controller reports** at the moment the
  cup is sealed onto that same cube's top face

Then a rigid transform (Kabsch/Umeyama, scale forced to 1) maps p_cam →
p_base, with an outlier pass that drops pairs beyond max(15 mm, 3×RMS)
and refits.

The intent is that the transform directly answers "the camera sees a
surface here → put the flange **here** to put the cup on it", so the
flange-to-cup distance is absorbed and never has to be measured. The
vacuum switch is what defines "contact", so it is a pressure measurement
rather than a visual judgement, and it is the same physical event as a
real pick.

**Assumption this rests on:** the flange→cup offset is **purely vertical**,
i.e. the wrist is at RPY = [180, 0, 0] for every sample.

## Attempt 1 — 12 points, RMS 10.95 mm

```
pass 1 (blob centroid)      RMS  9.24 mm   worst 18.41   scale 0.9892
pass 2 (top-face outline)   RMS 10.95 mm   worst 24.03   scale 0.9730
```

**Cause I found:** I had reused a detector written for a flat *plate*,
which takes the centroid of the whole coloured blob. On a flat plate that
is the top face. On a **cube seen at 45°** it averages the top face
together with a visible side face — both the same colour, similar area —
so the answer sits about 16 mm low and camera-ward, and the offset varies
with where the cube is in the view, so it distorts the fit rather than
cancelling.

Measured on a synthetic render of the same geometry: **16.2 mm** error.

## Attempt 2 — same 12 points, better detector, RMS 16.70 mm (worse)

I rewrote the measurement to isolate the top face without needing any
transform:

1. Fit the **dominant plane of the whole scene** in *camera* coordinates
   (SVD, robust trim) — this gives "up" without a hand-eye transform,
   which is the point, since none exists yet.
2. Take a generous 8 mm band off the top of the cube's point cloud, fit a
   plane to that band, then keep only points within **2.5 mm of that
   plane** (a single band always also catches the side face's top rim,
   which sits at the same height and drags the centroid camera-ward in
   proportion to band thickness — measured 10 mm band → 5.1 mm error,
   3 mm → 1.9 mm, linear).
3. Take the centre from **minAreaRect of the face**, not the mean, because
   a slanted surface is sampled unevenly (near half = more pixels/mm) and
   the mean is pulled ~2.5 mm toward the camera.

On the synthetic bench this goes **16.2 mm → 0.8 mm mean, 1.0 mm worst**,
and is stable at 1, 2 and 3 mm of simulated depth noise.

On the real cell it got **worse**:

```
12 pairs   RMS 16.70 mm   worst 32.48 mm   best-fit scale 0.9375
per-pair residuals:
  [17.13, 7.52, 9.42, 5.19, 9.62, 32.48, 24.14, 6.53, 15.7, 17.64, 6.99, 22.4]
```

## The clue I think explains it

The flange **Z at the moment of seal**, over the 12 touches:

```
165.5  169.8  179.1  171.7  172.2  177.8
172.3  172.1  168.8  177.3  164.2  160.3      -> spread 18.8 mm
```

Same cube, same floor, cup sealing on its top face — so this should
repeat to within the floor's flatness. It does not.

Between attempts I had switched the operator interface to the xArm's
**joint teaching mode (`set_mode(2)`)**, so the arm could be guided onto
each cube by hand instead of jogged with keys. In that mode the **wrist
orientation is free**, and I only ever recorded `pose[:3]` — I never
constrained or even checked RPY. If the wrist was tilted differently on
each touch, the flange→cup offset is no longer vertical, no rigid
transform relates the pairs, and both the residual and the scale go bad.

**Fix I am about to try:** hand-guide roughly over the cube, then have the
arm square the wrist to RPY = [180, 0, 0] via IK and **descend
automatically in 2 mm steps until the vacuum switch reports sealed**
(60 mm limit, floor-fenced). That removes both the tilt and the
variable press depth. I have also added a re-measure after each touch to
drop pairs where the cube slid more than 6 mm, and a report of the
seal-height spread.

## What I would like an expert's view on

1. **Is the pairing itself sound?** Pairing a camera-measured surface point
   with the flange pose at contact, and fitting a rigid transform — is
   this a legitimate eye-to-hand calibration, or am I better off with a
   checkerboard and a proper AX=XB solve? The appeal is that the tool
   length never has to be measured; the cost is that every sample depends
   on a physical touch.

2. **Does wrist tilt fully explain an 18.8 mm Z spread**, or should I
   suspect something else as well? Could a compliant/bellows cup sealing
   at different compressions account for a chunk of it?

3. **What does `best_scale = 0.9375` tell you?** My fit forces scale to 1
   (both frames are metric millimetres) and returns the best-fit uniform
   scale only as a diagnostic — it should land within ~1% of 1.0.
   A 6% deficit seems large for a rigid-transform violation alone.
   Depth scale error on a D435 at 0.7–1.0 m? Or is this exactly what a
   varying tool offset would produce?

4. **Should I solve for the tool offset as an unknown** instead of
   assuming it is vertical and constant — i.e. fit R, t **and** the
   flange→cup vector jointly? That would make wrist orientation
   harmless, at the cost of more unknowns and more samples.

5. **How many points, and spread how?** I am using 12 over roughly a
   400×300 mm area at one height. Is a single height a mistake for a
   transform that has to work over a range of heights?

6. **How much am I losing to the camera default preset?** High Accuracy is
   not exposed in this build. Is it worth changing the librealsense build
   for it, at this range and target size?

7. **Is a 30–40 mm cube simply too small a target?** I do not have a flat
   coloured plate 100–150 mm across, which is why I went to the touch
   method in the first place. If a bigger target is the real answer, what
   is the minimum that would make this comfortable?

Happy to share the raw pairs, the point cloud, or the detector code if
useful.

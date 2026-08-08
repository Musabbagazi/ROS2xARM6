#!/usr/bin/env python3
"""Self-calibrating camera-to-arm setup, v3 (YOLO detector, dynamic Z).

PHASE A (automatic, ~1 min, every calibration): the arm watches a cube,
makes two small sideways moves (+40mm X, +40mm Y) and MEASURES how image
pixels map to arm millimeters (rotation, scale, axis flips). It then
auto-centers the camera over the cube to PROVE the mapping and
self-corrects the sign if needed. Saves calib3.json.

PHASE B (one-time EVER, ~2 min): drive the open fingers around one cube
at its mid-height with the WASD keys, press ENTER. This measures the
XY anchor, the vertical constant zC (which lets every future grab
height come from depth alone - any cube, anywhere, any size) and the
grip stall anchor. Saves grip_ref.json. You only redo this if the
camera mount, the fingers or the SCAN pose change - NOT per cube spot.

Needs cube_model.pt (run capture_dataset.bat + train_cubes.bat first).

Usage: python auto_calibrate3.py [robot_ip]
Keep the e-stop in hand. The cube must NOT move during calibration.
"""
import os
import sys

import numpy as np
from xarm.wrapper import XArmAPI

import vision_common as vc
import vision3 as v3

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.197"

STEP_MM = 40.0        # size of the two calibration moves
CENTER = (320.0, 240.0)
TEACH_START_Z = 340.0  # auto-descend here before the finger teach


def jog_to(arm, dx=0.0, dy=0.0, dz=0.0, dr=0.0):
    """Small relative move as an IK-checked JOINT move. 0 on success."""
    code, p = arm.get_position(is_radian=False)
    if code != 0:
        print("  position read failed (%s)" % code)
        return 1
    target = [p[0] + dx, p[1] + dy, p[2] + dz, p[3], p[4], p[5] + dr]
    try:
        vc.moveto(arm, "jog", target)
        return 0
    except RuntimeError as e:
        print("  jog failed: %s" % e)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        code, ew = arm.get_err_warn_code()
        if code != 0 or ew[0] != 0:
            raise RuntimeError("arm did not recover from a failed jog "
                               "(error %s)" % (ew if code == 0 else code))
        return 1


def phase_a(arm, cam, scan_angles):
    """Measure the pixel->mm Jacobian and prove it by self-centering."""
    ref = cam.stable("cal3A_ref")
    if ref is None:
        raise RuntimeError("no stable cube in view - move it toward the "
                           "middle and rerun")
    p_ref = np.array(ref["pixel"])
    print("Cube seen at pixel [%.0f, %.0f], depth %.0fmm, conf %.2f"
          % (p_ref[0], p_ref[1], ref["depth_mm"], ref["conf"]))

    deltas = []
    for name, (dx, dy) in (("X", (STEP_MM, 0.0)), ("Y", (0.0, STEP_MM))):
        pose = list(vc.SCAN_POSE)
        pose[0] += dx
        pose[1] += dy
        vc.moveto(arm, "calibration move +%s" % name, pose)
        shot = cam.stable("cal3A_%s" % name, n=5, min_found=3)
        vc.movej(arm, "back to SCAN", scan_angles)
        if shot is None:
            raise RuntimeError("lost the cube during the %s move" % name)
        deltas.append((np.array(shot["pixel"]) - p_ref) / STEP_MM)

    K = np.column_stack(deltas)          # px change per mm of ARM motion
    if abs(np.linalg.det(K)) < 1e-4:
        raise RuntimeError("degenerate calibration (cube moved?) - rerun")
    J = -np.linalg.inv(K)                # px -> mm of CUBE offset
    scale = float(np.sqrt(abs(np.linalg.det(J))))
    print("Mapping measured: %.2f mm/pixel" % scale)
    if not 0.3 <= scale <= 3.0:
        raise RuntimeError("mm/pixel %.2f is implausible - rerun" % scale)

    # ---------- self-test: center the camera over the cube ----------
    MAX_CENTER_STEP = 80.0
    shot = cam.stable("cal3A_center", n=5, min_found=3)
    centered = False
    for attempt in range(6):
        if shot is None:
            raise RuntimeError("lost the cube while centering")
        p = np.array(shot["pixel"])
        err_px = float(np.linalg.norm(p - np.array(CENTER)))
        print("centering: cube at [%.0f, %.0f], %.0f px from center"
              % (p[0], p[1], err_px))
        if err_px < 15.0:
            centered = True
            break
        move = J @ (p - np.array(CENTER))
        mag = float(np.linalg.norm(move))
        if mag > MAX_CENTER_STEP:
            move = move * (MAX_CENTER_STEP / mag)
        if jog_to(arm, dx=float(move[0]), dy=float(move[1])) != 0:
            raise RuntimeError("centering move failed - is the cube too "
                               "close to the edge of the workspace?")
        check = cam.stable("cal3A_centerchk%d" % attempt, n=5, min_found=3)
        if check is None:
            raise RuntimeError("lost the cube while centering")
        after = float(np.linalg.norm(np.array(check["pixel"])
                                     - np.array(CENTER)))
        if after > err_px * 1.1 + 5.0:
            print("  moved the WRONG way - flipping the mapping sign")
            J = -J
            if jog_to(arm, dx=-2.0 * float(move[0]),
                      dy=-2.0 * float(move[1])) != 0:
                raise RuntimeError("sign-heal move failed - rerun")
            shot = cam.stable("cal3A_centerflip%d" % attempt, n=5,
                              min_found=3)
        else:
            shot = check
    if not centered:
        raise RuntimeError("could not center on the cube - rerun")
    print("Self-test passed: camera is centered over the cube.")
    # d_ref: the depth J was measured at. J scales with depth, so users
    # of J must rescale when their reference depth differs (a reused
    # grip_ref may have been taught on a different-height cube).
    return J, scale, float(ref["depth_mm"])


def teach_grip_ref(arm, cam, scan_angles):
    """One-time finger teach -> grip_ref.json. Cube must not move."""
    print("\n===== ONE-TIME GRAB TEACH =====")
    print("The arm descends near the cube; drive the OPEN fingers so they")
    print("straddle the cube at its MID-HEIGHT, then press ENTER.")
    code, cur = arm.get_position(is_radian=False)
    if code == 0:
        try:
            vc.moveto(arm, "descend to teach height",
                      [cur[0], cur[1], TEACH_START_Z, 180.0, 0.0, 0.0])
        except RuntimeError as e:
            print("(auto-descend skipped: %s)" % e)

    g0, _ = vc.wasd_jog(arm, title="ONE-TIME GRAB TEACH")
    if g0 is None:
        raise RuntimeError("could not read the grab pose")
    print("Grab pose recorded:", [round(v, 1) for v in g0])

    stall = vc.grip(arm, "close to measure the cube", 0)
    if stall is None or stall <= 15:
        raise RuntimeError("the fingers closed on nothing - they were not "
                           "around the cube; rerun")
    print("Grip stalls at %d pulses on this cube." % stall)
    vc.grip(arm, "open", 850)

    # half-way reference view (for the mid-descent correction), taken at
    # the SCAN yaw so the runtime look matches it whatever the cube's
    # rotation is
    look_pose = [g0[0], g0[1], g0[2] + v3.LOOK_H, 180.0, 0.0,
                 vc.SCAN_POSE[5]]
    vc.moveto(arm, "look reference pose", look_pose)
    look = cam.stable("cal3B_look", n=5, min_found=3)
    p_look0 = look["pixel"] if look else None
    d_look0 = look["depth_mm"] if look else None
    if p_look0 is None:
        print("NOTE: cube not visible from half-way height - picking will "
              "run without mid-descent correction.")

    # the anchor: the same (unmoved!) cube seen from the exact SCAN pose
    vc.movej(arm, "SCAN pose", scan_angles)
    anchor = cam.stable("cal3B_anchor", n=5, min_found=3)
    if anchor is None:
        raise RuntimeError("lost the cube at the end - rerun")
    if not 250.0 <= anchor["depth_mm"] <= 750.0:
        raise RuntimeError("anchor depth %.0fmm is implausible - rerun"
                           % anchor["depth_mm"])

    h0 = anchor["height_mm"]
    floor0 = anchor["floor_mm"] or (anchor["depth_mm"] + h0)
    zc = v3.compute_zc(g0[2], anchor["depth_mm"], h0)
    if abs(zc) > 400.0:
        raise RuntimeError("zC %.0fmm is implausible - the fingers were "
                           "probably not at the cube's mid-height; rerun"
                           % zc)

    ref = {
        "note": "One-time grab reference. Redo ONLY if the camera mount, "
                "the fingers or the SCAN pose change.",
        "scan_pose": list(vc.SCAN_POSE),
        "g0": [round(g0[0], 1), round(g0[1], 1), round(g0[2], 1),
               round(g0[5], 1)],
        "p0": anchor["pixel"],
        "d0": anchor["depth_mm"],
        "w0_mm": anchor["width_mm"],
        "w0_px": anchor["width_px"],
        "angle0": anchor["angle_deg"],
        "h0": h0,
        "floor0": round(floor0, 1),
        "stall0": int(stall),
        "zC": round(zc, 1),
        "look_h": v3.LOOK_H,
        "p_look0": p_look0,
        "d_look0": d_look0,
    }
    v3.save_grip_ref(ref)
    print("grip_ref.json saved (zC = %.1fmm, stall %d, cube %.0fmm)."
          % (zc, stall, anchor["width_mm"]))
    return ref


def main():
    print("Connecting to xArm6 at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    try:
        vc.setup_arm(arm)
    except RuntimeError as e:
        print("ABORT:", e)
        arm.disconnect()
        return

    scan_angles = vc.ik(arm, vc.SCAN_POSE)
    if scan_angles is None:
        print("ABORT: SCAN pose not reachable")
        arm.disconnect()
        return

    ref = v3.load_grip_ref()
    reuse = False
    if ref is not None:
        if ref.get("scan_pose") and list(ref["scan_pose"]) != \
                list(vc.SCAN_POSE):
            print("\nSaved grab reference was taught under a DIFFERENT "
                  "SCAN pose - it is invalid and will be re-taught.")
            ref = None
        else:
            print("\nSaved one-time grab reference found "
                  "(zC %.1fmm, stall %d, taught on a %.0fmm cube)."
                  % (ref["zC"], ref["stall0"], ref["w0_mm"]))
            reuse = input("Reuse it? [Y/n] ").strip().lower() != "n"

    print("\nPut ONE cube (any color) near the middle of the floor.")
    print("It must NOT move until calibration is done.")
    if input("Start? [y/N] ").strip().lower() != "y":
        print("Aborted."); arm.disconnect(); return

    try:
        with v3.Camera() as cam:
            vc.grip(arm, "open", 850)
            vc.movej(arm, "SCAN pose", scan_angles)

            J, scale, d_ref = phase_a(arm, cam, scan_angles)
            v3.save_calib3({
                "version": 3,
                "scan_pose": vc.SCAN_POSE,
                "J": J.tolist(),
                "mm_per_px": scale,
                "d_ref": round(d_ref, 1),
            })
            print("calib3.json saved.")

            if not reuse:
                ref = teach_grip_ref(arm, cam, scan_angles)
            else:
                vc.movej(arm, "SCAN pose", scan_angles)

        print("\n===== calibration v3 complete =====")
        print("mm/pixel %.2f | zC %.1f | stall anchor %d @ %.0fmm cube"
              % (scale, ref["zC"], ref["stall0"], ref["w0_mm"]))
        print("Ready: run vision_pick3.bat")
    except RuntimeError as e:
        print("\nSTOPPED:", e)
    finally:
        try:
            arm.set_mode(0)
            arm.set_state(0)
        except Exception:
            pass
        arm.disconnect()


if __name__ == "__main__":
    main()

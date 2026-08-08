#!/usr/bin/env python3
"""Shared helpers for the xArm6 + RealSense vision scripts."""
import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs

from camera_test import detect_red_cube, CAPT

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "calib.json")
CALIB2_FILE = os.path.join(HERE, "calib2.json")

# The one fixed pose the camera always scans from (straight down).
SCAN_POSE = [450.0, 70.0, 580.0, 180.0, 0.0, 0.0]

# Speed dial. The arm's own limits (read from the controller) are
# 270 deg/s and 1146 deg/s^2, so these are still well inside them -
# 60 is 22% of the joint speed limit. Motion dominates the pick cycle;
# camera, inference and gripper are fractions of a second next to it.
# Most of our moves are SHORT (look -> align -> grasp), and short moves
# are limited by ACCELERATION, not top speed - which is why mvacc
# matters as much as the speed here.
# If the arm ever trips collision detection (error 31) or a protective
# stop, halve JOINT_ACC first, then JOINT_SPEED.
JOINT_SPEED = 60
JOINT_ACC = 1000
MAX_JOINT_SWEEP = 120.0

GRIPPER_WEIGHT = 0.82
GRIPPER_COG = [0.0, 0.0, 48.0]


# ------------------------- camera -------------------------

def capture(tag="capture"):
    """Grab one aligned frame set; detect the red cube.

    Returns (found_dict_or_None, floor_depth_mm_or_None).
    found_dict gains "height_mm" (cube height above the floor).
    Saves annotated images to captures/<tag>_*.png
    """
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            pipe.wait_for_frames()
        frames = align.process(pipe.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        intr = color.profile.as_video_stream_profile().intrinsics
        img = np.asanyarray(color.get_data())
        found, mask = detect_red_cube(img, depth, intr)

        floor_mm = None
        if found is not None:
            # floor depth = median depth in a ring around the cube.
            # Only accept samples slightly BELOW the cube top (3..150mm
            # farther) - reflective walls / glass give wild depths that
            # must not poison the height estimate.
            cx, cy = found["pixel"]
            top = found["depth_mm"]
            ring = []
            for r in (60, 75, 90):
                for t in range(0, 360, 20):
                    px = int(cx + r * np.cos(np.radians(t)))
                    py = int(cy + r * np.sin(np.radians(t)))
                    if 0 <= px < 640 and 0 <= py < 480 and mask[py, px] == 0:
                        d = depth.get_distance(px, py) * 1000.0
                        if top + 3.0 <= d <= top + 150.0:
                            ring.append(d)
            h = None
            if len(ring) >= 5:
                floor_mm = float(np.median(ring))
                h = floor_mm - top
            # sanity: a cube is as tall as it is wide. If the depth-based
            # height is missing or implausible, fall back to the width.
            w = found["width_mm"]
            if h is None or not (8.0 <= h <= 100.0) or abs(h - w) > 30.0:
                h = w
                found["height_est"] = True
            found["height_mm"] = round(h, 1)

        ann = img.copy()
        if found:
            box = np.intp(found["box"])
            cv2.drawContours(ann, [box], 0, (0, 255, 0), 2)
            label = "%.0fmm wide h=%.0fmm %.0fdeg" % (
                found["width_mm"], found.get("height_mm", -1),
                found["angle_deg"])
            cv2.putText(ann, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(CAPT, tag + "_annotated.png"), ann)
        return found, floor_mm
    finally:
        pipe.stop()


def wrap90(x):
    """Fold an angle difference into (-45, 45] - a square repeats every 90."""
    return ((x + 45.0) % 90.0) - 45.0


def capture_stable(tag="stable", n=8, min_found=4):
    """Detect the cube over n frames and take medians for stability.

    Returns a dict like detect_red_cube's plus robust "height_mm", or None.
    Angles are median-folded so square symmetry can't split the votes.
    """
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            pipe.wait_for_frames()
        hits = []
        last_img, last_found = None, None
        for _ in range(n):
            frames = align.process(pipe.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            intr = color.profile.as_video_stream_profile().intrinsics
            img = np.asanyarray(color.get_data())
            found, mask = detect_red_cube(img, depth, intr)
            last_img = img
            if found:
                hits.append(found)
                last_found = found
        if len(hits) < min_found:
            if last_img is not None:
                ann = last_img.copy()
                cv2.putText(ann, "NO STABLE CUBE (%d/%d frames)" %
                            (len(hits), n), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(CAPT, tag + "_annotated.png"), ann)
            return None

        a0 = hits[0]["angle_deg"]
        med = {
            "pixel": [float(np.median([h["pixel"][0] for h in hits])),
                      float(np.median([h["pixel"][1] for h in hits]))],
            "depth_mm": float(np.median([h["depth_mm"] for h in hits])),
            "width_mm": float(np.median([h["width_mm"] for h in hits])),
            "width_px": float(np.median([h["width_px"] for h in hits])),
            "angle_deg": a0 + float(np.median(
                [wrap90(h["angle_deg"] - a0) for h in hits])),
            "frames_found": len(hits),
        }
        # cube heuristic: as tall as it is wide
        med["height_mm"] = med["width_mm"]

        ann = last_img.copy()
        box = np.intp(last_found["box"])
        cv2.drawContours(ann, [box], 0, (0, 255, 0), 2)
        cv2.putText(ann, "%.0fmm %.0fdeg px[%.0f,%.0f] (%d/%d)" % (
            med["width_mm"], med["angle_deg"], med["pixel"][0],
            med["pixel"][1], len(hits), n), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(CAPT, tag + "_annotated.png"), ann)
        return med
    finally:
        pipe.stop()


# ------------------------- run log -------------------------

LOGS = os.path.join(HERE, "logs")


class _Tee(object):
    """Mirror a stream to a file so a crashed/closed console is not the
    only record of what happened."""

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, s):
        self._s.write(s)
        self._s.flush()
        self._f.write(s)
        self._f.flush()

    def flush(self):
        self._s.flush()
        self._f.flush()

    def isatty(self):
        return self._s.isatty()

    def fileno(self):
        return self._s.fileno()


def start_log(name):
    """Tee stdout+stderr into logs/<name>_<timestamp>.log. Returns the
    path (or None if the log file could not be opened - never fatal)."""
    import datetime
    import sys
    try:
        if not os.path.isdir(LOGS):
            os.makedirs(LOGS)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS, "%s_%s.log" % (name, stamp))
        fh = open(path, "w", encoding="utf-8", errors="replace")
    except Exception as e:
        print("(no run log: %s)" % e)
        return None
    fh.write("=== %s  %s ===\n" % (name, stamp))
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path


# ------------------------- arm -------------------------

GRIPPER_SPEED = 5000        # SDK range 1..5000 - no reason to dawdle,
                            # the stall detector is what stops it on a cube


def enable_gripper(arm):
    """(Re)arm the gripper: clear its fault, enable, position mode, speed.

    Also used as mid-run recovery. The gripper can silently lose its
    enable (any controller fault, e-stop, or a modbus hiccup does it) and
    then ACCEPTS position commands while never moving and still reporting
    error code 0 - so 'not moving' is the only symptom, and re-enabling is
    the cure. Returns True if the enable was accepted."""
    arm.clean_gripper_error()
    ok = arm.set_gripper_enable(True) == 0
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(GRIPPER_SPEED)
    return ok


def setup_arm(arm):
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    if not enable_gripper(arm):
        raise RuntimeError("gripper did not enable - check cable/power")
    gripper_supports_status(arm)    # report grasp sensing before any move
    arm.set_tcp_load(GRIPPER_WEIGHT, GRIPPER_COG)
    arm.set_state(0)
    code, ew = arm.get_err_warn_code()
    if code != 0 or ew[0] != 0:
        raise RuntimeError("arm still in error %s - check e-stop, retry" % ew)


def ik(arm, pose):
    code, angles = arm.get_inverse_kinematics(pose, input_is_radian=False,
                                              return_is_radian=False)
    return angles if (code == 0 and angles is not None) else None


class SweepRefused(RuntimeError):
    """The move needs a bigger joint swing than MAX_JOINT_SWEEP allows.

    A subclass of RuntimeError so every existing handler still catches
    it, but callers that can respond sensibly - by giving up on one cube
    rather than ending the run - can tell it apart. It is a property of
    where the arm is and where it is being asked to go (firmware 1.6.9
    seeds IK from the current joints and can return a flipped wrist), not
    a fault: nothing is broken and the next cube may be fine."""


def movej(arm, label, angles):
    print("-> %s" % label)
    c, cur = arm.get_servo_angle(is_radian=False)
    if c == 0 and cur:
        worst = max(abs(a - b) for a, b in zip(angles, cur))
        if worst > MAX_JOINT_SWEEP:
            raise SweepRefused("%s needs a %.0f deg joint sweep - refusing"
                               % (label, worst))
    code = arm.set_servo_angle(angle=angles, speed=JOINT_SPEED,
                               mvacc=JOINT_ACC, wait=True, is_radian=False)
    if code != 0:
        raise RuntimeError("%s move failed (code %s)" % (label, code))
    code, ew = arm.get_err_warn_code()
    if code == 0 and ew[0] != 0:
        raise RuntimeError("arm faulted (error %s) during %s" % (ew[0], label))
    # a warning is advisory, but it LATCHES and would make every later
    # command (especially the modbus gripper) report code=2 - clear it
    if code == 0 and ew[1] != 0:
        print("   (controller warning %s during %s - cleared)"
              % (ew[1], label))
        arm.clean_warn()


def moveto(arm, label, pose):
    angles = ik(arm, pose)
    if angles is None:
        raise RuntimeError("%s is not reachable" % label)
    movej(arm, label, angles)


def gripper_pos(arm):
    # NB: on a gripper fault the SDK returns a bare int error code instead
    # of a (code, pos) tuple - never unpack blindly.
    try:
        ret = arm.get_gripper_position()
    except Exception:
        return None
    if isinstance(ret, (tuple, list)) and len(ret) == 2:
        code, pos = ret
        return pos if (code == 0 and pos is not None) else None
    return None


HAS_WARN = 2            # APIState.HAS_WARN - "uncleared warning present"
GRIP_TOL = 60           # pulses an open move may fall short by


# ------------------- "am I holding something?" -------------------
#
# The gripper answers this itself. Its status register (bits 0-1) is the
# hardware's own verdict, measured at the fingers:
#     0 = stop state   - the fingers reached the commanded position,
#                        nothing was in the way
#     1 = move state   - still moving, no verdict yet
#     2 = grasp state  - the fingers were STOPPED BY AN OBJECT
# Available on gripper firmware >= 3.4.3 (ours reports 3.6.0).
#
# This replaces inferring a grab from the finger position and a
# vision-measured cube width. That inference was only ever an
# approximation of this signal, and it failed for real: a cube measured
# 3mm too wide made the predicted stall 3mm too high, the fingers stalled
# inside the "closed on nothing" tolerance, and the script opened the
# gripper on a cube it was holding. The status bit cannot be fooled that
# way - it does not depend on the cube's size, on PULSES_PER_MM, or on
# the stall0/w0_mm calibration pair at all.

HOLDING, EMPTY, UNKNOWN = "holding", "empty", "unknown"
GRIPPER_STATUS_MIN_VER = (3, 4, 3)
_supports_status = None         # cached per process (a modbus read each)


def gripper_supports_status(arm):
    """Does this gripper report the grasp-status register (fw >= 3.4.3)?

    Cached: the answer cannot change while the script runs, and each
    query costs three modbus reads."""
    global _supports_status
    if _supports_status is None:
        _supports_status = False
        try:
            code, ver = arm.get_gripper_version()
            if code == 0 and isinstance(ver, str):
                # '*.*.*' when the version read failed -> int() raises
                parts = tuple(int(x) for x in ver.split(".")[:3])
                _supports_status = parts >= GRIPPER_STATUS_MIN_VER
                print("   (gripper firmware %s - grasp sensing %s)"
                      % (ver, "available" if _supports_status
                         else "NOT available, falling back to finger "
                              "position"))
        except (ValueError, TypeError, AttributeError):
            pass
    return _supports_status


def grasp_status(arm):
    """Ask the gripper whether it is holding something.

    Returns HOLDING, EMPTY, or UNKNOWN (gripper too old, read failed, or
    the fingers are still moving so there is no verdict yet). Callers
    must treat UNKNOWN as "use another test", never as "empty" - opening
    on a bad verdict drops the cube."""
    if not gripper_supports_status(arm):
        return UNKNOWN
    try:
        code, status = arm.get_gripper_status()
    except Exception:
        return UNKNOWN
    if code != 0 or not isinstance(status, int):
        return UNKNOWN
    low = status & 0x03
    if low == 2:
        return HOLDING
    if low == 0:
        return EMPTY
    return UNKNOWN              # 1 = still moving - no verdict


def clear_warn(arm, where=""):
    """Clear a latched controller WARNING (errors stay fatal).

    xArm warnings latch: once set, the controller answers HAS_WARN(2) to
    every later command, and each modbus gripper call then logs an
    '[SDK][ERROR] ... -> code=2' line. Left alone this hides real gripper
    failures, so clear it - but print it, because e.g. warn 14 ("no
    solution") means something asked for an unreachable pose.
    Returns the warning code that was cleared, or 0."""
    try:
        code, ew = arm.get_err_warn_code()
    except Exception:
        return 0
    if code == 0 and ew and ew[1]:
        print("   (controller warning %s%s - cleared)" % (ew[1], where))
        arm.clean_warn()
        return ew[1]
    return 0


def grip(arm, label, position):
    """Command the gripper and VERIFY it moved.

    The SDK's wait=True is NOT a success signal: __check_gripper_position
    returns 0 as soon as the fingers stop changing, so a gripper that
    never moved at all reports success. An opening move that is silently
    dropped leaves the fingers closed and the following grab catches
    nothing - so opens are checked against the position actually reached.
    A stalled open almost always means the gripper lost its enable, so the
    retry re-enables it first. Closing moves legitimately stall early (on
    the cube), so they are not checked."""
    target = int(position)
    print("-> gripper %s (%d)" % (label, target))
    clear_warn(arm, " before gripper %s" % label)
    before = gripper_pos(arm)

    code = arm.set_gripper_position(target, wait=True)
    if code == HAS_WARN:                 # advisory, not a failure
        clear_warn(arm, " from gripper %s" % label)
        code = 0
    if code != 0:
        raise RuntimeError("gripper %s failed (code %s)" % (label, code))
    pos = gripper_pos(arm)

    opening = before is not None and target > before + 20
    if not (opening and pos is not None and pos < target - GRIP_TOL):
        return pos

    # It did not open. Re-arm the gripper and try again - twice, because
    # the enable sometimes only takes on the second pass.
    for attempt in (1, 2):
        print("   (gripper stuck at %d of %d - re-enabling and retrying "
              "[%d/2])" % (pos, target, attempt))
        clear_warn(arm)
        if not enable_gripper(arm):
            print("   (gripper enable was refused)")
        arm.set_gripper_position(target, wait=True)
        pos = gripper_pos(arm)
        if pos is None or pos >= target - GRIP_TOL:
            print("   (recovered - gripper at %s)" % pos)
            return pos
    raise RuntimeError(
        "gripper %s did not open (stuck at %d, wanted %d) even after "
        "re-enabling it - the next grab would close on nothing.\n"
        "         The fingers are almost certainly still clamped on a "
        "cube: free it by hand, then run '5 - Gripper Reset.bat'."
        % (label, pos, target))


# ------------------------- WASD keyboard jog -------------------------

def _flush_keys():
    import msvcrt
    while msvcrt.kbhit():
        msvcrt.getch()


def wasd_jog(arm, title="JOG"):
    """Drive the arm with single keypresses (Windows console).

    Returns (final_pose, waypoints) where waypoints are the poses the
    user recorded with SPACE. Finish with ENTER.
    """
    import msvcrt

    step = 10.0     # mm per keypress
    print("""
==== %s - drive with keys (no Enter needed) ====
   w / s : +X / -X        a / d : +Y / -Y
   q / e : Z up / Z down  z / c : rotate gripper -/+
   o / p : gripper more OPEN / more CLOSED (50 steps)
   1 2 3 4 5 : step size 1 / 5 / 10 / 20 / 40 mm
   SPACE : record this pose      i : show pose
   ENTER : finish
=================================================""" % title)

    waypoints = []

    def show():
        c, pp = arm.get_position(is_radian=False)
        g = gripper_pos(arm)
        if c == 0:
            print("  x=%.1f y=%.1f z=%.1f yaw=%.1f | grip=%s | step=%.0fmm"
                  % (pp[0], pp[1], pp[2], pp[5],
                     "?" if g is None else int(g), step))

    def jog(dx=0.0, dy=0.0, dz=0.0, dr=0.0):
        c, pp = arm.get_position(is_radian=False)
        if c != 0:
            print("  position read failed")
            return
        target = [pp[0] + dx, pp[1] + dy, pp[2] + dz, pp[3], pp[4],
                  pp[5] + dr]
        try:
            moveto(arm, "jog", target)
        except RuntimeError as e:
            print("  can't: %s" % e)
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(0)

    show()
    while True:
        ch = msvcrt.getch()
        try:
            k = ch.decode("ascii").lower()
        except Exception:
            continue
        if k == "\r":
            break
        elif k == "w":
            jog(dx=step)
        elif k == "s":
            jog(dx=-step)
        elif k == "a":
            jog(dy=step)
        elif k == "d":
            jog(dy=-step)
        elif k == "q":
            jog(dz=step)
        elif k == "e":
            jog(dz=-step)
        elif k == "z":
            jog(dr=-max(2.0, min(15.0, step)))
        elif k == "c":
            jog(dr=max(2.0, min(15.0, step)))
        elif k in "12345":
            step = {"1": 1.0, "2": 5.0, "3": 10.0, "4": 20.0,
                    "5": 40.0}[k]
            print("  step = %.0fmm" % step)
        elif k == "o":
            g = gripper_pos(arm)
            if g is not None:
                grip(arm, "open more", min(850, int(g) + 50))
        elif k == "p":
            g = gripper_pos(arm)
            if g is not None:
                grip(arm, "close more", max(0, int(g) - 50))
        elif k == " ":
            c, pp = arm.get_position(is_radian=False)
            if c == 0:
                waypoints.append([round(v, 1) for v in pp])
                print("  recorded waypoint %d: %s"
                      % (len(waypoints), waypoints[-1]))
        elif k == "i":
            show()
        _flush_keys()   # drop key-repeat backlog so the arm never runs away
        if k in "wsadqezc":
            show()

    c, pp = arm.get_position(is_radian=False)
    final = [round(v, 1) for v in pp] if c == 0 else None
    return final, waypoints


# ------------------------- calibration file -------------------------

def save_calib(data):
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_calib():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        return json.load(f)


GRAB_POSE_FILE = os.path.join(HERE, "grab_pose.json")


def save_grab_pose(pose_xyzyaw):
    with open(GRAB_POSE_FILE, "w") as f:
        json.dump({"note": "Reference grab pose [x, y, z, yaw]; cube must "
                           "sit at the same floor spot.",
                   "pose": [round(float(v), 1) for v in pose_xyzyaw]}, f,
                  indent=2)


def load_grab_pose():
    if not os.path.exists(GRAB_POSE_FILE):
        return None
    try:
        with open(GRAB_POSE_FILE) as f:
            pose = json.load(f).get("pose")
        if isinstance(pose, list) and len(pose) == 4:
            return [float(v) for v in pose]
    except Exception:
        pass
    return None


def save_calib2(data):
    with open(CALIB2_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_calib2():
    if not os.path.exists(CALIB2_FILE):
        return None
    with open(CALIB2_FILE) as f:
        return json.load(f)


def px_to_mm(calib2, dpixel, depth_ratio=1.0):
    """Map a pixel displacement to a base-frame mm displacement.

    J was measured at the SCAN height; when looking from closer, pixel
    shifts grow by (scan_depth / current_depth), so scale J down by
    depth_ratio = current_depth / scan_depth.
    """
    J = np.asarray(calib2["J"], dtype=float)
    d = np.asarray(dpixel, dtype=float)
    v = depth_ratio * (J @ d)
    return [float(v[0]), float(v[1])]


def fit_similarity(cam_pts, base_pts):
    """Least-squares 2D similarity transform cam(x,y) -> base(x,y).
    Returns dict with rotation R (2x2 list), translation t (2 list),
    scale s, per-point residuals (mm)."""
    A = np.asarray(cam_pts, dtype=float)
    B = np.asarray(base_pts, dtype=float)
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float(np.trace(np.diag(S) @ D) / (A0 ** 2).sum())
    t = cb - s * (R @ ca)
    pred = (s * (R @ A.T)).T + t
    resid = np.linalg.norm(pred - B, axis=1)
    return {
        "R": R.tolist(), "t": t.tolist(), "s": s,
        "residuals_mm": [round(float(r), 2) for r in resid],
        "rotation_deg": round(float(np.degrees(np.arctan2(R[1][0], R[0][0]))), 3),
    }


def cam_to_base(calib, cam_xy):
    R = np.asarray(calib["fit"]["R"])
    t = np.asarray(calib["fit"]["t"])
    s = calib["fit"]["s"]
    p = s * (R @ np.asarray(cam_xy, dtype=float)) + t
    return [float(p[0]), float(p[1])]

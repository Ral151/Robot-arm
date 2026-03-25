"""
Click-and-Go Demo
=================
Click anywhere in the camera window → Dobot arm moves to that point.
A synthesized top-down view shows workspace, current arm position, and target.

Wired into your existing utilities:
  - realsense_utils.py  (get_aligned_frames, frames_to_numpy, pixel_to_3d)
  - config/device_port.yaml  (camera_serial, device_port)
  - configs/calibration.yaml (camera_to_robot: rotation_matrix, translation)

Usage:
    python click_and_go.py                  # live mode
    python click_and_go.py --dry-run        # no robot needed
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── Existing utilities ───────────────────────────────────────────────────
from realsense_utils import (
    get_aligned_frames,
    frames_to_numpy,
    pixel_to_3d,
)

try:
    from pydobotplus import Dobot
except ImportError:
    Dobot = None


# ─── Config loaders ───────────────────────────────────────────────────────────

def load_device_config():
    """Read camera_serial and device_port from config/device_port.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("camera_serial"), cfg.get("device_port")


def initialize_pipeline(camera_serial=None):
    """
    Start RealSense pipeline the same way your existing AprilTag scripts do.
    Returns (pipeline, profile, align).
    """
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    if camera_serial:
        rs_config.enable_device(camera_serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile = pipeline.start(rs_config)
    align   = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_color_intrinsics(profile):
    """
    Intrinsics from the COLOR stream (required for deprojecting clicks on
    the color image). Your realsense_utils.get_camera_intrinsics() defaults
    to the depth stream, so we pull color here explicitly.
    """
    return (profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics())


def load_calibration(config_path: str):
    """
    Load hand-eye calibration.

    Expected YAML shape:
        calibration:
          camera_to_robot:
            translation: [x, y, z]      # metres
            rotation_matrix:
              - [r00, r01, r02]
              - [r10, r11, r12]
              - [r20, r21, r22]
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cal = cfg["calibration"]["camera_to_robot"]
    R   = np.array(cal["rotation_matrix"], dtype=np.float64)   # (3, 3)
    T   = np.array(cal["translation"],     dtype=np.float64)   # (3,) metres
    return R, T


# ─── Coordinate transform ─────────────────────────────────────────────────────

def camera_to_robot_mm(P_cam_m: np.ndarray, R: np.ndarray, T: np.ndarray):
    """
    P_robot = R @ P_cam + T    (everything in metres, output converted to mm)
    pixel_to_3d() already returns metres, so no extra scaling before this call.
    """
    return (R @ P_cam_m + T) * 1000.0   # mm


# ─── Top-down canvas ──────────────────────────────────────────────────────────

# !! Adjust these bounds to match your actual Dobot reach (mm) !!
WORKSPACE = {"x_min": 100, "x_max": 350, "y_min": -150, "y_max": 150}
CANVAS_W, CANVAS_H = 480, 480
HOVER_Z = 50.0    # mm – safe transit height
PICK_Z  = 10.0    # mm – height when "picking"


def _robot_to_canvas(rx, ry):
    u = int((rx - WORKSPACE["x_min"]) /
            (WORKSPACE["x_max"] - WORKSPACE["x_min"]) * CANVAS_W)
    v = int((ry - WORKSPACE["y_min"]) /
            (WORKSPACE["y_max"] - WORKSPACE["y_min"]) * CANVAS_H)
    return u, CANVAS_H - v     # flip so +y points up


def make_topdown_canvas(arm_mm=None, target_mm=None):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (30, 30, 30), dtype=np.uint8)

    # Grid lines
    for x in range(WORKSPACE["x_min"], WORKSPACE["x_max"], 50):
        cv2.line(canvas, _robot_to_canvas(x, WORKSPACE["y_min"]),
                          _robot_to_canvas(x, WORKSPACE["y_max"]), (55, 55, 55), 1)
    for y in range(WORKSPACE["y_min"], WORKSPACE["y_max"], 50):
        cv2.line(canvas, _robot_to_canvas(WORKSPACE["x_min"], y),
                          _robot_to_canvas(WORKSPACE["x_max"], y), (55, 55, 55), 1)

    # Workspace border
    cv2.rectangle(canvas,
                  _robot_to_canvas(WORKSPACE["x_min"], WORKSPACE["y_min"]),
                  _robot_to_canvas(WORKSPACE["x_max"], WORKSPACE["y_max"]),
                  (80, 80, 80), 1)

    # Robot base origin
    bu, bv = _robot_to_canvas(0, 0)
    cv2.drawMarker(canvas, (bu, bv), (200, 200, 200), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(canvas, "Base", (bu + 6, bv - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Clicked target
    if target_mm is not None:
        tu, tv = _robot_to_canvas(target_mm[0], target_mm[1])
        cv2.drawMarker(canvas, (tu, tv), (0, 230, 120),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(canvas, f"({target_mm[0]:.0f}, {target_mm[1]:.0f})",
                    (tu + 8, tv - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 230, 120), 1)

    # Arm current position
    if arm_mm is not None:
        au, av = _robot_to_canvas(arm_mm[0], arm_mm[1])
        cv2.line(canvas, (bu, bv), (au, av), (60, 100, 160), 1)
        cv2.circle(canvas, (au, av), 9, (60, 160, 255), -1)
        cv2.circle(canvas, (au, av), 9, (120, 200, 255), 1)
        cv2.putText(canvas, "Arm", (au + 12, av + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1)

    cv2.putText(canvas, "Top-down  |  click camera window to move",
                (8, CANVAS_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (100, 100, 100), 1)
    return canvas


# ─── Shared state ─────────────────────────────────────────────────────────────

class DemoState:
    def __init__(self):
        self.lock      = threading.Lock()
        self.target_mm = None
        self.arm_mm    = None
        self.click_px  = None
        self.moving    = False
        self.status    = "Click the camera window to move the arm"

    def set_target(self, target_mm, click_px):
        with self.lock:
            self.target_mm = target_mm
            self.click_px  = click_px

    def snapshot(self):
        with self.lock:
            return (self.target_mm, self.arm_mm,
                    self.click_px, self.moving, self.status)


# ─── Robot worker (background thread) ────────────────────────────────────────

def robot_worker(state: DemoState, robot, dry_run: bool):
    last_target = None
    while True:
        target_mm = state.snapshot()[0]

        if target_mm is not None and target_mm is not last_target:
            last_target = target_mm
            x, y, z = target_mm

            with state.lock:
                state.moving = True
                state.status = f"Moving → ({x:.0f}, {y:.0f}, {z:.0f}) mm …"

            if not dry_run and robot is not None:
                try:
                    robot.move_to(x, y, HOVER_Z, 0, wait=True)  # lift first
                    robot.move_to(x, y, z,        0, wait=True)  # then descend
                    pos = robot.get_pose()
                    with state.lock:
                        state.arm_mm = (pos.position.x,
                                        pos.position.y,
                                        pos.position.z)
                except Exception as e:
                    with state.lock:
                        state.status = f"Robot error: {e}"
            else:
                time.sleep(0.8)     # simulate move delay
                with state.lock:
                    state.arm_mm = (x, y, z)

            with state.lock:
                state.moving = False
                state.status = f"Reached ({x:.0f}, {y:.0f}) mm — click again"

        time.sleep(0.05)


# ─── Mouse callback ───────────────────────────────────────────────────────────

def make_mouse_callback(state: DemoState, frame_store: dict,
                        intrinsics, R_cal, T_cal):
    """
    frame_store["depth"] is updated every main-loop iteration so the callback
    always has the freshest depth frame without blocking the camera loop.

    pixel_to_3d() is your existing realsense_utils function — it wraps
    rs2_deproject_pixel_to_point and returns [X, Y, Z] in metres (or None).
    """
    def on_mouse(event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state.moving:
            return   # ignore clicks while arm is in motion

        depth_frame = frame_store.get("depth")
        if depth_frame is None:
            return

        P_cam = pixel_to_3d(intrinsics, u, v, depth_frame)
        if P_cam is None:
            with state.lock:
                state.status = "No depth at that pixel — try again"
            return

        P_robot = camera_to_robot_mm(np.array(P_cam, dtype=np.float64),
                                     R_cal, T_cal)

        x = float(np.clip(P_robot[0], WORKSPACE["x_min"], WORKSPACE["x_max"]))
        y = float(np.clip(P_robot[1], WORKSPACE["y_min"], WORKSPACE["y_max"]))
        z = PICK_Z   # fixed pick height; use float(P_robot[2]) if Z is calibrated

        state.set_target((x, y, z), (u, v))

    return on_mouse


# ─── Camera-view overlay ──────────────────────────────────────────────────────

def draw_overlay(frame, click_px, status, moving):
    out = frame.copy()
    if click_px is not None:
        u, v = click_px
        color = (0, 80, 255) if moving else (0, 230, 120)
        cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(out, (u, v), 14, color, 1)

    bar_h   = 28
    overlay = out.copy()
    cv2.rectangle(overlay, (0, out.shape[0] - bar_h),
                  (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, status, (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 255) if moving else (200, 255, 200), 1)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="../calibration/calibration.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without robot (test camera + transform)")
    args = parser.parse_args()

    # Load config
    camera_serial, dobot_port = load_device_config()
    R_cal, T_cal = load_calibration(args.calibration)
    print(f"Camera serial : {camera_serial}")
    print(f"Dobot port    : {dobot_port}")
    print(f"R_cal:\n{R_cal}\nT_cal (m): {T_cal}")

    # Robot
    dry_run = args.dry_run or (Dobot is None)
    robot   = None
    if not dry_run:
        try:
            robot = Dobot(port=dobot_port)
            robot.move_to(200, 0, HOVER_Z, 0, wait=True)
            print("Robot connected and homed.")
        except Exception as e:
            print(f"WARNING: Robot connection failed ({e}) → dry-run mode")
            dry_run = True

    # Camera  (mirrors your AprilTag script setup)
    pipeline, profile, align = initialize_pipeline(camera_serial)
    intrinsics = get_color_intrinsics(profile)
    print(f"Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")

    # State + background thread
    state       = DemoState()
    frame_store = {"depth": None}

    threading.Thread(target=robot_worker,
                     args=(state, robot, dry_run), daemon=True).start()

    # Windows
    WIN_CAM = "Camera View  (click to move)"
    WIN_TOP = "Top-down View"
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_TOP, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WIN_CAM,  50,  50)
    cv2.moveWindow(WIN_TOP, 720,  50)

    cv2.setMouseCallback(
        WIN_CAM,
        make_mouse_callback(state, frame_store, intrinsics, R_cal, T_cal)
    )

    print("\nDemo running — click the camera window.  Q to quit.")
    if dry_run:
        print("(Dry-run mode: arm simulated)")

    try:
        while True:
            # ── get_aligned_frames / frames_to_numpy = your existing utils ──
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            if color_frame is None or depth_frame is None:
                continue

            frame_store["depth"] = depth_frame

            color_img, _ = frames_to_numpy(color_frame, depth_frame)

            target_mm, arm_mm, click_px, moving, status = state.snapshot()

            cv2.imshow(WIN_CAM, draw_overlay(color_img, click_px, status, moving))
            cv2.imshow(WIN_TOP, make_topdown_canvas(arm_mm, target_mm))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
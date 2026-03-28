"""
Click-and-Go Demo
=================
This demo shows how to:
1. Capture a mouse click on the camera window
2. Look up the real-world depth at that pixel
3. Convert the pixel to a 3D point in the camera frame
4. Transform that point into robot coordinates using calibration
5. Send the robot arm to that position

Click on the camera image to move the arm there.
Press 'q' to quit.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from realsense_utils import (
    get_aligned_frames,
    frames_to_numpy,
    pixel_to_3d,
)

try:
    from pydobotplus import Dobot
except ImportError:
    Dobot = None   # will fall back to dry-run automatically


# ========================================
# Step 1: Load config files
# ========================================

def load_device_config():
    """Read camera serial number and robot COM port from device_port.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("camera_serial"), cfg.get("device_port")


def load_calibration(config_path: str):
    """
    Load the hand-eye calibration result.

    The calibration is two things:
      R — a 3x3 rotation matrix  (how the camera is rotated relative to the robot)
      T — a 3-element translation (where the camera origin is, in metres)

    Together they let us convert any point seen by the camera into
    coordinates the robot understands.

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
    R = np.array(cal["rotation_matrix"], dtype=np.float64)  # shape (3, 3)
    T = np.array(cal["translation"],     dtype=np.float64)  # shape (3,) metres
    return R, T


# ========================================
# Step 2: Start the RealSense camera
# ========================================

def initialize_pipeline(camera_serial=None):
    """
    Start the RealSense camera — same pattern as realsense_basic.py.
    We need both color (to display) and depth (to measure distance).
    """
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    if camera_serial:
        rs_config.enable_device(camera_serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile = pipeline.start(rs_config)

    # align=color means every depth pixel lines up with the color pixel
    # at the same (u, v) — essential for accurate click-to-depth lookup
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_color_intrinsics(profile):
    """
    Get the color camera's intrinsic parameters (fx, fy, cx, cy).

    Intrinsics describe the lens geometry — they are what let us convert
    a 2D pixel + depth value into a real 3D position in space.

    Note: realsense_utils.get_camera_intrinsics() defaults to the depth
    stream. We explicitly request color here because the user clicks on
    the color image, so we need color intrinsics for accuracy.
    """
    return (profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics())


# ========================================
# Step 3: The core math — pixel → robot coords
# ========================================

def pixel_to_robot_mm(u, v, depth_frame, intrinsics, R_cal, T_cal):
    """
    Convert a clicked pixel (u, v) all the way to robot coordinates.

    The journey has three steps:

      (u, v) pixel
          ↓  pixel_to_3d()
      [X, Y, Z] in CAMERA frame  (metres, origin = camera lens)
          ↓  R @ P + T
      [X, Y, Z] in ROBOT frame   (metres, origin = robot base)
          ↓  × 1000
      [X, Y, Z] in ROBOT frame   (mm — what Dobot move_to() expects)

    Returns (x_mm, y_mm, z_mm) or None if depth is invalid at that pixel.
    """
    # Step 3a: look up the real-world depth at this pixel
    # pixel_to_3d uses the pinhole camera model:
    #   X = (u - cx) * depth / fx
    #   Y = (v - cy) * depth / fy
    #   Z = depth
    P_cam = pixel_to_3d(intrinsics, u, v, depth_frame)
    if P_cam is None:
        return None   # depth sensor returned 0 — too close or reflective surface

    # Step 3b: apply the calibration transform
    # P_robot = R @ P_camera + T
    # This rotates and shifts the point from "camera's perspective"
    # to "robot's perspective"
    P_cam_arr  = np.array(P_cam, dtype=np.float64)  # metres
    P_robot_m  = R_cal @ P_cam_arr + T_cal           # still metres
    P_robot_mm = P_robot_m * 1000.0                  # → mm for Dobot

    return float(P_robot_mm[0]), float(P_robot_mm[1]), float(P_robot_mm[2])


# ========================================
# Step 4: Top-down canvas (synthesized "third-person view")
# ========================================

# !! Adjust these to match your physical workspace (mm) !!
WORKSPACE = {"x_min": 100, "x_max": 350, "y_min": -150, "y_max": 150}
CANVAS_W, CANVAS_H = 480, 480

# Safe heights for arm movement — tune before live robot use
HOVER_Z = 50.0   # arm rises to this height before moving sideways
PICK_Z  = 10.0   # arm descends to this height at the target


def _robot_to_canvas(rx, ry):
    """Map robot mm coordinates → canvas pixel position."""
    u = int((rx - WORKSPACE["x_min"]) /
            (WORKSPACE["x_max"] - WORKSPACE["x_min"]) * CANVAS_W)
    v = int((ry - WORKSPACE["y_min"]) /
            (WORKSPACE["y_max"] - WORKSPACE["y_min"]) * CANVAS_H)
    return u, CANVAS_H - v   # flip V so +y points upward on screen


def make_topdown_canvas(arm_mm=None, target_mm=None):
    """
    Draw a bird's-eye diagram of the workspace.
      Grey cross  = robot base (origin)
      Blue dot    = current arm position
      Green cross = clicked target
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (30, 30, 30), dtype=np.uint8)

    # Grid lines every 50 mm
    for x in range(WORKSPACE["x_min"], WORKSPACE["x_max"], 50):
        cv2.line(canvas, _robot_to_canvas(x, WORKSPACE["y_min"]),
                          _robot_to_canvas(x, WORKSPACE["y_max"]), (55, 55, 55), 1)
    for y in range(WORKSPACE["y_min"], WORKSPACE["y_max"], 50):
        cv2.line(canvas, _robot_to_canvas(WORKSPACE["x_min"], y),
                          _robot_to_canvas(WORKSPACE["x_max"], y), (55, 55, 55), 1)

    # Workspace boundary box
    cv2.rectangle(canvas,
                  _robot_to_canvas(WORKSPACE["x_min"], WORKSPACE["y_min"]),
                  _robot_to_canvas(WORKSPACE["x_max"], WORKSPACE["y_max"]),
                  (80, 80, 80), 1)

    # Robot base at origin
    bu, bv = _robot_to_canvas(0, 0)
    cv2.drawMarker(canvas, (bu, bv), (200, 200, 200), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(canvas, "Base", (bu + 6, bv - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Target marker (where you clicked)
    if target_mm is not None:
        tu, tv = _robot_to_canvas(target_mm[0], target_mm[1])
        cv2.drawMarker(canvas, (tu, tv), (0, 230, 120),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(canvas, f"({target_mm[0]:.0f}, {target_mm[1]:.0f})",
                    (tu + 8, tv - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 230, 120), 1)

    # Arm dot (where the arm currently is)
    if arm_mm is not None:
        au, av = _robot_to_canvas(arm_mm[0], arm_mm[1])
        cv2.line(canvas, (bu, bv), (au, av), (60, 100, 160), 1)   # reach line
        cv2.circle(canvas, (au, av), 9, (60, 160, 255), -1)
        cv2.circle(canvas, (au, av), 9, (120, 200, 255), 1)
        cv2.putText(canvas, "Arm", (au + 12, av + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1)

    cv2.putText(canvas, "Top-down view  |  click camera window to move",
                (8, CANVAS_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (100, 100, 100), 1)
    return canvas


# ========================================
# Step 5: Shared state between threads
# ========================================

class DemoState:
    """
    Thread-safe container for everything the main loop and robot thread
    need to share: target position, current arm position, last click pixel,
    and a status message for the HUD.
    """
    def __init__(self):
        self.lock      = threading.Lock()
        self.target_mm = None    # (x, y, z) where you last clicked
        self.arm_mm    = None    # (x, y, z) where the arm currently is
        self.click_px  = None    # (u, v) last clicked pixel on camera image
        self.moving    = False   # True while arm is in motion
        self.status    = "Click the camera window to move the arm"

    def set_target(self, target_mm, click_px):
        with self.lock:
            self.target_mm = target_mm
            self.click_px  = click_px

    def snapshot(self):
        """Return a consistent copy of all state values at one moment."""
        with self.lock:
            return (self.target_mm, self.arm_mm,
                    self.click_px, self.moving, self.status)


# ========================================
# Step 6: Robot movement thread
# ========================================

def robot_worker(state: DemoState, robot, dry_run: bool):
    """
    Runs in the background — watches for new targets and moves the arm.

    Kept separate from the main loop so the camera feed never freezes
    while the arm is moving (move_to with wait=True blocks for ~1-2s).
    """
    last_target = None

    while True:
        target_mm = state.snapshot()[0]

        # Only act when there is a new target we haven't moved to yet
        if target_mm is not None and target_mm is not last_target:
            last_target = target_mm
            x, y, z = target_mm

            with state.lock:
                state.moving = True
                state.status = f"Moving → ({x:.0f}, {y:.0f}, {z:.0f}) mm …"

            if not dry_run and robot is not None:
                try:
                    # Rise to hover height first so the arm doesn't drag
                    # sideways across any objects sitting on the table
                    robot.move_to(x, y, HOVER_Z, 0, wait=True)
                    # Descend to pick height at the target location
                    robot.move_to(x, y, z, 0, wait=True)

                    # Read back actual position to update the canvas dot
                    # NOTE: if this crashes with AttributeError, check whether
                    # your pydobotplus uses pos.x or pos.position.x and match
                    # whatever visualize_arm_pose.py uses
                    pos = robot.get_pose()
                    with state.lock:
                        state.arm_mm = (pos.position.x,
                                        pos.position.y,
                                        pos.position.z)
                except Exception as e:
                    with state.lock:
                        state.status = f"Robot error: {e}"
            else:
                # Dry-run: pretend the arm moved after a short delay
                time.sleep(0.8)
                with state.lock:
                    state.arm_mm = (x, y, z)

            with state.lock:
                state.moving = False
                state.status = f"Reached ({x:.0f}, {y:.0f}) mm — click again"

        time.sleep(0.05)   # poll 20x per second — keeps CPU usage low


# ========================================
# Step 7: Mouse click handler
# ========================================

def make_mouse_callback(state: DemoState, frame_store: dict,
                        intrinsics, R_cal, T_cal):
    """
    Returns the function OpenCV calls whenever the mouse moves or clicks
    inside the camera window.

    frame_store["depth"] is updated every frame by the main loop, so the
    callback always has fresh depth data without touching the camera pipeline.
    """
    def on_mouse(event, u, v, flags, param):
        # We only care about left-button clicks — ignore moves, right-clicks etc.
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Ignore clicks while arm is already moving — prevents queue buildup
        if state.moving:
            return

        depth_frame = frame_store.get("depth")
        if depth_frame is None:
            return

        # ── Core conversion: pixel → robot mm ────────────────────────────
        result = pixel_to_robot_mm(u, v, depth_frame, intrinsics, R_cal, T_cal)

        if result is None:
            # Depth was 0 — happens on shiny, transparent, or very close surfaces
            with state.lock:
                state.status = "No depth at that pixel — try clicking elsewhere"
            return

        rx, ry, rz = result

        # Clamp to safe workspace bounds so the arm never tries to reach
        # somewhere it physically cannot go
        x = float(np.clip(rx, WORKSPACE["x_min"], WORKSPACE["x_max"]))
        y = float(np.clip(ry, WORKSPACE["y_min"], WORKSPACE["y_max"]))
        z = PICK_Z   # fixed height for now; swap to rz if Z is calibrated

        print(f"Click at pixel ({u}, {v})")
        print(f"  → Camera frame : ({rx:.1f}, {ry:.1f}, {rz:.1f}) mm")
        print(f"  → Robot target : ({x:.1f}, {y:.1f}, {z:.1f}) mm")

        state.set_target((x, y, z), (u, v))

    return on_mouse


# ========================================
# Step 8: Camera window overlay
# ========================================

def draw_overlay(frame, click_px, status, moving):
    """
    Draw a crosshair at the clicked pixel and a status bar at the bottom.
    Green = idle, blue = arm is moving.
    """
    out = frame.copy()

    if click_px is not None:
        u, v = click_px
        color = (0, 80, 255) if moving else (0, 230, 120)
        cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 24, 2)
        cv2.circle(out, (u, v), 14, color, 1)

    # Semi-transparent black bar for the status text
    bar_h   = 28
    overlay = out.copy()
    cv2.rectangle(overlay, (0, out.shape[0] - bar_h),
                  (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, status, (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 255) if moving else (200, 255, 200), 1)
    return out


# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="../calibration/calibration.yaml",
                        help="Path to hand-eye calibration YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without robot (camera + click transform only)")
    args = parser.parse_args()

    print("=" * 70)
    print("Click-and-Go Demo")
    print("=" * 70)

    # ── Step 1: Load configs ─────────────────────────────────────────────────
    print("\n[Step 1] Loading config files...")
    camera_serial, dobot_port = load_device_config()
    R_cal, T_cal = load_calibration(args.calibration)
    print(f"  Camera serial : {camera_serial}")
    print(f"  Dobot port    : {dobot_port}")
    print(f"  Calibration R :\n{R_cal}")
    print(f"  Calibration T : {T_cal} m")

    # ── Step 2: Connect robot ────────────────────────────────────────────────
    print("\n[Step 2] Connecting to robot...")
    dry_run = args.dry_run or (Dobot is None)
    robot   = None

    if not dry_run:
        try:
            robot = Dobot(port=dobot_port)
            robot.move_to(200, 0, HOVER_Z, 0, wait=True)   # move to home
            print("  Robot connected and homed.")
        except Exception as e:
            print(f"  WARNING: Robot connection failed ({e}) → dry-run mode")
            dry_run = True
    else:
        print("  Dry-run mode — robot not connected.")

    # ── Step 3: Start camera ─────────────────────────────────────────────────
    print("\n[Step 3] Initializing RealSense camera...")
    pipeline, profile, align = initialize_pipeline(camera_serial)
    intrinsics = get_color_intrinsics(profile)
    print(f"  Resolution  : {intrinsics.width} x {intrinsics.height}")
    print(f"  Focal length: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
    print(f"  Principal pt: cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")

    # ── Step 4: Set up shared state and robot thread ─────────────────────────
    print("\n[Step 4] Starting robot worker thread...")
    state       = DemoState()
    frame_store = {"depth": None}   # main loop writes here; mouse cb reads it

    threading.Thread(target=robot_worker,
                     args=(state, robot, dry_run), daemon=True).start()

    # ── Step 5: Open windows ─────────────────────────────────────────────────
    print("\n[Step 5] Opening display windows...")
    WIN_CAM = "Camera View  (click to move)"
    WIN_TOP = "Top-down View"
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_TOP, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WIN_CAM,  50,  50)
    cv2.moveWindow(WIN_TOP, 660,  50)   # side-by-side (reduce if off-screen)

    # Register the mouse callback — OpenCV calls on_mouse() automatically
    # on any mouse event inside WIN_CAM
    cv2.setMouseCallback(
        WIN_CAM,
        make_mouse_callback(state, frame_store, intrinsics, R_cal, T_cal)
    )

    print("\nDemo running — click the camera window.  Press 'q' to quit.")
    if dry_run:
        print("(Dry-run mode: arm movement is simulated)\n")

    # ── Step 6: Main loop ────────────────────────────────────────────────────
    try:
        while True:
            # Grab the latest aligned color + depth frame pair
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            if color_frame is None or depth_frame is None:
                continue

            # Expose depth frame to the mouse callback
            # (dict write is atomic in CPython — no lock needed here)
            frame_store["depth"] = depth_frame

            # Convert color frame to a numpy image for OpenCV drawing
            color_img, _ = frames_to_numpy(color_frame, depth_frame)

            # Read a consistent snapshot of all shared state
            target_mm, arm_mm, click_px, moving, status = state.snapshot()

            # Render both windows
            cv2.imshow(WIN_CAM, draw_overlay(color_img, click_px, status, moving))
            cv2.imshow(WIN_TOP, make_topdown_canvas(arm_mm, target_mm))

            # waitKey(1) pumps the OpenCV event loop — without this, windows
            # freeze and mouse callbacks never fire
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # ── Step 7: Cleanup ──────────────────────────────────────────────────
        pipeline.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.close()
        print("\nShutdown complete.")
        print("=" * 70)


if __name__ == "__main__":
    main()
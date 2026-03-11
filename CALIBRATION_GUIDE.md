# Hand-Eye Calibration Guide

## Overview

This guide explains how to calibrate the camera-to-robot transformation for the Robot Arm Challenge. The AprilTag is mounted on the **gripper** (not the workspace), making this an **eye-to-hand** calibration setup.

## Setup

### Hardware Configuration

```
         [Camera] (Fixed above workspace)
              |
              | (looking down)
              v
    ┌─────────────────────┐
    │   Workspace Area    │
    │                     │
    │    [Dobot Robot] ───┼──► [Gripper + AprilTag]
    │                     │
    │  [Source Bin]       │
    │  [Sorting Bins]     │
    └─────────────────────┘
```

- **Camera**: Fixed position above robot (does NOT move)
- **AprilTag**: Attached to gripper (moves with robot)
- **Calibration Type**: Eye-to-hand (camera external, tag on end-effector)

## Calibration Theory

### What We're Computing

The transformation matrix that converts coordinates from the camera frame to the robot base frame:

```
P_robot = R * P_camera + T
```

Where:

- `R`: 3×3 rotation matrix
- `T`: 3×1 translation vector (in metres)
- `P_camera`: Point in camera coordinates
- `P_robot`: Point in robot base coordinates

### Why AprilTag on Gripper?

The AprilTag on the gripper allows us to:

1. Know the gripper's position in camera coordinates (via tag detection)
2. Know the gripper's position in robot coordinates (via robot kinematics)
3. Compute the transformation between these two frames

## Calibration Process

### Step 1: Prepare Calibration Setup

1. **Attach AprilTag firmly to gripper** (provided by committee)
   - Ensure tag is flat and visible from camera
   - Note the tag ID (update in `configs/calibration.yaml`)

2. **Position camera above workspace**
   - Camera should see entire workspace
   - AprilTag should be visible when gripper is in workspace

3. **Measure AprilTag physical size**
   - Measure the **black square** side length in metres
   - Update `tag_size` in `configs/calibration.yaml`
   - Example: 50mm tag → `tag_size: 0.05`

### Step 2: Collect Calibration Data

You need to collect multiple robot poses where the AprilTag is visible:

```python
# Example calibration data collection script
import cv2
import numpy as np
from pupil_apriltags import Detector
from pydobot import Dobot

# Initialize
camera = cv2.VideoCapture(0)
robot = Dobot(port="/dev/ttyUSB0")
detector = Detector(families="tag36h11")

# Define calibration poses (vary position and orientation)
calibration_poses = [
    {"x": 200, "y": 0, "z": 50, "r": 0},
    {"x": 200, "y": 50, "z": 50, "r": 0},
    {"x": 200, "y": -50, "z": 50, "r": 0},
    {"x": 250, "y": 0, "z": 50, "r": 0},
    {"x": 150, "y": 0, "z": 50, "r": 0},
    {"x": 200, "y": 0, "z": 80, "r": 0},
    {"x": 200, "y": 0, "z": 30, "r": 45},
    # Add more poses for better calibration
]

camera_to_tag = []  # List of tag poses in camera frame
robot_to_gripper = []  # List of gripper poses in robot frame

for pose in calibration_poses:
    # Move robot to pose
    robot.move_to(pose["x"], pose["y"], pose["z"], pose["r"], wait=True)
    time.sleep(1)  # Let robot settle

    # Capture frame
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTag
    tags = detector.detect(gray, estimate_tag_pose=True,
                          camera_params=(fx, fy, cx, cy),
                          tag_size=0.05)

    if tags:
        tag = tags[0]
        # Store camera-to-tag transform
        camera_to_tag.append({
            "R": tag.pose_R,
            "t": tag.pose_t
        })

        # Store robot-to-gripper transform (from robot kinematics)
        robot_to_gripper.append({
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "r": pose["r"]
        })

# Now solve for camera-to-robot transformation
# using hand-eye calibration algorithms (see Step 3)
```

### Step 3: Compute Calibration

Use OpenCV's hand-eye calibration function:

```python
import cv2
import numpy as np

# Convert data to OpenCV format
R_gripper2base = []  # Gripper to robot base
t_gripper2base = []
R_tag2cam = []  # AprilTag to camera
t_tag2cam = []

# ... populate lists from collected data ...

# Solve hand-eye calibration (eye-to-hand case)
R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_gripper2base,
    t_gripper2base,
    R_tag2cam,
    t_tag2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("Camera to Robot Rotation Matrix:")
print(R_cam2base)
print("\nCamera to Robot Translation (metres):")
print(t_cam2base)
```

### Step 4: Update Configuration

Update `configs/calibration.yaml` with computed values:

```yaml
calibration:
  apriltag:
    family: tag36h11
    tag_size: 0.05 # Measured tag size in metres
    tag_id: 0 # Tag ID from committee

  # Computed from calibration
  camera_to_robot:
    translation: [0.35, 0.12, 0.85] # [x, y, z] in metres
    rotation_matrix:
      - [0.9998, -0.0123, 0.0156]
      - [0.0125, 0.9999, -0.0045]
      - [-0.0155, 0.0048, 0.9998]
```

### Step 5: Verify Calibration

Test the calibration accuracy:

```bash
python scripts/test_transform.py
```

**Visual verification**:

1. Place an object at a known robot position
2. Detect it with camera + YOLO
3. Transform pixel coordinates to robot coordinates
4. Check if transformed coordinates match known position

**Acceptable error**: ±5mm for pick-and-place tasks

## Camera Intrinsic Calibration

Before hand-eye calibration, you should calibrate camera intrinsics:

### Using Checkerboard

1. Print a checkerboard pattern (9×6 squares, 25mm each)
2. Capture 15-20 images from different angles
3. Run OpenCV calibration:

```python
import cv2
import numpy as np
import glob

# Checkerboard dimensions
pattern_size = (9, 6)
square_size = 0.025  # 25mm in metres

# Prepare object points
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

obj_points = []  # 3D points in real world
img_points = []  # 2D points in image

images = glob.glob('data/calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

# Calibrate
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
```

Update `configs/camera.yaml`:

```yaml
intrinsics:
  fx: 800.5 # camera_matrix[0, 0]
  fy: 801.2 # camera_matrix[1, 1]
  cx: 639.8 # camera_matrix[0, 2]
  cy: 359.6 # camera_matrix[1, 2]
  distortion_coeffs: [0.1, -0.3, 0.001, 0.002, 0.15]
```

## Calibration Tips

### For Best Results

1. **Vary robot poses widely**
   - Different X, Y, Z positions
   - Different orientations (r)
   - Cover entire workspace

2. **Ensure good tag visibility**
   - Tag should be clearly visible in all poses
   - Avoid motion blur (let robot settle)
   - Good lighting (consistent, no glare)

3. **Collect enough data**
   - Minimum: 10 poses
   - Recommended: 20-30 poses
   - More data = better accuracy

4. **Check for outliers**
   - If one pose gives bad detection, exclude it
   - Reprojection error should be consistent

### Common Issues

**AprilTag not detected**

- Check lighting (too bright/dark?)
- Check focus (blurry image?)
- Check tag size matches config
- Verify tag ID matches config

**High reprojection error**

- Recheck camera intrinsics
- Ensure robot positions are accurate
- Check for camera/tag movement during capture

**Inconsistent pick positions**

- Recalibrate with more poses
- Check if camera or AprilTag moved
- Verify robot repeatability

## Using Calibration in Code

The `CameraToRobotTransform` class handles coordinate conversion:

```python
from calibration.transforms import load_calibration

# Load calibration
config = load_config("configs/calibration.yaml")
transform = load_calibration(config)

# Convert pixel to robot coordinates
pixel_coords = (320, 240)  # (u, v) in pixels
robot_coords = transform.image_to_robot(pixel_coords)
# Returns: {"x": 200.5, "y": 15.3, "z": 10.0} in mm

# Use with robot
robot.pick(robot_coords)
```

## Troubleshooting

| Problem                         | Solution                                        |
| ------------------------------- | ----------------------------------------------- |
| Tag detection fails             | Check tag ID, size, lighting, focus             |
| Inconsistent positions          | Recalibrate, check camera/tag stability         |
| Large position errors (>10mm)   | Redo intrinsic calibration first                |
| Works in center, fails at edges | Camera distortion - recalibrate intrinsics      |
| Depth (Z) is wrong              | Check tag_size measurement, use fixed bin depth |

## Competition Day Checklist

- [ ] Camera mounted securely (won't move)
- [ ] AprilTag attached firmly to gripper
- [ ] Calibration files updated and saved
- [ ] Test transformation accuracy (<5mm error)
- [ ] Backup calibration data
- [ ] Verify bin positions in robot.yaml
- [ ] Test full pick-and-place cycle

## Additional Resources

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Hand-Eye Calibration Theory](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
- [AprilTag Detection](https://github.com/AprilRobotics/apriltag)

Good luck with your calibration! 🎯

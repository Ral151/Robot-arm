# Robot-arm Sorting Challenge

**Autonomous sorting system for nuts, bolts, and screws using Dobot Magician robot arm with vision-guided gripper control.**

This project is designed for the Robot Arm Challenge competition where students build an autonomous system to sort random objects from a bin into designated locations within a time limit.

## System Overview

- **Robot**: Dobot Magician with gripper (clamping mechanism)
- **Camera**: Fixed position above the robot workspace
- **Detection**: YOLO object detection trained on nuts, bolts, and screws
- **Calibration**: Hand-eye calibration using AprilTag mounted on gripper
- **Operation**: Single-run autonomous sorting until time expires

## Competition Requirements

✅ Sort nuts, bolts, and screws from source bin  
✅ Place each type into designated sorting bins  
✅ Maximize items sorted correctly within time limit  
✅ Single program execution (runs continuously)  
✅ Fixed camera position above robot  
✅ AprilTag on gripper for calibration  
✅ Gripper-based grasping (not suction)

## Getting Started

### 1. Setting Up the Environment

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#using-miniconda-in-a-commercial-setting)

2. Run the Following Commands

```bash
# create a new environment
conda env create -f environment.yml

# activate environment
conda activate RobotArm
```

### 2. Hardware Setup

1. **Mount camera** above the robot workspace (fixed position)
2. **Attach gripper** to Dobot Magician end-effector
3. **Attach AprilTag** (provided by committee) to the gripper
4. **Position bins**:
   - Source bin (container plate) in robot reach
   - Three sorting bins for nuts, bolts, and screws
5. **Connect robot** via USB (typically `/dev/ttyUSB0` on Linux, `COM3` on Windows)

### 3. Calibration

**Important**: The AprilTag is mounted on the **gripper** (not workspace) for hand-eye calibration.

#### Camera-to-Gripper Calibration Process:

1. **Capture calibration images** with AprilTag visible:

   ```bash
   python scripts/test_camera.py  # Verify camera works
   ```

2. **Run calibration** (compute transformation matrix):
   - Move robot to multiple poses
   - Detect AprilTag in each pose
   - Compute camera-to-gripper transformation
   - Save results to `configs/calibration.yaml`

3. **Test transformation**:

   ```bash
   python scripts/test_transform.py
   ```

4. **Update `configs/calibration.yaml`** with computed rotation matrix and translation vector

### 4. Training YOLO Model

1. **Collect training data**:

   ```bash
   python scripts/collect_dataset.py --output-dir data/raw_images
   ```

   Press `s` to save frames, `q` to quit

2. **Label images** with nuts, bolts, and screws (use tools like Roboflow, LabelImg, or CVAT)

3. **Train YOLO model** (use Ultralytics YOLO):

   ```bash
   yolo train data=custom_data.yaml model=yolov8n.pt epochs=100
   ```

4. **Copy trained model** to `models/yolo/best.pt`

### 5. Testing Components

Test individual components before running full system:

```bash
# Test camera stream
python scripts/test_camera.py

# Test robot connection and movement
python scripts/test_robot.py

# Test coordinate transformation
python scripts/test_transform.py
```

### 6. Running the Competition

#### Demo Mode (with visualization):

```bash
python scripts/run_demo.py --model models/yolo/best.pt
```

- Shows live camera feed with detections
- Press `q` to stop
- Displays sorting statistics

#### Competition Mode (headless):

```bash
python src/main.py --model models/yolo/best.pt
```

- Runs continuously until Ctrl+C or time limit
- Logs all operations to `outputs/logs/`
- Prints final statistics on shutdown

## Configuration Files

### `configs/robot.yaml`

- Serial port and connection settings
- Gripper parameters (open/close values, delays)
- **Sorting bin locations** (x, y, z coordinates for each object type)
- Speed and acceleration limits
- Workspace boundaries

### `configs/camera.yaml`

- Camera device ID and resolution
- Intrinsic parameters (fx, fy, cx, cy)
- Distortion coefficients

### `configs/calibration.yaml`

- AprilTag settings (family, size, ID)
- **Camera-to-robot transformation matrix**
- Rotation matrix (3x3)
- Translation vector (3x1, in metres)

### `configs/classes.yaml`

- Object classes: `nut`, `bolt`, `screw`
- Detection confidence thresholds per class

## Project Structure

```
Robot-arm/
├── configs/               # Configuration files
│   ├── robot.yaml        # Robot & sorting bin positions
│   ├── camera.yaml       # Camera settings
│   ├── calibration.yaml  # Hand-eye calibration
│   └── classes.yaml      # YOLO classes (nut/bolt/screw)
├── src/
│   ├── main.py           # Competition entry point
│   ├── camera/           # Camera stream management
│   ├── robot/            # Dobot controller & gripper
│   ├── calibration/      # Coordinate transformations
│   ├── vision/           # YOLO detection & selection
│   └── utils/            # Logging and helpers
├── scripts/              # Testing and setup scripts
├── data/                 # Training images
├── models/yolo/          # Trained YOLO weights
└── outputs/              # Logs and debug frames
```

## Competition Strategy Tips

1. **Calibration is Critical**: Spend time getting accurate camera-to-robot transformation
2. **Train on Real Data**: Capture images in actual competition lighting and setup
3. **Optimize Speed**: Adjust robot speed/acceleration in `robot.yaml` for faster sorting
4. **Test Gripper Grip**: Tune `close_value` and `open_value` for reliable grasping
5. **Handle Failures**: System returns to home position on errors and continues
6. **Monitor Logs**: Check `outputs/logs/` for debugging

## Troubleshooting

**Robot not connecting**: Check serial port in `configs/robot.yaml` (Windows: COM3, Linux: /dev/ttyUSB0)

**Camera not found**: Verify `device_id` in `configs/camera.yaml` (usually 0 for default camera)

**Low detection confidence**: Retrain YOLO with more diverse training data

**Inaccurate positioning**: Recalibrate camera-to-robot transformation

**Gripper not gripping**: Adjust `close_value` in `robot.yaml` (higher = tighter grip)

## License

This project is for educational and competition purposes.

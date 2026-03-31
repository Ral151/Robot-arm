# Robot Arm Challenge - Competition Strategy Guide

## Competition Overview

**Objective**: Sort the maximum number of nuts, bolts, and screws into correct bins within the time limit.

**Scoring**:

- Points awarded for each correctly sorted item
- Team with most points wins
- Penalties for incorrect placements (if applicable)

## System Architecture

### Hardware Setup

```
┌──────────────────────────────────────┐
│   Camera (Fixed Above)               │
│              ↓                       │
│   ┌──────────────────────┐          │
│   │  Source Container    │          │
│   │  (Nuts/Bolts/Screws) │          │
│   └──────────────────────┘          │
│              ↑                       │
│         Dobot Robot                  │
│        (+ Gripper + Tag)             │
│              ↓                       │
│   ┌─────┐ ┌─────┐ ┌─────┐          │
│   │ Nut │ │Bolt │ │Screw│          │
│   │ Bin │ │ Bin │ │ Bin │          │
│   └─────┘ └─────┘ └─────┘          │
└──────────────────────────────────────┘
```

### Software Pipeline

```
Camera → Frame Capture → YOLO Detection → Target Selection
                                              ↓
Robot Execution ← Coordinate Transform ← Best Target
      ↓
Pick from Source → Move to Bin → Place → Repeat
```

## Pre-Competition Preparation

### 1. Hardware Tuning (1-2 weeks before)

**Gripper Calibration**:

```yaml
# Test different values in robot.yaml
gripper:
  close_value: 480 # Start here, adjust if needed
  open_value: 200 # Full open
  close_delay: 0.5 # Time to ensure grip
  open_delay: 0.3 # Time to ensure release
```

**Testing Steps**:

1. Test gripper with actual nuts, bolts, screws
2. Adjust `close_value` if items slip (increase) or gripper struggles (decrease)
3. Minimize delays while ensuring reliability
4. Test at different heights (bin may have varying levels)

**Camera Setup**:

- Mount camera at optimal height (test field of view)
- Ensure entire workspace visible
- Test various lighting conditions
- Lock all camera settings (exposure, focus, white balance)

### 2. Calibration (1 week before)

**Critical Success Factor**: Accurate calibration = accurate picks

Steps:

1. Mount AprilTag securely on gripper (verify it doesn't move)
2. Measure tag size precisely (use calipers)
3. Perform camera intrinsic calibration (see CALIBRATION_GUIDE.md)
4. Perform hand-eye calibration with 20+ poses
5. Verify calibration: test picks at known positions
6. **Acceptable error**: ±3mm for these small objects

**Verification Test**:

```python
# Place a test object at robot coordinates (200, 0, 10)
# Run detection and check if transformed coordinates match
# Repeat for all workspace corners and center
```

### 3. YOLO Training (2-3 weeks before)

**Dataset Collection**:

- Collect 500-1000 images minimum
- Use actual competition objects (nuts, bolts, screws)
- Vary:
  - Object positions and orientations
  - Lighting conditions
  - Object clustering (multiple items visible)
  - Different bin fill levels
  - Background variations

**Labeling Strategy**:

- Use Roboflow, LabelImg, or CVAT
- Be consistent with bounding boxes
- Include partially visible objects (realistic scenario)
- Label occluded objects carefully

**Training Tips**:

```bash
# Use YOLOv8 nano for speed (competition has time limit)
yolo train data=nuts_bolts_screws.yaml model=yolov8n.pt epochs=100 imgsz=640

# Fine-tune on competition-like images
yolo train data=nuts_bolts_screws.yaml model=runs/detect/train/weights/best.pt epochs=50
```

**Model Optimization**:

- Target inference time: <50ms per frame
- Confidence threshold: 0.55 (balance precision vs. recall)
- Test on frames similar to competition environment

### 4. Speed Optimization

**Robot Motion**:

```yaml
# configs/robot.yaml - Balance speed vs. accuracy
speed: 80 # mm/s (increase if robot can handle)
acceleration: 80 # mm/s^2
```

**Path Planning**:

- Minimize travel distance between bins
- Optimize bin positions in robot.yaml
- Consider grouping strategy (sort all of one type first?)

**Detection Rate**:

- Process every frame vs. every Nth frame?
- Current: Process all frames (most accurate)
- Alternative: Process every 3rd frame (faster, may miss items)

## Competition Day Strategy

### Setup (Before Timer Starts)

**Checklist**:

- [ ] Camera positioned and secured
- [ ] Gripper attached and tested
- [ ] AprilTag attached and visible
- [ ] Source bin positioned in workspace
- [ ] Sorting bins positioned at configured locations
- [ ] Robot connected and homed
- [ ] Calibration verified (test pick at known position)
- [ ] YOLO model loaded
- [ ] Backup files saved
- [ ] Terminal ready to launch program

**Final Configuration Check**:

```bash
# Test camera
python scripts/test_camera.py  # Verify camera works

# Test robot
python scripts/test_robot.py   # Verify robot connects and moves

# Quick calibration verification
# Place object at (200, 0, 10)
# Run: python scripts/run_demo.py
# Check if robot picks correctly
```

### During Competition

**One-Time Launch**:

```bash
# Production mode (no GUI, maximum speed)
python src/main.py --model models/yolo/best.pt

# Alternative with visualization (if allowed)
python scripts/run_demo.py --model models/yolo/best.pt
```

**What the System Does**:

1. Continuously scans for objects in source bin
2. Detects and classifies (nut/bolt/screw)
3. Picks highest-confidence target
4. Places in corresponding bin
5. Returns to scan
6. Repeats until time expires or Ctrl+C

**Monitoring**:

- Watch terminal output for statistics
- Track success rate
- Note any repeated failures (same object?)

### Error Handling

The system automatically handles:

- **No detection**: Continues scanning
- **Pick failure**: Returns to home, continues
- **Unknown class**: Logs warning, skips object
- **Robot error**: Returns to home, continues

**Manual Intervention** (if allowed by rules):

- If robot gets stuck: Press Ctrl+C, clear obstacle, restart
- If specific object always fails: Note for post-competition analysis

## Optimization Tactics

### Target Selection Priority

Current strategy (in `vision/target_selector.py`):

1. Highest confidence detection
2. Largest bounding box (tiebreaker)

**Alternative Strategies**:

**Closest-First**:

```python
# Modify target_selector.py to pick nearest objects first
def select(self, detections):
    # Sort by distance from robot home position
    detections.sort(key=lambda d: distance_to_robot(d.centroid))
    return detections[0] if detections else None
```

**Type-Grouping** (sort all nuts, then bolts, then screws):

```python
# Reduces travel distance between bins
current_target_type = "nut"
detections_of_type = [d for d in detections if d.label == current_target_type]
# Switch to next type when current type exhausted
```

### Speed vs. Accuracy Trade-offs

**Conservative (Reliable)**:

```yaml
speed: 70
confidence_threshold: 0.6
close_delay: 0.5
```

- Fewer errors
- Slower cycle time
- Good if penalties for mistakes

**Aggressive (Fast)**:

```yaml
speed: 100
confidence_threshold: 0.5
close_delay: 0.3
```

- Higher throughput
- May have more failures
- Good if no penalties for mistakes

### Bin Depth Handling

Since bin depth is fixed (container plate):

```yaml
# configs/robot.yaml
pickup_zone:
  z_bin: 10.0 # Fixed depth - measure actual bin depth!
```

**Important**:

- Measure actual bin depth precisely
- Account for object thickness
- Pick height = bin_depth + object_height/2

## Failure Analysis

### Common Issues and Solutions

| Issue                        | Cause                     | Solution                           |
| ---------------------------- | ------------------------- | ---------------------------------- |
| Gripper misses object        | Calibration drift         | Recalibrate before competition     |
| Object drops during transfer | Gripper not tight enough  | Increase `close_value`             |
| Slow sorting rate            | Conservative speeds       | Increase speed/acceleration        |
| False detections             | Low confidence threshold  | Increase threshold in classes.yaml |
| Missed detections            | High confidence threshold | Decrease threshold, retrain model  |
| Robot reaches limits         | Bin positions wrong       | Update positions in robot.yaml     |

### Pre-Competition Dry Run

**Simulate Competition**:

1. Load source bin with random nuts/bolts/screws
2. Start timer (use competition duration)
3. Launch program: `python src/main.py`
4. Let run until time expires
5. Count correctly sorted items
6. Calculate sorting rate (items/minute)
7. Identify bottlenecks

**Target Performance**:

- Goal: 30-50 items sorted (depends on time limit)
- Cycle time: ~10-15 seconds per item
- Success rate: >90%

## Competition Rules Considerations

**Verify with Rules**:

- Can you restart program if crash occurs?
- Are there penalties for incorrect placements?
- Is manual intervention allowed during run?
- What happens if time expires mid-pick?
- Can you adjust camera/bins after start?

**Adapt Strategy**:

- If restarts allowed: Optimize for speed (risk failures)
- If no restarts: Optimize for reliability
- If penalties exist: Increase confidence thresholds

## Team Roles

**During Setup**:

- **Hardware**: Position bins, secure camera/robot
- **Software**: Load configurations, verify connections
- **QA**: Run test picks, verify calibration

**During Competition**:

- **Monitor**: Watch system, note statistics
- **Recovery**: Ready to intervene if allowed
- **Timer**: Track remaining time

## Post-Competition

**Data Collection**:

- Save logs from `outputs/logs/`
- Record final statistics
- Note any failure modes
- Document lessons learned

**Analysis**:

- What was bottleneck? (detection, motion, gripper?)
- Which objects were hardest to sort?
- How to improve for next competition?

## Final Checklist

### T-1 Day

- [ ] All code tested and working
- [ ] YOLO model trained and validated
- [ ] Calibration completed and verified
- [ ] Gripper tuned for all object types
- [ ] Bin positions optimized
- [ ] Speed/acceleration tuned
- [ ] Full dry run completed
- [ ] Backup all code and configs

### T-0 (Competition Day)

- [ ] Hardware assembled and secured
- [ ] Connections verified
- [ ] Quick calibration check
- [ ] Test pick (before timer starts)
- [ ] Program ready to launch
- [ ] Team briefed on roles

**Launch Command Ready**:

```bash
conda activate RobotArm
cd ~/Robot-arm
python src/main.py --model models/yolo/best.pt
```

## Good Luck! 🏆

Remember:

- **Calibration** is everything
- **Reliability** > Speed (usually)
- **Test** in realistic conditions
- **Stay calm** during competition
- **Learn** from each run

You've got this! 🤖

# Robot-arm

This repository supports two workflows:
- [src/main.py](src/main.py): Fully automated sorting system (camera + detection + robot).
- [src/clickNgo.py](src/clickNgo.py): Interactive click-to-move demo.

## Getting Started

### 1. Set Up the Environment

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#using-miniconda-in-a-commercial-setting)

2. Run the following commands:

```bash
# clone the repository
git clone https://github.com/Ral151/Robot-arm.git

# create a new environment
conda env create -f environment.yml

# activate environment
conda activate RobotArm
```

### 2. Configure the System

Edit these config files before running:
- [configs/robot.yaml](configs/robot.yaml): Robot port, speed, and bin locations.
- [configs/camera.yaml](configs/camera.yaml): Camera settings and ROI.
- [configs/calibration.yaml](configs/calibration.yaml): Camera-to-robot calibration matrix.
- [configs/classes.yaml](configs/classes.yaml): Class labels, priorities, and thresholds.

Model weights:
- [bestV3.pt](https://drive.google.com/file/d/1NZvp3GvCKtf0V2yCnkHdSO_28TEzVu7I/view?usp=sharing) is the default model used by [src/main.py](src/main.py). Download it and place in the root of this folder. 

## Usage

### First Usage: Automated Sorting (main.py)

Run the full sorting system:

```bash
python -u src/main.py
```

What it does:
- Starts the RealSense camera.
- Detects objects with YOLO.
- Converts pixel coordinates to robot coordinates.
- Picks and places objects into configured bins.

### Second Usage: Click-to-Move Demo (clickNgo.py)

Run the interactive demo:

```bash
python -u src/clickNgo.py
```

What it does:
- Shows the camera view.
- Click a pixel to compute its 3D position and move the Dobot there.

## Notes

- Make sure the Dobot is connected and the COM port in [configs/robot.yaml](configs/robot.yaml) is correct.
- If the camera is not detected, update the camera settings in [configs/camera.yaml](configs/camera.yaml).
- Make sure the April Tag can be detected when running the program. Otherwise, it will not start. 

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from pathlib import Path

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CLASS_THRESHOLDS = {
    "long_bolt": 0.50,   
    "nut": 0.50,
    "short_bolt": 0.50,
}

# ROI coordinates
ROI_X1 = 325
ROI_Y1 = 201
ROI_X2 = 556
ROI_Y2 = 413

def ensure_realsense_available():
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense device found."
        )
    
    device_info = devices[0]
    device_name = device_info.get_info(rs.camera_info.name)
    print(f" RealSense device detected: {device_name}")
    return True

def object_detection():
    model = YOLO("bestV2.pt") # Object Detection Model
    try:
        ensure_realsense_available()
    except RuntimeError as e:   
        print(e)
        exit(1)
        
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)

    print(" RealSense started.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display_full = frame.copy()

            # Draw ROI rectangle on full frame
            cv2.rectangle(display_full, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

            # Extract ROI for detection
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            display_roi = roi.copy()

            results = model(roi, verbose=False)
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])


                    class_names = model.names
                    if 0 <= cls_id < len(class_names):
                        class_name = class_names[cls_id]
                    else:
                        class_name = f"class_{cls_id}"

                    # Apply per-class confidence threshold
                    class_threshold = CLASS_THRESHOLDS.get(class_name, 0.50)
                    if conf < class_threshold:
                        continue

                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Convert box to full-frame coordinates
                    x1_full = ROI_X1 + x1
                    y1_full = ROI_Y1 + y1
                    x2_full = ROI_X1 + x2
                    y2_full = ROI_Y1 + y2

                    # Center in ROI coordinates
                    cx_roi = int((x1 + x2) / 2)
                    cy_roi = int((y1 + y2) / 2)

                    # Convert center to full-frame coordinates
                    cx_full = ROI_X1 + cx_roi
                    cy_full = ROI_Y1 + cy_roi

                    label = f"{class_name} {conf:.2f}"

                    # Draw on ROI image
                    cv2.rectangle(display_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(display_roi, (cx_roi, cy_roi), 4, (0, 0, 255), cv2.FILLED)
                    cv2.putText(
                        display_roi,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        display_roi,
                        f"ROI({cx_roi},{cy_roi})",
                        (cx_roi + 5, cy_roi - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1
                    )

                    # Draw detection on full frame
                    cv2.rectangle(display_full, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
                    cv2.circle(display_full, (cx_full, cy_full), 5, (0, 0, 255), -1)
                    cv2.putText(
                        display_full,
                        label,
                        (x1_full, max(y1_full - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    
            cv2.imshow("Full Frame with ROI", display_full)
            cv2.imshow("YOLO on ROI Only", display_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
            print(" RealSense stopped.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
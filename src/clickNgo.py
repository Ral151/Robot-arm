


import argparse, signal, sys, yaml, cv2
import numpy as np
import pyrealsense2 as rs
from pydobotplus import Dobot
from typing import Dict
from calibration.apriltag_detection import get_apriltag_object
from calibration.transforms import calc_calibration,update_calib_yaml,get_target_coords
from utils.get_port import get_dobot_port
from Dobot.clickNgo_movement import move_dobot

from camera.rs_demo.realsense_utils import (
    initialize_pipeline,
    get_camera_intrinsics,
    get_aligned_frames,
    frames_to_numpy,
    pixel_to_3d,
    print_camera_info
)


def pixel_to_homogeneous_point(intrinsics, x: int, y: int, depth_frame):
    """
    Convert a pixel (x, y) and its depth into a 4x1 homogeneous column
    vector in the camera frame, ready for matrix multiplication.

    Returns:
        numpy array of shape (4, 1):
            [[X],   <- metres
             [Y],   <- metres
             [Z],   <- metres (depth)
             [0]]   <- homogeneous coordinate

        or None if depth is invalid at that pixel.

    Usage by teammate:
        P_camera = pixel_to_homogeneous_point(intrinsics, x, y, depth_frame)
        if P_camera is not None:
            P_robot = base_T_camera @ P_camera
    """
    point_3d = pixel_to_3d(intrinsics, x, y, depth_frame)

    if point_3d is None:
        return None

    X, Y, Z = point_3d

    return np.array([
        [X],
        [Y],
        [Z],
        [1]
    ], dtype=np.float64)
class RealSense3DConverter:
    """Interactive demo for converting 2D pixels to 3D coordinates"""
    
    def __init__(self):
        # Initialize pipeline
        self.pipeline, profile, config = initialize_pipeline(width=640, height=480, fps=30)
        
        # Get camera intrinsics
        self.intrinsics= get_camera_intrinsics(profile)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Set dobot device
        device_port = get_dobot_port()
        self.device = Dobot(port = device_port)
        self.device.home()
        
        # Store clicked point
        self.clicked_point = None
        
        print("=" * 70)
        print("RealSense SDK Demo 2: Pixel to 3D Conversion")
        print("=" * 70)
        print()
        print_camera_info(self.intrinsics)
        print("\n" + "=" * 70)
        print("HOW PIXEL TO 3D CONVERSION WORKS:")
        print("=" * 70)
        print("""
                The conversion from 2D pixel (x, y) to 3D point (X, Y, Z) uses:

                1. Depth value at pixel (x, y): Z = depth(x, y) in meters
                2. Camera intrinsics: focal length (fx, fy) and principal point (cx, cy)
                3. Pinhole camera model equations:

                X = (x - cx) * Z / fx
                Y = (y - cy) * Z / fy
                Z = depth(x, y)

                Where:
                - (x, y) = pixel coordinates
                - (cx, cy) = camera principal point (image center)
                - (fx, fy) = camera focal lengths in pixels
                - Z = depth value at that pixel
                - (X, Y, Z) = 3D coordinates in camera frame (meters)

                The RealSense SDK provides a helper function rs.rs2_deproject_pixel_to_point()
                that performs this calculation automatically.
                """)
        print("=" * 70)
        print("\nClick on the image to see 3D coordinates!")
        print("Press 'q' to quit\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select pixels"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
    
    def draw_crosshair(self, image, x, y, size=20, color=(0, 255, 0), thickness=2):
        """Draw a crosshair at the specified pixel location"""
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(image, (x, y), 5, color, -1)
    
    def run(self):
        """Main loop"""
        # Create window and set mouse callback
        window_name = 'RealSense: Click to get 3D coordinates'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        try:
            while True:
                # Get aligned frames
                color_frame, depth_frame = get_aligned_frames(self.pipeline, self.align)
                
                if color_frame is None or depth_frame is None:
                    continue
                
                # Convert to numpy array
                color_image, _ = frames_to_numpy(color_frame, depth_frame)
                
                # If a point was clicked, process it
                if self.clicked_point is not None:
                    x, y = self.clicked_point

                    # Convert pixel to 4x1 homogeneous point
                    P_camera = pixel_to_homogeneous_point(self.intrinsics, x, y, depth_frame)
                    
                    # Draw crosshair at clicked location
                    self.draw_crosshair(color_image, x, y)
                    
                    # Display information
                    if P_camera is not None:
                        X, Y, Z = P_camera[0,0], P_camera[1,0], P_camera[2,0]

                        # Convert to millimeters for easier reading
                        X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
                        
                        # Detect Apriltag
                        apriltag = get_apriltag_object(self.pipeline,self.align,self.intrinsics)
                        
                        # Get base_T_cam matrix
                        base_T_cam = calc_calibration(self.device,apriltag)
                        
                        # Apply matrix transformation
                        P_camera = np.array([[X_mm],[Y_mm],[Z_mm],[1.0]], dtype=np.float64)
                        target_coords = base_T_cam @ P_camera
                        X,Y,Z = target_coords[0][0],target_coords[1][0],target_coords[2][0]
                        move_dobot(self.device,X,Y,Z,0,True)
                        # Print to console
                        print(f"\n{'='*60}")
                        print(f"Pixel: ({x}, {y})")
                        print(f"3D Point in Camera Frame:")
                        print(f"  X = {X:7.1f} mm")
                        print(f"  Y = {Y:7.1f} mm")
                        print(f"  Z = {Z:7.1f} mm (depth)")
                        print(f"  Distance: {np.sqrt(X**2 + Y**2 + Z**2)*1000:.1f} mm")
                        print(f"P_camera (4x1):\n{P_camera}")
                        print(f"{'='*60}")
                        
                        # Draw info on image
                        info_text = [
                            f"Pixel: ({x}, {y})",
                            f"3D: X={X_mm:.1f}mm Y={Y_mm:.1f}mm Z={Z_mm:.1f}mm",
                            f"Distance: {np.sqrt(X**2 + Y**2 + Z**2)*1000:.1f}mm"
                        ]
                        
                        # Background for text
                        cv2.rectangle(color_image, (x + 15, y - 45), (x + 450, y + 15), 
                                    (0, 0, 0), -1)
                        cv2.rectangle(color_image, (x + 15, y - 45), (x + 450, y + 15), 
                                    (0, 255, 0), 2)
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(color_image, text, (x + 20, y - 25 + i * 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # Invalid depth
                        cv2.putText(color_image, "Invalid depth at this pixel", 
                                  (x + 20, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print(f"\nPixel ({x}, {y}): Invalid depth (too close or no data)")
                
                # Instructions
                cv2.putText(color_image, "Click anywhere to get 3D coordinates", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(color_image, "Press 'q' to quit", 
                          (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center crosshair for reference
                center_x, center_y = 320, 240
                cv2.line(color_image, (center_x - 10, center_y), (center_x + 10, center_y), 
                        (128, 128, 128), 1)
                cv2.line(color_image, (center_x, center_y - 10), (center_x, center_y + 10), 
                        (128, 128, 128), 1)
                cv2.putText(color_image, "Center", (center_x + 15, center_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                
                # Show image
                cv2.imshow(window_name, color_image)
                
                self.clicked_point = None
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Clear clicked point
                    self.clicked_point = None
                    print("\nCleared selection")
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("\n" + "=" * 70)
            print("Demo completed!")
            print("=" * 70)
            

def main():
    converter = RealSense3DConverter()
    P_camera = converter.run()


if __name__ == "__main__":
    main()
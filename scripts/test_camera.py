import numpy as np
from pupil_apriltags import Detector
import pyrealsense2 as rs
import yaml
import os
import cv2
from pathlib import Path
from datetime import datetime

def initialize_pipeline(serial=None):
    # Load camera serial from config
    config_path = os.path.join(os.path.dirname(__file__), '../configs/camera.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if serial is None:
        serial = 0
    print(f"Using camera serial: {serial}")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_camera_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy


def initialize_detector(tag_size):
    return Detector(families="tag36h11", nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0), tag_size


def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image, color_frame, depth_frame

def main():
    """Test RealSense camera with live visualization and snapshot capability."""
    # Create output directory for snapshots
    output_dir = Path("data/test_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RealSense Camera Test")
    print("="*60)
    print(f"Snapshots will be saved to: {output_dir}")
    print()
    print("Controls:")
    print("  's' - Save snapshot")
    print("  'q' - Quit")
    print("="*60)
    
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy = get_camera_intrinsics(profile)
    
    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    snapshot_count = 0
    frame_count = 0
    
    try:
        while True:
            color_image, depth_image, _, _ = process_frames(pipeline, align)
            if color_image is None or depth_image is None:
                continue
            
            frame_count += 1
            
            # Create display image with info overlay
            display_image = color_image.copy()
            
            # Show the color image
            cv2.imshow("RealSense Camera - Color", display_image)
            
            # Optionally show depth image (colorized)
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.03), 
            #     cv2.COLORMAP_JET
            # )
            # cv2.imshow("RealSense Camera - Depth", depth_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("s"):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                color_filename = output_dir / f"color_{timestamp}.jpg"
                
                cv2.imwrite(str(color_filename), color_image)
                
                snapshot_count += 1
                print(f"✓ Snapshot {snapshot_count} saved:")
                print(f"  Color: {color_filename}")
            
            elif key == ord("q"):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nCamera test complete. Total snapshots: {snapshot_count}")
        print(f"Images saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

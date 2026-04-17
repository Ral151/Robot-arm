import pyrealsense2 as rs
import os, yaml

def initialize_pipeline(serial=None):
    """Initialize RealSense pipeline"""
    config_path = os.path.join(os.path.dirname(__file__), '../configs/camera.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if serial is None:
        serial = config.get("camera", {}).get("serial", 0)
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
    """Get camera intrinsic parameters"""
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs
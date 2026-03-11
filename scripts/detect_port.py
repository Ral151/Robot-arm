import sys
from pathlib import Path
from serial.tools import list_ports

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.logger import get_logger

logger = get_logger("detect_port")

try:
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    logger.error("pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


import os
import yaml
from serial.tools import list_ports

def get_ports():
    return {port.device for port in list_ports.comports()}

def detect_dobot_port():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, 'robot.yaml')

    input("Please ensure the USB device is connected and press Enter...")
    ports_before = get_ports()

    input("Now, please unplug the USB device and press Enter...")
    ports_after = get_ports()

    device_port = ports_before - ports_after
    if not device_port:
        print("No new port detected. Please try again.")
        return
    device_port = device_port.pop()
    print(f"USB device detected on port: {device_port}")

    # Save detected device port
    return device_port

def update_robot_yaml(port: str):
    import yaml
    
    config_path = Path("configs/robot.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        config["robot"]["port"] = port
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Updated configs/robot.yaml with port: {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to update robot.yaml: {e}")
        return False


def main():
    """Main port detection workflow."""
    detected_port = detect_dobot_port()
    
    if detected_port:
        logger.info(f"\n port available: {detected_port}")
        if update_robot_yaml(detected_port):
            logger.info("\n Configuration updated successfully!")
            logger.info("You can now run: python scripts/test_robot.py")
        return

    else:
        logger.info("\n📝 To manually update the port:")
        logger.info("   Edit configs/robot.yaml")
        logger.info("   Set: robot.port to your COM port")
        logger.info("   Example: port: \"COM3\" (Windows) or port: \"/dev/ttyUSB0\" (Linux)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nCancelled by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

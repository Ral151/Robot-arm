
import os,yaml
from pydobotplus import Dobot
from serial.tools import list_ports

def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]

def check_port():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, 'device_port.yaml')

    input("Please ensure the USB device is connected and press Enter...")
    ports_before = {port.device for port in list_ports.comports()}

    input("Now, please unplug the USB device and press Enter...")
    ports_after = {port.device for port in list_ports.comports()}

    device_port = ports_before - ports_after
    if not device_port:
        print("No new port detected. Please try again.")
        return
    device_port = device_port.pop()
    print(f"USB device detected on port: {device_port}")

    # Save detected device port
    config_data = {'device_port': device_port}

    with open(config_file, 'w') as file:
        yaml.dump(config_data, file)

    print(f"Device port and camera serial(s) saved to {config_file}")
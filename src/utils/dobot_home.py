import os
import yaml
import time
from pydobotplus import Dobot


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "robot.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["robot"]["port"]

if __name__ == "__main__":
    port = get_dobot_port()
    device = Dobot(port=port)
    print("Homing the robotic arm...")
    device.home()
    device.close()
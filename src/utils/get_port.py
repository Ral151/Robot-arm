import os, yaml

def get_dobot_port():
    """Load Dobot port from config file"""
    config_file = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "robot.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["robot"]["port"]
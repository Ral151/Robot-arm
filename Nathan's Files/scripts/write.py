import os
import yaml
import time
from pydobotplus import Dobot

# ============== CONFIGURATION CONSTANTS ==============

# Jump motion parameters
JUMP_HEIGHT = 50  # mm - how high to lift during jump movement

# Pick and place locations (x, y, z, r)
PICK_LOCATION = (254,-40,77.5,0)  # Location to pick object
PLACE_LOCATION = (254,60,77.5,0)  # Location to place object

# Movement delays (seconds)
GRIP_DELAY = 1.0  # Time to wait after gripping
RELEASE_DELAY = 1.0  # Time to wait after releasing
MOVE_DELAY = 0.5  # Time to wait after movement

# ============== END CONFIGURATION ==============


def get_dobot_port():
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "device_port.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["device_port"]


def jump_to(device, x, y, z, r=0, jump_height=JUMP_HEIGHT):
    """
    Move to target position using a jump motion (lift up, move, descend).
    
    Args:
        device: Dobot device instance
        x, y, z: Target coordinates in mm
        r: Target rotation in degrees
        jump_height: Height to lift above current and target positions (mm)
    """
    # Get current position
    current_pose = device.get_pose()
    current_z = current_pose.position.z
    
    # Step 1: Lift to jump height from current position
    device.move_to(current_pose.position.x, current_pose.position.y, current_z, r, wait=True)
    time.sleep(MOVE_DELAY)
    
    # Step 2: Move horizontally to target XY at jump height
    print(f"  Moving to X={x:.1f}, Y={y:.1f} at Z={z:.1f}mm")
    device.move_to(x, y, z, r, wait=True)
    time.sleep(MOVE_DELAY)
    


def pick_and_place(device, pick_pos, place_pos):
    """
    Execute a pick and place operation between two positions.
    
    Args:
        device: Dobot device instance
        pick_pos: Tuple (x, y, z, r) for pick location
        place_pos: Tuple (x, y, z, r) for place location
    """
    print("\n=== Starting Pick and Place Operation ===")
    
    # Move to pick location
    print(f"\nMoving to PICK location: ({pick_pos[0]}, {pick_pos[1]}, {pick_pos[2]}, {pick_pos[3]})")
    jump_to(device, pick_pos[0], pick_pos[1], pick_pos[2], pick_pos[3])
    
    # Move to place location
    print(f"\nMoving to PLACE location: ({place_pos[0]}, {place_pos[1]}, {place_pos[2]}, {place_pos[3]})")
    jump_to(device, place_pos[0], place_pos[1], place_pos[2], place_pos[3])
    
    # Release object
    print("Releasing object...")
    device.grip(False)
    time.sleep(RELEASE_DELAY)
    
    print("\n=== Pick and Place Complete ===\n")

if __name__ == "__main__":
    port = get_dobot_port()
    device = Dobot(port=port)
    device.grip(True)
    try:
        # Execute pick and place task
        pick_and_place(device, PICK_LOCATION, PLACE_LOCATION)
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
    finally:
        device.move_to(200, 0, 100, 0, wait=True)  # Move to safe position
        print("Closing connection...")
        device.grip(False)  # Ensure gripper is open before closing
        device.close()
        print("Done!")
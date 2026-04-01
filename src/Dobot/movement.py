from pydobotplus import Dobot
import time

def move_dobot(device:Dobot,x,y,z,r,wait=True):
    pose = device.get_pose()
    device.move_to(pose.x,pose.y,max(pose.z,100),0,wait)
    time.sleep(2)
    device.move_to(x,y,z,r,wait)
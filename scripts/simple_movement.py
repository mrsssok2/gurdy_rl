#!/usr/bin/env python

import rospy
import time
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection
from joint_publisher import JointPub
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import numpy as np

class SimpleGurdyController:
    def __init__(self):
        rospy.logwarn("Initializing Simple Gurdy Controller...")
        
        # Initialize connections
        self.gazebo = GazeboConnection()
        self.controllers = ControllersConnection(namespace="gurdy")
        self.joint_pub = JointPub()
        
        # Define movement sequences
        self.base_position = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.move_forward = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.move_backward = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.turn_right = [-0.5, -0.1, -0.5, -0.1, -0.5, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.turn_left = [-0.1, -0.5, -0.1, -0.5, -0.1, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        # Publishers for debugging
        self.reward_pub = rospy.Publisher('/gurdy/reward', Float64, queue_size=1)
        
        rospy.logwarn("Simple Gurdy Controller initialized")
    
    def setup(self):
        rospy.logwarn("Setting up robot...")
        
        # Pause and reset simulation
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        
        # Remove gravity for initialization
        self.gazebo.change_gravity(0.0, 0.0, 0.0)
        
        # Reset controllers
        self.controllers.reset_gurdy_joint_controllers()
        
        # Check publisher connections
        self.joint_pub.check_publishers_connection()
        
        # Set initial position
        rospy.logwarn(f"Moving to initial position: {self.base_position}")
        self.joint_pub.move_joints(self.base_position)
        
        # Wait a moment for initialization
        rospy.sleep(1.0)
        
        # Restore gravity
        self.gazebo.change_gravity(0.0, 0.0, -9.81)
        
        # Unpause simulation
        self.gazebo.unpauseSim()
        
        rospy.logwarn("Robot setup complete")
        return True
    
    def run_movement_sequence(self):
        rospy.logwarn("Starting movement sequence...")
        
        # Test different movements
        movements = [
            ("Base Position", self.base_position),
            ("Move Forward", self.move_forward),
            ("Move Backward", self.move_backward),
            ("Turn Right", self.turn_right),
            ("Turn Left", self.turn_left),
            ("Base Position", self.base_position)
        ]
        
        for name, positions in movements:
            rospy.logwarn(f"Executing {name}...")
            
            # Publish the movement
            self.joint_pub.move_joints(positions)
            
            # Publish a fake reward for visualization
            reward_msg = Float64()
            reward_msg.data = 10.0
            self.reward_pub.publish(reward_msg)
            
            # Wait to see the effect
            rospy.sleep(2.0)
            
            rospy.logwarn(f"{name} executed")
        
        rospy.logwarn("Movement sequence completed")
        return True
    
    def shutdown(self):
        rospy.logwarn("Shutting down...")
        self.gazebo.pauseSim()
        rospy.logwarn("Shutdown complete")

if __name__ == "__main__":
    rospy.init_node('simple_gurdy_controller', anonymous=True, log_level=rospy.WARN)
    controller = SimpleGurdyController()
    
    try:
        # Setup the robot
        if controller.setup():
            # Run the movement sequence
            controller.run_movement_sequence()
    except Exception as e:
        rospy.logerr(f"Error occurred: {e}")
        import traceback
        rospy.logerr(f"Traceback: {traceback.format_exc()}")
    finally:
        controller.shutdown()
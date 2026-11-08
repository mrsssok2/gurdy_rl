#!/usr/bin/env python

import rospy
import math
import time
import copy
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class JointPub(object):
    def __init__(self):
        self.publishers_array = []
        
        # Upper leg joints
        self._upperleg1_joint_pub = rospy.Publisher('/gurdy/head_upperlegM1_joint_position_controller/command', Float64, queue_size=1)
        self._upperleg2_joint_pub = rospy.Publisher('/gurdy/head_upperlegM2_joint_position_controller/command', Float64, queue_size=1)
        self._upperleg3_joint_pub = rospy.Publisher('/gurdy/head_upperlegM3_joint_position_controller/command', Float64, queue_size=1)
        self._upperleg4_joint_pub = rospy.Publisher('/gurdy/head_upperlegM4_joint_position_controller/command', Float64, queue_size=1)
        self._upperleg5_joint_pub = rospy.Publisher('/gurdy/head_upperlegM5_joint_position_controller/command', Float64, queue_size=1)
        self._upperleg6_joint_pub = rospy.Publisher('/gurdy/head_upperlegM6_joint_position_controller/command', Float64, queue_size=1)
        
        # Lower leg joints
        self._lowerleg1_joint_pub = rospy.Publisher('/gurdy/upperlegM1_lowerlegM1_joint_position_controller/command', Float64, queue_size=1)
        self._lowerleg2_joint_pub = rospy.Publisher('/gurdy/upperlegM2_lowerlegM2_joint_position_controller/command', Float64, queue_size=1)
        self._lowerleg3_joint_pub = rospy.Publisher('/gurdy/upperlegM3_lowerlegM3_joint_position_controller/command', Float64, queue_size=1)
        self._lowerleg4_joint_pub = rospy.Publisher('/gurdy/upperlegM4_lowerlegM4_joint_position_controller/command', Float64, queue_size=1)
        self._lowerleg5_joint_pub = rospy.Publisher('/gurdy/upperlegM5_lowerlegM5_joint_position_controller/command', Float64, queue_size=1)
        self._lowerleg6_joint_pub = rospy.Publisher('/gurdy/upperlegM6_lowerlegM6_joint_position_controller/command', Float64, queue_size=1)
        
        # Add all publishers to array
        self.publishers_array.append(self._upperleg1_joint_pub)
        self.publishers_array.append(self._upperleg2_joint_pub)
        self.publishers_array.append(self._upperleg3_joint_pub)
        self.publishers_array.append(self._upperleg4_joint_pub)
        self.publishers_array.append(self._upperleg5_joint_pub)
        self.publishers_array.append(self._upperleg6_joint_pub)
        self.publishers_array.append(self._lowerleg1_joint_pub)
        self.publishers_array.append(self._lowerleg2_joint_pub)
        self.publishers_array.append(self._lowerleg3_joint_pub)
        self.publishers_array.append(self._lowerleg4_joint_pub)
        self.publishers_array.append(self._lowerleg5_joint_pub)
        self.publishers_array.append(self._lowerleg6_joint_pub)

    def set_init_pose(self, init_pose):
        """
        Sets joints to initial position
        :return: The init Pose
        """
        self.check_publishers_connection()
        self.move_joints(init_pose)

    def check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        
        # Check all publishers
        for publisher_object in self.publishers_array:
            publisher_name = publisher_object.name.split('/')[-2]
            while publisher_object.get_num_connections() == 0:
                rospy.logdebug(f"No subscribers to {publisher_name} yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is reset, time when backwards.
                    pass
            rospy.logdebug(f"{publisher_name} Publisher Connected")
        
        rospy.logdebug("All Joint Publishers READY")

    def move_joints(self, joints_array):
        """
        Move the joints to the given positions
        :param joints_array: Array of joint positions
        """
        i = 0
        for publisher_object in self.publishers_array:
            joint_value = Float64()
            joint_value.data = joints_array[i]
            rospy.logdebug(f"Joint {i} position: {joint_value.data}")
            publisher_object.publish(joint_value)
            i += 1

    def move_wheel(self, wheel_positions):
        """
        Move the wheels for locomotion
        :param wheel_positions: Array of wheel positions
        """
        self.move_joints(wheel_positions)

    def start_loop(self, rate_value=2.0):
        """
        Test function to move joints back and forth
        """
        rospy.logdebug("Start Loop")
        
        # Initial positions - all zeros
        pos1 = [0.0] * 12
        
        # Alternative positions - small movement on each joint
        pos2 = []
        for i in range(12):
            if i < 6:  # Upper legs
                pos2.append(-0.2)  # Small movement for upper legs
            else:  # Lower legs
                pos2.append(0.2)   # Small movement for lower legs
        
        position = "pos1"
        rate = rospy.Rate(rate_value)
        while not rospy.is_shutdown():
            if position == "pos1":
                self.move_joints(pos1)
                position = "pos2"
            else:
                self.move_joints(pos2)
                position = "pos1"
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('joint_publisher_node', log_level=rospy.WARN)
    joint_publisher = JointPub()
    rate_value = 8.0
    joint_publisher.start_loop(rate_value)
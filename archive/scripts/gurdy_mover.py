#!/usr/bin/env python
import rospy
import time
from math import pi, sin, cos, acos
import random
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class gurdyJointMover(object):
    def __init__(self):
        rospy.loginfo("Gurdy JointMover Initialising...")
        
        # Publishers for yaw joints (head upper legs)
        self.pub_head_upperlegM1_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM1_yaw_joint_position_controller/command', Float64, queue_size=1)
        self.pub_head_upperlegM2_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM2_yaw_joint_position_controller/command', Float64, queue_size=1)
        self.pub_head_upperlegM3_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM3_yaw_joint_position_controller/command', Float64, queue_size=1)
        self.pub_head_upperlegM4_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM4_yaw_joint_position_controller/command', Float64, queue_size=1)
        self.pub_head_upperlegM5_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM5_yaw_joint_position_controller/command', Float64, queue_size=1)
        self.pub_head_upperlegM6_yaw_joint_position = rospy.Publisher('/gurdy/head_upperlegM6_yaw_joint_position_controller/command', Float64, queue_size=1)

        # Publishers for upper legs
        self.pub_upperlegM1_joint_position = rospy.Publisher('/gurdy/head_upperlegM1_joint_position_controller/command', Float64, queue_size=1)
        self.pub_upperlegM2_joint_position = rospy.Publisher('/gurdy/head_upperlegM2_joint_position_controller/command', Float64, queue_size=1)
        self.pub_upperlegM3_joint_position = rospy.Publisher('/gurdy/head_upperlegM3_joint_position_controller/command', Float64, queue_size=1)
        self.pub_upperlegM4_joint_position = rospy.Publisher('/gurdy/head_upperlegM4_joint_position_controller/command', Float64, queue_size=1)
        self.pub_upperlegM5_joint_position = rospy.Publisher('/gurdy/head_upperlegM5_joint_position_controller/command', Float64, queue_size=1)
        self.pub_upperlegM6_joint_position = rospy.Publisher('/gurdy/head_upperlegM6_joint_position_controller/command', Float64, queue_size=1)

        # Publishers for lower legs
        self.pub_lowerlegM1_joint_position = rospy.Publisher('/gurdy/upperlegM1_lowerlegM1_joint_position_controller/command', Float64, queue_size=1)
        self.pub_lowerlegM2_joint_position = rospy.Publisher('/gurdy/upperlegM2_lowerlegM2_joint_position_controller/command', Float64, queue_size=1)
        self.pub_lowerlegM3_joint_position = rospy.Publisher('/gurdy/upperlegM3_lowerlegM3_joint_position_controller/command', Float64, queue_size=1)
        self.pub_lowerlegM4_joint_position = rospy.Publisher('/gurdy/upperlegM4_lowerlegM4_joint_position_controller/command', Float64, queue_size=1)
        self.pub_lowerlegM5_joint_position = rospy.Publisher('/gurdy/upperlegM5_lowerlegM5_joint_position_controller/command', Float64, queue_size=1)
        self.pub_lowerlegM6_joint_position = rospy.Publisher('/gurdy/upperlegM6_lowerlegM6_joint_position_controller/command', Float64, queue_size=1)

        # Add prismatic joints
        self.basefoot_peg_M1_basefoot_peg_M1_joint_joint_position = rospy.Publisher('/gurdy/basefoot_peg_M1_basefoot_peg_M1_joint_joint_position_controller/command', Float64, queue_size=1)
        self.basefoot_peg_M2_basefoot_peg_M2_joint_joint_position = rospy.Publisher('/gurdy/basefoot_peg_M2_basefoot_peg_M2_joint_joint_position_controller/command', Float64, queue_size=1)
        self.basefoot_peg_M3_basefoot_peg_M3_joint_joint_position = rospy.Publisher('/gurdy/basefoot_peg_M3_basefoot_peg_M3_joint_joint_position_controller/command', Float64, queue_size=1)

        # Initialize Twist message and subscribers
        self.twist_value = Twist()
        rospy.Subscriber("/cmd_vel", Twist, self._cmd_vel_callback)

        # Joint states topic name and initialization
        self.joint_states_topic_name = "/gurdy/joint_states"
        gurdy_joints_data = self._check_joint_states_ready()
        
        if gurdy_joints_data is not None:
            self.gurdy_joint_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.position))
        
        rospy.Subscriber(self.joint_states_topic_name, JointState, self.gurdy_joints_callback)

    def _cmd_vel_callback(self, msg):
        """ Callback function for cmd_vel topic. """
        # Process the incoming Twist message here.
        pass  # Implement your logic here

    def _check_joint_states_ready(self):
        self.joint_states = None
        rospy.logdebug("Waiting for "+str(self.joint_states_topic_name)+" to be READY...")
        
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = rospy.wait_for_message(self.joint_states_topic_name, JointState, timeout=5.0)
                rospy.logdebug("Current "+str(self.joint_states_topic_name)+" READY=>")
            except:
                rospy.logerr("Current "+str(self.joint_states_topic_name)+" not ready yet, retrying")
        
        return self.joint_states

    def move_gurdy_all_joints(self,
                               upperlegs_angles,
                               lowerlegs_values,
                               headarm_value,
                               headforearm_value,
                               upperlegs_yaw_angles,
                               basefoot_values):
        
        upperlegs_positions = [Float64(data) for data in upperlegs_angles]
        lowerlegs_positions = [Float64(data) for data in lowerlegs_values]
        
        # Publish all joints
        for i in range(6):
            getattr(self, f'pub_head_upperlegM{i+1}_yaw_joint_position').publish(Float64(upperlegs_yaw_angles[i]))
            getattr(self, f'pub_upperlegM{i+1}_joint_position').publish(upperlegs_positions[i])
            getattr(self, f'pub_lowerlegM{i+1}_joint_position').publish(lowerlegs_positions[i])

        # Move prismatic joints
        self.gurdy_move_ony_primsatic(*basefoot_values)

    def gurdy_move_ony_primsatic(self, foot_m1, foot_m2, foot_m3):
         """ Moves only the prismatic joints """
        
         basefoot_pegs = [Float64(data) for data in [foot_m1, foot_m2, foot_m3]]
        
         for i in range(3):
             getattr(self, f'basefoot_peg_M{i+1}_basefoot_peg_M{i+1}_joint_joint_position').publish(basefoot_pegs[i])

    def gurdy_joints_callback(self, msg):
         """ sensor_msgs/JointState """
        
         self.gurdy_joint_dictionary = dict(zip(msg.name, msg.position))

    def convert_angle_to_unitary(self, angle):
         """ Removes complete revolutions from angle and converts to positive equivalent if the angle is negative """
         complete_rev = 2 * pi
         mod_angle = int(angle / complete_rev)
         clean_angle = angle - mod_angle * complete_rev
        
         return clean_angle + (2 * pi if clean_angle < 0 else 0)

    def assertAlmostEqualAngles(self, x, y):
         c2 = (sin(x) - sin(y)) ** 2 + (cos(x) - cos(y)) ** 2
         angle_diff = acos((2.0 - c2) / 2.0)
         return angle_diff

    def gurdy_check_continuous_joint_value(self, joint_name, value, error=0.01):
         """ Check the joint by name is near the value given """
         joint_reading = self.gurdy_joint_dictionary.get(joint_name)

         if joint_reading is None:
             print(f"No data about joint: {joint_name}")
             return False

         clean_reading = self.convert_angle_to_unitary(joint_reading)
         clean_value = self.convert_angle_to_unitary(value)

         dif_angles = self.assertAlmostEqualAngles(clean_reading, clean_value)
         similar = dif_angles <= error

         if not similar:
             rospy.logerr(f"The joint {joint_name} hasn't reached yet {value}, difference={dif_angles}>{error}")
         
         return similar

    def gurdy_movement_look(self,
                             upperlegs_angles,
                             lowerlegs_values,
                             headarm_value,
                             headforearm_value,
                             upperlegs_yaw_angles,
                             basefoot_values):
        
         check_rate = 5.0
         rate = rospy.Rate(check_rate)

         while not all([self.gurdy_check_continuous_joint_value(f'head_upperlegM{i+1}_joint', angle) 
                        for i, angle in enumerate(upperlegs_angles)]) and not rospy.is_shutdown():
             
             print("Loop gurdy_movement_look")
             self.move_gurdy_all_joints(
                 upperlegs_angles,
                 lowerlegs_values,
                 headarm_value,
                 headforearm_value,
                 upperlegs_yaw_angles,
                 basefoot_values)

             rate.sleep()

    def gurdy_init_pos_sequence(self):
         """ Initialize joint positions """
        
         upper_leg_angles = [-1.55] * 6  # Example initialization for all upper legs
         lower_leg_angles = [-1.55] * 6   # Example initialization for all lower legs
       
         headarm_angle = 0.0
         headforearm_angle = 0.0
       
         basefoot_values = [0.0] * 3       # Example initialization for prismatic joints

         self.gurdy_movement_look(
             upper_leg_angles,
             lower_leg_angles,
             headarm_angle,
             headforearm_angle,
             [0.0] * 6,   # Yaw angles for each leg
             basefoot_values)

if __name__ == '__main__':
    try:
         rospy.init_node('gurdyJointMover')
         mover = gurdyJointMover()
         mover.gurdy_init_pos_sequence()
         rospy.spin()
    except rospy.ROSInterruptException:
         pass



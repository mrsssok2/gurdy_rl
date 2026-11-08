#!/usr/bin/env python

import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, acos
import random
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class gurdyJointPhaseAnalyzer(object):

    def __init__(self):
        rospy.init_node('joint_phase_analyzer', anonymous=True)
        rospy.loginfo("Gurdy Joint Phase Analyzer Initialising...")

        # Publishers for joint commands
        self.pub_upperlegM1_joint_position = rospy.Publisher(
            '/gurdy/head_upperlegM1_joint_position_controller/command',
            Float64,
            queue_size=1)
        self.pub_upperlegM2_joint_position = rospy.Publisher(
            '/gurdy/head_upperlegM2_joint_position_controller/command',
            Float64,
            queue_size=1)
        self.pub_upperlegM3_joint_position = rospy.Publisher(
            '/gurdy/head_upperlegM3_joint_position_controller/command',
            Float64,
            queue_size=1)
        self.pub_lowerlegM1_joint_position = rospy.Publisher(
            '/gurdy/upperlegM1_lowerlegM1_joint_position_controller/command',
            Float64,
            queue_size=1)
        self.pub_lowerlegM2_joint_position = rospy.Publisher(
            '/gurdy/upperlegM2_lowerlegM2_joint_position_controller/command',
            Float64,
            queue_size=1)
        self.pub_lowerlegM3_joint_position = rospy.Publisher(
            '/gurdy/upperlegM3_lowerlegM3_joint_position_controller/command',
            Float64,
            queue_size=1)

        # Data collection structures
        self.joint_data = {
            "head_upperlegM1_joint": {"angles": [], "velocities": []},
            "head_upperlegM2_joint": {"angles": [], "velocities": []},
            "head_upperlegM3_joint": {"angles": [], "velocities": []},
            "upperlegM1_lowerlegM1_joint": {"angles": [], "velocities": []},
            "upperlegM2_lowerlegM2_joint": {"angles": [], "velocities": []},
            "upperlegM3_lowerlegM3_joint": {"angles": [], "velocities": []}
        }

        # Joint states topic setup
        self.joint_states_topic_name = "/gurdy/joint_states"
        gurdy_joints_data = self._check_joint_states_ready()
        if gurdy_joints_data is not None:
            self.gurdy_joint_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.position))
            self.gurdy_velocity_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.velocity))
            rospy.Subscriber(self.joint_states_topic_name, JointState, self.gurdy_joints_callback)

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

    def move_gurdy_all_joints(self, upperlegM1_angle, upperlegM2_angle, upperlegM3_angle, 
                              lowerlegM1_value, lowerlegM2_value, lowerlegM3_value):
        upperlegM1 = Float64()
        upperlegM1.data = upperlegM1_angle
        upperlegM2 = Float64()
        upperlegM2.data = upperlegM2_angle
        upperlegM3 = Float64()
        upperlegM3.data = upperlegM3_angle

        lowerlegM1 = Float64()
        lowerlegM1.data = lowerlegM1_value
        lowerlegM2 = Float64()
        lowerlegM2.data = lowerlegM2_value
        lowerlegM3 = Float64()
        lowerlegM3.data = lowerlegM3_value

        self.pub_upperlegM1_joint_position.publish(upperlegM1)
        self.pub_upperlegM2_joint_position.publish(upperlegM2)
        self.pub_upperlegM3_joint_position.publish(upperlegM3)

        self.pub_lowerlegM1_joint_position.publish(lowerlegM1)
        self.pub_lowerlegM2_joint_position.publish(lowerlegM2)
        self.pub_lowerlegM3_joint_position.publish(lowerlegM3)

    def gurdy_joints_callback(self, msg):
        """
        Callback to process joint state messages and store angle and velocity data
        """
        self.gurdy_joint_dictionary = dict(zip(msg.name, msg.position))
        self.gurdy_velocity_dictionary = dict(zip(msg.name, msg.velocity))
        
        # Store joint data for phase diagrams
        for joint_name in self.joint_data.keys():
            if joint_name in self.gurdy_joint_dictionary:
                self.joint_data[joint_name]["angles"].append(self.gurdy_joint_dictionary[joint_name])
                self.joint_data[joint_name]["velocities"].append(self.gurdy_velocity_dictionary[joint_name])

    def convert_angle_to_unitary(self, angle):
        """
        Removes complete revolutions from angle and converts to positive equivalent
        if the angle is negative
        :param angle: Has to be in radians
        :return:
        """
        # Convert to angle between [0,2π)
        complete_rev = 2 * pi
        mod_angle = int(angle / complete_rev)
        clean_angle = angle - mod_angle * complete_rev
        # Convert Negative angles to their corresponding positive values
        if clean_angle < 0:
            clean_angle += 2 * pi
        return clean_angle

    def assertAlmostEqualAngles(self, x, y):
        c2 = (sin(x) - sin(y)) ** 2 + (cos(x) - cos(y)) ** 2
        angle_diff = acos((2.0 - c2) / 2.0)
        return angle_diff

    def gurdy_check_continuous_joint_value(self, joint_name, value, error=0.1):
        """
        Check if the joint is near the target value
        """
        joint_reading = self.gurdy_joint_dictionary.get(joint_name)
        if not joint_reading:
            print("Joint name not found:", joint_name)
            return False
            
        clean_joint_reading = self.convert_angle_to_unitary(angle=joint_reading)
        clean_value = self.convert_angle_to_unitary(angle=value)

        dif_angles = self.assertAlmostEqualAngles(clean_joint_reading, clean_value)
        similar = dif_angles <= error

        return similar

    def gurdy_movement_look(self, upperlegM1_angle, upperlegM2_angle, upperlegM3_angle, 
                            lowerlegM1_value, lowerlegM2_value, lowerlegM3_value):
        """
        Move joints to specified positions and wait until they reach those positions
        """
        check_rate = 5.0
        position_upperlegM1 = upperlegM1_angle
        position_upperlegM2 = upperlegM2_angle
        position_upperlegM3 = upperlegM3_angle

        position_lowerlegM1 = lowerlegM1_value
        position_lowerlegM2 = lowerlegM2_value
        position_lowerlegM3 = lowerlegM3_value

        similar_upperlegM1 = False
        similar_upperlegM2 = False
        similar_upperlegM3 = False

        similar_lowerlegM1 = False
        similar_lowerlegM2 = False
        similar_lowerlegM3 = False

        rate = rospy.Rate(check_rate)
        while not (similar_upperlegM1 and similar_upperlegM2 and similar_upperlegM3 and 
                   similar_lowerlegM1 and similar_lowerlegM2 and similar_lowerlegM3) and not rospy.is_shutdown():
            self.move_gurdy_all_joints(position_upperlegM1,
                                       position_upperlegM2,
                                       position_upperlegM3,
                                       position_lowerlegM1,
                                       position_lowerlegM2,
                                       position_lowerlegM3)
            similar_upperlegM1 = self.gurdy_check_continuous_joint_value(joint_name="head_upperlegM1_joint",
                                                                         value=position_upperlegM1)
            similar_upperlegM2 = self.gurdy_check_continuous_joint_value(joint_name="head_upperlegM2_joint",
                                                                         value=position_upperlegM2)
            similar_upperlegM3 = self.gurdy_check_continuous_joint_value(joint_name="head_upperlegM3_joint",
                                                                         value=position_upperlegM3)
            similar_lowerlegM1 = self.gurdy_check_continuous_joint_value(joint_name="upperlegM1_lowerlegM1_joint",
                                                                         value=position_lowerlegM1)
            similar_lowerlegM2 = self.gurdy_check_continuous_joint_value(joint_name="upperlegM2_lowerlegM2_joint",
                                                                         value=position_lowerlegM2)
            similar_lowerlegM3 = self.gurdy_check_continuous_joint_value(joint_name="upperlegM3_lowerlegM3_joint",
                                                                         value=position_lowerlegM3)
            rate.sleep()

    def gurdy_init_pos_sequence(self):
        """
        Initialize the robot position
        """
        upperlegM1_angle = -1.55
        upperlegM2_angle = -1.55
        upperlegM3_angle = -1.55
        lowerlegM1_angle = 0.0
        lowerlegM2_angle = 0.0
        lowerlegM3_angle = 0.0
        self.gurdy_movement_look(upperlegM1_angle,
                                 upperlegM2_angle,
                                 upperlegM3_angle,
                                 lowerlegM1_angle,
                                 lowerlegM2_angle,
                                 lowerlegM3_angle)

        lowerlegM1_angle = -1.55
        lowerlegM2_angle = -1.55
        lowerlegM3_angle = -1.55
        self.gurdy_movement_look(upperlegM1_angle,
                                 upperlegM2_angle,
                                 upperlegM3_angle,
                                 lowerlegM1_angle,
                                 lowerlegM2_angle,
                                 lowerlegM3_angle)

    def gurdy_hop(self, num_hops=15):
        """
        Execute hopping motion and collect data for phase diagrams
        """
        upper_delta = 1
        basic_angle = -1.55
        angle_change = random.uniform(0.2, 0.7)
        upperlegM_angle = basic_angle
        lowerlegM_angle = basic_angle - upper_delta * angle_change * 2.0

        # Clear previous data
        for joint in self.joint_data.keys():
            self.joint_data[joint]["angles"] = []
            self.joint_data[joint]["velocities"] = []

        for repetitions in range(num_hops):
            self.gurdy_movement_look(upperlegM_angle,
                                     upperlegM_angle,
                                     upperlegM_angle,
                                     lowerlegM_angle,
                                     lowerlegM_angle,
                                     lowerlegM_angle)

            upper_delta *= -1
            if upper_delta < 0:
                upperlegM_angle = basic_angle + angle_change
            else:
                upperlegM_angle = basic_angle
            lowerlegM_angle = basic_angle - upper_delta * angle_change * 2.0

    def gurdy_moverandomly(self, num_moves=10):
        """
        Move joints randomly to collect varied data for phase diagrams
        """
        # Clear previous data
        for joint in self.joint_data.keys():
            self.joint_data[joint]["angles"] = []
            self.joint_data[joint]["velocities"] = []
            
        for _ in range(num_moves):
            upperlegM1_angle = random.uniform(-1.55, 0.0)
            upperlegM2_angle = random.uniform(-1.55, 0.0)
            upperlegM3_angle = random.uniform(-1.55, 0.0)
            lowerlegM1_angle = random.uniform(-2.9, pi/2)
            lowerlegM2_angle = random.uniform(-2.9, pi/2)
            lowerlegM3_angle = random.uniform(-2.9, pi/2)
            self.gurdy_movement_look(upperlegM1_angle,
                                    upperlegM2_angle,
                                    upperlegM3_angle,
                                    lowerlegM1_angle,
                                    lowerlegM2_angle,
                                    lowerlegM3_angle)

    def plot_phase_diagrams(self):
        """
        Plot phase diagrams for all 6 joints
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Gurdy Joint Phase Diagrams (Joint Velocity vs. Joint Angle)', fontsize=16)
        
        joint_names = list(self.joint_data.keys())
        
        for i in range(2):
            for j in range(3):
                idx = i * 3 + j
                if idx < len(joint_names):
                    joint_name = joint_names[idx]
                    
                    angles = self.joint_data[joint_name]["angles"]
                    velocities = self.joint_data[joint_name]["velocities"]
                    
                    if angles and velocities:
                        axs[i, j].scatter(angles, velocities, s=2, alpha=0.6)
                        axs[i, j].plot(angles, velocities, 'r-', linewidth=0.5, alpha=0.3)
                        axs[i, j].set_title(joint_name)
                        axs[i, j].set_xlabel('Joint Angle (rad)')
                        axs[i, j].set_ylabel('Joint Velocity (rad/s)')
                        axs[i, j].grid(True)
                    else:
                        axs[i, j].text(0.5, 0.5, 'No data available', 
                                      horizontalalignment='center',
                                      verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('gurdy_phase_diagrams.png')
        plt.show()

    def collect_data_and_plot(self):
        """
        Main function to collect joint data and plot phase diagrams
        """
        rospy.loginfo("Initializing Gurdy position...")
        self.gurdy_init_pos_sequence()
        
        rospy.loginfo("Collecting data with random movements...")
        self.gurdy_moverandomly(num_moves=5)
        
        rospy.loginfo("Collecting data with hopping motions...")
        self.gurdy_hop(num_hops=5)
        
        rospy.loginfo("Plotting phase diagrams...")
        self.plot_phase_diagrams()
        
        rospy.loginfo("Phase diagram analysis complete!")

if __name__ == "__main__":
    try:
        gurdy_analyzer = gurdyJointPhaseAnalyzer()
        gurdy_analyzer.collect_data_and_plot()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python

import rospy
import time
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, acos, sqrt
import random
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

class gurdyEnergyAnalyzer(object):

    def __init__(self):
        rospy.init_node('joint_energy_analyzer', anonymous=True)
        rospy.loginfo("Gurdy Joint Energy Analyzer Initialising...")

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
            "head_upperlegM1_joint": {"angles": [], "potential_energy": []},
            "head_upperlegM2_joint": {"angles": [], "potential_energy": []},
            "head_upperlegM3_joint": {"angles": [], "potential_energy": []},
            "upperlegM1_lowerlegM1_joint": {"angles": [], "potential_energy": []},
            "upperlegM2_lowerlegM2_joint": {"angles": [], "potential_energy": []},
            "upperlegM3_lowerlegM3_joint": {"angles": [], "potential_energy": []}
        }
        
        # System-level data collection
        self.system_data = {"time": [], "total_potential_energy": []}

        # Joint states topic setup
        self.joint_states_topic_name = "/gurdy/joint_states"
        gurdy_joints_data = self._check_joint_states_ready()
        if gurdy_joints_data is not None:
            self.gurdy_joint_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.position))
            self.gurdy_velocity_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.velocity))
            self.gurdy_effort_dictionary = dict(zip(gurdy_joints_data.name, gurdy_joints_data.effort))
            rospy.Subscriber(self.joint_states_topic_name, JointState, self.gurdy_joints_callback)

        # Parameters for energy calculation
        self.g = 9.81  # gravitational acceleration (m/s²)
        
        # Approximate masses and lengths for Gurdy's legs (in kg and meters)
        # These are estimates - replace with actual values if available
        self.upper_leg_mass = 0.5  # kg
        self.lower_leg_mass = 0.3  # kg
        self.upper_leg_length = 0.2  # meters
        self.lower_leg_length = 0.2  # meters
        
        # Spring constants (approximate values, replace with actual values if available)
        self.upper_leg_spring_constant = 5.0  # N/m
        self.lower_leg_spring_constant = 3.0  # N/m
        
        # Equilibrium positions (where spring force is zero)
        self.upper_leg_equilibrium = -1.0  # radians
        self.lower_leg_equilibrium = -1.0  # radians

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

    def calculate_potential_energy(self, joint_name, joint_angle):
        """
        Calculate potential energy for a joint based on its position
        This combines gravitational potential energy and spring potential energy
        """
        # Determine joint type
        is_upper_leg = "head_upperleg" in joint_name
        
        # Calculate energy based on joint type
        if is_upper_leg:
            mass = self.upper_leg_mass
            length = self.upper_leg_length
            spring_constant = self.upper_leg_spring_constant
            equilibrium = self.upper_leg_equilibrium
        else:  # lower leg
            mass = self.lower_leg_mass
            length = self.lower_leg_length
            spring_constant = self.lower_leg_spring_constant
            equilibrium = self.lower_leg_equilibrium
        
        # Calculate gravitational potential energy
        # For simplicity, height change is approximated as length * (1 - cos(angle))
        height_change = length * (1 - cos(joint_angle))
        gravitational_energy = mass * self.g * height_change
        
        # Calculate spring potential energy (using Hooke's law: PE = 0.5 * k * x^2)
        displacement = joint_angle - equilibrium
        spring_energy = 0.5 * spring_constant * displacement**2
        
        # Total potential energy
        total_energy = gravitational_energy + spring_energy
        
        return total_energy

    def calculate_total_system_energy(self):
        """
        Calculate the total potential energy of the entire Gurdy robot system
        by summing the potential energy of all joints
        """
        total_energy = 0.0
        
        # Get current joint positions
        for joint_name in self.joint_data.keys():
            if joint_name in self.gurdy_joint_dictionary:
                angle = self.gurdy_joint_dictionary[joint_name]
                joint_energy = self.calculate_potential_energy(joint_name, angle)
                total_energy += joint_energy
        
        return total_energy

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
        Callback to process joint state messages and store angle and energy data
        """
        self.gurdy_joint_dictionary = dict(zip(msg.name, msg.position))
        self.gurdy_velocity_dictionary = dict(zip(msg.name, msg.velocity))
        self.gurdy_effort_dictionary = dict(zip(msg.name, msg.effort))
        
        # Store joint data and calculate energy
        for joint_name in self.joint_data.keys():
            if joint_name in self.gurdy_joint_dictionary:
                angle = self.gurdy_joint_dictionary[joint_name]
                energy = self.calculate_potential_energy(joint_name, angle)
                
                self.joint_data[joint_name]["angles"].append(angle)
                self.joint_data[joint_name]["potential_energy"].append(energy)
        
        # Calculate and store total system energy
        total_energy = self.calculate_total_system_energy()
        self.system_data["time"].append(rospy.get_time())
        self.system_data["total_potential_energy"].append(total_energy)

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

    def gurdy_sweep_joints(self):
        """
        Sweep joints through their ranges to collect energy data at different positions
        """
        # Clear previous data
        for joint in self.joint_data.keys():
            self.joint_data[joint]["angles"] = []
            self.joint_data[joint]["potential_energy"] = []
        
        self.system_data = {"time": [], "total_potential_energy": []}
        
        # For upper leg joints (range: -1.55 to 0.0)
        step = 0.15
        for angle in np.arange(-1.55, 0.01, step):
            self.gurdy_movement_look(angle, angle, angle, 
                                    -1.55, -1.55, -1.55)
            rospy.sleep(0.5)  # Allow time to collect data

        # Reset to initial position
        self.gurdy_init_pos_sequence()
        
        # For lower leg joints (range: -2.9 to 1.5708)
        step = 0.4
        for angle in np.arange(-2.9, 1.5, step):
            self.gurdy_movement_look(-1.55, -1.55, -1.55, 
                                    angle, angle, angle)
            rospy.sleep(0.5)  # Allow time to collect data
        
        # Reset to initial position
        self.gurdy_init_pos_sequence()

    def gurdy_hop(self, num_hops=10):
        """
        Execute hopping motion and collect energy data
        """
        upper_delta = 1
        basic_angle = -1.55
        angle_change = random.uniform(0.2, 0.7)
        upperlegM_angle = basic_angle
        lowerlegM_angle = basic_angle - upper_delta * angle_change * 2.0

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

    def plot_energy_diagrams(self):
        """
        Plot potential energy vs. joint position diagrams for all 6 joints
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Gurdy Joint Potential Energy vs. Position', fontsize=16)
        
        joint_names = list(self.joint_data.keys())
        
        for i in range(2):
            for j in range(3):
                idx = i * 3 + j
                if idx < len(joint_names):
                    joint_name = joint_names[idx]
                    
                    angles = self.joint_data[joint_name]["angles"]
                    energies = self.joint_data[joint_name]["potential_energy"]
                    
                    if angles and energies:
                        # Sort data points by angle to get a cleaner plot
                        sorted_indices = np.argsort(angles)
                        sorted_angles = [angles[i] for i in sorted_indices]
                        sorted_energies = [energies[i] for i in sorted_indices]
                        
                        axs[i, j].scatter(sorted_angles, sorted_energies, s=2, alpha=0.6)
                        axs[i, j].plot(sorted_angles, sorted_energies, 'g-', linewidth=0.8, alpha=0.6)
                        axs[i, j].set_title(joint_name)
                        axs[i, j].set_xlabel('Joint Position (rad)')
                        axs[i, j].set_ylabel('Potential Energy (J)')
                        axs[i, j].grid(True)
                    else:
                        axs[i, j].text(0.5, 0.5, 'No data available', 
                                      horizontalalignment='center',
                                      verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('gurdy_potential_energy_diagrams.png')
        plt.show()

    def plot_total_system_energy(self):
        """
        Plot the total potential energy of the system over time
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.system_data["time"], self.system_data["total_potential_energy"], 'b-')
        plt.title('Gurdy Total System Potential Energy Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Total Potential Energy (J)')
        plt.grid(True)
        plt.savefig('gurdy_total_system_energy.png')
        plt.show()
        
    def plot_total_system_energy_vs_configuration(self):
        """
        Plot the total potential energy of the system against a configuration parameter
        For simplicity, we'll use the average position of upper leg joints as the configuration parameter
        """
        # Calculate average upper leg position for each time point
        config_param = []
        for i in range(len(self.system_data["time"])):
            time_index = i
            
            # Find closest angle measurements for this time
            upper_leg_positions = []
            for joint_name in ["head_upperlegM1_joint", "head_upperlegM2_joint", "head_upperlegM3_joint"]:
                if time_index < len(self.joint_data[joint_name]["angles"]):
                    upper_leg_positions.append(self.joint_data[joint_name]["angles"][time_index])
            
            if upper_leg_positions:
                avg_position = sum(upper_leg_positions) / len(upper_leg_positions)
                config_param.append(avg_position)
            else:
                # Skip this time point if no corresponding joint data
                continue
                
        # Only plot points where we have both configuration and energy data
        valid_indices = min(len(config_param), len(self.system_data["total_potential_energy"]))
        
        if valid_indices > 0:
            # Plot energy vs configuration
            plt.figure(figsize=(10, 6))
            plt.scatter(config_param[:valid_indices], 
                        self.system_data["total_potential_energy"][:valid_indices],
                        c='blue', alpha=0.7)
            plt.title('Gurdy Total System Potential Energy vs. Configuration')
            plt.xlabel('Average Upper Leg Position (rad)')
            plt.ylabel('Total Potential Energy (J)')
            plt.grid(True)
            plt.savefig('gurdy_total_system_energy_vs_config.png')
            plt.show()
        else:
            rospy.logwarn("Not enough data to plot system energy vs. configuration")

    def collect_data_and_plot(self):
        """
        Main function to collect joint energy data and plot diagrams
        """
        rospy.loginfo("Initializing Gurdy position...")
        self.gurdy_init_pos_sequence()
        
        rospy.loginfo("Sweeping joints through ranges...")
        self.gurdy_sweep_joints()
        
        rospy.loginfo("Collecting data with hopping motions...")
        self.gurdy_hop(num_hops=5)
        
        rospy.loginfo("Plotting individual joint potential energy diagrams...")
        self.plot_energy_diagrams()
        
        rospy.loginfo("Plotting total system potential energy over time...")
        self.plot_total_system_energy()
        
        rospy.loginfo("Plotting total system potential energy vs. configuration...")
        self.plot_total_system_energy_vs_configuration()
        
        rospy.loginfo("Potential energy analysis complete!")

if __name__ == "__main__":
    try:
        gurdy_analyzer = gurdyEnergyAnalyzer()
        gurdy_analyzer.collect_data_and_plot()
    except rospy.ROSInterruptException:
        pass
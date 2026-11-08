#!/usr/bin/env python

import rospy
import copy
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
import tf
import numpy
import math

class GurdyState(object):

    def __init__(self, max_height, min_height, abs_max_roll, abs_max_pitch, list_of_observations, joint_limits, episode_done_criteria, joint_increment_value=0.05, done_reward=-1000.0, alive_reward=10.0, desired_force=7.08, desired_yaw=0.0, weight_r1=1.0, weight_r2=1.0, weight_r3=1.0, weight_r4=1.0, weight_r5=1.0, discrete_division=10, maximum_base_linear_acceleration=3000.0, maximum_base_angular_velocity=20.0, maximum_joint_effort=10.0, wheel_increment=0.1):
        rospy.logdebug("Starting GurdyState Class object...")
        self.desired_world_point = Vector3(0.0, 0.0, 0.0)
        self._min_height = min_height
        self._max_height = max_height
        self._abs_max_roll = abs_max_roll
        self._abs_max_pitch = abs_max_pitch
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._alive_reward = alive_reward
        self._desired_force = desired_force
        self._desired_yaw = desired_yaw

        self._weight_r1 = weight_r1
        self._weight_r2 = weight_r2
        self._weight_r3 = weight_r3
        self._weight_r4 = weight_r4
        self._weight_r5 = weight_r5

        self._list_of_observations = list_of_observations

        # Dictionary with the max and min of each of the joints
        self._joint_limits = joint_limits

        # Maximum base linear acceleration values
        self.maximum_base_linear_acceleration = maximum_base_linear_acceleration

        # Maximum Angular Velocity value
        self.maximum_base_angular_velocity = maximum_base_angular_velocity

        self.maximum_joint_effort = maximum_joint_effort

        # List of all the Done Episode Criteria
        self._episode_done_criteria = episode_done_criteria
        assert len(self._episode_done_criteria) != 0, "Episode_done_criteria list is empty. Minimum one value"

        self._discrete_division = discrete_division

        self.wheel_increment = wheel_increment
        # We init the observation ranges and We create the bins now for all the observations
        self.init_bins()

        self.base_position = Point()
        self.base_orientation = Quaternion()
        self.base_angular_velocity = Vector3()
        self.base_linear_acceleration = Vector3()
        self.contact_force = Vector3()
        self.joints_state = JointState()

        # Odom we only use it for the height detection and planar position
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # We use the IMU for orientation and linearacceleration detection
        rospy.Subscriber("/gurdy/imu/data", Imu, self.imu_callback)
        # We use it to get the contact force, to know if its in the air or stumping too hard.
        rospy.Subscriber("/lowerleg_contactsensor_state", ContactsState, self.contact_callback)
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/gurdy/joint_states", JointState, self.joints_state_callback)

    def init_bins(self):
        """
        We init the Bins for the discretization of the observations
        """
        self.height_bins = numpy.linspace(self._min_height, self._max_height, self._discrete_division)
        self.roll_bins = numpy.linspace(-self._abs_max_roll, self._abs_max_roll, self._discrete_division)
        self.pitch_bins = numpy.linspace(-self._abs_max_pitch, self._abs_max_pitch, self._discrete_division)
        
        self.linearacc_bins = numpy.linspace(-self.maximum_base_linear_acceleration,
                                             self.maximum_base_linear_acceleration,
                                             self._discrete_division)
        
        self.angularvel_bins = numpy.linspace(-self.maximum_base_angular_velocity,
                                             self.maximum_base_angular_velocity,
                                             self._discrete_division)
        
        self.joint_position_bins = numpy.linspace(-math.pi, math.pi, self._discrete_division)
        
        self.joint_effort_bins = numpy.linspace(-self.maximum_joint_effort, self.maximum_joint_effort, self._discrete_division)
        
    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        data_pose = None
        while data_pose is None and not rospy.is_shutdown():
            try:
                data_pose = rospy.wait_for_message("/odom", Odometry, timeout=0.1)
                self.base_position = data_pose.pose.pose.position
                rospy.logdebug("Current odom READY")
            except:
                rospy.logdebug("Current odom pose not ready yet, retrying for getting robot base_position")

        imu_data = None
        while imu_data is None and not rospy.is_shutdown():
            try:
                imu_data = rospy.wait_for_message("/gurdy/imu/data", Imu, timeout=0.1)
                self.base_orientation = imu_data.orientation
                self.base_angular_velocity = imu_data.angular_velocity
                self.base_linear_acceleration = imu_data.linear_acceleration
                rospy.logdebug("Current imu_data READY")
            except:
                rospy.logdebug("Current imu_data not ready yet, retrying for getting robot base_orientation, and base_linear_acceleration")

        contacts_data = None
        while contacts_data is None and not rospy.is_shutdown():
            try:
                contacts_data = rospy.wait_for_message("/lowerleg_contactsensor_state", ContactsState, timeout=0.1)
                for state in contacts_data.states:
                    self.contact_force = state.total_wrench.force
                rospy.logdebug("Current contacts_data READY")
            except:
                rospy.logdebug("Current contacts_data not ready yet, retrying")

        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/gurdy/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def set_desired_world_point(self, x, y, z):
        """
        Point where you want the Gurdy to be
        :return:
        """
        self.desired_world_point.x = x
        self.desired_world_point.y = y
        self.desired_world_point.z = z

    def get_base_height(self):
        height = self.base_position.z
        rospy.logdebug("BASE-HEIGHT="+str(height))
        return height

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion(
            [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w])

        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def get_base_angular_velocity(self):
        return self.base_angular_velocity

    def set_joint_states(self, joints_increment_array):
        """
        Sets the desired joint positions by publishing to the joints controllers
        :param joints_increment_array: Array with the increment/decrement of each joint position
        :return:
        """
        # We need to import the JointPub here to avoid circular imports
        from joint_publisher import JointPub
        
        joint_publisher_object = JointPub()
        joint_publisher_object.check_publishers_connection()
        
        # Get current joint states
        current_jstates = self.get_joint_states().position
        
        # Calculate new joint positions from increments
        new_positions = []
        for i in range(len(current_jstates)):
            new_positions.append(current_jstates[i] + joints_increment_array[i])
        
        # Apply limits to the new positions
        for i in range(len(new_positions)):
            # Get the joint name
            if i < 6:  # Upper leg joints
                joint_name = f"wheel{i+1}"
            else:      # Lower leg joints
                joint_name = f"wheel{i+1}"
                
            # Apply limits if defined for this joint
            if joint_name in self._joint_limits:
                limits = self._joint_limits[joint_name]
                min_value = limits.get("min", -float('inf'))
                max_value = limits.get("max", float('inf'))
                new_positions[i] = max(min(new_positions[i], max_value), min_value)
        
        # Log what we're doing
        rospy.logwarn(f"Current positions: {[round(pos, 2) for pos in current_jstates[:6]]}")
        rospy.logwarn(f"Increments: {[round(inc, 2) for inc in joints_increment_array[:6]]}")
        rospy.logwarn(f"New positions: {[round(pos, 2) for pos in new_positions[:6]]}")
        
        # Send the joint commands
        joint_publisher_object.move_joints(new_positions)
        
        # Wait a bit for controllers to apply the commands
        rospy.sleep(0.1)

    def get_base_linear_acceleration(self):
        return self.base_linear_acceleration

    def get_distance_from_point(self, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((self.base_position.x, self.base_position.y, self.base_position.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_contact_force_magnitude(self):
        """
        You will see that because the X axis is the one pointing downwards, it will be the one with
        higher value when touching the floor
        For a Robot of total mas of 0.55Kg, a gravity of 9.81 m/sec**2, Weight = 0.55*9.81=5.39 N
        Falling from around 5centimetres ( negligible height ), we register peaks around
        Fx = 7.08 N
        :return:
        """
        contact_force = self.contact_force
        contact_force_np = numpy.array((contact_force.x, contact_force.y, contact_force.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    def get_joint_states(self):
        return self.joints_state

    def odom_callback(self, msg):
        self.base_position = msg.pose.pose.position

    def imu_callback(self, msg):
        self.base_orientation = msg.orientation
        self.base_angular_velocity = msg.angular_velocity
        self.base_linear_acceleration = msg.linear_acceleration

    def contact_callback(self, msg):
        """
        /lowerleg_contactsensor_state/states[0]/contact_positions ==> PointContact in World
        /lowerleg_contactsensor_state/states[0]/contact_normals ==> NormalContact in World

        ==> One is an array of all the forces, the other total,
         and are relative to the contact link referred to in the sensor.
        /lowerleg_contactsensor_state/states[0]/wrenches[]
        /lowerleg_contactsensor_state/states[0]/total_wrench
        :param msg:
        :return:
        """
        for state in msg.states:
            self.contact_force = state.total_wrench.force

    def joints_state_callback(self, msg):
        self.joints_state = msg

    def gurdy_height_ok(self):
        height_ok = self._min_height <= self.get_base_height() < self._max_height
        return height_ok

    def gurdy_orientation_ok(self):
        orientation_rpy = self.get_base_rpy()
        roll_ok = self._abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self._abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok

    def calculate_reward_joint_position(self, weight=1.0):
        """
        We calculate reward base on the joints configuration. The more near 0 the better.
        :return:
        """
        acumulated_joint_pos = 0.0
        for joint_pos in self.joints_state.position:
            # Abs to remove sign influence, it doesnt matter the direction of turn.
            acumulated_joint_pos += abs(joint_pos)
            rospy.logdebug("calculate_reward_joint_position>>acumulated_joint_pos=" + str(acumulated_joint_pos))
        reward = weight * acumulated_joint_pos
        rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
        return reward

    def calculate_reward_joint_effort(self, weight=1.0):
        """
        We calculate reward base on the joints effort readings. The more near 0 the better.
        :return:
        """
        acumulated_joint_effort = 0.0
        for joint_effort in self.joints_state.effort:
            # Abs to remove sign influence, it doesnt matter the direction of the effort.
            acumulated_joint_effort += abs(joint_effort)
            rospy.logdebug("calculate_reward_joint_effort>>joint_effort=" + str(joint_effort))
            rospy.logdebug("calculate_reward_joint_effort>>acumulated_joint_effort=" + str(acumulated_joint_effort))
        reward = weight * acumulated_joint_effort
        rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(reward))
        return reward

    def calculate_reward_contact_force(self, weight=1.0):
        """
        We calculate reward base on the contact force.
        The nearest to the desired contact force the better.
        We use exponential to magnify big departures from the desired force.
        Default ( 7.08 N ) desired force was taken from reading of the robot touching the ground
        with the feet under its own weight.
        :return:
        """
        force_magnitude = self.get_contact_force_magnitude()
        force_displacement = abs(self._desired_force - force_magnitude)
        rospy.logdebug("calculate_reward_contact_force>>force_magnitude=" + str(force_magnitude))
        rospy.logdebug("calculate_reward_contact_force>>force_displacement=" + str(force_displacement))
        # Negative because we want to reward small force displacement
        reward = weight * -1.0 * force_displacement
        rospy.logdebug("calculate_reward_contact_force>>reward=" + str(reward))
        return reward

    def calculate_reward_orientation(self, weight=1.0):
        """
        We calculate the reward based on the orientation.
        The more its closser to 0 the better because it means its upright
        desired_yaw is the desired_orientation=0 by default.
        :return:
        """
        orientation = self.get_base_rpy()
        
        # Abs because it doesnt matter the direction of turn
        roll_displacement = abs(orientation.x)
        pitch_displacement = abs(orientation.y)
        yaw_displacement = abs(self._desired_yaw - orientation.z)
        
        rospy.logdebug("calculate_reward_orientation>>roll_displacement=" + str(roll_displacement))
        rospy.logdebug("calculate_reward_orientation>>pitch_displacement=" + str(pitch_displacement))
        rospy.logdebug("calculate_reward_orientation>>yaw_displacement=" + str(yaw_displacement))
        
        # We want to reward being close to 0 in roll and pitch, and close to desired yaw
        reward = weight * (-1.0 * roll_displacement - 1.0 * pitch_displacement - 1.0 * yaw_displacement)
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_reward_distance_from_des_point(self, weight=1.0):
        """
        We calculate the distance from the desired point.
        The closser the better
        :param weight:
        :return:
        """
        distance = self.get_distance_from_point(self.desired_world_point)
        reward = weight * -1.0 * distance
        return reward

    def calculate_total_reward(self):
        """
        We calculate the total reward using the different weights
        :return:
        """
        r1 = self.calculate_reward_joint_position(self._weight_r1)
        r2 = self.calculate_reward_joint_effort(self._weight_r2)
        r3 = self.calculate_reward_contact_force(self._weight_r3)
        r4 = self.calculate_reward_orientation(self._weight_r4)
        r5 = self.calculate_reward_distance_from_des_point(self._weight_r5)

        # We add all the rewards
        total_reward = r1 + r2 + r3 + r4 + r5

        rospy.logdebug("###############")
        rospy.logdebug("r1 joint_position=" + str(r1))
        rospy.logdebug("r2 joint_effort=" + str(r2))
        rospy.logdebug("r3 contact_force=" + str(r3))
        rospy.logdebug("r4 orientation=" + str(r4))
        rospy.logdebug("r5 distance=" + str(r5))
        rospy.logdebug("total_reward=" + str(total_reward))
        rospy.logdebug("###############")

        return total_reward

    def get_observations(self):
        """
        Returns the state based on the observation selected
        :return:
        """
        observation = []
        
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_desired_point":
                observation.append(self.get_distance_from_point(self.desired_world_point))
            elif obs_name == "base_roll":
                observation.append(self.get_base_rpy().x)
            elif obs_name == "base_pitch":
                observation.append(self.get_base_rpy().y)
            elif obs_name == "base_yaw":
                observation.append(self.get_base_rpy().z)
            elif obs_name == "contact_force":
                observation.append(self.get_contact_force_magnitude())
            elif obs_name == "joint_states_haa":
                observation.append(self.get_joint_states().position[0])
            elif obs_name == "joint_states_hfe":
                observation.append(self.get_joint_states().position[1])
            elif obs_name == "joint_states_kfe":
                observation.append(self.get_joint_states().position[2])
            else:
                rospy.logerr("Observation Asked does not exist==>"+str(obs_name))

        return observation

    def get_state_as_string(self, observation):
        """
        This function will do two things:
        1) It will make discrete the observations
        2) Will convert the discrete observations in to state tags to be used as state for the Q-table
        :param observation:
        :return: state
        """
        observations_discrete = self.assign_bins(observation)
        state_discrete = ''.join(str(int(x)) for x in observations_discrete)
        return state_discrete

    def assign_bins(self, observation):
        """
        Assigns bins to the observations to make them discrete
        :param observation:
        :return: The bin value indexes
        """
        
        observation_bins = []
        
        for i in range(len(observation)):
            obs_name = self._list_of_observations[i]
            
            if obs_name == "distance_from_desired_point":
                # We consider the distance is always positive
                state = observation[i]
                min_value = 0.0
                max_value = self.max_distance
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
            
            elif obs_name == "base_roll":
                state = observation[i]
                min_value = -self._abs_max_roll
                max_value = self._abs_max_roll
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
            
            elif obs_name == "base_pitch":
                state = observation[i]
                min_value = -self._abs_max_pitch
                max_value = self._abs_max_pitch
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
            
            elif obs_name == "base_yaw":
                state = observation[i]
                min_value = -math.pi
                max_value = math.pi
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
            
            elif obs_name == "contact_force":
                state = observation[i]
                min_value = 0.0
                max_value = self._max_contact_force
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
            
            # For all joints
            else:
                state = observation[i]
                min_value = -math.pi
                max_value = math.pi
                num_bins = self._discrete_division
                observation_bins.append(int(self.get_bin_index(state, min_value, max_value, num_bins)))
        
        return observation_bins

    def get_bin_index(self, state, min_value, max_value, num_bins):
        """
        Given a state value, returns which bin it belongs to
        :param state: The state value
        :param min_value: The minimum value
        :param max_value: The maximum value
        :param num_bins: The number of bins
        :return: The bin index
        """
        if state <= min_value:
            return 0
        elif state >= max_value:
            return num_bins - 1
        else:
            return int(((state - min_value) / (max_value - min_value)) * num_bins)

    def init_joints_pose(self, init_jointstate_array):
        """
        Initialize the joints to a certain position
        :param init_jointstate_array:
        :return:
        """
        self.move_joints_to_jointstate_array(init_jointstate_array)
        return init_jointstate_array

    def move_joints_to_jointstate_array(self, jointstate_array):
        """
        It moves all the joints to the given position
        :param jointstate_array:
        :return:
        """
        
        # We get current Joints values
        current_jstates = self.get_joint_states().position
        
        # We calculate the increment or decrement needed to go to the desired position
        increment = []
        for i in range(len(current_jstates)):
            increment.append(jointstate_array[i] - current_jstates[i])
        
        # We send the command to make the increment
        self.set_joint_states(increment)

    def process_data(self):
        """
        We return the total reward based on the state in which we are in and if its done or not
        :return: reward, done
        """
        
        # We retrieve the reward
        reward = self.calculate_total_reward()
        
        # We check if its done
        done = self.is_done()
        
        # If its done because it went outside, we remove the reward
        if done:
            reward = self._done_reward
        else:
            # It means it passed all the checks and its still alive
            reward += self._alive_reward

        return reward, done

    def is_done(self):
        """
        Checks if the episode is done based on the observations
        """
        episode_done = False
        
        for done_criteria in self._episode_done_criteria:
            if done_criteria == "height_out_bounds":
                if not self.gurdy_height_ok():
                    rospy.logdebug("height_out_bounds")
                    episode_done = True
            elif done_criteria == "orientation_out_bounds":
                if not self.gurdy_orientation_ok():
                    rospy.logdebug("orientation_out_bounds")
                    episode_done = True

        return episode_done

    def get_action_to_position(self, action):
        """
        Here we have the actions that control the wheel joints
        :param action: Integer representing action taken
        :return: next_action_position, False if there's no need to move wheels
        """
        
        # We get current Joint values
        current_joint_states = self.get_joint_states()
        current_joint_positions = current_joint_states.position
        
        # 6 upper wheels, 6 lower wheels
        wheel_positions = list(current_joint_positions)
        
        # Increase the movement magnitude to make it more visible
        movement_magnitude = self.wheel_increment * 5.0  # Increase by factor of 5
        
        if action == 0:  # Increment all wheel positions (Move Forward)
            wheel_positions[0] += movement_magnitude  # Upper wheel 1
            wheel_positions[1] += movement_magnitude  # Upper wheel 2
            wheel_positions[2] += movement_magnitude  # Upper wheel 3
            wheel_positions[3] += movement_magnitude  # Upper wheel 4
            wheel_positions[4] += movement_magnitude  # Upper wheel 5
            wheel_positions[5] += movement_magnitude  # Upper wheel 6
            
        elif action == 1:  # Decrement all wheel positions (Move Backward)
            wheel_positions[0] -= movement_magnitude  # Upper wheel 1
            wheel_positions[1] -= movement_magnitude  # Upper wheel 2
            wheel_positions[2] -= movement_magnitude  # Upper wheel 3
            wheel_positions[3] -= movement_magnitude  # Upper wheel 4
            wheel_positions[4] -= movement_magnitude  # Upper wheel 5
            wheel_positions[5] -= movement_magnitude  # Upper wheel 6
            
        elif action == 2:  # Turn right (increment left wheels, decrement right wheels)
            wheel_positions[0] += movement_magnitude  # Upper left wheel 1
            wheel_positions[2] += movement_magnitude  # Upper left wheel 3
            wheel_positions[3] += movement_magnitude  # Upper left wheel 4
            wheel_positions[1] -= movement_magnitude  # Upper right wheel 2
            wheel_positions[4] -= movement_magnitude  # Upper right wheel 5
            wheel_positions[5] -= movement_magnitude  # Upper right wheel 6
            
        elif action == 3:  # Turn left (decrement left wheels, increment right wheels)
            wheel_positions[0] -= movement_magnitude  # Upper left wheel 1
            wheel_positions[2] -= movement_magnitude  # Upper left wheel 3
            wheel_positions[3] -= movement_magnitude  # Upper left wheel 4
            wheel_positions[1] += movement_magnitude  # Upper right wheel 2
            wheel_positions[4] += movement_magnitude  # Upper right wheel 5
            wheel_positions[5] += movement_magnitude  # Upper right wheel 6
            
        elif action == 4:  # Don't move
            return current_joint_positions, False
        
        # We clamp the values to the limits
        for i in range(len(wheel_positions)):
            joint_name = f"wheel{i+1}"
            if joint_name in self._joint_limits:
                limits = self._joint_limits[joint_name]
                wheel_positions[i] = max(min(wheel_positions[i], limits["max"]), limits["min"])
        
        rospy.logwarn(f"Moving wheels to positions: {[round(pos, 2) for pos in wheel_positions[:6]]}")
        return wheel_positions, True
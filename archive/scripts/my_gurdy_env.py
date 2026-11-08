#! /usr/bin/env python

import numpy
import rospy
from gym import spaces
import gurdy_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

max_episode_steps = 10000 # Can be any Value

register(
        id='MyGurdyWalkEnv-v0',
        entry_point='my_gurdy_env:MyGurdyWalkEnv',
        max_episode_steps=max_episode_steps,
    )

class MyGurdyWalkEnv(gurdy_env.GurdyEnv):
    def __init__(self):
        
        # Variables needed to be set here
        number_actions = rospy.get_param('/gurdy/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        """
        We set the Observation space for the following observations:
        gurdy_observations = [
            linear_speed_x,
            linear_speed_y,
            angular_speed_z,
            roll,
            pitch,
            yaw,
            joint1_position, joint1_velocity,
            joint2_position, joint2_velocity,
            ...
            joint12_position, joint12_velocity,
        ]
        """
        
        # Actions and Observations
        self.linear_speed_x_limit = rospy.get_param('/gurdy/linear_speed_x_limit')
        self.linear_speed_y_limit = rospy.get_param('/gurdy/linear_speed_y_limit')
        self.angular_speed_z_limit = rospy.get_param('/gurdy/angular_speed_z_limit')
        self.max_roll_angle = rospy.get_param('/gurdy/max_roll_angle')
        self.max_pitch_angle = rospy.get_param('/gurdy/max_pitch_angle')
        self.max_yaw_angle = rospy.get_param('/gurdy/max_yaw_angle')
        self.joint_limits = rospy.get_param('/gurdy/joint_limits')
        self.joint_velocity_limits = rospy.get_param('/gurdy/joint_velocity_limits')
        
        # Building observation space
        obs_high = []
        # Linear and angular speeds
        obs_high.extend([self.linear_speed_x_limit, self.linear_speed_y_limit, self.angular_speed_z_limit])
        # Orientation angles
        obs_high.extend([self.max_roll_angle, self.max_pitch_angle, self.max_yaw_angle])
        # Joint positions and velocities
        for i in range(12):  # 12 joints in total
            obs_high.extend([self.joint_limits, self.joint_velocity_limits])
        
        self.observation_space = spaces.Box(-numpy.array(obs_high), numpy.array(obs_high))
        
        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Variables that we retrieve through the param server
        self.init_joint_positions = rospy.get_param("/gurdy/init_joint_positions")
        
        # Get Observations
        self.start_point = Point()
        self.start_point.x = rospy.get_param("/gurdy/init_robot_pose/x")
        self.start_point.y = rospy.get_param("/gurdy/init_robot_pose/y")
        self.start_point.z = rospy.get_param("/gurdy/init_robot_pose/z")
        
        # Rewards
        self.forward_reward_weight = rospy.get_param("/gurdy/forward_reward_weight")
        self.orientation_reward_weight = rospy.get_param("/gurdy/orientation_reward_weight")
        self.joints_effort_reward_weight = rospy.get_param("/gurdy/joints_effort_reward_weight")
        self.end_episode_points = rospy.get_param("/gurdy/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the GurdyEnv
        super(MyGurdyWalkEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        for i, joint_name in enumerate(self.controllers_list[1:]):  # Skip joint_state_controller
            position = self.init_joint_positions[i]
            self.move_joint_position(joint_name, position)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.total_distance_moved = 0.0
        self.current_distance = self.get_distance_from_start_point(self.start_point)
        
        # For Info Purposes
        self.cumulated_reward = 0.0

    def _set_action(self, action):
        """
        Converts action index to movement of the 12 joints
        Action space is discrete with the following actions:
        0: Home position - all joints at initial positions
        1-6: Move upper leg joints with increasing position
        7-12: Move upper leg joints with decreasing position
        13-18: Move lower leg joints with increasing position
        19-24: Move lower leg joints with decreasing position
        """
        joint_names = [name for name in self.controllers_list if name != "joint_state_controller"]
        joint_positions = list(self.get_joint_positions().values())
        
        if action == 0:  # Home position
            for i, joint_name in enumerate(joint_names):
                self.move_joint_position(joint_name, self.init_joint_positions[i])
        
        elif 1 <= action <= 6:  # Increase upper leg joint position
            joint_idx = action - 1  # 0-based index for joint (0-5 for upper leg joints)
            joint_name = joint_names[joint_idx]
            current_pos = joint_positions[joint_idx]
            new_pos = min(current_pos + 0.1, 0.0)  # Limit to max position (0.0)
            self.move_joint_position(joint_name, new_pos)
        
        elif 7 <= action <= 12:  # Decrease upper leg joint position
            joint_idx = action - 7  # 0-based index for joint (0-5 for upper leg joints)
            joint_name = joint_names[joint_idx]
            current_pos = joint_positions[joint_idx]
            new_pos = max(current_pos - 0.1, -1.55)  # Limit to min position (-1.55)
            self.move_joint_position(joint_name, new_pos)
        
        elif 13 <= action <= 18:  # Increase lower leg joint position
            joint_idx = (action - 13) + 6  # 0-based index for joint (6-11 for lower leg joints)
            joint_name = joint_names[joint_idx]
            current_pos = joint_positions[joint_idx]
            new_pos = min(current_pos + 0.1, 1.5708)  # Limit to max position (1.5708)
            self.move_joint_position(joint_name, new_pos)
        
        elif 19 <= action <= 24:  # Decrease lower leg joint position
            joint_idx = (action - 19) + 6  # 0-based index for joint (6-11 for lower leg joints)
            joint_name = joint_names[joint_idx]
            current_pos = joint_positions[joint_idx]
            new_pos = max(current_pos - 0.1, -2.9)  # Limit to min position (-2.9)
            self.move_joint_position(joint_name, new_pos)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robot's observations
        :return: observations
        """
        # Get robot orientation in RPY
        roll, pitch, yaw = self.get_orientation_euler()

        # Get robot linear and angular velocities
        linear_speed_x = self.odom.twist.twist.linear.x
        linear_speed_y = self.odom.twist.twist.linear.y
        angular_speed_z = self.odom.twist.twist.angular.z

        # Get joint positions and velocities
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.joints.name)):
            # Skip base_joint if it exists
            if "base" not in self.joints.name[i]:
                joint_positions.append(self.joints.position[i])
                joint_velocities.append(self.joints.velocity[i])

        # Combine all observations into a single list
        gurdy_observations = [
            linear_speed_x,
            linear_speed_y,
            angular_speed_z,
            roll,
            pitch,
            yaw,
        ]

        # Add joint positions and velocities
        for i in range(len(joint_positions)):
            gurdy_observations.append(joint_positions[i])
            gurdy_observations.append(joint_velocities[i])
        
        return gurdy_observations

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        roll = observations[3]
        pitch = observations[4]
        
        # Check if robot has fallen (roll or pitch too large)
        if abs(roll) > self.max_roll_angle:
            rospy.logerr("WRONG Robot Roll Orientation==>" + str(roll))
            return True
        elif abs(pitch) > self.max_pitch_angle:
            rospy.logerr("WRONG Robot Pitch Orientation==>" + str(pitch))
            return True
        else:
            return False

    def _compute_reward(self, observations, done):
        """
        Calculate reward based on observations
        """
        if not done:
            # Reward for forward movement
            x_speed = observations[0]
            forward_reward = x_speed * self.forward_reward_weight
            
            # Penalty for orientation deviation from upright
            roll = observations[3]
            pitch = observations[4]
            orientation_penalty = -1 * (abs(roll) + abs(pitch)) * self.orientation_reward_weight
            
            # Penalty for excessive joint effort (based on joint velocities)
            joint_velocities = observations[7::2]  # Get all velocities from observations
            joint_effort_penalty = -1 * sum([abs(vel) for vel in joint_velocities]) * self.joints_effort_reward_weight
            
            # Combine rewards
            reward = forward_reward + orientation_penalty + joint_effort_penalty
        else:
            # Penalty for episode end (falling over)
            reward = -1 * self.end_episode_points

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward

    # Internal TaskEnv Methods
    def get_distance_from_start_point(self, start_point):
        """
        Calculates the distance from the given point and the current position
        given by odometry
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(start_point,
                                              self.odom.pose.pose.position)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a point, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
    
        distance = numpy.linalg.norm(a - b)
    
        return distance
    
    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

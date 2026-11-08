#!/usr/bin/env python

import rospy
import numpy
import math
from gym import spaces
import gurdy_single_leg_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

max_episode_steps = 10000 # Can be adjusted as needed

register(
    id='GurdyWalkEnv-v0',
    entry_point='gurdy_env:GurdyWalkEnv',
    max_episode_steps=max_episode_steps,
)

class GurdyWalkEnv(gurdy_single_leg_env.GurdySingleLegEnv):
    def __init__(self):
        # Get action and observation parameters from ROS parameter server
        number_actions = rospy.get_param('/gurdy/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # Define observation space parameters
        self.max_body_roll = rospy.get_param('/gurdy/max_body_roll')
        self.max_body_pitch = rospy.get_param('/gurdy/max_body_pitch')
        self.max_linear_speed = rospy.get_param('/gurdy/max_linear_speed')
        self.max_angular_speed = rospy.get_param('/gurdy/max_angular_speed')
        
        # Create observation space
        high = numpy.array([
            self.max_body_roll,
            self.max_body_pitch,
            self.max_linear_speed,
            self.max_angular_speed,
            self.max_body_roll,  # Additional body orientation
            self.max_body_pitch  # Additional body orientation
        ])
        
        self.observation_space = spaces.Box(-high, high)
        
        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Variables from parameter server
        self.start_point = Point()
        self.start_point.x = rospy.get_param("/gurdy/init_pose/x")
        self.start_point.y = rospy.get_param("/gurdy/init_pose/y")
        self.start_point.z = rospy.get_param("/gurdy/init_pose/z")
        
        # Reward parameters
        self.move_distance_reward_weight = rospy.get_param("/gurdy/move_distance_reward_weight")
        self.linear_speed_reward_weight = rospy.get_param("/gurdy/linear_speed_reward_weight")
        self.body_orientation_reward_weight = rospy.get_param("/gurdy/body_orientation_reward_weight")
        self.end_episode_points = rospy.get_param("/gurdy/end_episode_points")

        self.cumulated_steps = 0.0

        # Initialize parent class
        super(GurdyWalkEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its initial pose"""
        # You might need to implement leg positioning here
        return True

    def _init_env_variables(self):
        """
        Initializes variables needed at the start of each episode
        """
        self.total_distance_moved = 0.0
        self.current_distance = self.get_distance_from_start_point(self.start_point)
        self.cumulated_reward = 0.0

    def _set_action(self, action):
        """
        Convert action to joint movements
        Actions could be:
        0-5: Move specific leg joint
        6: Stabilize body
        7: Reset pose
        """
        if action < 6:
            # Move specific leg joint
            leg_index = action
            # You'll need to implement specific joint movement logic
            # This is a placeholder - actual implementation depends on your robot's specifics
            joint_angle = self.roll_speed_increment_value  # Adjust as needed
            self.move_leg_joint(leg_index, joint_angle)
        elif action == 6:
            # Stabilize body
            self.stabilize_body()
        elif action == 7:
            # Reset pose
            self.reset_pose()

    def _get_obs(self):
        """
        Define robot observations
        Includes:
        1. Body Roll
        2. Body Pitch
        3. Linear Speed
        4. Angular Speed
        5-6. Additional body orientation metrics
        """
        # Get body orientation
        roll, pitch, yaw = self.get_orientation_euler()
        
        # Get linear and angular speeds
        linear_speed = self.get_linear_speed()
        angular_speed = self.get_angular_speed()
        
        observations = [
            round(roll, 1),
            round(pitch, 1),
            round(linear_speed, 1),
            round(angular_speed, 1),
            round(roll, 1),  # Additional orientation metric
            round(pitch, 1)  # Additional orientation metric
        ]
        
        rospy.logdebug("Observations==>"+str(observations))
        return observations

    def _is_done(self, observations):
        """
        Determine if episode is done based on body orientation
        """
        roll_angle = observations[0]
        pitch_angle = observations[1]

        if abs(roll_angle) > self.max_body_roll:
            rospy.logerr("WRONG Gurdy Roll Orientation==>" + str(roll_angle))
            done = True
        elif abs(pitch_angle) > self.max_body_pitch:
            rospy.logerr("WRONG Gurdy Pitch Orientation==>" + str(pitch_angle))
            done = True
        else:
            done = False

        return done

    def _compute_reward(self, observations, done):
        """
        Compute reward based on various factors
        """
        if not done:
            # Distance-based reward
            current_distance = observations[1]  # Assuming this is distance
            delta_distance = current_distance - self.current_distance
            reward_distance = delta_distance * self.move_distance_reward_weight
            self.current_distance = current_distance

            # Linear speed reward
            linear_speed = observations[2]
            reward_linear_speed = linear_speed * self.linear_speed_reward_weight

            # Body orientation reward (minimize roll and pitch)
            roll_angle = observations[0]
            pitch_angle = observations[1]
            reward_body_orientation = -1 * (abs(roll_angle) + abs(pitch_angle)) * self.body_orientation_reward_weight

            # Total reward
            reward = round(reward_distance, 0) + \
                     round(reward_linear_speed, 0) + \
                     round(reward_body_orientation, 0)
        else:
            reward = -1 * self.end_episode_points

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward

    # Helper methods (to be implemented based on your robot's specifics)
    def get_orientation_euler(self):
        # Convert quaternion to euler angles
        orientation_list = [
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w
        ]
        return euler_from_quaternion(orientation_list)

    def get_linear_speed(self):
        # Get linear speed from odometry
        return self.odom.twist.twist.linear.x

    def get_angular_speed(self):
        # Get angular speed from odometry
        return self.odom.twist.twist.angular.z

    def get_distance_from_start_point(self, start_point):
        # Calculate distance from start point
        current_pos = self.odom.pose.pose.position
        a = numpy.array((start_point.x, start_point.y, start_point.z))
        b = numpy.array((current_pos.x, current_pos.y, current_pos.z))
        return numpy.linalg.norm(a - b)

    def move_leg_joint(self, leg_index, angle):
        # Placeholder for moving specific leg joint
        # Actual implementation depends on your robot's control mechanism
        rospy.logwarn(f"Moving leg {leg_index} joint by {angle} degrees")

    def stabilize_body(self):
        # Placeholder for body stabilization
        rospy.logwarn("Attempting to stabilize body")

    def reset_pose(self):
        # Placeholder for resetting robot pose
        rospy.logwarn("Resetting robot pose")
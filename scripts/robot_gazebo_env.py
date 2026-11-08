#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection
from std_msgs.msg import Bool
import os

class RobotGazeboEnv:
    """
    Parent class for robot environments using Gazebo
    """
    def __init__(self, controllers_list, robot_name_space, reset_controls=False, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):
        """
        Initialize the robot environment
        
        Args:
            controllers_list (list): List of controllers to manage
            robot_name_space (str): Namespace of the robot
            reset_controls (bool): Whether to reset controls on init
            start_init_physics_parameters (bool): Whether to initialize physics parameters
            reset_world_or_sim (str): "SIMULATION", "WORLD", or "NO_RESET_SIM"
        """
        # Parameters
        self.controllers_list = controllers_list
        self.robot_name_space = robot_name_space
        self.reset_controls = reset_controls
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        
        # Initialize Gazebo connection
        self.gazebo = GazeboConnection(
            start_init_physics_parameters=self.start_init_physics_parameters,
            reset_world_or_sim=self.reset_world_or_sim
        )
        
        # Initialize controllers connection
        self.controllers_object = ControllersConnection(
            namespace=self.robot_name_space,
            controllers_list=self.controllers_list
        )
        
        # Create reward publisher
        self.reward_pub = rospy.Publisher('/gurdy/reward', Float64, queue_size=1)
        
        # Initialize environment
        self._check_all_systems_ready()
        
        rospy.loginfo("Finished RobotGazeboEnvInit")
    
    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()
    
    def _get_obs(self):
        """
        Get observations from the environment
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """
        Initialize environment variables
        """
        raise NotImplementedError()
    
    def _set_action(self, action):
        """
        Set action in the environment
        """
        raise NotImplementedError()
    
    def _is_done(self, observations):
        """
        Check if episode is done
        """
        raise NotImplementedError()
    
    def _compute_reward(self, observations, done):
        """
        Compute the reward of a given state
        """
        raise NotImplementedError()
    
    def _env_setup(self, initial_qpos):
        """
        Setup for the initial positions of each joint
        """
        raise NotImplementedError()
    
    def _set_init_pose(self):
        """
        Sets the robot in its init pose
        """
        raise NotImplementedError()
    
    def _publish_reward_topic(self, reward, episode_number=1):
        """
        Publish reward on a ROS topic for monitoring
        """
        reward_msg = Float64()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)
    
    def reset(self):
        """
        Reset the environment
        """
        rospy.logdebug("Resetting RobotGazeboEnvironment")
        self.gazebo.pauseSim()
        
        # Reset controllers if specified
        if self.reset_controls:
            self.controllers_object.reset_controllers()
        
        # Reset the simulation
        self.gazebo.resetSim()
        
        # Set initial robot pose
        self._set_init_pose()
        
        # Unpause simulation
        self.gazebo.unpauseSim()
        
        # Call the gazebo reset
        self.gazebo.resetSim()
        self.gazebo.pauseSim()
        self.gazebo.unpauseSim()
        
        # Initialize environment variables
        self._init_env_variables()
        
        # Check all systems ready
        self._check_all_systems_ready()
        
        # Get observation
        observation = self._get_obs()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            observation: Environment observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Pause simulation
        self.gazebo.pauseSim()
        
        # Execute action
        self._set_action(action)
        
        # Resume simulation
        self.gazebo.unpauseSim()
        
        # Get observation
        observation = self._get_obs()
        
        # Check if done
        done = self._is_done(observation)
        
        # Compute reward
        reward = self._compute_reward(observation, done)
        
        # Publish reward for monitoring
        self._publish_reward_topic(reward)
        
        return observation, reward, done, {}
    
    def seed(self, seed=None):
        """
        Function executed when closing the environment
        
        Args:
            seed (int): Random seed
            
        Returns:
            list: List containing the seed
        """
        return [seed]
    
    def close(self):
        """
        Function executed when closing the environment
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        # Pause simulation
        self.gazebo.pauseSim()
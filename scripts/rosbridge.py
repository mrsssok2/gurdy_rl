#!/usr/bin/env python
"""
ROS Bridge for Reinforcement Learning Environments

This module provides a bridge between ROS Gym environments and RL algorithms, 
handling namespacing and environment management.
"""
import rospy
import gym
import numpy as np
import math
from gym import spaces
from std_msgs.msg import String

class GurdyEnvBridge:
    """Bridge between Gurdy ROS environment and RL algorithms"""
    def __init__(self, env_name, namespace=None):
        """
        Initialize the environment bridge.
        
        Args:
            env_name (str): Name of the Gym environment
            namespace (str, optional): ROS namespace for the environment
        """
        self.env_name = env_name
        self.namespace = namespace
        
        # Create the environment
        self.env = gym.make(env_name)
        
        # Store action and observation space for convenience
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Initialize ROS communication
        self._setup_ros_communication()
    
    def _setup_ros_communication(self):
        """Setup ROS communication for the environment"""
        if self.namespace:
            # Create a publisher for environment status
            status_topic = f"/{self.namespace}/env_status"
            self.status_pub = rospy.Publisher(status_topic, String, queue_size=10)
            
            rospy.loginfo(f"Created environment bridge for {self.env_name} in namespace {self.namespace}")
        else:
            self.status_pub = None
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        observation = self.env.reset()
        
        # Publish status if available
        if self.status_pub:
            self.status_pub.publish(String("Environment reset"))
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        observation, reward, done, info = self.env.step(action)
        
        # Publish status if available
        if self.status_pub and done:
            status_msg = f"Episode completed with reward: {reward:.2f}"
            self.status_pub.publish(String(status_msg))
        
        return observation, reward, done, info
    
    def close(self):
        """Close the environment"""
        self.env.close()

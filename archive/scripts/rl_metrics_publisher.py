#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64, Bool

class RLMetricsPublisher:
    """
    Helper class to publish metrics from RL algorithms for visualization
    """
    def __init__(self, algorithm_name):
        """
        Initialize publishers for algorithm metrics
        
        Args:
            algorithm_name (str): Name of the RL algorithm (e.g., 'dqn', 'sarsa')
        """
        self.prefix = f'/gurdy/{algorithm_name.lower()}'
        
        # Create publishers
        self.reward_pub = rospy.Publisher(f'{self.prefix}/episode_reward', Float64, queue_size=10)
        self.avg_reward_pub = rospy.Publisher(f'{self.prefix}/avg_reward', Float64, queue_size=10)
        self.head_height_pub = rospy.Publisher(f'{self.prefix}/head_height', Float64, queue_size=10)
        self.fall_pub = rospy.Publisher(f'{self.prefix}/fall', Bool, queue_size=10)
        self.episode_pub = rospy.Publisher(f'{self.prefix}/current_episode', Float64, queue_size=10)
        
    def publish_episode_reward(self, reward):
        """
        Publish the current episode's reward
        
        Args:
            reward (float): Reward value
        """
        msg = Float64()
        msg.data = reward
        self.reward_pub.publish(msg)
        
    def publish_avg_reward(self, avg_reward):
        """
        Publish the average reward
        
        Args:
            avg_reward (float): Average reward value
        """
        msg = Float64()
        msg.data = avg_reward
        self.avg_reward_pub.publish(msg)
        
    def publish_head_height(self, height):
        """
        Publish the head height (stability measure)
        
        Args:
            height (float): Head height value in meters
        """
        msg = Float64()
        msg.data = height
        self.head_height_pub.publish(msg)
        
    def publish_fall(self, has_fallen):
        """
        Publish whether the robot has fallen
        
        Args:
            has_fallen (bool): True if fallen, False otherwise
        """
        msg = Bool()
        msg.data = has_fallen
        self.fall_pub.publish(msg)
        
    def publish_episode(self, episode):
        """
        Publish the current episode number
        
        Args:
            episode (int): Current episode number
        """
        msg = Float64()
        msg.data = float(episode)
        self.episode_pub.publish(msg)
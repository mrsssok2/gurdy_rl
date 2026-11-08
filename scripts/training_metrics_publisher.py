#!/usr/bin/env python
import rospy
import numpy as np
import os
from std_msgs.msg import Float32, String, Float32MultiArray, MultiArrayDimension
from collections import defaultdict
import json

class TrainingMetricsPublisher:
    """
    Node that subscribes to metrics from different RL algorithms and republishes them
    in a standardized format for visualization.
    """
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('training_metrics_publisher', anonymous=True)
        
        # Create publishers for different metrics
        self.reward_pub = rospy.Publisher('/training_metrics/rewards', Float32MultiArray, queue_size=10)
        self.avg_reward_pub = rospy.Publisher('/training_metrics/avg_rewards', Float32MultiArray, queue_size=10)
        self.loss_pub = rospy.Publisher('/training_metrics/losses', Float32MultiArray, queue_size=10)
        self.secondary_loss_pub = rospy.Publisher('/training_metrics/secondary_losses', Float32MultiArray, queue_size=10)
        self.height_pub = rospy.Publisher('/training_metrics/heights', Float32MultiArray, queue_size=10)
        self.status_pub = rospy.Publisher('/training_metrics/status', String, queue_size=10)
        
        # Store metrics for each algorithm
        self.metrics = {
            'algorithms': [],
            'rewards': defaultdict(list),
            'avg_rewards': defaultdict(list),
            'losses': defaultdict(list),
            'secondary_losses': defaultdict(list),
            'heights': defaultdict(list),
            'falls': defaultdict(list)
        }
        
        # Subscribe to metrics from each algorithm
        self.create_subscribers()
        
        # Set up timer to publish aggregated metrics periodically
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_metrics)
        
        rospy.loginfo("Training metrics publisher initialized")
    
    def create_subscribers(self):
        """Create subscribers for all known algorithms"""
        algorithms = ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac']
        
        for algo in algorithms:
            # Track which algorithms are actually running
            if self.check_algorithm_running(algo):
                self.metrics['algorithms'].append(algo)
                
                # Create subscribers for each metric type
                rospy.Subscriber(f'/{algo}/episode_reward', Float32, 
                                self.reward_callback, callback_args=algo)
                rospy.Subscriber(f'/{algo}/avg_reward', Float32, 
                                self.avg_reward_callback, callback_args=algo)
                rospy.Subscriber(f'/{algo}/loss', Float32, 
                                self.loss_callback, callback_args=algo)
                rospy.Subscriber(f'/{algo}/secondary_loss', Float32, 
                                self.secondary_loss_callback, callback_args=algo)
                rospy.Subscriber(f'/{algo}/head_height', Float32, 
                                self.height_callback, callback_args=algo)
                rospy.Subscriber(f'/{algo}/fallen', Float32, 
                                self.fall_callback, callback_args=algo)
        
        rospy.loginfo(f"Monitoring algorithms: {self.metrics['algorithms']}")
    
    def check_algorithm_running(self, algo):
        """Check if an algorithm's node is running by looking for its parameters"""
        try:
            # Get parameter list and check if any are related to this algorithm
            params = rospy.get_param_names()
            return any(param.startswith(f"/{algo}/") for param in params)
        except:
            return False
    
    def reward_callback(self, msg, algorithm):
        """Callback for episode reward messages"""
        self.metrics['rewards'][algorithm].append(msg.data)
    
    def avg_reward_callback(self, msg, algorithm):
        """Callback for average reward messages"""
        self.metrics['avg_rewards'][algorithm].append(msg.data)
    
    def loss_callback(self, msg, algorithm):
        """Callback for loss messages"""
        self.metrics['losses'][algorithm].append(msg.data)
    
    def secondary_loss_callback(self, msg, algorithm):
        """Callback for secondary loss messages (like critic loss)"""
        self.metrics['secondary_losses'][algorithm].append(msg.data)
    
    def height_callback(self, msg, algorithm):
        """Callback for head height messages"""
        self.metrics['heights'][algorithm].append(msg.data)
    
    def fall_callback(self, msg, algorithm):
        """Callback for fall messages"""
        if msg.data > 0.5:  # Consider it a fall if the message is > 0.5
            self.metrics['falls'][algorithm].append(len(self.metrics['rewards'][algorithm]) - 1)
    
    def create_multi_array_msg(self, data_dict):
        """
        Create a Float32MultiArray message from a dictionary of algorithm data
        
        Args:
            data_dict (dict): Dictionary mapping algorithm names to lists of values
            
        Returns:
            Float32MultiArray: ROS message containing the data
        """
        msg = Float32MultiArray()
        
        # Set dimensions
        num_algorithms = len(self.metrics['algorithms'])
        if num_algorithms == 0:
            return msg
            
        # Find the maximum length of data among all algorithms
        max_len = max([len(data_dict.get(algo, [])) for algo in self.metrics['algorithms']])
        if max_len == 0:
            return msg
            
        # Create dimensions for the array
        # First dimension is algorithm index, second is the data points
        msg.layout.dim = [
            MultiArrayDimension("algorithms", num_algorithms, num_algorithms * max_len),
            MultiArrayDimension("data_points", max_len, max_len)
        ]
        msg.layout.data_offset = 0
        
        # Flatten the data into a single array
        data = []
        for algo in self.metrics['algorithms']:
            algo_data = data_dict.get(algo, [])
            # Pad with NaN for missing data
            padded_data = algo_data + [float('nan')] * (max_len - len(algo_data))
            data.extend(padded_data)
        
        msg.data = data
        return msg
    
    def publish_metrics(self, event=None):
        """Publish all metrics periodically"""
        # Create and publish rewards message
        self.reward_pub.publish(self.create_multi_array_msg(self.metrics['rewards']))
        
        # Create and publish average rewards message
        self.avg_reward_pub.publish(self.create_multi_array_msg(self.metrics['avg_rewards']))
        
        # Create and publish losses message
        self.loss_pub.publish(self.create_multi_array_msg(self.metrics['losses']))
        
        # Create and publish secondary losses message
        self.secondary_loss_pub.publish(self.create_multi_array_msg(self.metrics['secondary_losses']))
        
        # Create and publish heights message
        self.height_pub.publish(self.create_multi_array_msg(self.metrics['heights']))
        
        # Create and publish status message (contains algorithm names and falls)
        status = {
            'algorithms': self.metrics['algorithms'],
            'falls': self.metrics['falls']
        }
        self.status_pub.publish(json.dumps(status))

if __name__ == '__main__':
    try:
        publisher = TrainingMetricsPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

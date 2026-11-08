#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from collections import defaultdict
import os
import json
import rospy
from std_msgs.msg import Float32, String, Float32MultiArray, MultiArrayDimension

class ComparisonVisualizer:
    """
    Node that visualizes performance metrics from different RL algorithms
    in real-time for comparison.
    """
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('comparison_visualizer', anonymous=True)
        
        # Set up plot refresh rate (in seconds)
        self.refresh_rate = 1.0
        
        # Set up matplotlib to use TkAgg backend for interactive plotting
        plt.switch_backend('TkAgg')
        
        # Data storage
        self.algorithms = []
        self.rewards = defaultdict(list)
        self.avg_rewards = defaultdict(list)
        self.losses = defaultdict(list)
        self.secondary_losses = defaultdict(list)
        self.heights = defaultdict(list)
        self.falls = defaultdict(list)
        
        # Get output directory from parameter
        self.outdir = rospy.get_param("/gurdy/outdir", "/home/user/catkin_ws/src/my_gurdy_description/output")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # Create comparison directory
        self.comparison_dir = os.path.join(self.outdir, "comparison")
        if not os.path.exists(self.comparison_dir):
            os.makedirs(self.comparison_dir)
        
        # Colors for different algorithms
        self.colors = {
            'qlearn': 'blue',
            'sarsa': 'green',
            'dqn': 'red',
            'policy_gradient': 'purple',
            'ppo': 'orange',
            'sac': 'cyan'
        }
        
        # Set up the plot
        self.setup_plot()
        
        # Subscribe to metrics
        self.setup_subscribers()
        
        # Set up animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.refresh_rate*1000
        )
        
        # Display the plot
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        
        rospy.loginfo("Comparison visualizer initialized")
        
        # Set up shutdown hook
        rospy.on_shutdown(self.save_final_plots)
    
    def setup_plot(self):
        """Set up the plot layout and initial elements"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('RL Algorithm Comparison')
        self.fig.suptitle('Reinforcement Learning Algorithm Comparison', fontsize=16)
        
        # Adjust the layout
        gs = self.fig.add_gridspec(3, 2)
        
        # Create subplots
        self.ax_reward = self.fig.add_subplot(gs[0, 0])  # Episode rewards
        self.ax_avg_reward = self.fig.add_subplot(gs[0, 1])  # Average rewards
        self.ax_loss = self.fig.add_subplot(gs[1, 0])  # Primary loss
        self.ax_secondary_loss = self.fig.add_subplot(gs[1, 1])  # Secondary loss
        self.ax_height = self.fig.add_subplot(gs[2, 0])  # Head height
        self.ax_falls = self.fig.add_subplot(gs[2, 1])  # Fall count
        
        # Configure subplots
        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True)
        
        self.ax_avg_reward.set_title('Average Rewards')
        self.ax_avg_reward.set_xlabel('Episode')
        self.ax_avg_reward.set_ylabel('Avg Reward')
        self.ax_avg_reward.grid(True)
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True)
        
        self.ax_secondary_loss.set_title('Secondary Loss')
        self.ax_secondary_loss.set_xlabel('Episode')
        self.ax_secondary_loss.set_ylabel('Loss')
        self.ax_secondary_loss.grid(True)
        
        self.ax_height.set_title('Head Height (Stability)')
        self.ax_height.set_xlabel('Episode')
        self.ax_height.set_ylabel('Height (m)')
        self.ax_height.grid(True)
        
        self.ax_falls.set_title('Fall Count')
        self.ax_falls.set_xlabel('Algorithm')
        self.ax_falls.set_ylabel('Number of Falls')
        self.ax_falls.grid(True)
        
        # Ensure integer ticks for x-axis in falls plot
        self.ax_falls.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Dict to store line objects for each algorithm
        self.lines = {}
        self.fall_bars = None
        
        # Dict to store fall markers
        self.fall_markers = {}
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    def setup_subscribers(self):
        """Set up subscribers to the metrics topics"""
        rospy.Subscriber('/training_metrics/rewards', Float32MultiArray, 
                        self.rewards_callback)
        rospy.Subscriber('/training_metrics/avg_rewards', Float32MultiArray, 
                        self.avg_rewards_callback)
        rospy.Subscriber('/training_metrics/losses', Float32MultiArray, 
                        self.losses_callback)
        rospy.Subscriber('/training_metrics/secondary_losses', Float32MultiArray, 
                        self.secondary_losses_callback)
        rospy.Subscriber('/training_metrics/heights', Float32MultiArray, 
                        self.heights_callback)
        rospy.Subscriber('/training_metrics/status', String, 
                        self.status_callback)
    
    def parse_multiarray(self, msg, data_dict):
        """
        Parse a Float32MultiArray message into a dictionary
        
        Args:
            msg (Float32MultiArray): Message containing the data
            data_dict (dict): Dictionary to update with the parsed data
            
        Returns:
            dict: Updated dictionary
        """
        if not msg.layout.dim or len(msg.layout.dim) != 2:
            return data_dict
            
        num_algorithms = msg.layout.dim[0].size
        data_points = msg.layout.dim[1].size
        
        if num_algorithms == 0 or data_points == 0:
            return data_dict
        
        # Make sure we have algorithm names
        if not self.algorithms:
            rospy.logwarn("No algorithm names available yet")
            return data_dict
            
        # Check if we have the right number of algorithms
        if num_algorithms != len(self.algorithms):
            rospy.logwarn(f"Mismatch in algorithm count: {num_algorithms} vs {len(self.algorithms)}")
            return data_dict
        
        # Extract data for each algorithm
        for i, algo in enumerate(self.algorithms):
            start_idx = i * data_points
            end_idx = start_idx + data_points
            
            # Extract data for this algorithm
            algo_data = msg.data[start_idx:end_idx]
            
            # Filter out NaN values
            valid_data = [x for x in algo_data if not np.isnan(x)]
            
            # Update dictionary
            data_dict[algo] = valid_data
                
        return data_dict
    
    def rewards_callback(self, msg):
        """Process rewards message"""
        self.parse_multiarray(msg, self.rewards)
    
    def avg_rewards_callback(self, msg):
        """Process average rewards message"""
        self.parse_multiarray(msg, self.avg_rewards)
    
    def losses_callback(self, msg):
        """Process losses message"""
        self.parse_multiarray(msg, self.losses)
    
    def secondary_losses_callback(self, msg):
        """Process secondary losses message"""
        self.parse_multiarray(msg, self.secondary_losses)
    
    def heights_callback(self, msg):
        """Process heights message"""
        self.parse_multiarray(msg, self.heights)
    
    def status_callback(self, msg):
        """Process status message"""
        try:
            status = json.loads(msg.data)
            self.algorithms = status.get('algorithms', [])
            
            # Update falls data
            for algo, fall_episodes in status.get('falls', {}).items():
                self.falls[algo] = fall_episodes
        except json.JSONDecodeError:
            rospy.logerr("Failed to parse status message")
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Clear axes
        self.ax_reward.clear()
        self.ax_avg_reward.clear()
        self.ax_loss.clear()
        self.ax_secondary_loss.clear()
        self.ax_height.clear()
        self.ax_falls.clear()
        
        # Reset lines dict
        self.lines = {}
        self.fall_markers = {}
        
        # Configure axes again (after clearing)
        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True)
        
        self.ax_avg_reward.set_title('Average Rewards')
        self.ax_avg_reward.set_xlabel('Episode')
        self.ax_avg_reward.set_ylabel('Avg Reward')
        self.ax_avg_reward.grid(True)
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True)
        
        self.ax_secondary_loss.set_title('Secondary Loss')
        self.ax_secondary_loss.set_xlabel('Episode')
        self.ax_secondary_loss.set_ylabel('Loss')
        self.ax_secondary_loss.grid(True)
        
        self.ax_height.set_title('Head Height (Stability)')
        self.ax_height.set_xlabel('Episode')
        self.ax_height.set_ylabel('Height (m)')
        self.ax_height.grid(True)
        
        self.ax_falls.set_title('Fall Count')
        self.ax_falls.set_xlabel('Algorithm')
        self.ax_falls.set_ylabel('Number of Falls')
        self.ax_falls.grid(True)
        
        # Plot data for each algorithm
        for algo in self.algorithms:
            color = self.colors.get(algo, 'gray')
            
            # Plot reward data
            if algo in self.rewards and self.rewards[algo]:
                episodes = range(len(self.rewards[algo]))
                self.lines[f"{algo}_reward"] = self.ax_reward.plot(
                    episodes, self.rewards[algo], color=color, label=algo)[0]
            
            # Plot average reward data
            if algo in self.avg_rewards and self.avg_rewards[algo]:
                episodes = range(len(self.avg_rewards[algo]))
                self.lines[f"{algo}_avg_reward"] = self.ax_avg_reward.plot(
                    episodes, self.avg_rewards[algo], color=color, label=algo)[0]
            
            # Plot loss data
            if algo in self.losses and self.losses[algo]:
                episodes = range(len(self.losses[algo]))
                self.lines[f"{algo}_loss"] = self.ax_loss.plot(
                    episodes, self.losses[algo], color=color, label=algo)[0]
            
            # Plot secondary loss data
            if algo in self.secondary_losses and self.secondary_losses[algo]:
                episodes = range(len(self.secondary_losses[algo]))
                self.lines[f"{algo}_secondary_loss"] = self.ax_secondary_loss.plot(
                    episodes, self.secondary_losses[algo], color=color, label=algo)[0]
            
            # Plot height data
            if algo in self.heights and self.heights[algo]:
                episodes = range(len(self.heights[algo]))
                self.lines[f"{algo}_height"] = self.ax_height.plot(
                    episodes, self.heights[algo], color=color, label=algo)[0]
                
                # Plot fall markers
                if algo in self.falls and self.falls[algo]:
                    fall_x = [ep for ep in self.falls[algo] if ep < len(self.heights[algo])]
                    fall_y = [self.heights[algo][ep] for ep in fall_x]
                    self.fall_markers[algo] = self.ax_height.scatter(
                        fall_x, fall_y, color=color, marker='x', s=50)
        
        # Plot fall count as a bar chart
        if self.algorithms:
            fall_counts = [len(self.falls.get(algo, [])) for algo in self.algorithms]
            self.fall_bars = self.ax_falls.bar(
                range(len(self.algorithms)), fall_counts, color=[self.colors.get(algo, 'gray') for algo in self.algorithms])
            self.ax_falls.set_xticks(range(len(self.algorithms)))
            self.ax_falls.set_xticklabels(self.algorithms, rotation=45)
        
        # Add legends
        if self.algorithms:
            self.ax_reward.legend()
            self.ax_avg_reward.legend()
            self.ax_loss.legend()
            self.ax_secondary_loss.legend()
            self.ax_height.legend()
        
        # Save current plot
        try:
            self.fig.savefig(os.path.join(self.comparison_dir, "comparison_plot.png"))
        except Exception as e:
            rospy.logwarn(f"Failed to save plot: {e}")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    def save_final_plots(self):
        """Save final comparison plots when the node is shutting down"""
        try:
            # Save main comparison plot
            self.fig.savefig(os.path.join(self.comparison_dir, "final_comparison.png"))
            
            # Save individual algorithm plots
            for algo in self.algorithms:
                self.save_algorithm_plot(algo)
                
            rospy.loginfo("Saved final comparison plots")
        except Exception as e:
            rospy.logerr(f"Failed to save final plots: {e}")
    
    def save_algorithm_plot(self, algorithm):
        """Save detailed plot for a specific algorithm"""
        # Create a new figure for this algorithm
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Get the algorithm color
        color = self.colors.get(algorithm, 'blue')
        
        # Plot episode and average rewards
        ax1 = axes[0]
        if algorithm in self.rewards and self.rewards[algorithm]:
            episodes = range(len(self.rewards[algorithm]))
            ax1.plot(episodes, self.rewards[algorithm], color=color, label='Episode Reward')
            
        if algorithm in self.avg_rewards and self.avg_rewards[algorithm]:
            episodes = range(len(self.avg_rewards[algorithm]))
            ax1.plot(episodes, self.avg_rewards[algorithm], 'r-', label='Average Reward')
            
        ax1.set_title(f'{algorithm} - Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        ax1.legend()
        
        # Plot losses
        ax2 = axes[1]
        if algorithm in self.losses and self.losses[algorithm]:
            episodes = range(len(self.losses[algorithm]))
            ax2.plot(episodes, self.losses[algorithm], color=color, label='Primary Loss')
            
        if algorithm in self.secondary_losses and self.secondary_losses[algorithm]:
            episodes = range(len(self.secondary_losses[algorithm]))
            ax2.plot(episodes, self.secondary_losses[algorithm], 'm-', label='Secondary Loss')
            
        ax2.set_title(f'{algorithm} - Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
        
        # Plot stability and falls
        ax3 = axes[2]
        if algorithm in self.heights and self.heights[algorithm]:
            episodes = range(len(self.heights[algorithm]))
            ax3.plot(episodes, self.heights[algorithm], color=color, label='Head Height')
            
            # Plot fall markers
            if algorithm in self.falls and self.falls[algorithm]:
                fall_x = [ep for ep in self.falls[algorithm] if ep < len(self.heights[algorithm])]
                fall_y = [self.heights[algorithm][ep] for ep in fall_x]
                ax3.scatter(fall_x, fall_y, c='r', marker='x', s=50, label='Falls')
                
        ax3.set_title(f'{algorithm} - Stability')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Height (m)')
        ax3.grid(True)
        ax3.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, f"{algorithm}_detailed.png"))
        plt.close(fig)  # Close the figure to free memory

if __name__ == '__main__':
    try:
        visualizer = ComparisonVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
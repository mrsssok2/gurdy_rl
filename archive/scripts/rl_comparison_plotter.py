#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg for interactive plotting
import matplotlib.pyplot as plt
import yaml
import os
from std_msgs.msg import Float32

class RLComparisonPlotter:
    def __init__(self):
        """Initialize the RL Comparison Plotter"""
        rospy.init_node('rl_comparison_plotter', anonymous=True)
        
        # Load configuration
        config_file = rospy.get_param('~config_file', os.path.join(os.path.dirname(__file__), '../config/plotter_config.yaml'))
        rospy.loginfo(f"Loading configuration from {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr(f"Error loading config: {e}")
            self.config = {
                'algorithms': ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac'],
                'colors': ['blue', 'green', 'red', 'purple', 'orange', 'cyan'],
                'window_size': 10,
                'update_frequency': 1.0,
                'max_episodes': 100
            }
        
        # Initialize data storage
        self.algorithms = self.config.get('algorithms', [])
        self.colors = self.config.get('colors', [])
        self.data = {algo: {'rewards': [], 'avg_rewards': []} for algo in self.algorithms}
        
        # Create subscribers for each algorithm
        self.reward_subscribers = {}
        for algo in self.algorithms:
            topic = f"/gurdy_{algo}/episode_reward"
            self.reward_subscribers[algo] = rospy.Subscriber(
                topic, Float32, self.reward_callback, callback_args=algo)
            rospy.loginfo(f"Subscribed to {topic}")
        
        # Create test publisher to verify plotting works
        self.test_publisher = rospy.Publisher('/test_reward', Float32, queue_size=10)
        
        # Initialize the plot with explicit figure size and DPI for better visibility
        plt.ion()  # Turn on interactive mode
        plt.rcParams['figure.figsize'] = [12, 10]
        plt.rcParams['figure.dpi'] = 100
        
        self.fig, self.axes = plt.subplots(2, 1)
        self.fig.canvas.set_window_title('RL Algorithm Comparison')
        
        self.axes[0].set_title('Episode Rewards', fontsize=14)
        self.axes[0].set_xlabel('Episode', fontsize=12)
        self.axes[0].set_ylabel('Reward', fontsize=12)
        self.axes[0].grid(True)
        
        self.axes[1].set_title('Average Rewards (Moving Window)', fontsize=14)
        self.axes[1].set_xlabel('Episode', fontsize=12)
        self.axes[1].set_ylabel('Average Reward', fontsize=12)
        self.axes[1].grid(True)
        
        # Create empty plot lines for each algorithm
        self.reward_lines = {}
        self.avg_reward_lines = {}
        
        for algo, color in zip(self.algorithms, self.colors):
            self.reward_lines[algo], = self.axes[0].plot([], [], color=color, label=algo, linewidth=2)
            self.avg_reward_lines[algo], = self.axes[1].plot([], [], color=color, label=algo, linewidth=2)
        
        # Add legends
        self.axes[0].legend(loc='best', fontsize=10)
        self.axes[1].legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Force the window to appear and remain on top
        plt.show(block=False)
        self.fig.canvas.draw()
        
        # Generate some initial test data to verify plotting
        self.publish_test_data()
        
        # Set update timer
        self.update_frequency = self.config.get('update_frequency', 1.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_frequency), self.update_plot)
        
        rospy.loginfo("RL Comparison Plotter initialized. Showing initial test data.")
    
    def publish_test_data(self):
        """Publish test data for demonstration"""
        rospy.loginfo("Publishing test reward data")
        for i in range(5):
            msg = Float32()
            msg.data = float(i * 2)
            self.test_publisher.publish(msg)
            
            # Also add this to our first algorithm to show something on plot
            if self.algorithms:
                algo = self.algorithms[0]
                self.data[algo]['rewards'].append(float(i * 2))
                self.data[algo]['avg_rewards'].append(float(i * 1.5))
                
            rospy.sleep(0.1)
    
    def reward_callback(self, msg, algo):
        """Callback for episode reward topics"""
        reward = msg.data
        self.data[algo]['rewards'].append(reward)
        
        # Calculate moving average
        window_size = min(self.config.get('window_size', 10), len(self.data[algo]['rewards']))
        avg_reward = np.mean(self.data[algo]['rewards'][-window_size:])
        self.data[algo]['avg_rewards'].append(avg_reward)
        
        rospy.loginfo(f"Received reward {reward:.2f} for {algo}")
    
    def update_plot(self, event):
        """Update the plot with latest data"""
        for algo in self.algorithms:
            rewards = self.data[algo]['rewards']
            avg_rewards = self.data[algo]['avg_rewards']
            
            if rewards:
                episodes = range(len(rewards))
                self.reward_lines[algo].set_data(episodes, rewards)
                self.avg_reward_lines[algo].set_data(episodes, avg_rewards)
        
        # Adjust axes limits
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        # Redraw the figure
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Add a small pause to ensure window updates
        except Exception as e:
            rospy.logerr(f"Error updating plot: {e}")
    
    def save_results(self):
        """Save the comparison results to files"""
        # Create results directory if it doesn't exist
        results_dir = os.path.expanduser('~/catkin_ws/rl_comparison_results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save plot as image
        plt.savefig(os.path.join(results_dir, 'rl_comparison_plot.png'), dpi=150)
        
        # Save data as CSV
        for algo in self.algorithms:
            if not self.data[algo]['rewards']:
                continue  # Skip if no data
                
            algo_data = {
                'episode': list(range(len(self.data[algo]['rewards']))),
                'reward': self.data[algo]['rewards'],
                'avg_reward': self.data[algo]['avg_rewards']
            }
            
            filename = os.path.join(results_dir, f'{algo}_results.csv')
            with open(filename, 'w') as f:
                f.write('episode,reward,avg_reward\n')
                for i in range(len(algo_data['episode'])):
                    f.write(f"{algo_data['episode'][i]},{algo_data['reward'][i]},{algo_data['avg_reward'][i]}\n")
            
            rospy.loginfo(f"Saved results for {algo} to {filename}")
    
    def run(self):
        """Run the node"""
        rospy.loginfo("Starting RL Comparison Plotter")
        rospy.on_shutdown(self.save_results)
        
        # Instead of spinning, we'll use a custom loop to keep the plot alive
        rate = rospy.Rate(5)  # 5 Hz
        while not rospy.is_shutdown():
            plt.pause(0.1)  # Keep the plot window responsive
            rate.sleep()

if __name__ == '__main__':
    try:
        plotter = RLComparisonPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        pass
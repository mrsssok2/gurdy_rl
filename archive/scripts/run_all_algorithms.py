#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import subprocess
import signal
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

class AlgorithmRunner:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('algorithm_runner', anonymous=True)
        
        # Load parameters from ROS parameter server
        self.config = self.load_parameters()
        
        # Create output directory
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        
        # Store all results
        self.all_results = {}
        
        # Set up the plot
        self.setup_plot()
    
    def load_parameters(self):
        """Load parameters from the ROS parameter server"""
        config = {
            'algorithms': rospy.get_param('~algorithms', ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac']),
            'colors': {
                'qlearn': 'blue',
                'sarsa': 'green',
                'dqn': 'red',
                'policy_gradient': 'purple',
                'ppo': 'orange',
                'sac': 'cyan'
            },
            'episodes_per_algorithm': rospy.get_param('~episodes_per_algorithm', 100),
            'output_dir': rospy.get_param('~output_dir', '/home/user/catkin_ws/src/my_gurdy_description/results'),
            'plot_update_freq': rospy.get_param('~plot_update_freq', 10),  # Update plot every 10 episodes
            'save_freq': rospy.get_param('~save_freq', 20),  # Save data every 20 episodes
            'package_name': rospy.get_param('~package_name', 'my_gurdy_description')
        }
        return config
    
    def setup_plot(self):
        """Set up the matplotlib plot for visualization"""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot lines for episode rewards
        self.episode_lines = {}
        for algorithm in self.config['algorithms']:
            color = self.config['colors'].get(algorithm, 'black')
            line, = self.ax1.plot([], [], color=color, label=algorithm)
            self.episode_lines[algorithm] = line
        
        # Plot lines for average rewards
        self.avg_lines = {}
        for algorithm in self.config['algorithms']:
            color = self.config['colors'].get(algorithm, 'black')
            line, = self.ax2.plot([], [], color=color, label=algorithm)
            self.avg_lines[algorithm] = line
        
        # Configure the axes
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_title('Episode Rewards')
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Average Reward')
        self.ax2.set_title('Average Rewards (Moving Window)')
        self.ax2.legend()
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plot(self):
        """Update the plot with current data"""
        max_episodes = 0
        min_reward = float('inf')
        max_reward = float('-inf')
        
        # Update each algorithm's lines
        for algorithm in self.all_results:
            episode_rewards = self.all_results[algorithm]['episode_rewards']
            avg_rewards = self.all_results[algorithm]['avg_rewards']
            
            if len(episode_rewards) > 0:
                episodes = range(len(episode_rewards))
                max_episodes = max(max_episodes, len(episode_rewards))
                
                # Update min and max reward values for axis scaling
                if episode_rewards:
                    min_reward = min(min_reward, min(episode_rewards))
                    max_reward = max(max_reward, max(episode_rewards))
                
                # Update the data for the lines
                self.episode_lines[algorithm].set_data(episodes, episode_rewards)
                self.avg_lines[algorithm].set_data(episodes, avg_rewards)
        
        # Adjust the plot limits if we have data
        if max_episodes > 0 and min_reward != float('inf') and max_reward != float('-inf'):
            padding = (max_reward - min_reward) * 0.1 if max_reward > min_reward else 0.1
            
            self.ax1.set_xlim(0, max(10, max_episodes))
            self.ax1.set_ylim(min_reward - padding, max_reward + padding)
            
            self.ax2.set_xlim(0, max(10, max_episodes))
            self.ax2.set_ylim(min_reward - padding, max_reward + padding)
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_results(self):
        """Save the results to pickle files and plot"""
        # Save all algorithm data
        for algorithm in self.all_results:
            file_path = os.path.join(self.config['output_dir'], f'{algorithm}_data.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(self.all_results[algorithm], f)
            rospy.loginfo(f"Saved data for {algorithm} to {file_path}")
        
        # Save the plot
        plot_path = os.path.join(self.config['output_dir'], 'algorithm_comparison.png')
        plt.savefig(plot_path)
        rospy.loginfo(f"Saved plot to {plot_path}")
    
    def extract_rewards_from_log(self, log_file):
        """Extract episode rewards from a log file"""
        rewards = []
        with open(log_file, 'r') as f:
            for line in f:
                if "Episode reward:" in line:
                    try:
                        reward = float(line.split("Episode reward:")[1].split()[0])
                        rewards.append(reward)
                    except (ValueError, IndexError):
                        pass
        return rewards
    
    def run_algorithm(self, algorithm):
        """Run a specific algorithm and collect results"""
        rospy.loginfo(f"Starting algorithm: {algorithm}")
        
        # Prepare log file
        log_file = os.path.join(self.config['output_dir'], f'{algorithm}_log.txt')
        with open(log_file, 'w') as f:
            f.write(f"Starting {algorithm} training\n")
        
        # Launch the algorithm
        launch_file = f"{algorithm}_launch.launch"
        if algorithm == "qlearn":
            launch_file = "train_gurdy.launch"  # Special case for Q-learning
        if algorithm == "policy_gradient":
            launch_file = "policy_gradient.launch"  # Special case for policy gradient
        if algorithm == "ppo":
            launch_file = "ppo_launch.launch"  # Special case for PPO
        if algorithm == "sac":
            launch_file = "sac_launch.launch"  # Special case for SAC
        if algorithm == "dqn":
            launch_file = "launch_dqn.launch"  # Special case for DQN
        if algorithm == "sarsa":
            launch_file = "launch_sarsa.launch"  # Special case for SARSA
            
        cmd = ['roslaunch', self.config['package_name'], launch_file]
        
        # Start the process
        with open(log_file, 'a') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log)
            
            episode_count = 0
            episode_rewards = []
            avg_rewards = []
            
            try:
                # Monitor the log file for episode rewards
                while episode_count < self.config['episodes_per_algorithm'] and process.poll() is None:
                    time.sleep(1)  # Check every second
                    
                    # Extract rewards from log
                    rewards = self.extract_rewards_from_log(log_file)
                    new_episodes = len(rewards) - episode_count
                    
                    if new_episodes > 0:
                        # Get new rewards
                        new_rewards = rewards[episode_count:episode_count+new_episodes]
                        episode_rewards.extend(new_rewards)
                        
                        # Update episode count
                        episode_count += new_episodes
                        
                        # Calculate average reward
                        window_size = min(20, len(episode_rewards))
                        if window_size > 0:
                            avg = sum(episode_rewards[-window_size:]) / window_size
                            avg_rewards.append(avg)
                        
                        # Log progress
                        rospy.loginfo(f"{algorithm}: Completed {episode_count} episodes, latest reward: {new_rewards[-1]}")
                        
                        # Update plot periodically
                        if episode_count % self.config['plot_update_freq'] == 0:
                            self.all_results[algorithm] = {
                                'episode_rewards': episode_rewards,
                                'avg_rewards': avg_rewards
                            }
                            self.update_plot()
                        
                        # Save data periodically
                        if episode_count % self.config['save_freq'] == 0:
                            self.save_results()
                
                # If we have enough episodes or the process ended, terminate it
                if process.poll() is None:
                    rospy.loginfo(f"Terminating {algorithm} after {episode_count} episodes")
                    process.terminate()
                    process.wait(timeout=10)
            
            except Exception as e:
                rospy.logerr(f"Error running {algorithm}: {e}")
                if process.poll() is None:
                    process.terminate()
            
            # Store final results
            self.all_results[algorithm] = {
                'episode_rewards': episode_rewards,
                'avg_rewards': avg_rewards
            }
            
            rospy.loginfo(f"Completed {algorithm} with {len(episode_rewards)} episodes")
    
    def run_all(self):
        """Run all algorithms sequentially"""
        for algorithm in self.config['algorithms']:
            self.run_algorithm(algorithm)
            
            # Update plot after each algorithm
            self.update_plot()
            
            # Save results
            self.save_results()
            
            # Wait a moment between algorithms
            rospy.sleep(5)
        
        # Final update and save
        self.update_plot()
        self.save_results()
        
        rospy.loginfo("All algorithms completed. Results saved.")
        
        # Keep the plot open until Ctrl+C
        rospy.loginfo("Press Ctrl+C to exit")
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    try:
        runner = AlgorithmRunner()
        runner.run_all()
    except rospy.ROSInterruptException:
        pass
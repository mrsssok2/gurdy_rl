#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from collections import defaultdict
import os
import time
import json
import subprocess
import signal
import sys

class AlgorithmData:
    """Store data for a single RL algorithm"""
    def __init__(self, name):
        self.name = name
        self.rewards = []
        self.avg_rewards = []
        self.losses = []
        self.secondary_losses = []
        self.heights = []
        self.falls = []

class TrainingVisualizer:
    """
    Visualizes performance metrics from different RL algorithms
    in real-time for comparison.
    """
    def __init__(self):
        # Set up refresh rate (in seconds)
        self.refresh_rate = 1.0
        
        # Set up matplotlib to use TkAgg backend for interactive plotting
        plt.switch_backend('TkAgg')
        
        # Algorithm data storage
        self.algorithms = {}
        self.algorithm_names = ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac']
        
        # Initialize algorithm data objects
        for algo_name in self.algorithm_names:
            self.algorithms[algo_name] = AlgorithmData(algo_name)
        
        # Colors for different algorithms
        self.colors = {
            'qlearn': 'blue',
            'sarsa': 'green',
            'dqn': 'red',
            'policy_gradient': 'purple',
            'ppo': 'orange',
            'sac': 'cyan'
        }
        
        # Output directory for saving plots
        self.outdir = os.path.expanduser("~/rl_comparison_plots")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # Start the algorithm processes
        self.processes = self.start_training_processes()
        
        # Set up the plot
        self.setup_plot()
        
        # Set up animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.refresh_rate*1000
        )
        
        # Display the plot
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        
        print("Training visualizer initialized")
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def start_training_processes(self):
        """
        Start the training processes for each algorithm
        Returns a dictionary of subprocess objects
        """
        processes = {}
        
        # Start each algorithm in a separate process
        print("Starting training processes...")
        
        # Q-learning (Original)
        processes['qlearn'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/start_training.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2.0)  # Wait a bit before starting the next algorithm
        
        # SARSA
        processes['sarsa'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/train_gurdy_sarsa.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2.0)
        
        # DQN
        processes['dqn'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/train_gurdy_dqn.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2.0)
        
        # Policy Gradient
        processes['policy_gradient'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/train_gurdy_policy_gradient.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2.0)
        
        # PPO
        processes['ppo'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/train_gurdy_ppo.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2.0)
        
        # SAC
        processes['sac'] = subprocess.Popen(
            ["python", os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts/train_gurdy_sac.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("All training processes started")
        return processes
    
    def signal_handler(self, sig, frame):
        """Handle termination signals to clean up processes"""
        print("Termination signal received, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources and terminate subprocesses"""
        print("Cleaning up...")
        
        # Terminate all processes
        for name, process in self.processes.items():
            if process.poll() is None:  # If process is still running
                print(f"Terminating {name} process...")
                process.terminate()
                try:
                    # Wait for process to terminate
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    print(f"Force killing {name} process...")
                    process.kill()
        
        # Save final plots
        self.save_final_plots()
    
    def setup_plot(self):
        """Set up the plot layout and initial elements"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.set_window_title('RL Algorithm Comparison')
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
        self.fall_markers = {}
        self.fall_bars = None
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    def collect_data(self):
        """
        Collect data from algorithm processes by reading output files
        For simplicity, this implementation uses simulated data
        In a real implementation, it would read from the actual output files
        """
        # For demonstration purposes, we'll update with simulated data
        for algo_name, algo_data in self.algorithms.items():
            # Check if process is still running
            if algo_name in self.processes and self.processes[algo_name].poll() is not None:
                print(f"Warning: {algo_name} process has terminated")
            
            # In a real implementation, this would read from files
            # Just for simulation in this stub
            if len(algo_data.rewards) < 100:  # Simulate up to 100 episodes
                # Add random data for demonstration
                algo_data.rewards.append(np.random.normal(0, 1) * (len(algo_data.rewards) / 10))
                
                # Calculate running average
                if len(algo_data.rewards) > 0:
                    window = min(20, len(algo_data.rewards))
                    avg = sum(algo_data.rewards[-window:]) / window
                    algo_data.avg_rewards.append(avg)
                
                # Add example loss data
                algo_data.losses.append(np.random.exponential(1) * np.exp(-len(algo_data.losses) / 50))
                
                # Add secondary loss data for algorithms that have it
                if algo_name in ['ppo', 'sac']:
                    algo_data.secondary_losses.append(np.random.exponential(0.5) * np.exp(-len(algo_data.secondary_losses) / 60))
                
                # Add head height data
                algo_data.heights.append(0.15 + np.random.normal(0, 0.03))
                
                # Randomly add falls
                if np.random.random() < 0.05:  # 5% chance of falling
                    algo_data.falls.append(len(algo_data.rewards) - 1)
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Collect data
        self.collect_data()
        
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
        
        # Reconfigure axes after clearing
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
        active_algorithms = []
        
        for algo_name, algo_data in self.algorithms.items():
            # Only plot algorithms with data
            if not algo_data.rewards:
                continue
                
            active_algorithms.append(algo_name)
            color = self.colors.get(algo_name, 'gray')
            
            # Plot reward data
            episodes = range(len(algo_data.rewards))
            self.lines[f"{algo_name}_reward"] = self.ax_reward.plot(
                episodes, algo_data.rewards, color=color, label=algo_name)[0]
            
            # Plot average reward data
            if algo_data.avg_rewards:
                episodes = range(len(algo_data.avg_rewards))
                self.lines[f"{algo_name}_avg_reward"] = self.ax_avg_reward.plot(
                    episodes, algo_data.avg_rewards, color=color, label=algo_name)[0]
            
            # Plot loss data
            if algo_data.losses:
                episodes = range(len(algo_data.losses))
                self.lines[f"{algo_name}_loss"] = self.ax_loss.plot(
                    episodes, algo_data.losses, color=color, label=algo_name)[0]
            
            # Plot secondary loss data
            if algo_data.secondary_losses:
                episodes = range(len(algo_data.secondary_losses))
                self.lines[f"{algo_name}_secondary_loss"] = self.ax_secondary_loss.plot(
                    episodes, algo_data.secondary_losses, color=color, label=algo_name)[0]
            
            # Plot height data
            if algo_data.heights:
                episodes = range(len(algo_data.heights))
                self.lines[f"{algo_name}_height"] = self.ax_height.plot(
                    episodes, algo_data.heights, color=color, label=algo_name)[0]
                
                # Plot fall markers
                if algo_data.falls:
                    fall_x = [ep for ep in algo_data.falls if ep < len(algo_data.heights)]
                    fall_y = [algo_data.heights[ep] for ep in fall_x]
                    self.fall_markers[algo_name] = self.ax_height.scatter(
                        fall_x, fall_y, color=color, marker='x', s=50)
        
        # Plot fall count as a bar chart
        if active_algorithms:
            fall_counts = [len(self.algorithms[algo].falls) for algo in active_algorithms]
            self.fall_bars = self.ax_falls.bar(
                range(len(active_algorithms)), fall_counts, 
                color=[self.colors.get(algo, 'gray') for algo in active_algorithms])
            self.ax_falls.set_xticks(range(len(active_algorithms)))
            self.ax_falls.set_xticklabels(active_algorithms, rotation=45)
        
        # Add legends
        if active_algorithms:
            self.ax_reward.legend()
            self.ax_avg_reward.legend()
            self.ax_loss.legend()
            self.ax_secondary_loss.legend()
            self.ax_height.legend()
        
        # Save current plot
        try:
            self.fig.savefig(os.path.join(self.outdir, "comparison_plot.png"))
        except Exception as e:
            print(f"Failed to save plot: {e}")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    def save_final_plots(self):
        """Save final comparison plots when the node is shutting down"""
        try:
            # Save main comparison plot
            self.fig.savefig(os.path.join(self.outdir, "final_comparison.png"))
            
            # Save individual algorithm plots
            for algo_name in self.algorithm_names:
                if algo_name in self.algorithms and self.algorithms[algo_name].rewards:
                    self.save_algorithm_plot(algo_name)
                    
            print("Saved final comparison plots")
        except Exception as e:
            print(f"Failed to save final plots: {e}")
    
    def save_algorithm_plot(self, algorithm):
        """Save detailed plot for a specific algorithm"""
        # Create a new figure for this algorithm
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Get the algorithm color
        color = self.colors.get(algorithm, 'blue')
        algo_data = self.algorithms[algorithm]
        
        # Plot episode and average rewards
        ax1 = axes[0]
        if algo_data.rewards:
            episodes = range(len(algo_data.rewards))
            ax1.plot(episodes, algo_data.rewards, color=color, label='Episode Reward')
            
        if algo_data.avg_rewards:
            episodes = range(len(algo_data.avg_rewards))
            ax1.plot(episodes, algo_data.avg_rewards, 'r-', label='Average Reward')
            
        ax1.set_title(f'{algorithm} - Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        ax1.legend()
        
        # Plot losses
        ax2 = axes[1]
        if algo_data.losses:
            episodes = range(len(algo_data.losses))
            ax2.plot(episodes, algo_data.losses, color=color, label='Primary Loss')
            
        if algo_data.secondary_losses:
            episodes = range(len(algo_data.secondary_losses))
            ax2.plot(episodes, algo_data.secondary_losses, 'm-', label='Secondary Loss')
            
        ax2.set_title(f'{algorithm} - Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
        
        # Plot stability and falls
        ax3 = axes[2]
        if algo_data.heights:
            episodes = range(len(algo_data.heights))
            ax3.plot(episodes, algo_data.heights, color=color, label='Head Height')
            
            # Plot fall markers
            if algo_data.falls:
                fall_x = [ep for ep in algo_data.falls if ep < len(algo_data.heights)]
                fall_y = [algo_data.heights[ep] for ep in fall_x]
                ax3.scatter(fall_x, fall_y, c='r', marker='x', s=50, label='Falls')
                
        ax3.set_title(f'{algorithm} - Stability')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Height (m)')
        ax3.grid(True)
        ax3.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"{algorithm}_detailed.png"))
        plt.close(fig)  # Close the figure to free memory

def main():
    """Main function"""
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    try:
        # Run until interrupted
        while True:
            plt.pause(0.1)  # Keep the plot updated
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up when exiting
        visualizer.cleanup()
        print("Exiting")

if __name__ == '__main__':
    main()
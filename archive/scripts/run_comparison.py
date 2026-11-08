#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from collections import defaultdict
import os
import time
import threading
import sys
import signal

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
        
        # Simulated data parameters (for demonstration)
        self.trend = np.random.normal(0.3, 0.1)  # Learning trend (different for each algorithm)
        self.volatility = np.random.uniform(0.5, 1.5)  # Reward volatility
        self.fall_probability = np.random.uniform(0.01, 0.1)  # Probability of falling

class ComparisonVisualizer:
    """
    Simple visualization tool for comparing RL algorithms
    with simulated data for demonstration.
    """
    def __init__(self):
        # Set up refresh rate (in seconds)
        self.refresh_rate = 0.5
        
        # Set up matplotlib to use TkAgg backend for interactive plotting
        plt.switch_backend('TkAgg')
        
        # Algorithm data
        self.algorithms = {}
        self.algorithm_names = ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac']
        
        # Initialize algorithm data with different characteristics
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
        self.outdir = "rl_comparison_plots"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Set up the plot
        self.setup_plot()
        
        # Running flag
        self.running = True
        
        # Episode counter
        self.episode = 0
        self.max_episodes = 100
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self.update_data_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Set up animation for the plot
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.refresh_rate*1000
        )
        
        print("Comparison visualizer initialized")
    
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
        self.fall_markers = {}
        self.fall_bars = None
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        print("Termination signal received, shutting down...")
        self.running = False
        self.save_final_plots()
        sys.exit(0)
    
    def update_data_thread(self):
        """Thread to update algorithm data periodically"""
        while self.running and self.episode < self.max_episodes:
            # Generate new data
            self.generate_episode_data()
            
            # Increment episode counter
            self.episode += 1
            
            # Simulate training time
            time.sleep(0.2)
    
    def generate_episode_data(self):
        """Generate simulated data for one episode for all algorithms"""
        for algo_name, algo_data in self.algorithms.items():
            # Generate reward with learning trend and noise
            progress = min(1.0, self.episode / 50.0)  # Learning progress (0 to 1)
            base_reward = algo_data.trend * progress * 10  # Base reward increases with progress
            noise = np.random.normal(0, algo_data.volatility)  # Random noise
            reward = base_reward + noise
            
            # Add reward to data
            algo_data.rewards.append(reward)
            
            # Calculate running average reward
            window = min(20, len(algo_data.rewards))
            avg = sum(algo_data.rewards[-window:]) / window
            algo_data.avg_rewards.append(avg)
            
            # Generate loss data (typically decreases over time)
            loss = 1.0 / (1.0 + 0.05 * self.episode) + np.random.normal(0, 0.1)
            loss = max(0.01, loss)  # Ensure positive loss
            algo_data.losses.append(loss)
            
            # Generate secondary loss for algorithms that have it
            if algo_name in ['ppo', 'sac']:
                secondary_loss = 0.8 / (1.0 + 0.05 * self.episode) + np.random.normal(0, 0.08)
                secondary_loss = max(0.01, secondary_loss)
                algo_data.secondary_losses.append(secondary_loss)
            
            # Generate head height (stability)
            # Better algorithms maintain height better
            stability_factor = 0.5 + 0.5 * (algo_data.trend / 0.5)  # 0.5 to 1.0 based on trend
            base_height = 0.15 * stability_factor
            height_noise = np.random.normal(0, 0.02)
            height = base_height + height_noise
            height = max(0.02, height)  # Minimum height
            algo_data.heights.append(height)
            
            # Randomly add falls (probability decreases with progress)
            fall_chance = algo_data.fall_probability * (1.0 - progress * 0.8)
            if np.random.random() < fall_chance:
                algo_data.falls.append(self.episode)
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Skip if no data
        if self.episode == 0:
            return
        
        # Clear axes
        self.ax_reward.clear()
        self.ax_avg_reward.clear()
        self.ax_loss.clear()
        self.ax_secondary_loss.clear()
        self.ax_height.clear()
        self.ax_falls.clear()
        
        # Reset line objects
        self.lines = {}
        self.fall_markers = {}
        
        # Reconfigure axes
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
        
        # Add episode counter
        plt.figtext(0.5, 0.01, f"Episode: {self.episode}/{self.max_episodes}", 
                   ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Save current plot periodically
        if self.episode % 10 == 0:
            try:
                self.fig.savefig(os.path.join(self.outdir, "comparison_plot.png"))
            except Exception as e:
                print(f"Failed to save plot: {e}")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Check if we've reached the end
        if self.episode >= self.max_episodes and self.running:
            self.running = False
            self.save_final_plots()
            print(f"Completed {self.max_episodes} episodes. Final plots saved.")
    
    def save_final_plots(self):
        """Save final comparison plots"""
        try:
            # Save main comparison plot
            self.fig.savefig(os.path.join(self.outdir, "final_comparison.png"))
            
            # Save individual algorithm plots
            for algo_name in self.algorithm_names:
                if algo_name in self.algorithms and self.algorithms[algo_name].rewards:
                    self.save_algorithm_plot(algo_name)
                    
            print("Saved final comparison plots to:", self.outdir)
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
    print("Starting RL Algorithm Comparison Visualizer...")
    print("This will simulate and visualize the training of 6 RL algorithms")
    print("Press Ctrl+C to exit")
    
    # Create visualizer
    visualizer = ComparisonVisualizer()
    
    try:
        # Display the plot and keep it updated
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Ensure plots are saved when exiting
        if visualizer.running:
            visualizer.running = False
            visualizer.save_final_plots()
        print("Exiting")

if __name__ == '__main__':
    main()
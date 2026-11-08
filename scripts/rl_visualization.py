#!/usr/bin/env python
"""
Real-time Visualization Tool for Reinforcement Learning Algorithms

This module provides real-time visualization of multiple RL algorithms
running in parallel on the Gurdy robot.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import threading
import time
import rospy
from collections import defaultdict

# Use TkAgg backend for interactive plotting
matplotlib.use('TkAgg')

class RLVisualization:
    def __init__(self, algorithms, update_interval=1.0):
        """
        Initialize visualization for multiple RL algorithms.
        
        Args:
            algorithms (list): List of algorithm names to visualize
            update_interval (float): Interval between plot updates in seconds
        """
        self.algorithms = algorithms
        self.update_interval = update_interval
        
        # Data storage for each algorithm
        self.data = {algo: {
            'episode_rewards': [],
            'avg_rewards': [],
            'q_values': [],
            'head_heights': [],
            'fall_episodes': [],
            'current_episode': 0,
            'latest_reward': 0,
            'latest_q': 0,
            'latest_height': 0,
            'latest_fall': False,
            'training_time': 0
        } for algo in algorithms}
        
        # Set colors for each algorithm
        self.colors = {
            'qlearn': 'blue',
            'sarsa': 'green',
            'dqn': 'red',
            'ppo': 'purple',
            'sac': 'orange',
            'pg': 'brown'
        }
        
        # Setup plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(12, 8))
        
        # 1. Episode rewards
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.reward_lines = {}
        self.avg_reward_lines = {}
        for algo in algorithms:
            line, = self.ax1.plot([], [], 
                                 alpha=0.3, 
                                 color=self.colors.get(algo, 'gray'))
            self.reward_lines[algo] = line
            
            avg_line, = self.ax1.plot([], [], 
                                     alpha=0.8,
                                     linewidth=2,
                                     color=self.colors.get(algo, 'gray'), 
                                     label=algo)
            self.avg_reward_lines[algo] = avg_line
            
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_title('Episode Rewards')
        self.ax1.legend()
        
        # 2. Average Q-values
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.q_lines = {}
        for algo in algorithms:
            line, = self.ax2.plot([], [], 
                                 color=self.colors.get(algo, 'gray'), 
                                 label=algo)
            self.q_lines[algo] = line
            
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Q-value')
        self.ax2.set_title('Average Q-values')
        self.ax2.legend()
        
        # 3. Head heights (stability)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.height_lines = {}
        self.fall_markers = {}
        for algo in algorithms:
            line, = self.ax3.plot([], [], 
                                 color=self.colors.get(algo, 'gray'), 
                                 label=algo)
            self.height_lines[algo] = line
            
            # Placeholder for fall markers
            scatter = self.ax3.scatter([], [], 
                                      marker='x', 
                                      s=50, 
                                      color=self.colors.get(algo, 'gray'),
                                      alpha=0.7)
            self.fall_markers[algo] = scatter
            
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Height (m)')
        self.ax3.set_title('Robot Stability (Head Height)')
        self.ax3.legend()
        
        # 4. Current performance metrics
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.performance_bars = self.ax4.bar(
            np.arange(len(algorithms)), 
            np.zeros(len(algorithms)),
            color=[self.colors.get(algo, 'gray') for algo in algorithms]
        )
        self.ax4.set_xticks(np.arange(len(algorithms)))
        self.ax4.set_xticklabels(algorithms)
        self.ax4.set_title('Current Episode Reward')
        self.ax4.set_ylabel('Reward')
        
        # Add text for episode info
        self.episode_text = self.fig.text(0.5, 0.01, '', ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Animation setup
        self.running = True
        self.anim = None
        self.viz_thread = None
        
    def update_data(self, algorithm, episode, reward, q_value, head_height, fall=False, training_time=0):
        """
        Update data for a specific algorithm.
        
        Args:
            algorithm (str): Algorithm name
            episode (int): Current episode number
            reward (float): Episode reward
            q_value (float): Average Q-value
            head_height (float): Robot head height
            fall (bool): Whether robot fell in this episode
            training_time (float): Training time in seconds
        """
        if algorithm not in self.data:
            rospy.logwarn(f"Algorithm {algorithm} not found in visualization data")
            return
            
        data = self.data[algorithm]
        
        # Make sure we have enough elements in the arrays
        if len(data['episode_rewards']) <= episode:
            # Extend arrays with None values up to current episode
            extension = [None] * (episode - len(data['episode_rewards']) + 1)
            data['episode_rewards'].extend(extension)
            data['avg_rewards'].extend(extension)
            data['q_values'].extend(extension)
            data['head_heights'].extend(extension)
            
        # Update episode data
        data['episode_rewards'][episode] = reward
        
        # Calculate running average (last 100 episodes)
        valid_rewards = [r for r in data['episode_rewards'][-100:] if r is not None]
        avg_reward = np.mean(valid_rewards) if valid_rewards else 0
        data['avg_rewards'][episode] = avg_reward
        
        # Update other metrics
        data['q_values'][episode] = q_value
        data['head_heights'][episode] = head_height
        
        # Record fall
        if fall:
            data['fall_episodes'].append(episode)
            
        # Update latest values for real-time display
        data['current_episode'] = episode
        data['latest_reward'] = reward
        data['latest_q'] = q_value
        data['latest_height'] = head_height
        data['latest_fall'] = fall
        data['training_time'] = training_time
    
    def _update_plot(self, frame):
        """Update function for animation"""
        if not self.running:
            return
            
        # Update episode rewards plot
        for algo, line in self.reward_lines.items():
            data = self.data[algo]
            rewards = data['episode_rewards']
            
            # Filter out None values
            episodes = [i for i, r in enumerate(rewards) if r is not None]
            valid_rewards = [r for r in rewards if r is not None]
            
            if episodes:
                line.set_data(episodes, valid_rewards)
                
        # Update average rewards
        for algo, line in self.avg_reward_lines.items():
            data = self.data[algo]
            avgs = data['avg_rewards']
            
            # Filter out None values
            episodes = [i for i, r in enumerate(avgs) if r is not None]
            valid_avgs = [r for r in avgs if r is not None]
            
            if episodes:
                line.set_data(episodes, valid_avgs)
                
        # Update Q-values plot
        for algo, line in self.q_lines.items():
            data = self.data[algo]
            q_values = data['q_values']
            
            # Filter out None values
            episodes = [i for i, q in enumerate(q_values) if q is not None]
            valid_q_values = [q for q in q_values if q is not None]
            
            if episodes:
                line.set_data(episodes, valid_q_values)
                
        # Update head heights plot
        for algo, line in self.height_lines.items():
            data = self.data[algo]
            heights = data['head_heights']
            
            # Filter out None values
            episodes = [i for i, h in enumerate(heights) if h is not None]
            valid_heights = [h for h in heights if h is not None]
            
            if episodes:
                line.set_data(episodes, valid_heights)
                
        # Update fall markers
        for algo, scatter in self.fall_markers.items():
            data = self.data[algo]
            falls = data['fall_episodes']
            heights = data['head_heights']
            
            if falls:
                # Only include falls for episodes with valid height data
                fall_x = [ep for ep in falls if ep < len(heights) and heights[ep] is not None]
                fall_y = [heights[ep] for ep in falls if ep < len(heights) and heights[ep] is not None]
                
                if fall_x:
                    scatter.set_offsets(np.column_stack([fall_x, fall_y]))
                    
        # Update performance bars
        latest_rewards = [self.data[algo]['latest_reward'] for algo in self.algorithms]
        for i, bar in enumerate(self.performance_bars):
            bar.set_height(latest_rewards[i])
            
        # Update episode text
        episodes_text = [f"{algo}: {self.data[algo]['current_episode']}" for algo in self.algorithms]
        self.episode_text.set_text("Current Episodes: " + ", ".join(episodes_text))
        
        # Adjust axis limits
        max_ep = max([self.data[algo]['current_episode'] for algo in self.algorithms])
        if max_ep > 0:
            # Reward plot limits
            all_rewards = []
            for algo in self.algorithms:
                rewards = [r for r in self.data[algo]['episode_rewards'] if r is not None]
                if rewards:
                    all_rewards.extend(rewards)
                    
            if all_rewards:
                min_r = min(all_rewards)
                max_r = max(all_rewards)
                buffer = max(1, (max_r - min_r) * 0.1)
                self.ax1.set_xlim(0, max_ep)
                self.ax1.set_ylim(min_r - buffer, max_r + buffer)
                
            # Q-value plot limits
            all_q = []
            for algo in self.algorithms:
                q_values = [q for q in self.data[algo]['q_values'] if q is not None]
                if q_values:
                    all_q.extend(q_values)
                    
            if all_q:
                min_q = min(all_q)
                max_q = max(all_q)
                buffer = max(0.1, (max_q - min_q) * 0.1)
                self.ax2.set_xlim(0, max_ep)
                self.ax2.set_ylim(min_q - buffer, max_q + buffer)
                
            # Height plot limits
            all_heights = []
            for algo in self.algorithms:
                heights = [h for h in self.data[algo]['head_heights'] if h is not None]
                if heights:
                    all_heights.extend(heights)
                    
            if all_heights:
                max_h = max(all_heights)
                self.ax3.set_xlim(0, max_ep)
                self.ax3.set_ylim(0, max_h * 1.1)
                
            # Performance bar limits
            if latest_rewards:
                min_latest = min(latest_rewards)
                max_latest = max(latest_rewards)
                buffer = max(1, (max_latest - min_latest) * 0.1)
                self.ax4.set_ylim(min_latest - buffer, max_latest + buffer)
        
        return (list(self.reward_lines.values()) + 
                list(self.avg_reward_lines.values()) + 
                list(self.q_lines.values()) + 
                list(self.height_lines.values()) + 
                list(self.fall_markers.values()) + 
                list(self.performance_bars) +
                [self.episode_text])
    
    def _run_animation(self):
        """Run matplotlib animation in a separate thread"""
        self.anim = FuncAnimation(
            self.fig, self._update_plot, interval=int(self.update_interval * 1000),
            blit=True, cache_frame_data=False
        )
        plt.show()
        self.running = False
    
    def show(self):
        """Start visualization in a separate thread"""
        if self.viz_thread is not None and self.viz_thread.is_alive():
            return
            
        self.running = True
        self.viz_thread = threading.Thread(target=self._run_animation)
        self.viz_thread.daemon = True
        self.viz_thread.start()
    
    def is_active(self):
        """Check if visualization is still active"""
        return self.running and self.viz_thread is not None and self.viz_thread.is_alive()
    
    def close(self):
        """Close visualization"""
        self.running = False
        if self.anim:
            self.anim.event_source.stop()
        plt.close(self.fig)

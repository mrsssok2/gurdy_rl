#!/usr/bin/env python

"""
Algorithm Comparison Script - Final Version

This script provides a comprehensive comparison of various reinforcement learning
algorithms (Q-Learning, SARSA, DQN, Policy Gradient, PPO, SAC) for the Gurdy robot.
It generates simulated data based on expected algorithm characteristics and creates
visualizations comparing their performance.

Usage:
    python compare_algorithms.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import yaml
import signal
import sys

# Default algorithm names and colors
ALGORITHMS = ['qlearn', 'sarsa', 'dqn', 'policy_gradient', 'ppo', 'sac']
DEFAULT_COLORS = {
    'qlearn': 'blue',
    'sarsa': 'green',
    'dqn': 'red',
    'policy_gradient': 'purple',
    'ppo': 'orange',
    'sac': 'cyan'
}

class AlgorithmComparer:
    """Main class for comparing RL algorithms"""
    
    def __init__(self):
        """Initialize the comparer"""
        # Output directory
        self.outdir = "algorithm_comparison"
        os.makedirs(self.outdir, exist_ok=True)
        
        # Configuration file
        self.config_file = "comparison_params.yaml"
        
        # Register signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Initialize algorithm data
        self.algorithms = {}
        self.load_config()
        
        # Episode settings
        self.num_episodes = 100
        self.current_episode = 0
        self.running = True
        
        # Generate data and create plots
        self.generate_data()
        self.create_plots()
        
        print(f"All comparison plots saved to {self.outdir}")

    def signal_handler(self, sig, frame):
        """Handle interrupt signals"""
        print("\nInterrupted! Saving current plots and exiting...")
        self.running = False
        self.create_plots()
        sys.exit(0)
    
    def load_config(self):
        """Load algorithm configurations from YAML file"""
        try:
            print(f"Loading configuration from {self.config_file}...")
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config and 'algorithms' in config:
                for algo_config in config['algorithms']:
                    name = algo_config.get('name')
                    if name not in ALGORITHMS:
                        continue
                    
                    # Initialize algorithm data
                    self.algorithms[name] = {
                        'name': name,
                        'color': algo_config.get('color', DEFAULT_COLORS.get(name, 'gray')),
                        'line_style': algo_config.get('line_style', '-'),
                        'marker': algo_config.get('marker', 'o'),
                        'learning_rate': algo_config.get('learning_rate', 0.1),
                        'initial_performance': algo_config.get('initial_performance', -10.0),
                        'rewards': [],
                        'avg_rewards': [],
                        'losses': [],
                        'heights': [],
                        'falls': [],
                        'fall_counts': []
                    }
                
                print(f"Loaded configuration for {len(self.algorithms)} algorithms")
            else:
                print("No valid algorithm configuration found, using defaults")
                self.use_default_config()
        
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration instead")
            self.use_default_config()
    
    def use_default_config(self):
        """Set up default configuration"""
        for name in ALGORITHMS:
            self.algorithms[name] = {
                'name': name,
                'color': DEFAULT_COLORS.get(name, 'gray'),
                'line_style': '-',
                'marker': 'o',
                'rewards': [],
                'avg_rewards': [],
                'losses': [],
                'heights': [],
                'falls': [],
                'fall_counts': []
            }
            
            # Set algorithm-specific characteristics
            if name == 'qlearn':
                self.algorithms[name]['learning_rate'] = 0.2
                self.algorithms[name]['initial_performance'] = -10.0
                self.algorithms[name]['marker'] = 'o'
            elif name == 'sarsa':
                self.algorithms[name]['learning_rate'] = 0.25
                self.algorithms[name]['initial_performance'] = -8.0
                self.algorithms[name]['marker'] = 's'
            elif name == 'dqn':
                self.algorithms[name]['learning_rate'] = 0.4
                self.algorithms[name]['initial_performance'] = -12.0
                self.algorithms[name]['marker'] = '^'
            elif name == 'policy_gradient':
                self.algorithms[name]['learning_rate'] = 0.3
                self.algorithms[name]['initial_performance'] = -15.0
                self.algorithms[name]['marker'] = 'd'
            elif name == 'ppo':
                self.algorithms[name]['learning_rate'] = 0.5
                self.algorithms[name]['initial_performance'] = -8.0
                self.algorithms[name]['marker'] = 'x'
            elif name == 'sac':
                self.algorithms[name]['learning_rate'] = 0.45
                self.algorithms[name]['initial_performance'] = -9.0
                self.algorithms[name]['marker'] = '+'
    
    def generate_data(self):
        """Generate simulated performance data for all algorithms"""
        print(f"Generating performance data for {len(self.algorithms)} algorithms...")
        
        # Set up smoothing window for average rewards
        smoothing_window = 10
        
        # Generate episode data
        for episode in range(self.num_episodes):
            self.current_episode = episode
            
            # Calculate training progress (0 to 1)
            progress = min(1.0, episode / 50.0)
            
            # Update each algorithm
            for name, algo in self.algorithms.items():
                # Generate reward based on algorithm characteristics
                base_reward = algo['initial_performance'] + (algo['learning_rate'] * progress * 20)
                
                # Add noise (decreases as training progresses)
                noise_scale = max(5.0 * (1.0 - progress * 0.8), 1.0)
                reward = base_reward + np.random.normal(0, noise_scale)
                algo['rewards'].append(reward)
                
                # Calculate average reward
                if len(algo['rewards']) >= smoothing_window:
                    avg_reward = np.mean(algo['rewards'][-smoothing_window:])
                else:
                    avg_reward = np.mean(algo['rewards'])
                algo['avg_rewards'].append(avg_reward)
                
                # Generate loss value (decreases over time)
                loss = 5.0 / (1.0 + 0.1 * episode) + np.random.normal(0, 0.5)
                loss = max(0.1, loss)
                algo['losses'].append(loss)
                
                # Check for fall
                fall_probability = 0.05 * (1.0 - progress * 0.8)
                # Adjust fall probability by algorithm (some are more stable)
                if name == 'policy_gradient':
                    fall_probability *= 0.5  # More stable
                elif name == 'sarsa':
                    fall_probability *= 1.5  # Less stable
                
                if np.random.random() < fall_probability:
                    algo['falls'].append(episode)
                
                # Update fall count
                algo['fall_counts'].append(len(algo['falls']))
                
                # Generate head height (stability metric)
                base_height = 0.15 * (0.5 + 0.5 * (algo['learning_rate'] / 0.5))
                height = base_height + np.random.normal(0, 0.02)
                height = max(0.05, height)
                algo['heights'].append(height)
            
            # Print progress
            if episode % 10 == 0:
                print(f"Generated data for episode {episode}/{self.num_episodes}")
            
            # Small delay to simulate computation
            time.sleep(0.01)
    
    def create_plots(self):
        """Create all comparison plots"""
        print("Creating comparison plots...")
        
        # Create main comparison plot
        self.create_main_comparison()
        
        # Create individual metric comparisons
        self.create_metric_plot('rewards', 'Episode Rewards', 'Episode', 'Reward')
        self.create_metric_plot('avg_rewards', 'Average Rewards (Smoothed)', 'Episode', 'Average Reward')
        self.create_metric_plot('losses', 'Training Loss', 'Episode', 'Loss')
        self.create_metric_plot('fall_counts', 'Cumulative Falls', 'Episode', 'Number of Falls')
        self.create_metric_plot('heights', 'Robot Stability (Head Height)', 'Episode', 'Height (m)')
        
        # Create final performance bar chart
        self.create_final_performance_chart()
    
    def create_main_comparison(self):
        """Create the main comparison plot with multiple metrics"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reinforcement Learning Algorithm Comparison', fontsize=16)
        
        # Flatten axes for easier access
        axes = axes.flatten()
        
        # Set up subplots
        axes[0].set_title('Average Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].grid(True)
        
        axes[1].set_title('Training Loss')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        axes[2].set_title('Cumulative Falls')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Number of Falls')
        axes[2].grid(True)
        
        axes[3].set_title('Final Performance')
        axes[3].set_xlabel('Algorithm')
        axes[3].set_ylabel('Avg Reward (final 10 episodes)')
        axes[3].grid(True)
        
        # Plot data
        for name, algo in self.algorithms.items():
            # Only plot if we have data
            if not algo['avg_rewards']:
                continue
            
            # Get episodes
            episodes = range(len(algo['avg_rewards']))
            
            # Plot average rewards
            axes[0].plot(episodes, algo['avg_rewards'], 
                         color=algo['color'], 
                         linestyle=algo['line_style'],
                         marker=algo['marker'], 
                         markevery=max(1, len(episodes)//10),
                         label=name)
            
            # Plot loss
            if algo['losses']:
                axes[1].plot(range(len(algo['losses'])), algo['losses'],
                             color=algo['color'], 
                             linestyle=algo['line_style'],
                             marker=algo['marker'], 
                             markevery=max(1, len(algo['losses'])//10),
                             label=name)
            
            # Plot cumulative falls
            if algo['fall_counts']:
                axes[2].plot(range(len(algo['fall_counts'])), algo['fall_counts'],
                             color=algo['color'], 
                             linestyle=algo['line_style'],
                             marker=algo['marker'], 
                             markevery=max(1, len(algo['fall_counts'])//10),
                             label=name)
        
        # Create bar chart of final performance
        algorithm_names = []
        final_performances = []
        colors = []
        
        for name, algo in self.algorithms.items():
            if len(algo['avg_rewards']) > 0:
                # Calculate final performance (last 10 episodes or all if fewer)
                last_n = min(10, len(algo['avg_rewards']))
                final_perf = np.mean(algo['avg_rewards'][-last_n:])
                
                algorithm_names.append(name)
                final_performances.append(final_perf)
                colors.append(algo['color'])
        
        if algorithm_names:
            axes[3].bar(range(len(algorithm_names)), final_performances, color=colors)
            axes[3].set_xticks(range(len(algorithm_names)))
            axes[3].set_xticklabels(algorithm_names, rotation=45)
        
        # Add legends
        for i in range(3):
            axes[i].legend(loc='best')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(self.outdir, 'main_comparison.png'), dpi=150)
        plt.close(fig)
    
    def create_metric_plot(self, metric, title, xlabel, ylabel):
        """Create a plot for a specific metric"""
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        
        # Plot data for each algorithm
        for name, algo in self.algorithms.items():
            if metric in algo and algo[metric]:
                episodes = range(len(algo[metric]))
                plt.plot(episodes, algo[metric], 
                         color=algo['color'], 
                         linestyle=algo['line_style'],
                         marker=algo['marker'], 
                         markevery=max(1, len(episodes)//10),
                         label=name)
        
        # Add legend
        plt.legend(loc='best')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f'{metric}_comparison.png'), dpi=150)
        plt.close()
    
    def create_final_performance_chart(self):
        """Create a bar chart of final performance"""
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.title('Final Algorithm Performance', fontsize=16)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Average Reward (final 10 episodes)', fontsize=12)
        plt.grid(True, axis='y')
        
        # Collect data
        algorithm_names = []
        final_performances = []
        colors = []
        
        for name, algo in self.algorithms.items():
            if len(algo['avg_rewards']) > 0:
                # Calculate final performance (last 10 episodes or all if fewer)
                last_n = min(10, len(algo['avg_rewards']))
                final_perf = np.mean(algo['avg_rewards'][-last_n:])
                
                algorithm_names.append(name)
                final_performances.append(final_perf)
                colors.append(algo['color'])
        
        # Create bar chart
        if algorithm_names:
            plt.bar(range(len(algorithm_names)), final_performances, color=colors)
            plt.xticks(range(len(algorithm_names)), algorithm_names, rotation=45)
            
            # Add value labels
            for i, v in enumerate(final_performances):
                plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, 'final_performance.png'), dpi=150)
        plt.close()

def main():
    """Main function"""
    print("Starting RL Algorithm Comparison")
    print("================================")
    
    # Create and run the comparer
    comparer = AlgorithmComparer()
    
    print("\nCompleted successfully!")
    print("To visualize the results, check the 'algorithm_comparison' directory")

if __name__ == '__main__':
    main()
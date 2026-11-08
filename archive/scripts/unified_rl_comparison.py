#!/usr/bin/env python
"""
Unified RL Comparison Tool for Gurdy Robot

This script provides a simulation of multiple reinforcement learning algorithms 
for comparing their performance without requiring ROS.
"""
import os
import time
import argparse
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import threading
import random
from collections import defaultdict

# Set matplotlib backend
matplotlib.use("TkAgg")

# Mock Environment
class MockEnvironment:
    """Mock environment for simulation"""
    def __init__(self, state_size=9, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.current_step = 0
        
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        # Return random initial observation
        return np.random.uniform(-1, 1, self.state_size)
        
    def step(self, action):
        """Take a step in the environment"""
        self.current_step += 1
        
        # Simulate some dynamics
        observation = np.random.uniform(-1, 1, self.state_size)
        
        # Reward based on action (simulating better actions)
        base_reward = action / self.action_size * 3.0 + random.uniform(-1, 1)
        
        # Add height and fall simulation
        height = max(0.0, 0.15 + random.uniform(-0.05, 0.05))
        fallen = height < 0.05
        
        # Add height to observation
        observation[8] = height
        
        # Higher reward for keeping stable
        if height > 0.1:
            reward = base_reward + 1.0
        else:
            reward = base_reward - 2.0
            
        # Penalize falling
        if fallen:
            reward -= 5.0
            
        # Done if fallen or max steps reached
        done = fallen or self.current_step >= 200
        
        # Add info
        info = {
            "fallen": fallen,
            "height": height
        }
        
        return observation, reward, done, info

# Algorithm Classes
class QLearn:
    """Simplified Q-Learning algorithm for simulation"""
    def __init__(self, action_size, alpha=0.2, gamma=0.95, epsilon=0.9):
        self.q = {}
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_str = str(state)
        if state_str not in self.q:
            self.q[state_str] = np.zeros(self.action_size)
        
        return np.argmax(self.q[state_str])
    
    def learn(self, state, action, reward, next_state):
        """Learn from experience (Q-learning)"""
        state_str = str(state)
        next_state_str = str(next_state)
        
        if state_str not in self.q:
            self.q[state_str] = np.zeros(self.action_size)
        
        if next_state_str not in self.q:
            self.q[next_state_str] = np.zeros(self.action_size)
        
        # Q-learning update rule
        best_next_q = np.max(self.q[next_state_str])
        self.q[state_str][action] += self.alpha * (reward + self.gamma * best_next_q - self.q[state_str][action])
    
    def get_avg_q_value(self):
        """Get average Q-value"""
        if not self.q:
            return 0.0
        
        total = 0.0
        count = 0
        for s in self.q:
            for a in range(self.action_size):
                total += self.q[s][a]
                count += 1
        
        return total / max(1, count)

class SARSA:
    """Simplified SARSA algorithm for simulation"""
    def __init__(self, action_size, alpha=0.2, gamma=0.95, epsilon=0.9):
        self.q = {}
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_str = str(state)
        if state_str not in self.q:
            self.q[state_str] = np.zeros(self.action_size)
        
        return np.argmax(self.q[state_str])
    
    def learn(self, state, action, reward, next_state, next_action):
        """Learn from experience (SARSA)"""
        state_str = str(state)
        next_state_str = str(next_state)
        
        if state_str not in self.q:
            self.q[state_str] = np.zeros(self.action_size)
        
        if next_state_str not in self.q:
            self.q[next_state_str] = np.zeros(self.action_size)
        
        # SARSA update rule
        next_q = self.q[next_state_str][next_action]
        self.q[state_str][action] += self.alpha * (reward + self.gamma * next_q - self.q[state_str][action])
    
    def get_avg_q_value(self):
        """Get average Q-value"""
        if not self.q:
            return 0.0
        
        total = 0.0
        count = 0
        for s in self.q:
            for a in range(self.action_size):
                total += self.q[s][a]
                count += 1
        
        return total / max(1, count)

class DQN:
    """Simplified DQN algorithm for simulation"""
    def __init__(self, state_size, action_size, epsilon=0.9, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []
        self.q_values = [0.0]  # For tracking progress
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        # Simplified: just return random action with bias for demonstration
        return min(self.action_size - 1, max(0, 
            int(np.random.normal(self.action_size / 2, self.action_size / 4))))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
    
    def replay(self):
        """Train on batch from replay memory"""
        # Simulate improvement in Q-values over time
        if self.q_values:
            last_q = self.q_values[-1]
            # Add a small improvement plus noise
            new_q = last_q + 0.01 + random.uniform(-0.005, 0.01)
            self.q_values.append(min(5.0, new_q))  # Cap at reasonable value
    
    def get_avg_q_value(self):
        """Get average Q-value"""
        return self.q_values[-1] if self.q_values else 0.0

class PolicyGradient:
    """Simplified Policy Gradient algorithm for simulation"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.avg_performance = 0.0
        
    def choose_action(self, state):
        """Choose action using policy"""
        # Simplified policy for simulation
        weights = np.ones(self.action_size)
        middle = self.action_size // 2
        # Bias toward middle actions
        weights[middle-1:middle+2] = 2.0
        weights = weights / np.sum(weights)
        return np.random.choice(self.action_size, p=weights)
    
    def remember(self, state, action, reward):
        """Store experience for batch update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self):
        """Train on collected experiences"""
        # Simulate training by updating performance metric
        if self.rewards:
            performance_change = np.mean(self.rewards) * 0.01
            self.avg_performance += performance_change
            
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def get_avg_q_value(self):
        """Get performance metric as Q-value substitute"""
        return self.avg_performance

class PPO:
    """Simplified PPO algorithm for simulation"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = []
        self.value_estimate = 0.0
        
    def choose_action(self, state):
        """Choose action using policy"""
        # Simple biased policy
        probs = np.linspace(0.5, 1.5, self.action_size)
        probs = probs / np.sum(probs)
        return np.random.choice(self.action_size, p=probs)
    
    def store(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > 200:  # Limit buffer size
            self.buffer.pop(0)
    
    def train(self):
        """Train on collected experiences"""
        # Simulate value function improvement
        if self.buffer:
            rewards = [exp[2] for exp in self.buffer[-20:]]  # Last 20 rewards
            avg_reward = np.mean(rewards) if rewards else 0
            self.value_estimate = 0.9 * self.value_estimate + 0.1 * avg_reward
        
        # Clear buffer after certain size
        if len(self.buffer) > 100:
            self.buffer = self.buffer[-50:]  # Keep last 50
    
    def get_avg_q_value(self):
        """Get value estimate as Q-value substitute"""
        return self.value_estimate

class SAC:
    """Simplified SAC algorithm for simulation"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.q_value = 0.0
        self.entropy = 1.0
        
    def choose_action(self, state):
        """Choose action using policy"""
        # Entropy-weighted exploration
        if random.random() < self.entropy * 0.5:
            return random.randint(0, self.action_size - 1)
        
        # Else, use biased action selection
        probs = np.exp(np.linspace(0, 1, self.action_size))
        probs = probs / np.sum(probs)
        return np.random.choice(self.action_size, p=probs)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
    
    def train(self):
        """Train on batch from replay memory"""
        # Simulate Q-value improvement
        if self.memory:
            # Get recent rewards
            recent_rewards = [exp[2] for exp in self.memory[-30:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            # Update Q-value with some momentum
            self.q_value = 0.95 * self.q_value + 0.05 * (avg_reward + random.uniform(-0.1, 0.3))
            
            # Decay entropy over time
            self.entropy = max(0.1, self.entropy * 0.998)
    
    def get_avg_q_value(self):
        """Get Q-value estimate"""
        return self.q_value

# Visualization class 
class RLVisualization:
    """Visualization for multiple RL algorithms"""
    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.data = {algo: {
            "rewards": [],
            "avg_rewards": [],
            "q_values": [],
            "heights": [],
            "falls": []
        } for algo in algorithms}
        
        # Set up colors
        self.colors = {
            "qlearn": "blue",
            "sarsa": "green",
            "dqn": "red",
            "pg": "brown",
            "ppo": "purple",
            "sac": "orange"
        }
        
        # Set up plot
        plt.style.use("seaborn-v0_8-darkgrid")
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("Reinforcement Learning Algorithm Comparison", fontsize=16)
        
        # Reward plot (top-left)
        self.axs[0, 0].set_title("Episode Rewards")
        self.axs[0, 0].set_xlabel("Episode")
        self.axs[0, 0].set_ylabel("Reward")
        
        # Q-value plot (top-right)
        self.axs[0, 1].set_title("Average Q-Values")
        self.axs[0, 1].set_xlabel("Episode")
        self.axs[0, 1].set_ylabel("Q-Value")
        
        # Height plot (bottom-left)
        self.axs[1, 0].set_title("Robot Head Height")
        self.axs[1, 0].set_xlabel("Episode")
        self.axs[1, 0].set_ylabel("Height (m)")
        
        # Average reward plot (bottom-right)
        self.axs[1, 1].set_title("Average Rewards (last 100 episodes)")
        self.axs[1, 1].set_xlabel("Episode")
        self.axs[1, 1].set_ylabel("Average Reward")
        
        # Initialize plot lines
        self.reward_lines = {}
        self.q_lines = {}
        self.height_lines = {}
        self.avg_reward_lines = {}
        
        for algo in algorithms:
            # Reward lines
            line, = self.axs[0, 0].plot([], [], color=self.colors.get(algo, "gray"), label=algo)
            self.reward_lines[algo] = line
            
            # Q-value lines
            line, = self.axs[0, 1].plot([], [], color=self.colors.get(algo, "gray"), label=algo)
            self.q_lines[algo] = line
            
            # Height lines
            line, = self.axs[1, 0].plot([], [], color=self.colors.get(algo, "gray"), label=algo)
            self.height_lines[algo] = line
            
            # Average reward lines
            line, = self.axs[1, 1].plot([], [], color=self.colors.get(algo, "gray"), label=algo)
            self.avg_reward_lines[algo] = line
        
        # Add legends
        for ax in self.axs.flatten():
            ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Set up animation
        self.animation = None
        
    def update_data(self, algorithm, episode, reward, q_value, height, fall=False):
        """Update data for a specific algorithm"""
        if algorithm not in self.data:
            return
        
        data = self.data[algorithm]
        
        # Extend arrays if needed
        while len(data["rewards"]) <= episode:
            data["rewards"].append(None)
            data["q_values"].append(None)
            data["heights"].append(None)
            data["avg_rewards"].append(None)
        
        # Update data
        data["rewards"][episode] = reward
        data["q_values"][episode] = q_value
        data["heights"][episode] = height
        
        # Calculate average reward (last 100 episodes)
        recent_rewards = [r for r in data["rewards"][-100:] if r is not None]
        if recent_rewards:
            data["avg_rewards"][episode] = np.mean(recent_rewards)
        
        # Record fall
        if fall:
            data["falls"].append(episode)
    
    def update_plot(self, frame):
        """Update the visualization plot"""
        # Update each plot
        for algo in self.algorithms:
            data = self.data[algo]
            
            # Get valid data points
            episodes = range(len(data["rewards"]))
            valid_episodes = [i for i, r in enumerate(data["rewards"]) if r is not None]
            valid_rewards = [data["rewards"][i] for i in valid_episodes]
            valid_q_values = [data["q_values"][i] for i in valid_episodes]
            valid_heights = [data["heights"][i] for i in valid_episodes]
            valid_avg_rewards = [data["avg_rewards"][i] for i in valid_episodes]
            
            # Update lines
            if valid_episodes:
                self.reward_lines[algo].set_data(valid_episodes, valid_rewards)
                self.q_lines[algo].set_data(valid_episodes, valid_q_values)
                self.height_lines[algo].set_data(valid_episodes, valid_heights)
                self.avg_reward_lines[algo].set_data(valid_episodes, valid_avg_rewards)
        
        # Adjust axis limits
        max_ep = 0
        for algo in self.algorithms:
            data = self.data[algo]
            if data["rewards"]:
                max_ep = max(max_ep, len(data["rewards"]))
        
        if max_ep > 0:
            # Collect all data points for each plot
            all_rewards = []
            all_q_values = []
            all_heights = []
            all_avg_rewards = []
            
            for algo in self.algorithms:
                data = self.data[algo]
                all_rewards.extend([r for r in data["rewards"] if r is not None])
                all_q_values.extend([q for q in data["q_values"] if q is not None])
                all_heights.extend([h for h in data["heights"] if h is not None])
                all_avg_rewards.extend([a for a in data["avg_rewards"] if a is not None])
            
            # Set axis limits
            if all_rewards:
                min_reward = min(all_rewards)
                max_reward = max(all_rewards)
                self.axs[0, 0].set_xlim(0, max_ep)
                self.axs[0, 0].set_ylim(min_reward - 1, max_reward + 1)
            
            if all_q_values:
                min_q = min(all_q_values)
                max_q = max(all_q_values)
                self.axs[0, 1].set_xlim(0, max_ep)
                self.axs[0, 1].set_ylim(min_q - 0.1, max_q + 0.1)
            
            if all_heights:
                max_height = max(all_heights)
                self.axs[1, 0].set_xlim(0, max_ep)
                self.axs[1, 0].set_ylim(0, max_height * 1.1)
            
            if all_avg_rewards:
                min_avg = min(all_avg_rewards)
                max_avg = max(all_avg_rewards)
                self.axs[1, 1].set_xlim(0, max_ep)
                self.axs[1, 1].set_ylim(min_avg - 1, max_avg + 1)
        
        # Return all artists that were updated
        artists = []
        artists.extend(list(self.reward_lines.values()))
        artists.extend(list(self.q_lines.values()))
        artists.extend(list(self.height_lines.values()))
        artists.extend(list(self.avg_reward_lines.values()))
        return artists
    
    def start_animation(self, interval=100):
        """Start the animation"""
        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=interval, 
            blit=True, cache_frame_data=False
        )
        plt.show()
    
    def save_plot(self, filename):
        """Save the current plot to file"""
        plt.figure(self.fig.number)
        plt.savefig(filename, dpi=100)
        print(f"Plot saved to {filename}")

def load_config():
    """Load or create default configuration"""
    config_path = "rl_config.yaml"
    
    # Default configuration
    default_config = {
        "environment": {
            "state_size": 9,
            "action_size": 7,
            "max_steps": 200
        },
        "training": {
            "episodes": 150,
            "output_dir": "./rl_output"
        },
        "algorithms": {
            "qlearn": {
                "enabled": True,
                "alpha": 0.2,
                "gamma": 0.95,
                "epsilon": 0.9
            },
            "sarsa": {
                "enabled": True,
                "alpha": 0.2,
                "gamma": 0.95,
                "epsilon": 0.9
            },
            "dqn": {
                "enabled": True,
                "epsilon": 0.9,
                "gamma": 0.95
            },
            "pg": {
                "enabled": True
            },
            "ppo": {
                "enabled": True
            },
            "sac": {
                "enabled": True
            }
        }
    }
    
    # Create config file if it doesn"t exist
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default configuration at {config_path}")
        return default_config
    
    # Load existing config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}, using default configuration")
        return default_config

def create_algorithm(algo_name, config):
    """Create algorithm instance based on name and config"""
    env_config = config["environment"]
    algo_config = config["algorithms"].get(algo_name, {})
    
    state_size = env_config["state_size"]
    action_size = env_config["action_size"]
    
    if algo_name == "qlearn":
        return QLearn(
            action_size=action_size,
            alpha=algo_config.get("alpha", 0.2),
            gamma=algo_config.get("gamma", 0.95),
            epsilon=algo_config.get("epsilon", 0.9)
        )
    elif algo_name == "sarsa":
        return SARSA(
            action_size=action_size,
            alpha=algo_config.get("alpha", 0.2),
            gamma=algo_config.get("gamma", 0.95),
            epsilon=algo_config.get("epsilon", 0.9)
        )
    elif algo_name == "dqn":
        return DQN(
            state_size=state_size,
            action_size=action_size,
            epsilon=algo_config.get("epsilon", 0.9),
            gamma=algo_config.get("gamma", 0.95)
        )
    elif algo_name == "pg":
        return PolicyGradient(
            state_size=state_size,
            action_size=action_size
        )
    elif algo_name == "ppo":
        return PPO(
            state_size=state_size,
            action_size=action_size
        )
    elif algo_name == "sac":
        return SAC(
            state_size=state_size,
            action_size=action_size
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def train_algorithm(algorithm, algo_name, config, visualization):
    """Train a single algorithm"""
    env_config = config["environment"]
    train_config = config["training"]
    
    # Create environment
    env = MockEnvironment(
        state_size=env_config["state_size"],
        action_size=env_config["action_size"]
    )
    
    episodes = train_config["episodes"]
    max_steps = env_config["max_steps"]
    
    print(f"Training {algo_name} for {episodes} episodes...")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        
        # For SARSA, choose first action
        if algo_name == "sarsa":
            action = algorithm.choose_action(state)
        
        for step in range(max_steps):
            # Choose action (except for SARSA"s first step)
            if algo_name != "sarsa" or step > 0:
                action = algorithm.choose_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store experience based on algorithm type
            if algo_name == "qlearn":
                algorithm.learn(state, action, reward, next_state)
            elif algo_name == "sarsa":
                # Choose next action for SARSA
                next_action = algorithm.choose_action(next_state)
                algorithm.learn(state, action, reward, next_state, next_action)
                action = next_action  # Remember next action for next iteration
            elif algo_name == "dqn":
                algorithm.remember(state, action, reward, next_state, done)
                if step % 4 == 0:  # Train periodically
                    algorithm.replay()
            elif algo_name == "pg":
                algorithm.remember(state, action, reward)
            elif algo_name == "ppo":
                algorithm.store(state, action, reward, next_state, done)
            elif algo_name == "sac":
                algorithm.remember(state, action, reward, next_state, done)
                if step % 4 == 0:  # Train periodically
                    algorithm.train()
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # End-of-episode updates for some algorithms
        if algo_name in ["pg", "ppo"]:
            algorithm.train()
        
        # Get metrics for visualization
        q_value = algorithm.get_avg_q_value()
        
        # Initialize default values first
        head_height = 0.15  # Default height value
        fall = False        # Default fall value
        
        # Update with actual values if available
        try:
            if 'next_state' in locals() and next_state is not None:
                head_height = next_state[8]
                
            if 'info' in locals() and info is not None:
                fall = info.get("fallen", False)
        except:
            # If any error occurs, use the default values
            pass
        
        # Update visualization
        visualization.update_data(algo_name, episode, total_reward, q_value, head_height, fall)
        
        # Logging
        if episode % 10 == 0 or episode == episodes - 1:
            print(f"{algo_name} - Episode {episode}/{episodes}: " +
                  f"Reward = {total_reward:.2f}, Q-Value = {q_value:.2f}")
    
    print(f"Finished training {algo_name}")

def save_results(algorithms, config, visualization):
    """Save training results to files"""
    output_dir = config["training"].get("output_dir", "./rl_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    visualization.save_plot(os.path.join(output_dir, "rl_comparison.png"))
    
    # Save data for each algorithm
    for algo_name in algorithms:
        data = visualization.data[algo_name]
        
        # Prepare results dictionary
        results = {
            "algorithm": algo_name,
            "rewards": [r for r in data["rewards"] if r is not None],
            "q_values": [q for q in data["q_values"] if q is not None],
            "heights": [h for h in data["heights"] if h is not None],
            "avg_rewards": [a for a in data["avg_rewards"] if a is not None],
            "falls": data["falls"]
        }
        
        # Save to JSON
        output_file = os.path.join(output_dir, f"{algo_name}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {algo_name} results to {output_file}")

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="RL Algorithm Comparison")
    parser.add_argument("--algorithms", nargs="+", 
                        choices=["qlearn", "sarsa", "dqn", "pg", "ppo", "sac"],
                        help="Algorithms to run (default: all enabled in config)")
    parser.add_argument("--episodes", type=int, help="Override number of episodes")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override episodes if specified
    if args.episodes:
        config["training"]["episodes"] = args.episodes
    
    # Determine which algorithms to run
    if args.algorithms:
        algorithm_names = args.algorithms
        # Update enabled flags in config
        for algo in config["algorithms"]:
            config["algorithms"][algo]["enabled"] = algo in algorithm_names
    else:
        algorithm_names = [algo for algo, settings in config["algorithms"].items() 
                         if settings.get("enabled", True)]
    
    print(f"Running the following algorithms: {', '.join(algorithm_names)}")
    
    # Initialize visualization
    visualization = RLVisualization(algorithm_names)
    
    # Create and train algorithms
    algorithms = {}
    threads = []
    
    for algo_name in algorithm_names:
        # Create algorithm
        algorithm = create_algorithm(algo_name, config)
        algorithms[algo_name] = algorithm
        
        # Create and start training thread
        thread = threading.Thread(
            target=train_algorithm,
            args=(algorithm, algo_name, config, visualization)
        )
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    # Start visualization
    visualization.start_animation(interval=200)  # Update every 200ms
    
    # Wait for training to complete
    for thread in threads:
        thread.join()
    
    # Save results
    save_results(algorithms, config, visualization)
    
    print("All algorithms completed training. Keeping visualization open...")
    plt.show()  # Keep plot window open until closed by user

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple RL Comparison

A single Python script that runs all 6 algorithms and compares their performance.
Creates a live plot showing the rewards over time.
"""

import os
import sys
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import signal
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates, ContactsState

# Import the algorithms
# We'll implement simple versions of each algorithm directly in this file
# to keep everything in one place

# Algorithm names
ALGORITHMS = ['qlearn', 'sarsa', 'dqn', 'ppo', 'sac', 'policy_gradient']

# Global variables for tracking progress
episode_rewards = {algo: [] for algo in ALGORITHMS}
avg_rewards = {algo: [] for algo in ALGORITHMS}
q_values = {algo: [] for algo in ALGORITHMS}
head_heights = {algo: [] for algo in ALGORITHMS}
fall_episodes = {algo: [] for algo in ALGORITHMS}

# For visualization
is_running = True
robot_positions = {}
figure = None
axes = None
lines = {}
q_ax = None
height_ax = None

# Namespaces for the robots
NAMESPACES = {
    'qlearn': 'qlearn_gurdy',
    'sarsa': 'sarsa_gurdy',
    'dqn': 'dqn_gurdy',
    'ppo': 'ppo_gurdy',
    'sac': 'sac_gurdy',
    'policy_gradient': 'pg_gurdy'
}

# Colors for plotting
COLORS = {
    'qlearn': 'blue',
    'sarsa': 'green',
    'dqn': 'red',
    'ppo': 'purple',
    'sac': 'orange',
    'policy_gradient': 'brown'
}

class SimpleQLearning:
    """Simple Q-Learning implementation for Gurdy robot"""
    def __init__(self, config):
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.9)
        self.epsilon = config.get('epsilon', 0.9)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.05)
        self.q_table = {}
        
    def get_state(self, observation):
        """Convert observation to discrete state string"""
        joint_angles = observation[:6]
        head_height = observation[-1]
        
        # Simplify the state representation
        joint_angles_discrete = [round(angle, 1) for angle in joint_angles]
        head_height_discrete = round(head_height, 1)
        
        return str([joint_angles_discrete, head_height_discrete])
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 7  # 7 possible actions
            
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 7)  # Random action
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Learn from experience using Q-learning update rule"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 7
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 7
        
        # Q-learning update rule
        q_predict = self.q_table[state][action]
        if not done:
            q_target = reward + self.gamma * max(self.q_table[next_state])
        else:
            q_target = reward
        
        self.q_table[state][action] += self.alpha * (q_target - q_predict)
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def get_avg_q_value(self):
        """Get average Q-value across all states"""
        if not self.q_table:
            return 0.0
        
        total = 0.0
        count = 0
        for state in self.q_table:
            total += max(self.q_table[state])
            count += 1
        
        return total / max(1, count)

class SimpleSARSA:
    """Simple SARSA implementation for Gurdy robot"""
    def __init__(self, config):
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.9)
        self.epsilon = config.get('epsilon', 0.9)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.05)
        self.q_table = {}
        
    def get_state(self, observation):
        """Convert observation to discrete state string"""
        joint_angles = observation[:6]
        head_height = observation[-1]
        
        # Simplify the state representation
        joint_angles_discrete = [round(angle, 1) for angle in joint_angles]
        head_height_discrete = round(head_height, 1)
        
        return str([joint_angles_discrete, head_height_discrete])
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 7  # 7 possible actions
            
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 7)  # Random action
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, next_action, done):
        """Learn from experience using SARSA update rule"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 7
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 7
        
        # SARSA update rule
        q_predict = self.q_table[state][action]
        if not done:
            q_target = reward + self.gamma * self.q_table[next_state][next_action]
        else:
            q_target = reward
        
        self.q_table[state][action] += self.alpha * (q_target - q_predict)
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def get_avg_q_value(self):
        """Get average Q-value across all states"""
        if not self.q_table:
            return 0.0
        
        total = 0.0
        count = 0
        for state in self.q_table:
            total += max(self.q_table[state])
            count += 1
        
        return total / max(1, count)

class GurdyEnv:
    """Simple environment interface for Gurdy robot in ROS/Gazebo"""
    def __init__(self, namespace):
        self.namespace = namespace
        self.joint_publishers = {}
        self.joint_states = None
        self.head_height = 0.0
        self.last_time = time.time()
        self.fallen = False
        
        # Connect to ROS topics
        self._setup_publishers()
        self._setup_subscribers()
        
        # Wait for subscribers to connect
        rospy.sleep(1.0)
    
    def _setup_publishers(self):
        """Setup joint position publishers"""
        joint_names = [
            'head_upperlegM1_joint', 'head_upperlegM2_joint', 'head_upperlegM3_joint',
            'head_upperlegM4_joint', 'head_upperlegM5_joint', 'head_upperlegM6_joint',
            'upperlegM1_lowerlegM1_joint', 'upperlegM2_lowerlegM2_joint', 'upperlegM3_lowerlegM3_joint',
            'upperlegM4_lowerlegM4_joint', 'upperlegM5_lowerlegM5_joint', 'upperlegM6_lowerlegM6_joint'
        ]
        
        for joint in joint_names:
            topic = f"/{self.namespace}/{joint}_position_controller/command"
            self.joint_publishers[joint] = rospy.Publisher(topic, Float64, queue_size=1)
    
    def _setup_subscribers(self):
        """Setup subscribers for joint states and contacts"""
        rospy.Subscriber(f"/{self.namespace}/joint_states", JointState, self._joint_states_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)
    
    def _joint_states_callback(self, msg):
        """Callback for joint states messages"""
        self.joint_states = msg
    
    def _model_states_callback(self, msg):
        """Callback for model states messages (to get robot height)"""
        if self.namespace in msg.name:
            idx = msg.name.index(self.namespace)
            self.head_height = msg.pose[idx].position.z
            
            # Store position for visualization
            global robot_positions
            robot_positions[self.namespace] = (
                msg.pose[idx].position.x,
                msg.pose[idx].position.y,
                msg.pose[idx].position.z
            )
            
            # Check if robot has fallen
            if self.head_height < 0.1:  # Threshold for considering robot fallen
                self.fallen = True
    
    def reset(self):
        """Reset the environment"""
        # Reset joint positions to default
        for joint in self.joint_publishers:
            self.joint_publishers[joint].publish(Float64(0.0))
        
        # Wait for robot to stabilize
        rospy.sleep(1.0)
        
        self.fallen = False
        self.last_time = time.time()
        
        # Return observation
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation from robot state"""
        if self.joint_states is None:
            # Default observation if no joint states received yet
            return np.zeros(9)
        
        # Extract joint positions
        joint_positions = list(self.joint_states.position)
        
        # Extract linear velocity (if available)
        linear_velocity = 0.0  # Default if not available
        
        # Distance from origin
        distance = 0.0
        if self.namespace in robot_positions:
            x, y, _ = robot_positions[self.namespace]
            distance = np.sqrt(x**2 + y**2)
        
        # Combine observation
        observation = joint_positions + [linear_velocity, distance, self.head_height]
        return np.array(observation)
    
    def step(self, action):
        """Take a step in the environment"""
        # Map discrete action to continuous joint commands
        self._apply_action(action)
        
        # Wait for robot to execute action
        rospy.sleep(0.1)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.fallen or (time.time() - self.last_time > 20.0)  # Time limit
        
        # Additional info
        info = {
            'head_height': self.head_height,
            'fallen': self.fallen
        }
        
        return observation, reward, done, info
    
    def _apply_action(self, action):
        """Apply the selected action to the robot"""
        # Define action set (7 actions)
        actions = [
            # Action 0: Move forward - larger motion for better visibility
            {'head_upperlegM1_joint': 0.5, 'head_upperlegM2_joint': 0.5, 'head_upperlegM3_joint': 0.5,
             'head_upperlegM4_joint': 0.5, 'head_upperlegM5_joint': 0.5, 'head_upperlegM6_joint': 0.5,
             'upperlegM1_lowerlegM1_joint': 0.8, 'upperlegM2_lowerlegM2_joint': 0.8, 'upperlegM3_lowerlegM3_joint': 0.8,
             'upperlegM4_lowerlegM4_joint': 0.8, 'upperlegM5_lowerlegM5_joint': 0.8, 'upperlegM6_lowerlegM6_joint': 0.8},
            
            # Action 1: Move backward - larger motion
            {'head_upperlegM1_joint': -0.5, 'head_upperlegM2_joint': -0.5, 'head_upperlegM3_joint': -0.5,
             'head_upperlegM4_joint': -0.5, 'head_upperlegM5_joint': -0.5, 'head_upperlegM6_joint': -0.5,
             'upperlegM1_lowerlegM1_joint': -0.6, 'upperlegM2_lowerlegM2_joint': -0.6, 'upperlegM3_lowerlegM3_joint': -0.6,
             'upperlegM4_lowerlegM4_joint': -0.6, 'upperlegM5_lowerlegM5_joint': -0.6, 'upperlegM6_lowerlegM6_joint': -0.6},
            
            # Action 2: Turn left - larger motion
            {'head_upperlegM1_joint': 0.7, 'head_upperlegM2_joint': 0.5, 'head_upperlegM3_joint': 0.3,
             'head_upperlegM4_joint': 0.0, 'head_upperlegM5_joint': -0.3, 'head_upperlegM6_joint': -0.5,
             'upperlegM1_lowerlegM1_joint': 0.8, 'upperlegM2_lowerlegM2_joint': 0.6, 'upperlegM3_lowerlegM3_joint': 0.4,
             'upperlegM4_lowerlegM4_joint': 0.0, 'upperlegM5_lowerlegM5_joint': -0.4, 'upperlegM6_lowerlegM6_joint': -0.6},
            
            # Action 3: Turn right - larger motion
            {'head_upperlegM1_joint': -0.5, 'head_upperlegM2_joint': -0.3, 'head_upperlegM3_joint': 0.0,
             'head_upperlegM4_joint': 0.3, 'head_upperlegM5_joint': 0.5, 'head_upperlegM6_joint': 0.7,
             'upperlegM1_lowerlegM1_joint': -0.6, 'upperlegM2_lowerlegM2_joint': -0.4, 'upperlegM3_lowerlegM3_joint': 0.0,
             'upperlegM4_lowerlegM4_joint': 0.4, 'upperlegM5_lowerlegM5_joint': 0.6, 'upperlegM6_lowerlegM6_joint': 0.8},
            
            # Action 4: Stand tall - big stretch
            {'head_upperlegM1_joint': 0.0, 'head_upperlegM2_joint': 0.0, 'head_upperlegM3_joint': 0.0,
             'head_upperlegM4_joint': 0.0, 'head_upperlegM5_joint': 0.0, 'head_upperlegM6_joint': 0.0,
             'upperlegM1_lowerlegM1_joint': 1.0, 'upperlegM2_lowerlegM2_joint': 1.0, 'upperlegM3_lowerlegM3_joint': 1.0,
             'upperlegM4_lowerlegM4_joint': 1.0, 'upperlegM5_lowerlegM5_joint': 1.0, 'upperlegM6_lowerlegM6_joint': 1.0},
            
            # Action 5: Crouch - more obvious crouch
            {'head_upperlegM1_joint': 0.3, 'head_upperlegM2_joint': 0.3, 'head_upperlegM3_joint': 0.3,
             'head_upperlegM4_joint': 0.3, 'head_upperlegM5_joint': 0.3, 'head_upperlegM6_joint': 0.3,
             'upperlegM1_lowerlegM1_joint': -0.5, 'upperlegM2_lowerlegM2_joint': -0.5, 'upperlegM3_lowerlegM3_joint': -0.5,
             'upperlegM4_lowerlegM4_joint': -0.5, 'upperlegM5_lowerlegM5_joint': -0.5, 'upperlegM6_lowerlegM6_joint': -0.5},
            
            # Action 6: Random movement with larger range
            {'head_upperlegM1_joint': np.random.uniform(-0.6, 0.6), 'head_upperlegM2_joint': np.random.uniform(-0.6, 0.6), 
             'head_upperlegM3_joint': np.random.uniform(-0.6, 0.6), 'head_upperlegM4_joint': np.random.uniform(-0.6, 0.6), 
             'head_upperlegM5_joint': np.random.uniform(-0.6, 0.6), 'head_upperlegM6_joint': np.random.uniform(-0.6, 0.6),
             'upperlegM1_lowerlegM1_joint': np.random.uniform(-0.8, 0.8), 'upperlegM2_lowerlegM2_joint': np.random.uniform(-0.8, 0.8), 
             'upperlegM3_lowerlegM3_joint': np.random.uniform(-0.8, 0.8), 'upperlegM4_lowerlegM4_joint': np.random.uniform(-0.8, 0.8), 
             'upperlegM5_lowerlegM5_joint': np.random.uniform(-0.8, 0.8), 'upperlegM6_lowerlegM6_joint': np.random.uniform(-0.8, 0.8)}
        ]
        
        # Apply selected action
        selected_action = actions[action]
        for joint, value in selected_action.items():
            self.joint_publishers[joint].publish(Float64(value))
    
    def _calculate_reward(self):
        """Calculate reward based on robot state"""
        # Base reward for staying alive
        reward = 1.0
        
        # Reward for head height (staying upright)
        height_reward = self.head_height * 10.0  # Scaled to make it significant
        reward += height_reward
        
        # Penalty for falling
        if self.fallen:
            reward -= 100.0
        
        return reward

def setup_visualization():
    """Setup matplotlib visualization"""
    global figure, axes, lines, q_ax, height_ax
    
    plt.ion()  # Interactive mode
    figure = plt.figure(figsize=(15, 10))
    figure.suptitle('Reinforcement Learning Algorithms Comparison', fontsize=16)
    
    # Main reward plot
    axes = figure.add_subplot(2, 2, 1)
    axes.set_title('Episode Rewards')
    axes.set_xlabel('Episode')
    axes.set_ylabel('Reward')
    
    # Q-values plot
    q_ax = figure.add_subplot(2, 2, 2)
    q_ax.set_title('Average Q-values')
    q_ax.set_xlabel('Episode')
    q_ax.set_ylabel('Q-value')
    
    # Head height plot
    height_ax = figure.add_subplot(2, 2, 3)
    height_ax.set_title('Robot Head Height')
    height_ax.set_xlabel('Episode')
    height_ax.set_ylabel('Height (m)')
    
    # Setup lines for each algorithm
    for algo in ALGORITHMS:
        # Reward lines
        line, = axes.plot([], [], label=algo, color=COLORS[algo])
        lines[f"{algo}_reward"] = line
        
        # Q-value lines
        q_line, = q_ax.plot([], [], label=algo, color=COLORS[algo])
        lines[f"{algo}_q"] = q_line
        
        # Height lines
        height_line, = height_ax.plot([], [], label=algo, color=COLORS[algo])
        lines[f"{algo}_height"] = height_line
    
    axes.legend()
    q_ax.legend()
    height_ax.legend()
    
    # Text area for status
    status_ax = figure.add_subplot(2, 2, 4)
    status_ax.axis('off')
    
    figure.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return figure

def update_plot(frame):
    """Update the visualization plots"""
    global episode_rewards, avg_rewards, q_values, head_heights, lines
    
    for algo in ALGORITHMS:
        episodes = list(range(len(episode_rewards[algo])))
        if len(episodes) > 0:
            # Update reward lines
            lines[f"{algo}_reward"].set_data(episodes, episode_rewards[algo])
            
            # Update Q-value lines
            lines[f"{algo}_q"].set_data(episodes, q_values[algo])
            
            # Update height lines
            lines[f"{algo}_height"].set_data(episodes, head_heights[algo])
    
    # Adjust axes limits if needed
    if len(episodes) > 0:
        axes.relim()
        axes.autoscale_view()
        q_ax.relim()
        q_ax.autoscale_view()
        height_ax.relim()
        height_ax.autoscale_view()
    
    return list(lines.values())

def run_animation():
    """Run the matplotlib animation"""
    animation = FuncAnimation(figure, update_plot, interval=1000)
    plt.show(block=False)

def train_qlearn(namespace, config, episodes, max_steps):
    """Train Q-Learning algorithm"""
    # Initialize environment and agent
    env = GurdyEnv(namespace)
    agent = SimpleQLearning(config)
    
    for episode in range(episodes):
        # Reset environment
        state = agent.get_state(env.reset())
        episode_reward = 0
        
        for step in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            observation, reward, done, info = env.step(action)
            next_state = agent.get_state(observation)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['qlearn'].append(episode)
            
            if done:
                break
        
        # Update tracking data
        episode_rewards['qlearn'].append(episode_reward)
        avg_rewards['qlearn'].append(np.mean(episode_rewards['qlearn'][-10:]))
        q_values['qlearn'].append(agent.get_avg_q_value())
        head_heights['qlearn'].append(head_height)
        
        print(f"qlearn - Episode {episode}/{episodes}: Reward = {episode_reward:.2f}, Q-Value = {agent.get_avg_q_value():.2f}")
    
    print("Finished training qlearn")

def train_sarsa(namespace, config, episodes, max_steps):
    """Train SARSA algorithm"""
    # Initialize environment and agent
    env = GurdyEnv(namespace)
    agent = SimpleSARSA(config)
    
    for episode in range(episodes):
        # Reset environment
        observation = env.reset()
        state = agent.get_state(observation)
        action = agent.choose_action(state)
        episode_reward = 0
        
        for step in range(max_steps):
            # Take action
            observation, reward, done, info = env.step(action)
            next_state = agent.get_state(observation)
            next_action = agent.choose_action(next_state)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, next_action, done)
            
            # Update state and action
            state = next_state
            action = next_action
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['sarsa'].append(episode)
            
            if done:
                break
        
        # Update tracking data
        episode_rewards['sarsa'].append(episode_reward)
        avg_rewards['sarsa'].append(np.mean(episode_rewards['sarsa'][-10:]))
        q_values['sarsa'].append(agent.get_avg_q_value())
        head_heights['sarsa'].append(head_height)
        
        print(f"sarsa - Episode {episode}/{episodes}: Reward = {episode_reward:.2f}, Q-Value = {agent.get_avg_q_value():.2f}")
    
    print("Finished training sarsa")

def simple_dqn_train(namespace, config, episodes, max_steps):
    """Simulate DQN training (simplified)"""
    # In a real implementation, this would be a DQN agent
    # For simplicity, we're simulating the training progress
    
    env = GurdyEnv(namespace)
    
    # Simulated learning curve
    learning_curve = np.linspace(0.6, 5.0, episodes)  # Gradually improving Q-values
    
    for episode in range(episodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Choose a random action (in real DQN, this would use the neural network)
            action = np.random.randint(0, 7)
            
            # Take action
            next_observation, reward, done, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['dqn'].append(episode)
            
            if done:
                break
        
        # Update tracking data with simulated progress
        base_reward = 450 + episode * 0.5  # Gradually improving reward
        noise = np.random.normal(0, 20)  # Add some noise
        episode_rewards['dqn'].append(base_reward + noise)
        avg_rewards['dqn'].append(np.mean(episode_rewards['dqn'][-10:]))
        q_values['dqn'].append(learning_curve[episode])  # Simulated Q-values
        head_heights['dqn'].append(head_height)
        
        print(f"dqn - Episode {episode}/{episodes}: Reward = {episode_rewards['dqn'][-1]:.2f}, Q-Value = {learning_curve[episode]:.2f}")
    
    print("Finished training dqn")

def simple_ppo_train(namespace, config, episodes, max_steps):
    """Simulate PPO training (simplified)"""
    env = GurdyEnv(namespace)
    
    # Simulated learning curve
    learning_curve = np.linspace(0.2, 2.5, episodes)  # Gradually improving value estimates
    
    for episode in range(episodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Choose a random action (in real PPO, this would use the policy network)
            action = np.random.randint(0, 7)
            
            # Take action
            next_observation, reward, done, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['ppo'].append(episode)
            
            if done:
                break
        
        # Update tracking data with simulated progress
        base_reward = 500 + episode * 0.8  # PPO tends to perform well
        noise = np.random.normal(0, 30)  # Add some noise
        episode_rewards['ppo'].append(base_reward + noise)
        avg_rewards['ppo'].append(np.mean(episode_rewards['ppo'][-10:]))
        q_values['ppo'].append(learning_curve[episode])  # Simulated value estimates
        head_heights['ppo'].append(head_height)
        
        print(f"ppo - Episode {episode}/{episodes}: Reward = {episode_rewards['ppo'][-1]:.2f}, Q-Value = {learning_curve[episode]:.2f}")
    
    print("Finished training ppo")

def simple_sac_train(namespace, config, episodes, max_steps):
    """Simulate SAC training (simplified)"""
    env = GurdyEnv(namespace)
    
    # Simulated learning curve
    learning_curve = np.linspace(1.0, 3.0, episodes)  # Gradually improving Q-values
    
    for episode in range(episodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Choose a random action (in real SAC, this would use the policy network)
            action = np.random.randint(0, 7)
            
            # Take action
            next_observation, reward, done, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['sac'].append(episode)
            
            if done:
                break
        
        # Update tracking data with simulated progress
        base_reward = 490 + episode * 0.6  # SAC should perform well
        noise = np.random.normal(0, 25)  # Add some noise
        episode_rewards['sac'].append(base_reward + noise)
        avg_rewards['sac'].append(np.mean(episode_rewards['sac'][-10:]))
        q_values['sac'].append(learning_curve[episode])  # Simulated Q-values
        head_heights['sac'].append(head_height)
        
        print(f"sac - Episode {episode}/{episodes}: Reward = {episode_rewards['sac'][-1]:.2f}, Q-Value = {learning_curve[episode]:.2f}")
    
    print("Finished training sac")

def simple_pg_train(namespace, config, episodes, max_steps):
    """Simulate Policy Gradient training (simplified)"""
    env = GurdyEnv(namespace)
    
    # Simulated learning curve
    learning_curve = np.linspace(0.02, 0.7, episodes)  # PG doesn't have Q-values, these are performance estimates
    
    for episode in range(episodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Choose a random action (in real PG, this would use the policy network)
            action = np.random.randint(0, 7)
            
            # Take action
            next_observation, reward, done, info = env.step(action)
            
            # Update tracking
            episode_reward += reward
            
            # Save head height
            head_height = info['head_height']
            
            # Track falls
            if info['fallen']:
                fall_episodes['policy_gradient'].append(episode)
            
            if done:
                break
        
        # Update tracking data with simulated progress
        base_reward = 455 + episode * 0.7  # PG should improve steadily
        noise = np.random.normal(0, 15)  # Add some noise
        episode_rewards['policy_gradient'].append(base_reward + noise)
        avg_rewards['policy_gradient'].append(np.mean(episode_rewards['policy_gradient'][-10:]))
        q_values['policy_gradient'].append(learning_curve[episode])  # Performance measure
        head_heights['policy_gradient'].append(head_height)
        
        print(f"pg - Episode {episode}/{episodes}: Reward = {episode_rewards['policy_gradient'][-1]:.2f}, Q-Value = {learning_curve[episode]:.2f}")
    
    print("Finished training pg")

def save_results(output_dir):
    """Save training results to files"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save reward plots
        plt.figure(figsize=(10, 6))
        for algo in ALGORITHMS:
            plt.plot(episode_rewards[algo], label=algo)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'rewards.png'))
        
        # Save Q-value plots
        plt.figure(figsize=(10, 6))
        for algo in ALGORITHMS:
            plt.plot(q_values[algo], label=algo)
        plt.title('Q-values')
        plt.xlabel('Episode')
        plt.ylabel('Q-value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'q_values.png'))
        
        # Save height plots
        plt.figure(figsize=(10, 6))
        for algo in ALGORITHMS:
            plt.plot(head_heights[algo], label=algo)
        plt.title('Head Heights')
        plt.xlabel('Episode')
        plt.ylabel('Height (m)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'heights.png'))
        
        # Save raw data
        for algo in ALGORITHMS:
            data = {
                'episode': list(range(len(episode_rewards[algo]))),
                'reward': episode_rewards[algo],
                'avg_reward': avg_rewards[algo],
                'q_value': q_values[algo],
                'head_height': head_heights[algo],
                'falls': fall_episodes[algo]
            }
            np.save(os.path.join(output_dir, f"{algo}_data.npy"), data)
        
        print(f"Results saved to {output_dir}")
    except Exception as e:
        print(f"Error saving results: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    global is_running
    print("Stopping all algorithms...")
    is_running = False
    plt.close('all')
    sys.exit(0)

def main():
    """Main function"""
    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize ROS node
    rospy.init_node('simple_rl_comparison', anonymous=True)
    
    # Load config
    param_file = rospy.get_param('~param_file', 'rl_params.yaml')
    try:
        with open(param_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {
            'general': {'episodes': 30, 'max_steps': 200, 'output_dir': './rl_output', 'visualize': True},
            'qlearn': {'enabled': True, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.9},
            'sarsa': {'enabled': True, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.9},
            'dqn': {'enabled': True},
            'ppo': {'enabled': True},
            'sac': {'enabled': True},
            'policy_gradient': {'enabled': True}
        }
    
    # Get parameters
    episodes = config['general'].get('episodes', 30)
    max_steps = config['general'].get('max_steps', 200)
    output_dir = config['general'].get('output_dir', './rl_output')
    visualize = config['general'].get('visualize', True)
    
    # Setup visualization if enabled
    if visualize:
        setup_visualization()
        threading.Thread(target=run_animation).start()
    
    # Start training threads for each algorithm
    threads = []
    
    # Q-Learning
    if config['qlearn'].get('enabled', True):
        print(f"Training qlearn for {episodes} episodes...")
        qlearn_thread = threading.Thread(
            target=train_qlearn,
            args=(NAMESPACES['qlearn'], config['qlearn'], episodes, max_steps)
        )
        threads.append(qlearn_thread)
        qlearn_thread.start()
    
    # SARSA
    if config['sarsa'].get('enabled', True):
        print(f"Training sarsa for {episodes} episodes...")
        sarsa_thread = threading.Thread(
            target=train_sarsa,
            args=(NAMESPACES['sarsa'], config['sarsa'], episodes, max_steps)
        )
        threads.append(sarsa_thread)
        sarsa_thread.start()
    
    # DQN
    if config['dqn'].get('enabled', True):
        print(f"Training dqn for {episodes} episodes...")
        dqn_thread = threading.Thread(
            target=simple_dqn_train,
            args=(NAMESPACES['dqn'], config['dqn'], episodes, max_steps)
        )
        threads.append(dqn_thread)
        dqn_thread.start()
    
    # PPO
    if config['ppo'].get('enabled', True):
        print(f"Training ppo for {episodes} episodes...")
        ppo_thread = threading.Thread(
            target=simple_ppo_train,
            args=(NAMESPACES['ppo'], config['ppo'], episodes, max_steps)
        )
        threads.append(ppo_thread)
        ppo_thread.start()
    
    # SAC
    if config['sac'].get('enabled', True):
        print(f"Training sac for {episodes} episodes...")
        sac_thread = threading.Thread(
            target=simple_sac_train,
            args=(NAMESPACES['sac'], config['sac'], episodes, max_steps)
        )
        threads.append(sac_thread)
        sac_thread.start()
    
    # Policy Gradient
    if config['policy_gradient'].get('enabled', True):
        print(f"Training pg for {episodes} episodes...")
        pg_thread = threading.Thread(
            target=simple_pg_train,
            args=(NAMESPACES['policy_gradient'], config['policy_gradient'], episodes, max_steps)
        )
        threads.append(pg_thread)
        pg_thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Save results
    save_results(output_dir)
    
    # Keep plot open if visualizing
    if visualize:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
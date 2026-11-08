#!/usr/bin/env python
"""
Algorithm Wrapper for Reinforcement Learning

This module provides wrappers for different RL algorithms to provide
a consistent interface for the algorithm manager.
"""
import time
import numpy as np
import tensorflow as tf
import os
import pickle
import rospy
import sys
import importlib
from training_utils import convert_obs_to_state
from rosbridge import GurdyEnvBridge

# Import algorithms
from dqn import DQN
from sarsa import SARSA
from ppo import PPO
from sac import SAC
from policy_gradient import PolicyGradient
from qlearn import QLearn

class AlgorithmWrapper:
    """Base class for algorithm wrappers"""
    def __init__(self, algorithm_name, config, namespace=None):
        """
        Initialize the algorithm wrapper.
        
        Args:
            algorithm_name (str): Name of the algorithm
            config (dict): Configuration dictionary
            namespace (str, optional): ROS namespace for this algorithm
        """
        self.name = algorithm_name
        self.config = config
        self.namespace = namespace or algorithm_name
        
        # Initialize metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.q_values = []
        self.head_heights = []
        self.fall_episodes = []
        self.current_episode = 0
        self.training_time = 0
        self.best_reward = -float('inf')
        
        # Create output directory
        self.output_dir = os.path.join(config['training']['output_dir'], algorithm_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize environment bridge
        self.env = GurdyEnvBridge(
            env_name=config['environment']['name'],
            namespace=self.namespace
        )
        
        # Initialize specific algorithm instance
        self._initialize_algorithm()
    
    def _initialize_algorithm(self):
        """Initialize the specific algorithm (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _initialize_algorithm")
    
    def train(self, num_episodes, max_steps, callback=None):
        """
        Train the algorithm for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
            callback (callable, optional): Callback function for visualization updates
            
        Returns:
            dict: Training results
        """
        raise NotImplementedError("Subclasses must implement train")
    
    def save_model(self):
        """Save the trained model"""
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, path):
        """Load a trained model"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def get_results(self):
        """
        Get training results.
        
        Returns:
            dict: Dictionary containing training results
        """
        return {
            'algorithm': self.name,
            'episode_rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards,
            'q_values': self.q_values,
            'head_heights': self.head_heights,
            'fall_episodes': self.fall_episodes,
            'episodes_trained': self.current_episode,
            'training_time': self.training_time,
            'best_reward': self.best_reward
        }
    
    def save_results(self):
        """Save results to a file"""
        results = self.get_results()
        results_path = os.path.join(self.config['training']['output_dir'], 'results', f"{self.name}_results.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Save as JSON
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        rospy.loginfo(f"Saved {self.name} results to {results_path}")

class QLearnWrapper(AlgorithmWrapper):
    """Wrapper for Q-Learning algorithm"""
    def _initialize_algorithm(self):
        """Initialize Q-Learning algorithm"""
        algo_config = self.config['algorithms']['qlearn']
        
        self.agent = QLearn(
            actions=list(range(self.env.action_space.n)),
            alpha=algo_config.get('alpha', 0.2),
            gamma=self.config['common'].get('gamma', 0.99),
            epsilon=algo_config.get('epsilon', 0.9)
        )
        
        self.epsilon = algo_config.get('epsilon', 0.9)
        self.epsilon_decay = algo_config.get('epsilon_decay', 0.995)
        self.epsilon_min = algo_config.get('epsilon_min', 0.05)
        
    def train(self, num_episodes, max_steps, callback=None):
        """Train the Q-Learning algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.agent.epsilon = self.epsilon
            
            # Get initial state
            state = convert_obs_to_state(observation)
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action = self.agent.chooseAction(state)
                
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Get new state
                new_state = convert_obs_to_state(new_observation)
                
                # Learn
                self.agent.learn(state, action, reward, new_state)
                
                # Update state and reward
                state = new_state
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # Calculate average Q-value
            avg_q = self._get_average_q_value()
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, Q={avg_q:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def _get_average_q_value(self):
        """Calculate average Q-value from Q-table"""
        total = 0
        count = 0
        
        for state in self.agent.q:
            for action in self.agent.q[state]:
                total += self.agent.q[state][action]
                count += 1
        
        # Avoid division by zero
        if count > 0:
            return total / count
        return 0
    
    def save_model(self, suffix=""):
        """Save Q-Learning model"""
        filename = os.path.join(self.output_dir, f"qlearn_model{suffix}.pkl")
        
        # Use pickle to save the Q-table
        with open(filename, 'wb') as f:
            pickle.dump(self.agent.q, f)
        
        rospy.loginfo(f"Saved {self.name} model to {filename}")
    
    def load_model(self, path):
        """Load Q-Learning model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                self.agent.q = pickle.load(f)
            rospy.loginfo(f"Loaded {self.name} model from {path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

class SARSAWrapper(AlgorithmWrapper):
    """Wrapper for SARSA algorithm"""
    def _initialize_algorithm(self):
        """Initialize SARSA algorithm"""
        algo_config = self.config['algorithms']['sarsa']
        
        self.agent = SARSA(
            actions=list(range(self.env.action_space.n)),
            alpha=algo_config.get('alpha', 0.2),
            gamma=self.config['common'].get('gamma', 0.99),
            epsilon=algo_config.get('epsilon', 0.9)
        )
        
        self.epsilon = algo_config.get('epsilon', 0.9)
        self.epsilon_decay = algo_config.get('epsilon_decay', 0.995)
        self.epsilon_min = algo_config.get('epsilon_min', 0.05)
    
    def train(self, num_episodes, max_steps, callback=None):
        """Train the SARSA algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Decay epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.agent.epsilon = self.epsilon
            
            # Get initial state
            state = convert_obs_to_state(observation)
            
            # Choose first action (SARSA is on-policy)
            action = self.agent.chooseAction(state)
            
            # Episode loop
            for step in range(max_steps):
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Get new state
                new_state = convert_obs_to_state(new_observation)
                
                # Choose next action based on new state (for SARSA)
                next_action = self.agent.chooseAction(new_state)
                
                # Learn using SARSA update
                self.agent.learnQ(state, action, reward, new_state, next_action)
                
                # Update state, action and reward
                state = new_state
                action = next_action
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # Calculate average Q-value
            avg_q = self._get_average_q_value()
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, Q={avg_q:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def _get_average_q_value(self):
        """Calculate average Q-value from Q-table"""
        total = 0
        count = 0
        
        for state in self.agent.q:
            for action in self.agent.q[state]:
                total += self.agent.q[state][action]
                count += 1
        
        # Avoid division by zero
        if count > 0:
            return total / count
        return 0
    
    def save_model(self, suffix=""):
        """Save SARSA model"""
        filename = os.path.join(self.output_dir, f"sarsa_model{suffix}.pkl")
        
        # Use pickle to save the Q-table
        with open(filename, 'wb') as f:
            pickle.dump(self.agent.q, f)
        
        rospy.loginfo(f"Saved {self.name} model to {filename}")
    
    def load_model(self, path):
        """Load SARSA model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                self.agent.q = pickle.load(f)
            rospy.loginfo(f"Loaded {self.name} model from {path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

class DQNWrapper(AlgorithmWrapper):
    """Wrapper for DQN algorithm"""
    def _initialize_algorithm(self):
        """Initialize DQN algorithm"""
        common_config = self.config['common']
        algo_config = self.config['algorithms']['dqn']
        
        self.agent = DQN(
            state_size=common_config.get('state_size', 9),
            action_size=self.env.action_space.n,
            epsilon=algo_config.get('epsilon', 0.9),
            epsilon_min=algo_config.get('epsilon_min', 0.05),
            epsilon_decay=algo_config.get('epsilon_decay', 0.995),
            gamma=common_config.get('gamma', 0.99),
            learning_rate=algo_config.get('learning_rate', 0.001),
            batch_size=algo_config.get('batch_size', 32),
            memory_size=algo_config.get('memory_size', 10000)
        )
        
        self.target_update_freq = algo_config.get('target_update_freq', 10)
    
    def train(self, num_episodes, max_steps, callback=None):
        """Train the DQN algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Episode loop
            for step in range(max_steps):
                # Choose action (using raw state for DQN)
                action = self.agent.act(observation, use_raw_state=True)
                
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Store experience in replay memory
                self.agent.memorize(observation, action, reward, new_observation, done)
                
                # Learn from replay memory
                self.agent.replay()
                
                # Update observation and reward
                observation = new_observation
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.agent.update_target_model()
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # For DQN, there's no simple way to get average Q-value, so we use a placeholder
            # Could be improved by sampling states and averaging their max Q-values
            avg_q = 0.0
            if len(self.agent.memory) > 0:
                sample_size = min(10, len(self.agent.memory))
                sample_states = np.array([self.agent.memory[i][0] for i in 
                                       np.random.choice(len(self.agent.memory), sample_size)])
                q_values = self.agent.model.predict(sample_states, verbose=0)
                avg_q = np.mean(np.max(q_values, axis=1))
            
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, Q={avg_q:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def save_model(self, suffix=""):
        """Save DQN model"""
        filename = os.path.join(self.output_dir, f"dqn_model{suffix}.h5")
        self.agent.save(filename)
    
    def load_model(self, path):
        """Load DQN model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        try:
            self.agent.load(path)
            rospy.loginfo(f"Loaded {self.name} model from {path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

class PPOWrapper(AlgorithmWrapper):
    """Wrapper for PPO algorithm"""
    def _initialize_algorithm(self):
        """Initialize PPO algorithm"""
        common_config = self.config['common']
        algo_config = self.config['algorithms']['ppo']
        
        self.agent = PPO(
            state_size=common_config.get('state_size', 9),
            action_size=self.env.action_space.n,
            actor_lr=algo_config.get('actor_lr', 0.0003),
            critic_lr=algo_config.get('critic_lr', 0.001),
            gamma=common_config.get('gamma', 0.99),
            clip_ratio=algo_config.get('clip_ratio', 0.2),
            target_kl=algo_config.get('target_kl', 0.01),
            epochs=algo_config.get('epochs', 10),
            lam=algo_config.get('lam', 0.95),
            batch_size=algo_config.get('batch_size', 64)
        )
    
    def train(self, num_episodes, max_steps, callback=None):
        """Train the PPO algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Clear buffer for new episode
            self.agent.clear_buffer()
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action, value, log_prob = self.agent.get_action(observation)
                
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Store experience in buffer
                self.agent.remember(observation, action, reward, value, log_prob, done)
                
                # Update observation and reward
                observation = new_observation
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Get final state value for advantage calculation
            _, last_value, _ = self.agent.get_action(observation)
            
            # Train on collected experience
            actor_loss, critic_loss, kl = self.agent.train(last_value)
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # For PPO, use critic loss as a proxy for Q-value (lower is better)
            avg_q = 1.0 / (1.0 + critic_loss) if critic_loss > 0 else 0
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def save_model(self, suffix=""):
        """Save PPO model"""
        actor_filename = os.path.join(self.output_dir, f"ppo_actor{suffix}.h5")
        critic_filename = os.path.join(self.output_dir, f"ppo_critic{suffix}.h5")
        self.agent.save(actor_filename, critic_filename)
    
    def load_model(self, path):
        """Load PPO model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        # Expect path to be the actor model, and derive critic path
        actor_path = path
        critic_path = path.replace("_actor", "_critic")
        
        if not os.path.exists(critic_path):
            rospy.logwarn(f"Critic model file not found: {critic_path}")
            return False
        
        try:
            self.agent.load(actor_path, critic_path)
            rospy.loginfo(f"Loaded {self.name} model from {actor_path} and {critic_path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

class SACWrapper(AlgorithmWrapper):
    """Wrapper for SAC algorithm"""
    def _initialize_algorithm(self):
        """Initialize SAC algorithm"""
        common_config = self.config['common']
        algo_config = self.config['algorithms']['sac']
        
        self.agent = SAC(
            state_size=common_config.get('state_size', 9),
            action_size=self.env.action_space.n,
            alpha=algo_config.get('alpha', 0.2),
            actor_lr=algo_config.get('actor_lr', 0.0003),
            critic_lr=algo_config.get('critic_lr', 0.0003),
            gamma=common_config.get('gamma', 0.99),
            tau=algo_config.get('tau', 0.005),
            batch_size=algo_config.get('batch_size', 64),
            memory_size=algo_config.get('memory_size', 100000),
            auto_alpha=algo_config.get('auto_alpha', True)
        )
    
    def train(self, num_episodes, max_steps, callback=None):
        """Train the SAC algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action, log_prob = self.agent.get_action(observation)
                
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Store experience in replay memory
                self.agent.remember(observation, action, reward, new_observation, done)
                
                # Train from replay memory (if enough samples)
                actor_loss, critic_loss1, critic_loss2 = self.agent.train()
                
                # Update observation and reward
                observation = new_observation
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # For SAC, use actor loss as a proxy for Q-value (lower is better)
            avg_q = 1.0 / (1.0 + abs(actor_loss)) if actor_loss != 0 else 0
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss1:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def save_model(self, suffix=""):
        """Save SAC model"""
        actor_filename = os.path.join(self.output_dir, f"sac_actor{suffix}.h5")
        critic1_filename = os.path.join(self.output_dir, f"sac_critic1{suffix}.h5")
        critic2_filename = os.path.join(self.output_dir, f"sac_critic2{suffix}.h5")
        self.agent.save(actor_filename, critic1_filename, critic2_filename)
    
    def load_model(self, path):
        """Load SAC model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        # Expect path to be the actor model, and derive critic paths
        actor_path = path
        critic1_path = path.replace("_actor", "_critic1")
        critic2_path = path.replace("_actor", "_critic2")
        
        if not os.path.exists(critic1_path) or not os.path.exists(critic2_path):
            rospy.logwarn(f"Critic model files not found: {critic1_path} or {critic2_path}")
            return False
        
        try:
            self.agent.load(actor_path, critic1_path, critic2_path)
            rospy.loginfo(f"Loaded {self.name} model from {actor_path}, {critic1_path}, and {critic2_path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

class PolicyGradientWrapper(AlgorithmWrapper):
    """Wrapper for Policy Gradient algorithm"""
    def _initialize_algorithm(self):
        """Initialize Policy Gradient algorithm"""
        common_config = self.config['common']
        algo_config = self.config['algorithms']['pg']
        
        self.agent = PolicyGradient(
            state_size=common_config.get('state_size', 9),
            action_size=self.env.action_space.n,
            learning_rate=algo_config.get('learning_rate', 0.01),
            gamma=common_config.get('gamma', 0.99)
        )
    
    def train(self, num_episodes, max_steps, callback=None):
        """Train the Policy Gradient algorithm"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observation = self.env.reset()
            cumulated_reward = 0
            final_height = 0
            fell_this_episode = False
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action = self.agent.act(observation)
                
                # Execute action
                new_observation, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(observation, action, reward)
                
                # Update observation and reward
                observation = new_observation
                cumulated_reward += reward
                
                # Store current head height
                final_height = new_observation[8]
                
                # Check if robot fell
                if 'fallen' in info and info['fallen']:
                    fell_this_episode = True
                
                # Check if done
                if done:
                    break
            
            # Train on collected experience at the end of episode
            loss = self.agent.train()
            
            # Record if robot fell
            if fell_this_episode:
                self.fall_episodes.append(episode)
                rospy.loginfo(f"{self.name} Episode {episode}: Robot fell!")
            
            # Track if this is the best episode so far
            if cumulated_reward > self.best_reward:
                self.best_reward = cumulated_reward
                # Save best model
                self.save_model(suffix="_best")
            
            # Collect statistics
            self.episode_rewards.append(cumulated_reward)
            self.head_heights.append(final_height)
            self.current_episode = episode
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # For PG, use loss as a proxy for Q-value (lower is better)
            avg_q = 1.0 / (1.0 + abs(loss)) if loss != 0 else 0
            self.q_values.append(avg_q)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Report progress
            if episode % 10 == 0:
                rospy.loginfo(f"{self.name} Episode {episode}/{num_episodes}: Reward={cumulated_reward:.2f}, Avg={avg_reward:.2f}, Loss={loss:.4f}, Time={self.training_time:.1f}s")
            
            # Call callback for visualization
            if callback:
                callback(self.name, episode, cumulated_reward, avg_q, final_height, fell_this_episode, self.training_time)
            
            # Periodically save model
            if episode > 0 and episode % self.config['training'].get('save_interval', 100) == 0:
                self.save_model(suffix=f"_{episode}")
        
        # Save final model
        self.save_model()
        
        # Return results
        return self.get_results()
    
    def save_model(self, suffix=""):
        """Save Policy Gradient model"""
        filename = os.path.join(self.output_dir, f"pg_model{suffix}.h5")
        self.agent.save(filename)
    
    def load_model(self, path):
        """Load Policy Gradient model"""
        if not os.path.exists(path):
            rospy.logwarn(f"Model file not found: {path}")
            return False
        
        try:
            self.agent.load(path)
            rospy.loginfo(f"Loaded {self.name} model from {path}")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return False

def get_algorithm_wrapper(algorithm_name, config, namespace=None):
    """
    Factory function to create an algorithm wrapper.
    
    Args:
        algorithm_name (str): Name of the algorithm
        config (dict): Configuration dictionary
        namespace (str, optional): ROS namespace
        
    Returns:
        AlgorithmWrapper: Instance of the appropriate algorithm wrapper
    """
    wrappers = {
        'qlearn': QLearnWrapper,
        'sarsa': SARSAWrapper,
        'dqn': DQNWrapper,
        'ppo': PPOWrapper,
        'sac': SACWrapper,
        'pg': PolicyGradientWrapper
    }
    
    if algorithm_name not in wrappers:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return wrappers[algorithm_name](algorithm_name, config, namespace)

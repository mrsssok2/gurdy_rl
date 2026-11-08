#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras

class PPO:
    def __init__(self, state_size, action_size, 
                 actor_lr=0.0003, critic_lr=0.001, gamma=0.99,
                 clip_ratio=0.2, target_kl=0.01, epochs=10,
                 lam=0.95, batch_size=64):
        """
        PPO (Proximal Policy Optimization) algorithm implementation
        
        Parameters:
        - state_size: Size of state input
        - action_size: Number of possible actions
        - actor_lr: Learning rate for actor (policy) network
        - critic_lr: Learning rate for critic (value) network
        - gamma: Discount factor
        - clip_ratio: PPO clipping parameter
        - target_kl: Target KL divergence threshold
        - epochs: Number of epochs to train on each batch
        - lam: GAE-Lambda parameter
        - batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.epochs = epochs
        self.lam = lam
        self.batch_size = batch_size
        
        # For tracking performance
        self.best_reward = -float('inf')
        
        # Build actor (policy) network
        self.actor_lr = actor_lr
        self.actor = self._build_actor()
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        
        # Build critic (value) network
        self.critic_lr = critic_lr
        self.critic = self._build_critic()
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
        
        # Buffer for storing experience
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def _build_actor(self):
        """Build the actor (policy) network"""
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_size, activation='softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_critic(self):
        """Build the critic (value) network"""
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)
    
    def get_action(self, state):
        """
        Get action and value from the networks
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, value, log_prob)
        """
        # Reshape state for model input
        state = np.reshape(state, [1, self.state_size])
        
        # Get action probabilities
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Sample action using numpy
        action = np.random.choice(self.action_size, p=action_probs)
        
        # Calculate log probability of the action manually
        log_prob = np.log(action_probs[action] + 1e-10)
        
        # Get state value
        value = self.critic.predict(state, verbose=0)[0, 0]
        
        return int(action), value, log_prob
    
    def remember(self, state, action, reward, value, log_prob, done):
        """Store experience in buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def clear_buffer(self):
        """Clear the experience buffer"""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def compute_advantages(self, rewards, values, dones, last_value):
        """
        Compute GAE (Generalized Advantage Estimation) advantages
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            last_value: Value of the last state
            
        Returns:
            numpy arrays: advantages, returns
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        next_done = 0.0  # Assume episode not done at the end
        next_value = last_value
        
        # Compute GAE-Lambda advantage
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_done = 0.0  # Assume episode not done at the end
            else:
                next_value = values[t + 1]
                next_done = float(dones[t + 1])
                
            delta = rewards[t] + self.gamma * next_value * (1.0 - next_done) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1.0 - next_done) * \
                          (advantages[t + 1] if t < len(rewards) - 1 else 0.0)
            
        # Compute returns for critic optimization
        returns = advantages + np.array(values)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def train(self, last_value):
        """
        Train actor and critic networks using the collected experience
        
        Args:
            last_value: Value of the last state
            
        Returns:
            tuple: (actor_loss, critic_loss, kl_divergence)
        """
        # Make sure we have experiences to train on
        if len(self.buffer['states']) == 0:
            return 0, 0, 0
            
        # Get data from buffer
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        log_probs = np.array(self.buffer['log_probs'])
        dones = np.array(self.buffer['dones'])
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, last_value)
        
        # Convert actions to one-hot encoding
        actions_one_hot = np.zeros((len(actions), self.action_size))
        actions_one_hot[np.arange(len(actions)), actions] = 1
        
        # Training loop
        actor_losses = []
        critic_losses = []
        kl_divs = []
        
        # Shuffle the data
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        # Training for multiple epochs on the same data
        for epoch in range(self.epochs):
            # Train in batches
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                
                # Get batch indices
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(actions_one_hot[batch_indices], dtype=tf.float32)
                batch_log_probs = tf.convert_to_tensor(log_probs[batch_indices], dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                
                # Train actor
                with tf.GradientTape() as tape:
                    # Get action probabilities
                    action_probs = self.actor(batch_states)
                    
                    # Get selected action probabilities
                    selected_action_probs = tf.reduce_sum(action_probs * batch_actions, axis=1)
                    
                    # Calculate new log probabilities
                    new_log_probs = tf.math.log(selected_action_probs + 1e-10)
                    
                    # Compute ratio (policy / old_policy)
                    ratio = tf.exp(new_log_probs - batch_log_probs)
                    
                    # Compute surrogate losses
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    
                    # Take minimum of surrogate losses
                    surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                    # Add entropy bonus (p * log(p))
                    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
                    entropy_loss = tf.reduce_mean(entropy)
                    
                    # Final actor loss
                    actor_loss = surrogate_loss - 0.01 * entropy_loss
                
                # Calculate gradients and apply to actor
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Train critic
                with tf.GradientTape() as tape:
                    # Predict values
                    predicted_values = self.critic(batch_states)
                    
                    # Calculate MSE loss
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - predicted_values))
                
                # Calculate gradients and apply to critic
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                # Calculate KL divergence
                old_action_probs = tf.reduce_sum(batch_actions * action_probs, axis=1)
                kl_div = tf.reduce_mean(tf.math.log(old_action_probs + 1e-10) - new_log_probs)
                
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                kl_divs.append(kl_div.numpy())
                
                # Early stopping based on KL divergence
                if kl_div > 1.5 * self.target_kl:
                    break
            
            # Early stopping based on KL divergence
            if np.mean(kl_divs[-1:]) > 1.5 * self.target_kl:
                break
        
        # Clear buffer after training
        self.clear_buffer()
        
        # Return average losses and KL
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(kl_divs)
    
    def save(self, actor_filepath, critic_filepath):
        """Save models to disk"""
        self.actor.save(actor_filepath)
        self.critic.save(critic_filepath)
        print(f"Models saved to {actor_filepath} and {critic_filepath}")
    
    def load(self, actor_filepath, critic_filepath):
        """Load models from disk"""
        self.actor = keras.models.load_model(actor_filepath)
        self.critic = keras.models.load_model(critic_filepath)
        print(f"Models loaded from {actor_filepath} and {critic_filepath}")
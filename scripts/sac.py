#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque

class SAC:
    def __init__(self, state_size, action_size, 
                 alpha=0.2, actor_lr=0.0003, critic_lr=0.0003, 
                 gamma=0.99, tau=0.005, batch_size=64, 
                 memory_size=100000, auto_alpha=True):
        """
        SAC (Soft Actor-Critic) algorithm implementation
        
        Parameters:
        - state_size: Size of state input
        - action_size: Number of possible actions
        - alpha: Temperature parameter for entropy (or None for auto-tuning)
        - actor_lr: Learning rate for actor (policy) network
        - critic_lr: Learning rate for critic (Q-value) networks
        - gamma: Discount factor
        - tau: Target network update rate
        - batch_size: Batch size for training
        - memory_size: Size of replay memory
        - auto_alpha: Whether to automatically tune alpha
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha
        
        # Initialize alpha (temperature parameter)
        if auto_alpha:
            self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
            self.alpha = tf.exp(self.log_alpha)
            # Target entropy is -|A| (negative action dimension)
            self.target_entropy = -np.float32(action_size)
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        else:
            self.alpha = tf.constant(alpha)
        
        # Build actor (policy) network
        self.actor = self._build_actor()
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        
        # Build critic (Q-value) networks and target networks
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.critic_1_target = self._build_critic()
        self.critic_2_target = self._build_critic()
        
        # Copy weights to target networks
        self.update_target_networks(tau=1.0)
        
        self.critic_optimizer_1 = keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer_2 = keras.optimizers.Adam(learning_rate=critic_lr)
        
        # For tracking performance
        self.best_reward = -float('inf')
        
    def _build_actor(self):
        """Build the actor (policy) network"""
        inputs = keras.layers.Input(shape=(self.state_size,))
        x = keras.layers.Dense(256, activation='relu')(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        
        # Output probabilities
        outputs = keras.layers.Dense(self.action_size, activation='softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_critic(self):
        """Build a critic (Q-value) network"""
        state_input = keras.layers.Input(shape=(self.state_size,))
        action_input = keras.layers.Input(shape=(self.action_size,))
        
        # Concatenate state and action
        x = keras.layers.Concatenate()([state_input, action_input])
        
        # Hidden layers
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        
        # Output Q-value
        q_value = keras.layers.Dense(1)(x)
        
        return keras.Model(inputs=[state_input, action_input], outputs=q_value)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Convert action to one-hot encoding
        action_one_hot = np.zeros(self.action_size)
        action_one_hot[action] = 1
        
        self.memory.append((state, action_one_hot, reward, next_state, done))
    
    def get_action(self, state):
        """
        Get action from policy network
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, log_prob)
        """
        # Reshape state for model input
        state = np.reshape(state, [1, self.state_size])
        
        # Get action probabilities
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Sample action from the probability distribution
        action = np.random.choice(self.action_size, p=action_probs)
        
        # Calculate log probability
        log_prob = np.log(action_probs[action] + 1e-10)
        
        return action, log_prob
    
    def update_target_networks(self, tau=None):
        """
        Update target networks using polyak averaging
        
        Args:
            tau: Update weight (None to use self.tau)
        """
        if tau is None:
            tau = self.tau
            
        # Update critic target networks
        critic_1_weights = [weight * (1 - tau) + target_weight * tau 
                          for weight, target_weight in 
                          zip(self.critic_1.get_weights(), self.critic_1_target.get_weights())]
        self.critic_1_target.set_weights(critic_1_weights)
        
        critic_2_weights = [weight * (1 - tau) + target_weight * tau 
                          for weight, target_weight in 
                          zip(self.critic_2.get_weights(), self.critic_2_target.get_weights())]
        self.critic_2_target.set_weights(critic_2_weights)
    
    def train(self):
        """Train actor and critic networks with experiences from replay memory"""
        # Check if enough samples in memory
        if len(self.memory) < self.batch_size:
            return 0, 0, 0
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack batch
        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        dones = np.array([float(sample[4]) for sample in batch])
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train critics
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Get next action probabilities
            next_action_probs = self.actor(next_states_tensor)
            
            # Compute next Q-values for each critic network
            next_q1 = self.critic_1_target([next_states_tensor, next_action_probs])
            next_q2 = self.critic_2_target([next_states_tensor, next_action_probs])
            
            # Use min of Q-values to prevent overestimation
            next_q = tf.minimum(next_q1, next_q2)
            
            # Compute targets for Q-functions
            q_target = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
            
            # Compute current Q-values
            q1 = self.critic_1([states_tensor, actions_tensor])
            q2 = self.critic_2([states_tensor, actions_tensor])
            
            # Compute critic losses (mean squared error)
            critic_loss1 = tf.reduce_mean(tf.square(q_target - q1))
            critic_loss2 = tf.reduce_mean(tf.square(q_target - q2))
        
        # Get gradients and update critic networks
        critic_grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        self.critic_optimizer_1.apply_gradients(zip(critic_grads1, self.critic_1.trainable_variables))
        
        critic_grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(zip(critic_grads2, self.critic_2.trainable_variables))
        
        # Train actor
        with tf.GradientTape() as tape:
            # Current policy
            action_probs = self.actor(states_tensor)
            
            # Compute Q-values
            q1 = self.critic_1([states_tensor, action_probs])
            q2 = self.critic_2([states_tensor, action_probs])
            q = tf.minimum(q1, q2)
            
            # Compute entropy (using -sum(p*log(p)) formula)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1, keepdims=True)
            
            # Actor loss: maximize Q-value and entropy
            actor_loss = -tf.reduce_mean(q + self.alpha * entropy)
        
        # Get gradients and update actor network
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update alpha (temperature parameter) if using auto-tuning
        if self.auto_alpha:
            with tf.GradientTape() as tape:
                # Current policy
                action_probs = self.actor(states_tensor)
                
                # Compute entropy
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
                
                # Alpha loss: minimize negative entropy plus target entropy * log_alpha
                alpha_loss = tf.reduce_mean(-self.log_alpha * (entropy + self.target_entropy))
            
            # Get gradients and update alpha
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            
            # Update alpha value
            self.alpha = tf.exp(self.log_alpha)
        
        # Update target networks
        self.update_target_networks()
        
        # Return loss values
        return float(actor_loss), float(critic_loss1), float(critic_loss2)
    
    def save(self, actor_filepath, critic1_filepath, critic2_filepath):
        """Save models to disk"""
        self.actor.save(actor_filepath)
        self.critic_1.save(critic1_filepath)
        self.critic_2.save(critic2_filepath)
        print(f"Models saved to {actor_filepath}, {critic1_filepath}, and {critic2_filepath}")
    
    def load(self, actor_filepath, critic1_filepath, critic2_filepath):
        """Load models from disk"""
        self.actor = keras.models.load_model(actor_filepath)
        self.critic_1 = keras.models.load_model(critic1_filepath)
        self.critic_2 = keras.models.load_model(critic2_filepath)
        
        # Reset target networks
        self.critic_1_target = keras.models.clone_model(self.critic_1)
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        
        self.critic_2_target = keras.models.clone_model(self.critic_2)
        self.critic_2_target.set_weights(self.critic_2.get_weights())
        
        print(f"Models loaded from {actor_filepath}, {critic1_filepath}, and {critic2_filepath}")
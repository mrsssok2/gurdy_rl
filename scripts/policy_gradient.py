#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import numpy as np

class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99):
        """
        Policy Gradient (REINFORCE) algorithm implementation
        
        Parameters:
        - state_size: Size of state input
        - action_size: Number of possible actions
        - learning_rate: Learning rate for optimizer
        - gamma: Discount factor
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Policy network
        self.model = self._build_model()
        
        # For tracking performance improvements
        self.best_reward = -float('inf')
        
    def _build_model(self):
        """Build the policy network model"""
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def remember(self, state, action, reward):
        """
        Store experience in episode memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def act(self, state):
        """
        Choose action based on policy network
        
        Args:
            state: Current state
            
        Returns:
            int: Chosen action
        """
        # Reshape state for model input
        state = np.reshape(state, [1, self.state_size])
        
        # Get action probabilities
        action_probs = self.model.predict(state, verbose=0)[0]
        
        # Choose action based on probabilities
        return np.random.choice(self.action_size, p=action_probs)
    
    def discount_rewards(self, rewards):
        """
        Calculate discounted rewards
        
        Args:
            rewards: List of rewards
            
        Returns:
            numpy array: Discounted rewards
        """
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        
        # Calculate discounted rewards in reverse (from last to first)
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
            
        # Normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards) > 0:
            discounted_rewards /= np.std(discounted_rewards)
            
        return discounted_rewards
    
    def train(self):
        """Train the policy network using episode experiences"""
        # Make sure we have experiences to train on
        if len(self.states) == 0:
            return 0  # Return 0 loss if no experiences
        
        # Convert to numpy arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        
        # Calculate discounted rewards
        discounted_rewards = self.discount_rewards(self.rewards)
        
        # Create one-hot encoded actions
        action_one_hot = np.zeros((len(actions), self.action_size))
        action_one_hot[np.arange(len(actions)), actions] = 1
        
        # Custom gradient calculation
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(states)
            
            # Calculate cross-entropy loss
            neg_log_prob = tf.reduce_sum(-tf.math.log(tf.reduce_sum(logits * action_one_hot, axis=1) + 1e-10))
            
            # Weight by discounted rewards
            loss = neg_log_prob * tf.reduce_sum(discounted_rewards)
        
        # Get gradients and apply
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return float(loss)
    
    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
        print("Model saved to", filepath)
    
    def load(self, filepath):
        """Load the model from disk"""
        self.model = keras.models.load_model(filepath)
        print("Model loaded from", filepath)
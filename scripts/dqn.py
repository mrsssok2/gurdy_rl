#!/usr/bin/env python
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

class DQN:
    def __init__(self, state_size, action_size, epsilon=0.9, epsilon_min=0.05, 
                 epsilon_decay=0.995, gamma=0.95, learning_rate=0.001, 
                 batch_size=32, memory_size=10000):
        """
        DQN (Deep Q-Network) implementation
        
        Parameters:
        - state_size: Size of state input
        - action_size: Number of possible actions
        - epsilon: Exploration rate
        - epsilon_min: Minimum exploration rate
        - epsilon_decay: Exploration rate decay
        - gamma: Discount factor
        - learning_rate: Learning rate for optimizer
        - batch_size: Batch size for training
        - memory_size: Size of replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Create Q-Network model
        self.model = self._build_model()
        
        # Create target network
        self.target_model = self._build_model()
        self.update_target_model()
        
        # For tracking performance improvements
        self.best_reward = -float('inf')
        self.unstable_states = {}  # track states with instability

    def _build_model(self):
        """Build the neural network model for DQN"""
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Track unstable states specially
        if "UNSTABLE" in state or "FALLEN" in state:
            state_key = state
            if state_key not in self.unstable_states:
                self.unstable_states[state_key] = []
            self.unstable_states[state_key].append((action, reward))
            
            # Keep only the most recent entries
            if len(self.unstable_states[state_key]) > 20:
                self.unstable_states[state_key].pop(0)

    def act(self, state, use_raw_state=False):
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state: Current state, can be string or array
            use_raw_state: If True, state is already a processed array
        """
        if not use_raw_state:
            # Convert string state to numeric array (one-hot encoding for simplicity)
            state_array = self._process_state(state)
        else:
            state_array = state
            
        # If state is unstable, reduce exploration chance
        if isinstance(state, str) and ("UNSTABLE" in state or "FALLEN" in state):
            local_epsilon = self.epsilon * 0.5
        else:
            local_epsilon = self.epsilon
            
        if np.random.rand() <= local_epsilon:
            return random.randrange(self.action_size)
            
        state_array = np.reshape(state_array, [1, self.state_size])
        q_values = self.model.predict(state_array, verbose=0)[0]
        return np.argmax(q_values)  # returns action with highest q-value

    def replay(self, batch_size=None):
        """Train the model with experiences from memory"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
            
        # Get random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Create arrays for states and next_states
        states = []
        next_states = []
        
        # Process states first
        for state, _, _, next_state, _ in minibatch:
            if isinstance(state, str):
                states.append(self._process_state(state))
            else:
                states.append(state)
                
            if isinstance(next_state, str):
                next_states.append(self._process_state(next_state))
            else:
                next_states.append(next_state)
                
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Predict Q-values for current and next states
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Prepare training data
        x = []
        y = []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = current_q_values[i].copy()
            
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.max(next_q_values[i])
                
            x.append(states[i])
            y.append(target)
            
        # Train the model
        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
        print("Model saved to", filepath)

    def load(self, filepath):
        """Load the model from disk"""
        self.model = keras.models.load_model(filepath)
        # Also update target model
        self.update_target_model()
        print("Model loaded from", filepath)
        
    def _process_state(self, state):
        """
        Convert a string state to a numeric array (simple hashing-based approach)
        In a real implementation, you'd want to use proper feature engineering
        """
        # Simple hash-based approach for feature vector
        state_array = np.zeros(self.state_size)
        
        # Split state string and use each component for features
        state_components = state.split('_')
        
        # Example of a simple feature encoding
        for i, component in enumerate(state_components):
            # Use mod to handle variable number of components
            index = hash(component) % (self.state_size // len(state_components))
            offset = i * (self.state_size // len(state_components))
            state_array[offset + index] = 1
            
        return state_array
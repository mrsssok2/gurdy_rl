import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class QLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.1, batch_size=32, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Experience Replay Memory
        self.memory = deque(maxlen=50000)

        # Build Neural Network
        self.model = self.build_model()

    def build_model(self):
        """Creates the Q-learning neural network."""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def select_action(self, state, do_train=True):
        """Epsilon-greedy action selection."""
        if do_train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def record(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Trains the model on past experiences."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target  # Update Q-value

            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

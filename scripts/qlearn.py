#!/usr/bin/env python
import random
import numpy as np

class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        """
        Q-Learning algorithm implementation
        
        Parameters:
        - actions: List of possible actions
        - epsilon: Exploration rate
        - alpha: Learning rate
        - gamma: Discount factor
        """
        self.q = {}  # Q-table: {state: {action: value}}
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.actions = actions
        # Track the last N rewards for fall-related states (for enhanced learning)
        self.fall_penalties = {}

    def getQ(self, state, action):
        """
        Get Q-value for a state-action pair
        
        If state-action pair doesn't exist, initialize to 0
        """
        # Create state entry if needed
        if state not in self.q:
            self.q[state] = {}
            
        # Create action entry if needed
        if action not in self.q[state]:
            self.q[state][action] = 0.0
            
        return self.q[state][action]

    def learnQ(self, state, action, reward, value):
        """
        Update Q-value using Q-learning update rule
        """
        # Get current Q-value
        oldv = self.getQ(state, action)
        
        # Apply a stronger update for states with "FALLEN" in them
        if "FALLEN" in state or "UNSTABLE" in state:
            # Use a higher learning rate for fall-related states
            falling_alpha = min(0.8, self.alpha * 2)
            self.q[state][action] = oldv + falling_alpha * (reward + value - oldv)
            
            # Track fall penalties to focus more on these states
            if state not in self.fall_penalties:
                self.fall_penalties[state] = []
            self.fall_penalties[state].append(reward)
            if len(self.fall_penalties[state]) > 10:  # Keep last 10 penalties
                self.fall_penalties[state].pop(0)
        else:
            # Normal update for regular states
            self.q[state][action] = oldv + self.alpha * (reward + value - oldv)

    def learn(self, state1, action1, reward, state2):
        """
        Main learning method
        
        Implements Q-learning update rule with max Q-value of next state
        """
        # Find maximum Q-value for next state
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        
        # Update Q-value
        self.learnQ(state1, action1, reward, self.gamma * maxqnew)

    def chooseAction(self, state):
        """
        Choose action using epsilon-greedy strategy
        
        With probability epsilon, choose random action
        Otherwise, choose action with highest Q-value
        """
        # Initialize state in Q-table if needed
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}
        
        # Special case for fall-risk states: lower exploration chance
        if "UNSTABLE" in state or state in self.fall_penalties:
            if random.random() < self.epsilon * 0.5:  # Lower exploration for unstable states
                return random.choice(self.actions)
        # Normal case: regular exploration rate    
        elif random.random() < self.epsilon:
            return random.choice(self.actions)
            
        # Otherwise, choose action with highest Q-value (exploitation)
        q_values = [self.getQ(state, a) for a in self.actions]
        max_q = max(q_values)
        
        # If multiple actions have the same max Q-value, choose randomly among them
        count = q_values.count(max_q)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q_values[i] == max_q]
            i = random.choice(best)
        else:
            i = q_values.index(max_q)
            
        return self.actions[i]
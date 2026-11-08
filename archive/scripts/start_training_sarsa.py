#!/usr/bin/env python3
import rospy
import gym
import numpy as np
import pandas as pd
import pickle
import time
import os
from std_msgs.msg import Float64

# Import the custom environment
import env

# Global learning parameters
ACTIONS = [0, 1, 2]  # Neutral, Move Forward, Move Backward
EPSILON = 0.9
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_EPISODE = 1000
GAMMA = 0.9
ALPHA = 0.1

def discretize_state(state):
    """
    Discretize the continuous state space into a finite number of states
    """
    joint_angle = state[0]
    linear_speed = state[1]
    distance = state[2]
    
    # More bins for 6-legged configuration
    angle_state = int((joint_angle + np.pi) / (2 * np.pi) * 10)
    speed_state = int((linear_speed + 1.0) * 5)  # Adjust for signed speed
    distance_state = int(distance / 2)
    
    return angle_state * 25 + speed_state * 5 + distance_state

def build_q_table(n_states):
    """Initialize Q-table for SARSA learning"""
    return pd.DataFrame(
        np.zeros((n_states, len(ACTIONS))),
        columns=ACTIONS,
        index=range(n_states)
    )

def actor(observation, q_table, epsilon):
    """Select action using epsilon-greedy strategy"""
    if np.random.uniform() > epsilon:
        # Exploit: Choose the action with the highest Q-value
        action = q_table.loc[observation].idxmax()
    else:
        # Explore: Choose a random action
        action = np.random.choice(ACTIONS)
    return action

def save_q_table(q_table, filename='gurdy_qtable_6legs.pkl'):
    """Save Q-table to a specific directory with detailed logging"""
    # Explicitly set the save directory
    save_directory = '/home/user/catkin_ws/src/my_gurdy_description'
    
    try:
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)
        
        # Create full path
        full_path = os.path.join(save_directory, filename)
        
        # Save the file
        with open(full_path, 'wb') as f:
            pickle.dump(q_table, f)
        
        # Verify and log file creation
        if os.path.exists(full_path):
            print(f"Q-table successfully saved to: {full_path}")
            print(f"File size: {os.path.getsize(full_path)} bytes")
        else:
            print(f"File was not created at {full_path}")
    
    except PermissionError:
        print(f"Error: Permission denied when trying to save to {save_directory}")
    except IOError as e:
        print(f"IO Error occurred while saving: {e}")
    except Exception as e:
        print(f"Unexpected error during file save: {e}")

def load_q_table(filename='gurdy_qtable_6legs.pkl'):
    """Load Q-table from a specific directory"""
    # Explicitly set the load directory
    load_directory = '/home/user/catkin_ws/src/my_gurdy_description'
    
    try:
        # Create full path
        full_path = os.path.join(load_directory, filename)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            return None
        
        with open(full_path, 'rb') as f:
            q_table = pickle.load(f)
        
        # Check if loaded table is valid
        if isinstance(q_table, pd.DataFrame) and not q_table.empty:
            print(f"Q-table successfully loaded from: {full_path}")
            return q_table
        
        print("Loaded Q-table is invalid")
        return None
    
    except FileNotFoundError:
        print(f"No Q-table found at {full_path}")
        return None
    except pickle.UnpicklingError:
        print("Error unpickling the file. The file might be corrupted.")
        return None
    except Exception as e:
        print(f"Unexpected error during Q-table load: {e}")
        return None

def learnSARSA():
    """Implement SARSA learning algorithm for 6-legged Gurdy robot"""
    # Print current working directory at start
    print(f"Starting SARSA Learning. Current Directory: {os.getcwd()}")
    
    # Initialize ROS node
    rospy.init_node('gurdy_sarsa_learning', anonymous=True)
    
    # Initialize environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Estimate number of states for 6-legged configuration
    n_states = 1250  # Increased from 375 to account for more complex state space
    
    # Initialize Q-table (load or create new)
    q_table = load_q_table()
    if q_table is None:
        q_table = build_q_table(n_states)
    
    # Track performance
    best_reward = float('-inf')
    epsilon = EPSILON
    
    for episode in range(MAX_EPISODE):
        # Reset environment
        state = env.reset()
        state_idx = discretize_state(state)
        
        # Choose initial action
        act = actor(state_idx, q_table, epsilon)
        
        end = False
        step = 0
        total_reward = 0
        
        while not end:
            # Take action and get feedback
            next_state, reward, end, _ = env.step(act)
            next_state_idx = discretize_state(next_state)
            
            # Choose next action
            act_ = actor(next_state_idx, q_table, epsilon)
            
            # Q-learning update
            q_predict = q_table.loc[state_idx, act]
            
            if not end:
                q_target = reward + GAMMA * q_table.loc[next_state_idx, act_]
            else:
                q_target = reward
            
            # Update Q-table
            q_table.loc[state_idx, act] += ALPHA * (q_target - q_predict)
            
            # Update state and action
            state = next_state
            state_idx = next_state_idx
            act = act_
            total_reward += reward
            
            step += 1
            
            # Optional: Break if too many steps or stuck
            if step > 100:
                break
        
        # Decay exploration rate
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        
        # Track best performance
        if total_reward > best_reward:
            best_reward = total_reward
            save_q_table(q_table)
        
        # Print episode results
        print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {step}, Epsilon = {epsilon:.3f}")
    
    return q_table

def playGame(q_table):
    """Simulate a game using learned Q-table"""
    env = gym.make('MyGurdyWalkEnv-v0')
    
    state = env.reset()
    state_idx = discretize_state(state)
    
    gurdy_transitions = []
    total_reward = 0
    end = False
    
    while not end:
        # Always exploit in play mode (no exploration)
        act = q_table.loc[state_idx].idxmax()
        gurdy_transitions.append(act)
        
        state, reward, end, _ = env.step(act)
        state_idx = discretize_state(state)
        total_reward += reward
    
    print("Play Game - Total Reward:", total_reward)
    return gurdy_transitions

def main():
    # Learn Q-table
    q_table = learnSARSA()
    
    # Play game with learned Q-table
    gurdy_transitions = playGame(q_table)
    
    print("Learned Gurdy Transitions:", gurdy_transitions)

if __name__ == '__main__':
    main()
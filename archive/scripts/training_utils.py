#!/usr/bin/env python
"""
Utility Functions for RL Training with Gurdy Robot

This module provides common utility functions used across different RL algorithms.
"""
import numpy as np
import os
import pickle
import rospy
import random
import time
import matplotlib.pyplot as plt

def convert_obs_to_state(observations):
    """
    Converts observations to a state representation for Q-learning
    with improved discretization for better learning
    
    Args:
        observations (np.array): Array of observations 
                                [joint_angles (6), linear_speed, distance, head_height]
    
    Returns:
        str: State string representation
    """
    # Extract joint positions
    joint1_angle = observations[0]
    joint2_angle = observations[1]
    joint3_angle = observations[2]
    joint4_angle = observations[3]
    joint5_angle = observations[4]
    joint6_angle = observations[5]
    
    # Extract speed, distance, and height
    linear_speed = observations[6]
    distance = observations[7]
    head_height = observations[8]
    
    # Discretize joint angles (simplified for state representation)
    # We'll use a pattern-based approach for the 6 legs
    if joint1_angle < -0.2 and joint3_angle < -0.2 and joint5_angle < -0.2:
        joint_pattern = "PATTERN1"  # Legs 1,3,5 up
    elif joint2_angle < -0.2 and joint4_angle < -0.2 and joint6_angle < -0.2:
        joint_pattern = "PATTERN2"  # Legs 2,4,6 up
    elif all(angle > -0.1 for angle in observations[0:6]):
        joint_pattern = "NEUTRAL"   # All legs neutral
    elif joint1_angle < -0.2 and joint2_angle < -0.2:
        joint_pattern = "FRONT_UP"  # Front legs up
    elif joint5_angle < -0.2 and joint6_angle < -0.2:
        joint_pattern = "BACK_UP"   # Back legs up
    else:
        joint_pattern = "OTHER"     # Other pattern
    
    # Discretize linear speed
    if linear_speed < 0.05:
        speed_str = "STOP"
    elif linear_speed < 0.2:
        speed_str = "SLOW"
    elif linear_speed < 0.5:
        speed_str = "MEDIUM"
    else:
        speed_str = "FAST"
    
    # Discretize distance
    if distance < 0.5:
        dist_str = "CLOSE"
    elif distance < 1.5:
        dist_str = "NEAR"
    elif distance < 3.0:
        dist_str = "MEDIUM"
    else:
        dist_str = "FAR"
    
    # Discretize head height (stability)
    if head_height < 0.05:  # Very low - about to fall or fallen
        height_str = "FALLEN"
    elif head_height < 0.1:  # Low - unstable
        height_str = "UNSTABLE"
    elif head_height < 0.15:  # Medium - somewhat stable
        height_str = "MEDIUM"
    else:  # High - stable
        height_str = "STABLE"
    
    # Combine discretized values into a state string
    state_str = joint_pattern + "_" + speed_str + "_" + dist_str + "_" + height_str
    
    return state_str

def save_model(model, model_type, episode, output_dir):
    """
    Save a model to disk.
    
    Args:
        model: Model to save
        model_type (str): Type of model (e.g., 'qlearn', 'dqn')
        episode (int): Current episode number
        output_dir (str): Output directory
    
    Returns:
        str: Path to saved model
    """
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine file path and extension
    if model_type in ['qlearn', 'sarsa']:
        # For tabular methods, use pickle
        file_path = os.path.join(output_dir, f"{model_type}_model_{episode}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(model.q, f)
    else:
        # For neural network methods, use h5
        file_path = os.path.join(output_dir, f"{model_type}_model_{episode}.h5")
        if hasattr(model, 'save'):
            model.save(file_path)
        else:
            rospy.logwarn(f"Model of type {model_type} doesn't have a save method")
            return None
    
    rospy.loginfo(f"Saved {model_type} model to {file_path}")
    return file_path

def load_model(model, model_type, model_path):
    """
    Load a model from disk.
    
    Args:
        model: Model to load into
        model_type (str): Type of model
        model_path (str): Path to model file
    
    Returns:
        object: Loaded model
    """
    if not os.path.exists(model_path):
        rospy.logwarn(f"Model path does not exist: {model_path}")
        return model
    
    try:
        if model_type in ['qlearn', 'sarsa']:
            # For tabular methods, use pickle
            with open(model_path, 'rb') as f:
                model.q = pickle.load(f)
        else:
            # For neural network methods, use load method
            if hasattr(model, 'load'):
                model.load(model_path)
            else:
                rospy.logwarn(f"Model of type {model_type} doesn't have a load method")
                return model
        
        rospy.loginfo(f"Loaded {model_type} model from {model_path}")
        return model
    except Exception as e:
        rospy.logerr(f"Error loading model: {e}")
        return model

def plot_learning_curve(episode_rewards, avg_rewards, q_values, heights, fall_episodes, outdir, prefix=""):
    """
    Plot learning curve showing episode rewards over time and save it
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        q_values (list): List of average Q-values
        heights (list): List of head heights at the end of each episode
        fall_episodes (list): List of episodes where the robot fell
        outdir (str): Output directory
        prefix (str, optional): Prefix for the output file
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot episode rewards
    episodes = range(len(episode_rewards))
    ax1.plot(episodes, episode_rewards, 'b', label='Episode Reward')
    ax1.plot(episodes, avg_rewards, 'r', label='Average Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode and Average Rewards')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Q-values
    ax2.plot(episodes, q_values, 'g')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Q-value')
    ax2.set_title('Q-value Evolution')
    ax2.grid(True)
    
    # Plot stability (head height)
    ax3.plot(episodes, heights, 'm', label='Head Height')
    
    # Plot fall markers
    if fall_episodes:
        fall_x = [ep for ep in fall_episodes if ep < len(heights)]
        fall_y = [heights[ep] for ep in fall_episodes if ep < len(heights)]
        ax3.scatter(fall_x, fall_y, c='r', marker='x', s=50, label='Falls')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Robot Stability (Head Height)')
    ax3.legend()
    ax3.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Add timestamp to filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(outdir, f"{prefix}learning_curve_{timestamp}.png"))
    plt.close(fig)
    
    rospy.loginfo(f"Saved learning curve to {outdir}")

#!/usr/bin/env python
import gym
import rospy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from policy_gradient import PolicyGradient
from gym import wrappers
import math
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Make sure environment is registered
import env

# Set up the live plot
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
episode_line, = ax1.plot([], [], 'b-', label='Episode Reward')
avg_line, = ax1.plot([], [], 'r-', label='Average Reward')
loss_line, = ax2.plot([], [], 'g-')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Episode and Average Rewards')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show(block=False)

def extract_state_features(observations):
    """
    Extract features from raw observations for Policy Gradient
    
    Args:
        observations (np.array): Array of observations 
                                [joint_angles (6), linear_speed, distance, head_height]
    
    Returns:
        np.array: Feature array for Policy Gradient
    """
    # We'll use numeric values directly for Policy Gradient
    # Extract joint positions
    joint_angles = observations[0:6]
    
    # Extract speed, distance, and height
    linear_speed = observations[6]
    distance = observations[7]
    head_height = observations[8]
    
    # Normalize the values
    normalized_joints = joint_angles / 1.5  # Assuming max joint angle is about 1.5
    normalized_speed = np.array([linear_speed / 2.0])  # Assuming max speed is about 2.0
    normalized_distance = np.array([distance / 10.0])  # Assuming max distance is about 10.0
    normalized_height = np.array([head_height / 0.3])  # Assuming max height is about 0.3
    
    # Combine features
    return np.concatenate([
        normalized_joints, 
        normalized_speed, 
        normalized_distance, 
        normalized_height
    ])

def save_model(pg_object, episode, outdir):
    """
    Save the Policy Gradient model to a file
    
    Args:
        pg_object (PolicyGradient): Policy Gradient learning object
        episode (int): Current episode number
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save model
    filename = os.path.join(outdir, f"gurdy_pg_model_{episode}")
    pg_object.save(filename)

def load_model(pg_object, model_path):
    """
    Load a Policy Gradient model from a file
    
    Args:
        pg_object (PolicyGradient): Policy Gradient learning object
        model_path (str): Path to the model file
    
    Returns:
        PolicyGradient: Updated Policy Gradient learning object
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print("Model file does not exist:", model_path)
        return pg_object
    
    # Load model
    pg_object.load(model_path)
    return pg_object

def update_live_plot(episode_rewards, avg_rewards, losses):
    """
    Update the live plot with current data
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        losses (list): List of training losses
    """
    episodes = range(len(episode_rewards))
    
    # Update the data
    episode_line.set_data(episodes, episode_rewards)
    avg_line.set_data(episodes, avg_rewards)
    loss_line.set_data(episodes, losses)
    
    # Adjust the plot limits
    if len(episodes) > 0:
        ax1.set_xlim(0, max(10, len(episodes)))
        ax1.set_ylim(min(min(episode_rewards) - 1, -10), max(max(episode_rewards) + 1, 10))
        ax2.set_xlim(0, max(10, len(episodes)))
        if losses and max(losses) > 0:
            ax2.set_ylim(0, max(max(losses) * 1.1, 0.1))
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_learning_curve(episode_rewards, avg_rewards, losses, outdir):
    """
    Plot learning curve showing episode rewards over time and save it
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        losses (list): List of training losses
        outdir (str): Output directory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Make sure we have data to plot
    if len(episode_rewards) == 0:
        print("No episode data to plot")
        return
        
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Unpack axes
    ax1, ax2 = axes
    
    # Generate x-axis values
    episodes = range(len(episode_rewards))
    
    # Plot episode rewards
    ax1.plot(episodes, episode_rewards, 'b', label='Episode Reward')
    ax1.plot(episodes, avg_rewards, 'r', label='Average Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode and Average Rewards (Policy Gradient)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(episodes, losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Policy Gradient)')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "policy_gradient_learning_curve.png"))
    print(f"Learning curve saved to {os.path.join(outdir, 'policy_gradient_learning_curve.png')}")
    plt.close(fig)  # Close the figure to free memory

def main():
    # Initialize ROS node
    rospy.init_node('gurdy_policy_gradient', anonymous=True, log_level=rospy.INFO)
    
    # Create output directory
    outdir = rospy.get_param("/gurdy/outdir", "/home/user/catkin_ws/src/my_gurdy_description/output/policy_gradient")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set initial learning parameters
    learning_rate = rospy.get_param("/gurdy/learning_rate", 0.01)
    gamma = rospy.get_param("/gurdy/gamma", 0.99)
    
    # Number of episodes and steps
    nepisodes = rospy.get_param("/gurdy/nepisodes", 20)
    nsteps = rospy.get_param("/gurdy/nsteps", 200)
    save_model_freq = rospy.get_param("/gurdy/save_model_freq", 100)
    
    # Set state and action sizes
    # Using observation directly: 6 joint angles + speed + distance + height = 9 features
    state_size = 9  # Using direct observations
    action_size = env.action_space.n
    
    # Create Policy Gradient agent
    pg_agent = PolicyGradient(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    # Optional: Load a previously trained model
    model_path = rospy.get_param("/gurdy/load_model_path", "")
    if model_path and os.path.exists(model_path):
        pg_agent = load_model(pg_agent, model_path)
    
    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    losses = []  # Track training losses
    
    start_time = time.time()
    highest_reward = -float('inf')
    
    # Training loop
    for episode in range(nepisodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        
        # Episode loop
        for step in range(nsteps):
            # Get state features
            state = extract_state_features(observation)
            
            # Choose action
            action = pg_agent.act(state)
            
            # Execute action
            new_observation, reward, done, info = env.step(action)
            
            # Store in memory
            pg_agent.remember(state, action, reward)
            
            # Update state and reward
            observation = new_observation
            episode_reward += reward
            
            # Check if done
            if done:
                break
        
        # Train the model after the episode
        loss = pg_agent.train()
        
        # Keep track of highest reward
        if episode_reward > highest_reward:
            highest_reward = episode_reward
            rospy.loginfo(f"New highest reward: {highest_reward}")
        
        # Update statistics
        episode_rewards.append(episode_reward)
        losses.append(loss)
        
        # Calculate average reward over the last 20 episodes (or fewer if not enough yet)
        avg_over = min(20, len(episode_rewards))
        avg_reward = sum(episode_rewards[-avg_over:]) / avg_over
        avg_rewards.append(avg_reward)
        
        # Update live plot
        update_live_plot(episode_rewards, avg_rewards, losses)
        
        # Print episode information
        log_freq = rospy.get_param("/gurdy/log_freq", 1)
        if episode % log_freq == 0:
            elapsed_time = time.time() - start_time
            rospy.loginfo(f"Episode: {episode}/{nepisodes-1} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Loss: {loss:.5f} | "
                        f"Time: {elapsed_time:.1f}s")
        
        # Save model periodically
        if episode % save_model_freq == 0 and episode > 0:
            save_model(pg_agent, episode, outdir)
    
    # Save final model
    save_model(pg_agent, nepisodes-1, outdir)
    
    # Plot final learning curve
    plot_learning_curve(episode_rewards, avg_rewards, losses, outdir)
    
    # Print final statistics
    total_time = time.time() - start_time
    rospy.loginfo(f"Training completed in {total_time:.1f} seconds")
    rospy.loginfo(f"Highest reward achieved: {highest_reward:.2f}")
    
    return pg_agent, episode_rewards, avg_rewards, losses

if __name__ == '__main__':
    main()
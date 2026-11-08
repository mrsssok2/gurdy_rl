#!/usr/bin/env python
import gym
import rospy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sac import SAC
from gym import wrappers
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Make sure environment is registered
import env

# Set up the live plot
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
episode_line, = ax1.plot([], [], 'b-', label='Episode Reward')
avg_line, = ax1.plot([], [], 'r-', label='Average Reward')
actor_loss_line, = ax2.plot([], [], 'g-', label='Actor Loss')
critic1_loss_line, = ax2.plot([], [], 'm-', label='Critic 1 Loss')
critic2_loss_line, = ax2.plot([], [], 'c-', label='Critic 2 Loss')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Episode and Average Rewards')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Actor and Critics Losses')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show(block=False)

def extract_state_features(observations):
    """
    Extract features from raw observations for SAC
    
    Args:
        observations (np.array): Array of observations 
                                [joint_angles (6), linear_speed, distance, head_height]
    
    Returns:
        np.array: Feature array for SAC
    """
    # We'll use numeric values directly for SAC
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

def save_model(sac_agent, episode, outdir):
    """
    Save the SAC model to files
    
    Args:
        sac_agent (SAC): SAC learning object
        episode (int): Current episode number
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save model
    actor_filename = os.path.join(outdir, f"gurdy_sac_actor_{episode}")
    critic1_filename = os.path.join(outdir, f"gurdy_sac_critic1_{episode}")
    critic2_filename = os.path.join(outdir, f"gurdy_sac_critic2_{episode}")
    sac_agent.save(actor_filename, critic1_filename, critic2_filename)

def load_model(sac_agent, actor_path, critic1_path, critic2_path):
    """
    Load a SAC model from files
    
    Args:
        sac_agent (SAC): SAC learning object
        actor_path (str): Path to actor model file
        critic1_path (str): Path to critic1 model file
        critic2_path (str): Path to critic2 model file
    
    Returns:
        SAC: Updated SAC learning object
    """
    # Check if files exist
    if not os.path.exists(actor_path) or not os.path.exists(critic1_path) or not os.path.exists(critic2_path):
        print("Model files do not exist:", actor_path, critic1_path, critic2_path)
        return sac_agent
    
    # Load model
    sac_agent.load(actor_path, critic1_path, critic2_path)
    return sac_agent

def update_live_plot(episode_rewards, avg_rewards, actor_losses, critic1_losses, critic2_losses):
    """
    Update the live plot with current data
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        actor_losses (list): List of actor losses
        critic1_losses (list): List of critic 1 losses
        critic2_losses (list): List of critic 2 losses
    """
    episodes = range(len(episode_rewards))
    
    # Update the data
    episode_line.set_data(episodes, episode_rewards)
    avg_line.set_data(episodes, avg_rewards)
    actor_loss_line.set_data(episodes, actor_losses)
    critic1_loss_line.set_data(episodes, critic1_losses)
    critic2_loss_line.set_data(episodes, critic2_losses)
    
    # Adjust the plot limits
    if len(episodes) > 0:
        ax1.set_xlim(0, max(10, len(episodes)))
        ax1.set_ylim(min(min(episode_rewards) - 1, -10), max(max(episode_rewards) + 1, 10))
        ax2.set_xlim(0, max(10, len(episodes)))
        
        # Find max loss for scaling y-axis, but avoid excessive scaling due to outliers
        if actor_losses and critic1_losses and critic2_losses:
            max_loss = max(
                np.percentile(actor_losses, 95),
                np.percentile(critic1_losses, 95),
                np.percentile(critic2_losses, 95)
            )
            ax2.set_ylim(0, max(max_loss * 1.1, 0.1))
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_learning_curve(episode_rewards, avg_rewards, actor_losses, critic1_losses, critic2_losses, outdir):
    """
    Plot learning curve showing episode rewards over time and save it
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        actor_losses (list): List of actor losses
        critic1_losses (list): List of critic 1 losses
        critic2_losses (list): List of critic 2 losses
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
    ax1.set_title('Episode and Average Rewards (SAC)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(episodes, actor_losses, 'g', label='Actor Loss')
    ax2.plot(episodes, critic1_losses, 'm', label='Critic 1 Loss')
    ax2.plot(episodes, critic2_losses, 'c', label='Critic 2 Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Actor and Critics Losses (SAC)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sac_learning_curve.png"))
    print(f"Learning curve saved to {os.path.join(outdir, 'sac_learning_curve.png')}")
    plt.close(fig)  # Close the figure to free memory

def main():
    # Initialize ROS node
    rospy.init_node('gurdy_sac', anonymous=True, log_level=rospy.INFO)
    
    # Create output directory
    outdir = rospy.get_param("/gurdy/outdir", "/home/user/catkin_ws/src/my_gurdy_description/output/sac")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set SAC hyperparameters
    alpha = rospy.get_param("/gurdy/alpha", 0.2)
    actor_lr = rospy.get_param("/gurdy/actor_lr", 0.0003)
    critic_lr = rospy.get_param("/gurdy/critic_lr", 0.0003)
    gamma = rospy.get_param("/gurdy/gamma", 0.99)
    tau = rospy.get_param("/gurdy/tau", 0.005)
    batch_size = rospy.get_param("/gurdy/batch_size", 64)
    memory_size = rospy.get_param("/gurdy/memory_size", 100000)
    auto_alpha = rospy.get_param("/gurdy/auto_alpha", True)
    
    # Number of episodes and steps
    nepisodes = rospy.get_param("/gurdy/nepisodes", 20)
    nsteps = rospy.get_param("/gurdy/nsteps", 200)
    save_model_freq = rospy.get_param("/gurdy/save_model_freq", 100)
    
    # Training frequency (how often to run train)
    train_freq = rospy.get_param("/gurdy/train_freq", 1)
    
    # Set state and action sizes
    # Using observation directly: 6 joint angles + speed + distance + height = 9 features
    state_size = 9  # Using direct observations
    action_size = env.action_space.n
    
    # Create SAC agent
    sac_agent = SAC(
        state_size=state_size,
        action_size=action_size,
        alpha=alpha,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        memory_size=memory_size,
        auto_alpha=auto_alpha
    )
    
    # Optional: Load a previously trained model
    actor_path = rospy.get_param("/gurdy/load_actor_path", "")
    critic1_path = rospy.get_param("/gurdy/load_critic1_path", "")
    critic2_path = rospy.get_param("/gurdy/load_critic2_path", "")
    if (actor_path and critic1_path and critic2_path and
        os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path)):
        sac_agent = load_model(sac_agent, actor_path, critic1_path, critic2_path)
    
    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    
    start_time = time.time()
    highest_reward = -float('inf')
    
    # Training loop
    for episode in range(nepisodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(nsteps):
            # Get state features
            state = extract_state_features(observation)
            
            # Choose action
            action, _ = sac_agent.get_action(state)
            
            # Execute action
            new_observation, reward, done, info = env.step(action)
            
            # Get next state features
            next_state = extract_state_features(new_observation)
            
            # Store in memory
            sac_agent.remember(state, action, reward, next_state, done)
            
            # Train every train_freq steps
            if step % train_freq == 0:
                actor_loss, critic1_loss, critic2_loss = sac_agent.train()
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
            
            # Update state and reward
            observation = new_observation
            episode_reward += reward
            
            # Check if done
            if done:
                break
        
        # Keep track of highest reward
        if episode_reward > highest_reward:
            highest_reward = episode_reward
            rospy.loginfo(f"New highest reward: {highest_reward}")
        
        # Update statistics
        episode_rewards.append(episode_reward)
        
        # Calculate average reward over the last 20 episodes (or fewer if not enough yet)
        avg_over = min(20, len(episode_rewards))
        avg_reward = sum(episode_rewards[-avg_over:]) / avg_over
        avg_rewards.append(avg_reward)
        
        # Update live plot
        update_live_plot(episode_rewards, avg_rewards, actor_losses, critic1_losses, critic2_losses)
        
        # Print episode information
        log_freq = rospy.get_param("/gurdy/log_freq", 1)
        if episode % log_freq == 0:
            elapsed_time = time.time() - start_time
            rospy.loginfo(f"Episode: {episode}/{nepisodes-1} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Alpha: {float(sac_agent.alpha):.3f} | "
                        f"Time: {elapsed_time:.1f}s")
        
        # Save model periodically
        if episode % save_model_freq == 0 and episode > 0:
            save_model(sac_agent, episode, outdir)
    
    # Save final model
    save_model(sac_agent, nepisodes-1, outdir)
    
    # Plot final learning curve
    plot_learning_curve(episode_rewards, avg_rewards, actor_losses, critic1_losses, critic2_losses, outdir)
    
    # Print final statistics
    total_time = time.time() - start_time
    rospy.loginfo(f"Training completed in {total_time:.1f} seconds")
    rospy.loginfo(f"Highest reward achieved: {highest_reward:.2f}")
    
    return sac_agent, episode_rewards, avg_rewards, actor_losses, critic1_losses, critic2_losses

if __name__ == '__main__':
    main()
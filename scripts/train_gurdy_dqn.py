#!/usr/bin/env python
import gym
import rospy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from dqn import DQN
from gym import wrappers
import math
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Make sure environment is registered
import env

# Set up the live plot
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
episode_line, = ax1.plot([], [], 'b-', label='Episode Reward')
avg_line, = ax1.plot([], [], 'r-', label='Average Reward')
loss_line, = ax2.plot([], [], 'g-')
stability_line, = ax3.plot([], [], 'm-', label='Head Height')
fall_scatter = ax3.scatter([], [], c='r', marker='x', s=50, label='Falls')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Episode and Average Rewards')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

ax3.set_xlabel('Episode')
ax3.set_ylabel('Height (m)')
ax3.set_title('Robot Stability (Head Height)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show(block=False)

def extract_state_features(observations):
    """
    Extract features from raw observations for DQN
    
    Args:
        observations (np.array): Array of observations 
                                [joint_angles (6), linear_speed, distance, head_height]
    
    Returns:
        np.array: Feature array for DQN
    """
    # We'll use numeric values directly for DQN
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

def convert_obs_to_state(observations):
    """
    Converts observations to a state representation for DQN
    with improved discretization for better learning and logging
    
    Args:
        observations (np.array): Array of observations 
                                [joint_angles (6), linear_speed, distance, head_height]
    
    Returns:
        str: State string representation (for logging and visualization)
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

def save_model(dqn_object, episode, outdir):
    """
    Save the DQN model to a file
    
    Args:
        dqn_object (DQN): DQN learning object
        episode (int): Current episode number
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save model
    filename = os.path.join(outdir, f"gurdy_dqn_model_{episode}")
    dqn_object.save(filename)

def load_model(dqn_object, model_path):
    """
    Load a DQN model from a file
    
    Args:
        dqn_object (DQN): DQN learning object
        model_path (str): Path to the model file
    
    Returns:
        DQN: Updated DQN learning object
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print("Model file does not exist:", model_path)
        return dqn_object
    
    # Load model
    dqn_object.load(model_path)
    return dqn_object

def update_live_plot(episode_rewards, avg_rewards, losses, heights, fall_episodes):
    """
    Update the live plot with current data
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        losses (list): List of training losses
        heights (list): List of head heights at the end of each episode
        fall_episodes (list): List of episodes where the robot fell
    """
    episodes = range(len(episode_rewards))
    
    # Update the data
    episode_line.set_data(episodes, episode_rewards)
    avg_line.set_data(episodes, avg_rewards)
    loss_line.set_data(episodes, losses)
    stability_line.set_data(episodes, heights)
    
    # Update the fall markers
    if fall_episodes:
        fall_x = [ep for ep in fall_episodes if ep < len(heights)]
        fall_y = [heights[ep] for ep in fall_episodes if ep < len(heights)]
        fall_scatter.set_offsets(np.column_stack([fall_x, fall_y]))
    
    # Adjust the plot limits
    if len(episodes) > 0:
        ax1.set_xlim(0, max(10, len(episodes)))
        ax1.set_ylim(min(min(episode_rewards) - 1, -10), max(max(episode_rewards) + 1, 10))
        ax2.set_xlim(0, max(10, len(episodes)))
        if losses and max(losses) > 0:
            ax2.set_ylim(0, max(max(losses) * 1.1, 0.1))
        ax3.set_xlim(0, max(10, len(episodes)))
        ax3.set_ylim(0, max(0.3, max(heights) * 1.1))
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_learning_curve(episode_rewards, avg_rewards, losses, heights, fall_episodes, outdir):
    """
    Plot learning curve showing episode rewards over time and save it
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        losses (list): List of training losses
        heights (list): List of head heights at the end of each episode
        fall_episodes (list): List of episodes where the robot fell
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
        
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Unpack axes
    ax1, ax2, ax3 = axes
    
    # Generate x-axis values
    episodes = range(len(episode_rewards))
    
    # Plot episode rewards
    ax1.plot(episodes, episode_rewards, 'b', label='Episode Reward')
    ax1.plot(episodes, avg_rewards, 'r', label='Average Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode and Average Rewards (DQN)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(episodes, losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (DQN)')
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
    plt.savefig(os.path.join(outdir, "dqn_learning_curve.png"))
    print(f"Learning curve saved to {os.path.join(outdir, 'dqn_learning_curve.png')}")
    plt.close(fig)  # Close the figure to free memory

def main():
    # Initialize ROS node
    rospy.init_node('gurdy_dqn', anonymous=True, log_level=rospy.INFO)
    
    # Create output directory
    outdir = rospy.get_param("/gurdy/outdir", "/home/user/catkin_ws/src/my_gurdy_description/output/dqn")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set initial learning parameters
    epsilon = rospy.get_param("/gurdy/epsilon", 0.9)
    epsilon_min = rospy.get_param("/gurdy/min_epsilon", 0.05)
    epsilon_decay = rospy.get_param("/gurdy/epsilon_discount", 0.995)
    gamma = rospy.get_param("/gurdy/gamma", 0.95)
    learning_rate = rospy.get_param("/gurdy/learning_rate", 0.001)
    batch_size = rospy.get_param("/gurdy/batch_size", 32)
    
    # Number of episodes and steps
    nepisodes = rospy.get_param("/gurdy/nepisodes", 101)
    nsteps = rospy.get_param("/gurdy/nsteps", 200)
    save_model_freq = rospy.get_param("/gurdy/save_model_freq", 100)
    
    # Set state and action sizes
    # Using observation directly: 6 joint angles + speed + distance + height = 9 features
    state_size = 9  # Using direct observations
    action_size = env.action_space.n
    
    # Create DQN agent
    dqn_agent = DQN(
        state_size=state_size,
        action_size=action_size,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Optional: Load a previously trained model
    model_path = rospy.get_param("/gurdy/load_model_path", "")
    if model_path and os.path.exists(model_path):
        dqn_agent = load_model(dqn_agent, model_path)
    
    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    losses = []  # Track training losses
    head_heights = []  # Track head height at end of each episode
    fall_episodes = []  # Track episodes where the robot fell
    
    start_time = time.time()
    highest_reward = -float('inf')
    
    # Training loop
    for episode in range(nepisodes):
        # Reset environment
        observation = env.reset()
        cumulated_reward = 0
        final_height = 0  # Store final height for this episode
        fell_this_episode = False  # Track if robot fell this episode
        episode_loss = 0  # Track loss for this episode
        
        # Get initial state
        state = extract_state_features(observation)
        state_str = convert_obs_to_state(observation)  # For logging
        
        # Episode loop
        for step in range(nsteps):
            # Choose action
            action = dqn_agent.act(state, use_raw_state=True)
            
            # Execute action
            new_observation, reward, done, info = env.step(action)
            
            # Get new state
            next_state = extract_state_features(new_observation)
            next_state_str = convert_obs_to_state(new_observation)  # For logging
            
            # Store experience in replay memory
            dqn_agent.memorize(state, action, reward, next_state, done)
            
            # Train the model
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)
            
            # Update state and reward
            state = next_state
            state_str = next_state_str
            cumulated_reward += reward
            
            # Store current head height
            final_height = new_observation[8]
            
            # Check if robot fell
            if 'fallen' in info and info['fallen']:
                fell_this_episode = True
            
            # Check if done
            if done:
                break
        
        # Record if robot fell
        if fell_this_episode:
            fall_episodes.append(episode)
            rospy.logwarn(f"Episode {episode}: Robot fell!")
        
        # Update target model every few episodes
        if episode % 5 == 0:
            dqn_agent.update_target_model()
            
        # Keep track of highest reward
        if cumulated_reward > highest_reward:
            highest_reward = cumulated_reward
            rospy.loginfo(f"New highest reward: {highest_reward}")
        
        # Update statistics
        episode_rewards.append(cumulated_reward)
        head_heights.append(final_height)
        losses.append(0.0)  # Placeholder for loss (we don't track it directly in this implementation)
        
        # Calculate average reward over the last 20 episodes (or fewer if not enough yet)
        avg_over = min(20, len(episode_rewards))
        avg_reward = sum(episode_rewards[-avg_over:]) / avg_over
        avg_rewards.append(avg_reward)
        
        # Update live plot
        update_live_plot(episode_rewards, avg_rewards, losses, head_heights, fall_episodes)
        
        # Print episode information
        log_freq = rospy.get_param("/gurdy/log_freq", 1)
        if episode % log_freq == 0:
            elapsed_time = time.time() - start_time
            rospy.loginfo(f"Episode: {episode}/{nepisodes-1} | "
                        f"Reward: {cumulated_reward:.2f} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Epsilon: {dqn_agent.epsilon:.3f} | "
                        f"Time: {elapsed_time:.1f}s | "
                        f"Head Height: {final_height:.3f}")
        
        # Save model periodically
        if episode % save_model_freq == 0 and episode > 0:
            save_model(dqn_agent, episode, outdir)
    
    # Save final model
    save_model(dqn_agent, nepisodes-1, outdir)
    
    # Plot final learning curve
    plot_learning_curve(episode_rewards, avg_rewards, losses, head_heights, fall_episodes, outdir)
    
    # Print final statistics
    total_time = time.time() - start_time
    rospy.loginfo(f"Training completed in {total_time:.1f} seconds")
    rospy.loginfo(f"Highest reward achieved: {highest_reward:.2f}")
    
    return dqn_agent, episode_rewards, avg_rewards, losses, head_heights, fall_episodes

if __name__ == '__main__':
    main()
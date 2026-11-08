#!/usr/bin/env python
import gym
import rospy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from qlearn import QLearn  # Import the QLearn class directly
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
q_line, = ax2.plot([], [], 'g-')
stability_line, = ax3.plot([], [], 'm-', label='Head Height')
fall_scatter = ax3.scatter([], [], c='r', marker='x', s=50, label='Falls')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Episode and Average Rewards')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Q-value')
ax2.set_title('Q-value Evolution')
ax2.grid(True)

ax3.set_xlabel('Episode')
ax3.set_ylabel('Height (m)')
ax3.set_title('Robot Stability (Head Height)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show(block=False)

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

def save_model(qlearn_object, episode, outdir):
    """
    Save the Q-learning model to a file
    
    Args:
        qlearn_object (QLearn): Q-learning object
        episode (int): Current episode number
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save model
    filename = os.path.join(outdir, "gurdy_qlearn_model_" + str(episode) + ".pkl")
    
    # Use pickle to save the Q-table
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(qlearn_object.q, f)
    
    print("Model saved to", filename)

def load_model(qlearn_object, model_path):
    """
    Load a Q-learning model from a file
    
    Args:
        qlearn_object (QLearn): Q-learning object
        model_path (str): Path to the model file
    
    Returns:
        QLearn: Updated Q-learning object
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print("Model file does not exist:", model_path)
        return qlearn_object
    
    # Load model
    import pickle
    with open(model_path, 'rb') as f:
        qlearn_object.q = pickle.load(f)
    
    print("Model loaded from", model_path)
    return qlearn_object

def update_live_plot(episode_rewards, avg_rewards, q_values, heights, fall_episodes):
    """
    Update the live plot with current data
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        q_values (list): List of average Q-values
        heights (list): List of head heights at the end of each episode
        fall_episodes (list): List of episodes where the robot fell
    """
    episodes = range(len(episode_rewards))
    
    # Update the data
    episode_line.set_data(episodes, episode_rewards)
    avg_line.set_data(episodes, avg_rewards)
    q_line.set_data(episodes, q_values)
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
        ax2.set_ylim(min(min(q_values) - 0.1, 0), max(max(q_values) + 0.1, 1))
        ax3.set_xlim(0, max(10, len(episodes)))
        ax3.set_ylim(0, max(0.3, max(heights) * 1.1))
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_learning_curve(episode_rewards, avg_rewards, q_values, heights, fall_episodes, outdir):
    """
    Plot learning curve showing episode rewards over time and save it
    
    Args:
        episode_rewards (list): List of rewards for each episode
        avg_rewards (list): List of average rewards
        q_values (list): List of average Q-values
        heights (list): List of head heights at the end of each episode
        fall_episodes (list): List of episodes where the robot fell
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create figure with three subplots (for the saved file)
    fig_save, (ax1_save, ax2_save, ax3_save) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot episode rewards
    episodes = range(len(episode_rewards))
    ax1_save.plot(episodes, episode_rewards, 'b', label='Episode Reward')
    ax1_save.plot(episodes, avg_rewards, 'r', label='Average Reward')
    ax1_save.set_xlabel('Episode')
    ax1_save.set_ylabel('Reward')
    ax1_save.set_title('Episode and Average Rewards')
    ax1_save.legend()
    ax1_save.grid(True)
    
    # Plot Q-values
    ax2_save.plot(episodes, q_values)
    ax2_save.set_xlabel('Episode')
    ax2_save.set_ylabel('Average Q-value')
    ax2_save.set_title('Q-value Evolution')
    ax2_save.grid(True)
    
    # Plot stability (head height)
    ax3_save.plot(episodes, heights, 'm', label='Head Height')
    
    # Plot fall markers
    if fall_episodes:
        fall_x = [ep for ep in fall_episodes if ep < len(heights)]
        fall_y = [heights[ep] for ep in fall_episodes if ep < len(heights)]
        ax3_save.scatter(fall_x, fall_y, c='r', marker='x', s=50, label='Falls')
    
    ax3_save.set_xlabel('Episode')
    ax3_save.set_ylabel('Height (m)')
    ax3_save.set_title('Robot Stability (Head Height)')
    ax3_save.legend()
    ax3_save.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curve.png"))
    plt.close(fig_save)  # Close the saved figure, not the live plot

def get_average_q_value(qlearn_obj):
    """
    Calculate average Q-value from Q-table
    
    Args:
        qlearn_obj (QLearn): Q-learning object
        
    Returns:
        float: Average Q-value
    """
    # Calculate average Q-value
    total = 0
    count = 0
    
    for state in qlearn_obj.q:
        for action in qlearn_obj.q[state]:
            total += qlearn_obj.q[state][action]
            count += 1
    
    # Avoid division by zero
    if count > 0:
        return total / count
    return 0

def main():
    # Initialize ROS node
    rospy.init_node('gurdy_qlearn', anonymous=True, log_level=rospy.INFO)
    
    # Create output directory
    outdir = "/home/user/catkin_ws/src/my_gurdy_description/output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set initial learning parameters
    alpha = rospy.get_param("/gurdy/alpha", 0.2)  # Increased learning rate
    epsilon = rospy.get_param("/gurdy/epsilon", 0.9)
    gamma = rospy.get_param("/gurdy/gamma", 0.95)  # Increased discount factor
    epsilon_discount = rospy.get_param("/gurdy/epsilon_discount", 0.995)
    
    # Number of episodes and steps
    nepisodes = rospy.get_param("/gurdy/nepisodes", 1000)
    nsteps = rospy.get_param("/gurdy/nsteps", 200)
    
    # Create Q-learning agent
    qlearn_agent = QLearn(
        actions=range(env.action_space.n),
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    # Optional: Load a previously trained model
    model_path = rospy.get_param("/gurdy/load_model_path", "")
    if model_path != "":
        qlearn_agent = load_model(qlearn_agent, model_path)
    
    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    q_values = []
    head_heights = []  # Track head height at end of each episode
    fall_episodes = []  # Track episodes where the robot fell
    
    initial_epsilon = epsilon
    
    start_time = time.time()
    highest_reward = 0
    
    # Training loop
    for episode in range(nepisodes):
        # Reset environment
        observation = env.reset()
        cumulated_reward = 0
        final_height = 0  # Store final height for this episode
        fell_this_episode = False  # Track if robot fell this episode
        
        # Decay epsilon
        epsilon *= epsilon_discount
        epsilon = max(epsilon, 0.05)
        qlearn_agent.epsilon = epsilon
        
        # Get initial state
        state = convert_obs_to_state(observation)
        
        # Episode loop
        for step in range(nsteps):
            # Choose action
            action = qlearn_agent.chooseAction(state)
            
            # Execute action
            new_observation, reward, done, info = env.step(action)
            
            # Get new state
            new_state = convert_obs_to_state(new_observation)
            
            # Learn
            qlearn_agent.learn(state, action, reward, new_state)
            
            # Update state and reward
            state = new_state
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
        
        # Keep track of highest reward
        if cumulated_reward > highest_reward:
            highest_reward = cumulated_reward
        
        # Collect statistics
        episode_rewards.append(cumulated_reward)
        head_heights.append(final_height)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
        avg_rewards.append(avg_reward)
        
        # Calculate average Q-value
        avg_q = get_average_q_value(qlearn_agent)
        q_values.append(avg_q)
        
        # Update the live plot
        update_live_plot(episode_rewards, avg_rewards, q_values, head_heights, fall_episodes)
        
        # Log episode stats
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        rospy.loginfo(f"Episode: {episode+1}/{nepisodes} - Reward: {cumulated_reward:.2f} - Avg Reward: {avg_reward:.2f} - Height: {final_height:.3f} - Fell: {fell_this_episode} - Epsilon: {epsilon:.2f} - Time: {h:d}:{m:02d}:{s:02d}")
        
        # Save model periodically
        if episode % 100 == 0 and episode > 0:
            save_model(qlearn_agent, episode, outdir)
            
            # Save the learning curve
            plot_learning_curve(episode_rewards, avg_rewards, q_values, head_heights, fall_episodes, outdir)
        
        # Process any ROS callbacks and sleep a bit to allow GUI updates
        rospy.sleep(0.1)
    
    # Save final model
    save_model(qlearn_agent, "final", outdir)
    
    # Final learning curve
    plot_learning_curve(episode_rewards, avg_rewards, q_values, head_heights, fall_episodes, outdir)
    
    # Log training summary
    fall_count = len(fall_episodes)
    rospy.loginfo(f"Training completed! - Episodes: {nepisodes} - Highest Reward: {highest_reward:.2f} - Total Falls: {fall_count}")
    
    # Keep the plot open at the end
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the plot and block until window is closed
    
    # Close environment
    env.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        plt.close('all')  # Close all plots if ROS is interrupted
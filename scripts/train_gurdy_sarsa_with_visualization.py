#!/usr/bin/env python
import gym
import rospy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sarsa import SARSA
from gym import wrappers
from gurdy_visualization import GurdyVisualizer

# Make sure environment is registered
import env

def convert_obs_to_state(observations):
    """
    Converts observations to a state representation for SARSA
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
    
    # Discretize joint angles
    joints_str = ""
    for i, angle in enumerate([joint1_angle, joint2_angle, joint3_angle, joint4_angle, joint5_angle, joint6_angle]):
        if angle < -0.3:
            joints_str += "L" + str(i+1) + "_DOWN_"
        elif angle < -0.1:
            joints_str += "L" + str(i+1) + "_MID_"
        else:
            joints_str += "L" + str(i+1) + "_UP_"
    
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
    state_str = joints_str + speed_str + "_" + dist_str + "_" + height_str
    
    return state_str

def save_model(sarsa_object, episode, outdir):
    """
    Save the SARSA model to a file
    
    Args:
        sarsa_object (SARSA): SARSA learning object
        episode (int): Current episode number
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save model (Q-table)
    filename = os.path.join(outdir, f"gurdy_sarsa_model_{episode}.npy")
    np.save(filename, sarsa_object.q)
    print(f"Model saved to {filename}")

def load_model(sarsa_object, model_path):
    """
    Load a SARSA model from a file
    
    Args:
        sarsa_object (SARSA): SARSA learning object
        model_path (str): Path to the model file
    
    Returns:
        SARSA: Updated SARSA learning object
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print("Model file does not exist:", model_path)
        return sarsa_object
    
    # Load model (Q-table)
    sarsa_object.q = np.load(model_path, allow_pickle=True).item()
    print(f"Model loaded from {model_path}")
    return sarsa_object

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
    import matplotlib.pyplot as plt
    
    # Create figure if it doesn't exist
    if not hasattr(update_live_plot, 'fig'):
        update_live_plot.fig, update_live_plot.axes = plt.subplots(3, 1, figsize=(10, 12))
        update_live_plot.episode_line, = update_live_plot.axes[0].plot([], [], 'b-', label='Episode Reward')
        update_live_plot.avg_line, = update_live_plot.axes[0].plot([], [], 'r-', label='Average Reward')
        update_live_plot.q_line, = update_live_plot.axes[1].plot([], [], 'g-')
        update_live_plot.height_line, = update_live_plot.axes[2].plot([], [], 'm-', label='Head Height')
        update_live_plot.fall_scatter = update_live_plot.axes[2].scatter([], [], c='r', marker='x', s=50, label='Falls')
        
        update_live_plot.axes[0].set_xlabel('Episode')
        update_live_plot.axes[0].set_ylabel('Reward')
        update_live_plot.axes[0].set_title('Episode and Average Rewards')
        update_live_plot.axes[0].legend()
        update_live_plot.axes[0].grid(True)
        
        update_live_plot.axes[1].set_xlabel('Episode')
        update_live_plot.axes[1].set_ylabel('Average Q-value')
        update_live_plot.axes[1].set_title('Q-value Evolution')
        update_live_plot.axes[1].grid(True)
        
        update_live_plot.axes[2].set_xlabel('Episode')
        update_live_plot.axes[2].set_ylabel('Head Height (m)')
        update_live_plot.axes[2].set_title('Robot Stability')
        update_live_plot.axes[2].legend()
        update_live_plot.axes[2].grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
    
    # Generate x-axis values
    episodes = range(len(episode_rewards))
    
    # Update the data - ensure arrays are of same length
    episodes_arr = np.array(episodes)
    
    # Convert all data to numpy arrays and ensure they have the same length as episodes
    ep_rewards_arr = np.array(episode_rewards[:len(episodes)])
    avg_rewards_arr = np.array(avg_rewards[:len(episodes)])
    
    # Make sure q_values matches episodes length
    q_values_arr = np.array(q_values[:len(episodes)] if len(q_values) >= len(episodes) 
                           else q_values + [0] * (len(episodes) - len(q_values)))
    
    # Make sure heights matches episodes length
    heights_arr = np.array(heights[:len(episodes)] if len(heights) >= len(episodes)
                          else heights + [0] * (len(episodes) - len(heights)))
    
    # Set the data for each line
    update_live_plot.episode_line.set_data(episodes_arr, ep_rewards_arr)
    update_live_plot.avg_line.set_data(episodes_arr, avg_rewards_arr)
    update_live_plot.q_line.set_data(episodes_arr, q_values_arr)
    update_live_plot.height_line.set_data(episodes_arr, heights_arr)
    
    # Update the fall markers
    if fall_episodes:
        try:
            # Filter fall episodes that are within range of heights array
            fall_x = [ep for ep in fall_episodes if ep < len(heights)]
            # Use the corresponding height values for each fall episode
            fall_y = [heights[ep] if ep < len(heights) else 0 for ep in fall_episodes]
            
            # Ensure fall_x and fall_y have the same length
            min_len = min(len(fall_x), len(fall_y))
            fall_x = fall_x[:min_len]
            fall_y = fall_y[:min_len]
            
            if len(fall_x) > 0 and len(fall_y) > 0:
                # Create a 2D array of coordinates for the scatter plot
                offsets = np.column_stack([fall_x, fall_y])
                update_live_plot.fall_scatter.set_offsets(offsets)
        except Exception as e:
            print(f"Error updating fall markers: {e}")
    
    # Adjust the axis limits
    for i, ax in enumerate(update_live_plot.axes):
        if len(episodes) > 0:
            ax.set_xlim(0, max(10, len(episodes)))
    
    if len(episodes) > 0:
        update_live_plot.axes[0].set_ylim(min(min(episode_rewards) - 10, -100), max(max(episode_rewards) + 10, 100))
        if q_values:
            update_live_plot.axes[1].set_ylim(min(min(q_values) - 1, -10), max(max(q_values) + 1, 10))
        update_live_plot.axes[2].set_ylim(0, max(0.3, max(heights) * 1.1))
    
    # Redraw the figure
    update_live_plot.fig.canvas.draw()
    update_live_plot.fig.canvas.flush_events()

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
    import matplotlib.pyplot as plt
    
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Generate x-axis values
    episodes = range(len(episode_rewards))
    
    # Plot episode rewards
    axes[0].plot(episodes, episode_rewards, 'b-', label='Episode Reward')
    axes[0].plot(episodes, avg_rewards, 'r-', label='Average Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode and Average Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Q-values
    axes[1].plot(episodes, q_values, 'g-')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Q-value')
    axes[1].set_title('Q-value Evolution')
    axes[1].grid(True)
    
    # Plot heights and falls
    axes[2].plot(episodes, heights, 'm-', label='Head Height')
    if fall_episodes:
        try:
            # Filter and prepare data safely
            fall_x = [ep for ep in fall_episodes if ep < len(heights)]
            fall_y = [heights[ep] if ep < len(heights) else 0 for ep in fall_episodes]
            
            # Ensure consistent length
            min_len = min(len(fall_x), len(fall_y))
            fall_x = fall_x[:min_len]
            fall_y = fall_y[:min_len]
            
            if len(fall_x) > 0 and len(fall_y) > 0:
                axes[2].scatter(fall_x, fall_y, c='r', marker='x', s=50, label='Falls')
        except Exception as e:
            print(f"Error plotting fall markers: {e}")
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Head Height (m)')
    axes[2].set_title('Robot Stability')
    axes[2].legend()
    axes[2].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sarsa_learning_curve.png"))
    plt.close()

def get_average_q_value(sarsa_obj):
    """
    Calculate average Q-value from Q-table
    
    Args:
        sarsa_obj (SARSA): SARSA learning object
        
    Returns:
        float: Average Q-value
    """
    # Extract all Q-values
    q_values = []
    for state in sarsa_obj.q:
        q_values.extend(list(sarsa_obj.q[state].values()))
    
    # Calculate average
    if q_values:
        return np.mean(q_values)
    else:
        return 0.0

def collect_trajectory(env, sarsa_obj, num_steps=200, head_height_threshold=0.05):
    """
    Run the agent for a specified number of steps and collect the trajectory
    
    Args:
        env: Gym environment
        sarsa_obj (SARSA): SARSA agent
        num_steps (int): Maximum number of steps
        head_height_threshold (float): Threshold for fallen robot
        
    Returns:
        tuple: (joint_angles_sequence, head_heights, reward, fell)
    """
    # Reset environment
    observation = env.reset()
    
    # Initialize lists to store trajectory
    joint_angles_sequence = []
    head_heights = []
    
    # Initial state
    state = convert_obs_to_state(observation)
    
    # Track reward and if robot fell
    total_reward = 0
    fell = False
    
    # Run for specified number of steps
    for step in range(num_steps):
        # Choose action
        action = sarsa_obj.chooseAction(state)
        
        # Execute action
        new_observation, reward, done, info = env.step(action)
        
        # Store joint angles and head height
        joint_angles_sequence.append(observation[0:6].copy())
        head_heights.append(observation[8])
        
        # Update total reward
        total_reward += reward
        
        # Check if robot fell
        if observation[8] < head_height_threshold:
            fell = True
        
        # Update state
        state = convert_obs_to_state(new_observation)
        observation = new_observation
        
        # Check if done
        if done:
            break
    
    return joint_angles_sequence, head_heights, total_reward, fell

def main():
    # Initialize ROS node
    rospy.init_node('gurdy_sarsa_visualization', anonymous=True, log_level=rospy.INFO)
    
    # Create output directory
    outdir = rospy.get_param("/gurdy/outdir", "/home/user/catkin_ws/src/my_gurdy_description/output/sarsa")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set initial learning parameters
    alpha = rospy.get_param("/gurdy/alpha", 0.2)
    epsilon = rospy.get_param("/gurdy/epsilon", 0.9)
    gamma = rospy.get_param("/gurdy/gamma", 0.8)
    
    # Number of episodes and steps
    nepisodes = rospy.get_param("/gurdy/nepisodes", 20)
    nsteps = rospy.get_param("/gurdy/nsteps", 200)
    
    # Create SARSA object
    actions = range(env.action_space.n)
    sarsa_object = SARSA(actions, epsilon, alpha, gamma)
    
    # Optional: Load a previously trained model
    model_path = rospy.get_param("/gurdy/load_model_path", "")
    if model_path and os.path.exists(model_path):
        sarsa_object = load_model(sarsa_object, model_path)
    
    # Create visualizer
    visualizer = GurdyVisualizer()
    
    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    q_values = []
    head_heights = []
    fall_episodes = []
    
    # Training loop
    for episode in range(nepisodes):
        # Reset environment
        observation = env.reset()
        
        # Initialize trajectory collection
        if episode % 5 == 0 or episode == nepisodes - 1:  # Collect every 5 episodes and last episode
            joint_angles_sequence = []
            head_heights_sequence = []
            collect_trajectory_flag = True
        else:
            collect_trajectory_flag = False
        
        # Get initial state
        state = convert_obs_to_state(observation)
        action = sarsa_object.chooseAction(state)
        
        cumulated_reward = 0
        fell_this_episode = False
        
        # Episode loop
        for step in range(nsteps):
            # Execute the action
            new_observation, reward, done, info = env.step(action)
            
            # Collect trajectory data if needed
            if collect_trajectory_flag:
                joint_angles_sequence.append(observation[0:6].copy())
                head_heights_sequence.append(observation[8])
            
            # Visualize robot in real-time
            visualizer.update_visualization(observation[0:6], observation[8])
            
            # Get next state and action
            next_state = convert_obs_to_state(new_observation)
            next_action = sarsa_object.chooseAction(next_state)
            
            # Update Q-value with SARSA
            sarsa_object.learnQ(state, action, reward, next_state, next_action)
            
            # Update state, action and reward
            state = next_state
            action = next_action
            cumulated_reward += reward
            
            # Store current head height to track stability
            head_heights.append(observation[8])
            
            # Check if robot fell
            if observation[8] < 0.05:  # If head height is very low
                fell_this_episode = True
            
            # Update observation
            observation = new_observation
            
            # Check if done
            if done:
                break
        
        # Update statistics
        episode_rewards.append(cumulated_reward)
        q_values.append(get_average_q_value(sarsa_object))
        
        # Record if robot fell
        final_height = observation[8]
        if fell_this_episode:
            fall_episodes.append(episode)
            print(f"Episode {episode}: Robot fell!")
        
        # Keep track of head height
        head_heights.append(final_height)
        
        # Calculate average reward over the last 20 episodes (or fewer if not enough yet)
        avg_over = min(20, len(episode_rewards))
        avg_reward = sum(episode_rewards[-avg_over:]) / avg_over
        avg_rewards.append(avg_reward)
        
        # Update plot
        update_live_plot(episode_rewards, avg_rewards, q_values, head_heights, fall_episodes)
        
        # Create animation from collected trajectory
        if collect_trajectory_flag and joint_angles_sequence:
            anim_filename = os.path.join(outdir, f"gurdy_episode_{episode}.mp4")
            visualizer.create_animation(joint_angles_sequence, head_heights_sequence, 
                                        filename=anim_filename, fps=20)
            print(f"Created animation for episode {episode}: {anim_filename}")
        
        # Print episode info
        print(f"Episode: {episode}/{nepisodes-1} | "
              f"Reward: {cumulated_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Avg Q-value: {q_values[-1]:.2f} | "
              f"Head Height: {final_height:.3f} | "
              f"Epsilon: {sarsa_object.epsilon:.2f}")
        
        # Save model periodically
        if episode % 10 == 0 and episode > 0:
            save_model(sarsa_object, episode, outdir)
        
        # Decay epsilon (exploration rate)
        if sarsa_object.epsilon > 0.1:  # Don't go below 0.1
            sarsa_object.epsilon *= 0.95
    
    # Save final model
    save_model(sarsa_object, nepisodes-1, outdir)
    
    # Plot final learning curve
    plot_learning_curve(episode_rewards, avg_rewards, q_values, head_heights, fall_episodes, outdir)
    
    # Test best policy and create animation
    print("\nTesting best policy...")
    test_joint_angles, test_heights, test_reward, test_fell = collect_trajectory(env, sarsa_object)
    print(f"Test reward: {test_reward:.2f}, Fell: {test_fell}")
    
    # Create animation for best policy
    visualizer.create_animation(test_joint_angles, test_heights, 
                               filename=os.path.join(outdir, "gurdy_best_policy.mp4"), fps=20)
    
    # Close visualizer
    visualizer.close()
    
    return sarsa_object

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import gym
import rospy
import numpy as np
import time
import os
import json
import rospkg
import matplotlib.pyplot as plt
from dqn import DQN
from gym import wrappers
import math
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Make sure environment is registered
import env
from output_paths import get_outdir
from training_params import get_nepisodes

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
                                [leg pose codes (6), forward_velocity, forward_distance, head_height,
                                 last action one-hot (25), step fraction]
    
    Returns:
        np.array: Feature array for DQN
    """
    # We'll use numeric values directly for DQN
    # Target pose code of each leg 1-6, in [0, 1] (see env.OBS_LEG_CODES)
    leg_codes = np.asarray(observations[0:6], dtype=np.float32)
    
    # Extract signed forward velocity, signed forward distance, and height
    linear_speed = observations[6]  # Forward (x) velocity, negative = backwards
    distance = observations[7]      # Forward (x) distance from start, negative = behind
    head_height = observations[8]
    
    # Normalize the values
    normalized_speed = np.array([linear_speed / 2.0])  # Assuming max speed is about 2.0
    normalized_distance = np.array([distance / 10.0])  # Assuming max distance is about 10.0
    normalized_height = np.array([head_height / 0.3])  # Assuming max height is about 0.3
    
    # Gait phase: last action one-hot and step fraction, already in [0, 1]
    # (see env.OBS_LAST_ACTION and env.OBS_STEP_FRACTION)
    phase = np.asarray(observations[env.OBS_LAST_ACTION.start:env.OBS_SIZE], dtype=np.float32)
    
    # Combine features: one per observation entry (env.OBS_SIZE), which is
    # the state_size the agent is built with
    return np.concatenate([
        leg_codes,  # already in [0, 1]
        normalized_speed, 
        normalized_distance, 
        normalized_height,
        phase
    ])

def convert_obs_to_state(observations):
    """
    Converts observations to a state representation for DQN
    with improved discretization for better learning and logging
    
    Args:
        observations (np.array): Array of observations 
                                [leg pose codes (6), forward_velocity, forward_distance, head_height]
    
    Returns:
        str: State string representation (for logging and visualization)
    """
    # Current foot pose of each leg 1-6: GROUND_IN, GROUND_OUT, LIFT_IN or
    # LIFT_OUT (see env.ACTIONS)
    leg_states = env.leg_states_from_obs(observations)
    
    # Extract signed forward velocity, signed forward distance, and height
    linear_speed = observations[6]  # Forward (x) velocity, negative = backwards
    distance = observations[7]      # Forward (x) distance from start, negative = behind
    head_height = observations[8]
    
    # The leg poses identify the robot's pose exactly, since each action
    # sets one leg's pose. Each leg is lifted/grounded (L/G) x in/out (I/O),
    # e.g. "1GO_2GO_3GO_4LI_5GO_6GO"
    joint_pattern = "_".join(
        "%d%s%s" % (leg, "L" if env.POSE_FLAGS[pose][0] else "G",
                    "O" if env.POSE_FLAGS[pose][1] else "I")
        for leg, pose in zip(env.LEGS, leg_states))
    
    # Discretize signed forward velocity
    if linear_speed < -0.05:
        speed_str = "BACKWARD"  # Moving backwards
    elif linear_speed < 0.05:
        speed_str = "STOP"
    elif linear_speed < 0.2:
        speed_str = "SLOW"
    elif linear_speed < 0.5:
        speed_str = "MEDIUM"
    else:
        speed_str = "FAST"
    
    # Discretize signed forward distance from start
    if distance < -0.1:
        dist_str = "BEHIND"     # Ended up behind the start point
    elif distance < 0.5:
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
    # Last action, so states at different points of the gait cycle with the
    # same leg poses stay apart: "A7", or "A-" right after a reset
    last_action = env.last_action_from_obs(observations)
    action_str = "A-" if last_action is None else "A%d" % last_action
    
    state_str = (joint_pattern + "_" + speed_str + "_" + dist_str + "_" +
                 height_str + "_" + action_str)
    
    return state_str

def training_state_path(model_path):
    """
    Path of the training state file saved next to a model:
    gurdy_dqn_model_<episode> -> gurdy_dqn_model_<episode>_state.json
    """
    return os.path.normpath(model_path) + "_state.json"

def save_model(dqn_object, episode, outdir):
    """
    Save the DQN model, plus the training state needed to resume from it
    (last finished episode and epsilon), to the output directory

    Args:
        dqn_object (DQN): DQN learning object
        episode (int): Episode just finished
        outdir (str): Output directory
    """
    # Ensure directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Save model
    filename = os.path.join(outdir, f"gurdy_dqn_model_{episode}")
    dqn_object.save(filename)

    # Save training state, for resuming with load_model_path
    with open(training_state_path(filename), "w") as f:
        json.dump({"episode": episode, "epsilon": float(dqn_object.epsilon)}, f, indent=2)

def load_model(dqn_object, model_path):
    """
    Load a DQN model and, when there is one, its training state

    Args:
        dqn_object (DQN): DQN learning object
        model_path (str): Path to the saved model

    Returns:
        int: Episode to start from: the one after the saved episode, or 0
             if the model has no training state file
    """
    dqn_object.load(model_path)

    state_path = training_state_path(model_path)
    if not os.path.exists(state_path):
        rospy.logwarn("No training state file %s: starting at episode 0 with epsilon %.3f",
                      state_path, dqn_object.epsilon)
        return 0
    with open(state_path) as f:
        training_state = json.load(f)
    dqn_object.epsilon = training_state["epsilon"]
    start_episode = training_state["episode"] + 1
    rospy.loginfo("Resuming at episode %d with epsilon %.3f", start_episode, dqn_object.epsilon)
    return start_episode

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
    # Output directory inside the package (found at runtime); /gurdy/outdir overrides it
    outdir = get_outdir("output/dqn")
    
    # Create environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    # Set initial learning parameters
    epsilon = rospy.get_param("/gurdy/epsilon", 1.0)
    epsilon_min = rospy.get_param("/gurdy/min_epsilon", 0.05)
    epsilon_decay = rospy.get_param("/gurdy/epsilon_discount", 0.99)  # Per episode
    gamma = rospy.get_param("/gurdy/gamma", 0.99)
    learning_rate = rospy.get_param("/gurdy/learning_rate", 0.001)
    batch_size = rospy.get_param("/gurdy/batch_size", 32)
    
    # Number of episodes and steps
    nepisodes = get_nepisodes(500)
    nsteps = rospy.get_param("/gurdy/nsteps", 400)
    save_model_freq = rospy.get_param("/gurdy/save_model_freq", 25)
    
    # Set state and action sizes
    # Using observation directly: 6 leg pose codes + speed + distance + height = 9 features
    state_size = env.observation_space.shape[0]  # One feature per observation entry (env.OBS_SIZE)
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
    
    # Optional: resume from a saved model (and its epsilon and episode)
    start_episode = 0
    model_path = rospy.get_param("/gurdy/load_model_path", "")
    if model_path:
        if not os.path.isabs(model_path):
            model_path = os.path.join(rospkg.RosPack().get_path('my_gurdy_description'), model_path)
        if not os.path.exists(model_path):
            rospy.logerr("load_model_path does not exist: %s", model_path)
            return
        start_episode = load_model(dqn_agent, model_path)
        if start_episode >= nepisodes:
            rospy.logwarn("Saved model already finished episode %d of nepisodes %d: nothing to do",
                          start_episode - 1, nepisodes)
            return

    # Initialize statistics
    episode_rewards = []
    avg_rewards = []
    losses = []  # Track training losses
    head_heights = []  # Track head height at end of each episode
    fall_episodes = []  # Track episodes where the robot fell
    
    start_time = time.time()
    highest_reward = -float('inf')
    
    # Training loop
    for episode in range(start_episode, nepisodes):
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
            # Index into this run's lists (the plots), which start at
            # start_episode on a resumed run
            fall_episodes.append(episode - start_episode)
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

        # Decay exploration once per episode (1.0 -> 0.05 in about 300
        # episodes with epsilon_discount 0.99)
        dqn_agent.decay_epsilon()

        # Save model periodically, with the decayed epsilon the next
        # episode will use, so a resumed run picks up exactly here
        if episode % save_model_freq == 0 and episode > 0:
            save_model(dqn_agent, episode, outdir)

    # Save final model (unless the last episode was just saved)
    if (nepisodes - 1) % save_model_freq != 0:
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
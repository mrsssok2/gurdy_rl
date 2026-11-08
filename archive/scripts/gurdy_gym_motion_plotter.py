#!/usr/bin/env python
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from matplotlib.animation import FuncAnimation

class GurdyGymMotionPlotter:
    """
    Class for creating live plots of Gurdy robot motion data from a gym environment:
    - Velocity vs. Time
    - Velocity vs. Position
    - Velocity vs. Joint Position
    """
    def __init__(self, env, buffer_size=100, update_interval=100):
        """
        Initialize the motion plotter
        
        Args:
            env: Gym environment for Gurdy
            buffer_size (int): Maximum number of data points to keep in history
            update_interval (int): Update interval in milliseconds
        """
        self.env = env
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Initialize data buffers
        self.times = []
        self.joint_positions = []
        self.joint_velocities = []
        self.head_heights = []
        self.robot_velocities = []  # Linear velocity of the robot
        self.distances = []  # Distance traveled
        
        # Previous values for velocity calculation
        self.prev_joint_positions = None
        self.prev_time = None
        
        # Track the starting time
        self.start_time = time.time()
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation object
        self.ani = None
        
        # Running status
        self.is_running = False
    
    def setup_plots(self):
        """Initialize all subplot axes and lines for plotting"""
        # Top row: Velocity vs Time for each joint
        self.ax_vel_time = self.fig.add_subplot(221)
        self.vel_time_lines = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(6):
            line, = self.ax_vel_time.plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.vel_time_lines.append(line)
        self.ax_vel_time.set_xlabel('Time (s)')
        self.ax_vel_time.set_ylabel('Joint Velocity (rad/s)')
        self.ax_vel_time.set_title('Joint Velocity vs Time')
        self.ax_vel_time.grid(True)
        self.ax_vel_time.legend()
        
        # Top right: Robot Velocity vs Distance
        self.ax_vel_dist = self.fig.add_subplot(222)
        self.vel_dist_line, = self.ax_vel_dist.plot([], [], 'b-')
        self.vel_dist_scatter = self.ax_vel_dist.scatter([], [], c='r', s=50)
        self.ax_vel_dist.set_xlabel('Distance (m)')
        self.ax_vel_dist.set_ylabel('Robot Velocity (m/s)')
        self.ax_vel_dist.set_title('Robot Velocity vs Distance')
        self.ax_vel_dist.grid(True)
        
        # Bottom left: Joint 1-3 Velocity vs Position Phase Plot
        self.ax_phase1 = self.fig.add_subplot(223)
        self.phase_lines1 = []
        for i in range(3):
            line, = self.ax_phase1.plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.phase_lines1.append(line)
        self.ax_phase1.set_xlabel('Joint Position (rad)')
        self.ax_phase1.set_ylabel('Joint Velocity (rad/s)')
        self.ax_phase1.set_title('Phase Plot (Joints 1-3)')
        self.ax_phase1.grid(True)
        self.ax_phase1.legend()
        
        # Bottom right: Joint 4-6 Velocity vs Position Phase Plot
        self.ax_phase2 = self.fig.add_subplot(224)
        self.phase_lines2 = []
        for i in range(3):
            line, = self.ax_phase2.plot([], [], colors[i+3]+'-', label=f'Joint {i+4}')
            self.phase_lines2.append(line)
        self.ax_phase2.set_xlabel('Joint Position (rad)')
        self.ax_phase2.set_ylabel('Joint Velocity (rad/s)')
        self.ax_phase2.set_title('Phase Plot (Joints 4-6)')
        self.ax_phase2.grid(True)
        self.ax_phase2.legend()
        
        plt.tight_layout()
    
    def update_data(self, observation):
        """
        Update data buffers with current observation
        
        Args:
            observation: Observation from gym environment containing:
                         [joint_angles (6), linear_speed, distance, head_height]
        """
        with self.data_lock:
            current_time = time.time() - self.start_time
            
            # Extract data from observation
            joint_positions = observation[0:6].copy()
            linear_speed = observation[6]
            distance = observation[7]
            head_height = observation[8]
            
            # Calculate joint velocities
            if self.prev_joint_positions is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    joint_velocities = [(curr - prev) / dt for curr, prev in 
                                       zip(joint_positions, self.prev_joint_positions)]
                else:
                    joint_velocities = [0.0] * 6
            else:
                joint_velocities = [0.0] * 6
            
            # Update previous values
            self.prev_joint_positions = joint_positions.copy()
            self.prev_time = current_time
            
            # Append new data
            self.times.append(current_time)
            self.joint_positions.append(joint_positions)
            self.joint_velocities.append(joint_velocities)
            self.head_heights.append(head_height)
            self.robot_velocities.append(linear_speed)
            self.distances.append(distance)
            
            # Limit buffer size
            if len(self.times) > self.buffer_size:
                self.times = self.times[-self.buffer_size:]
                self.joint_positions = self.joint_positions[-self.buffer_size:]
                self.joint_velocities = self.joint_velocities[-self.buffer_size:]
                self.head_heights = self.head_heights[-self.buffer_size:]
                self.robot_velocities = self.robot_velocities[-self.buffer_size:]
                self.distances = self.distances[-self.buffer_size:]
    
    def reset_data(self):
        """Clear all data buffers"""
        with self.data_lock:
            self.times = []
            self.joint_positions = []
            self.joint_velocities = []
            self.head_heights = []
            self.robot_velocities = []
            self.distances = []
            self.prev_joint_positions = None
            self.prev_time = None
            self.start_time = time.time()
    
    def _update_plots(self, frame):
        """
        Update function for animation
        
        Args:
            frame: Frame number (not used)
            
        Returns:
            list: List of artists to update
        """
        with self.data_lock:
            if not self.times:  # If no data yet
                return self.vel_time_lines + [self.vel_dist_line, self.vel_dist_scatter] + self.phase_lines1 + self.phase_lines2
            
            # Convert lists to numpy arrays for easier manipulation
            times_array = np.array(self.times)
            joint_positions_array = np.array(self.joint_positions)
            joint_velocities_array = np.array(self.joint_velocities)
            robot_velocities_array = np.array(self.robot_velocities)
            distances_array = np.array(self.distances)
            
            # Update velocity vs time plot
            for i, line in enumerate(self.vel_time_lines):
                line.set_data(times_array, joint_velocities_array[:, i])
            self.ax_vel_time.relim()
            self.ax_vel_time.autoscale_view()
            
            # Update velocity vs distance plot
            self.vel_dist_line.set_data(distances_array, robot_velocities_array)
            if len(distances_array) > 0:
                self.vel_dist_scatter.set_offsets(
                    np.column_stack([distances_array[-1], robot_velocities_array[-1]])
                )
            self.ax_vel_dist.relim()
            self.ax_vel_dist.autoscale_view()
            
            # Update phase plots (velocity vs position)
            # First 3 joints
            for i, line in enumerate(self.phase_lines1):
                line.set_data(joint_positions_array[:, i], joint_velocities_array[:, i])
            self.ax_phase1.relim()
            self.ax_phase1.autoscale_view()
            
            # Last 3 joints
            for i, line in enumerate(self.phase_lines2):
                line.set_data(joint_positions_array[:, i+3], joint_velocities_array[:, i+3])
            self.ax_phase2.relim()
            self.ax_phase2.autoscale_view()
            
        # Return all artists
        return self.vel_time_lines + [self.vel_dist_line, self.vel_dist_scatter] + self.phase_lines1 + self.phase_lines2
    
    def start_plotting(self):
        """Start the animation for live plotting"""
        plt.ion()  # Turn on interactive mode
        self.ani = FuncAnimation(
            self.fig, 
            self._update_plots, 
            interval=self.update_interval,
            blit=True
        )
        plt.show(block=False)
    
    def stop_plotting(self):
        """Stop the animation"""
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
    
    def save_plots(self, filename="gurdy_gym_motion_plots.png"):
        """
        Save the current plots to a file
        
        Args:
            filename (str): Name of the file to save
        """
        plt.figure(self.fig.number)
        plt.savefig(filename)
        print(f"Plots saved to {filename}")
    
    def run_with_random_actions(self, num_episodes=5, max_steps=200):
        """
        Run the environment with random actions and plot the data
        
        Args:
            num_episodes (int): Number of episodes to run
            max_steps (int): Maximum steps per episode
        """
        self.start_plotting()
        self.is_running = True
        
        try:
            for episode in range(num_episodes):
                print(f"Episode {episode+1}/{num_episodes}")
                
                # Reset environment and data
                observation = self.env.reset()
                self.reset_data()
                
                # Run one episode
                for step in range(max_steps):
                    # Choose random action
                    action = self.env.action_space.sample()
                    
                    # Execute action
                    observation, reward, done, info = self.env.step(action)
                    
                    # Update data and plots
                    self.update_data(observation)
                    
                    # Check if done
                    if done:
                        print(f"Episode finished after {step+1} steps")
                        break
                
                # Save plots at the end of each episode
                self.save_plots(f"gurdy_motion_episode_{episode+1}.png")
                
                # Slight pause between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            self.is_running = False
            self.stop_plotting()
    
    def run_with_policy(self, policy_fn, num_episodes=5, max_steps=200):
        """
        Run the environment with a policy and plot the data
        
        Args:
            policy_fn: Function that takes observation and returns action
            num_episodes (int): Number of episodes to run
            max_steps (int): Maximum steps per episode
        """
        self.start_plotting()
        self.is_running = True
        
        try:
            for episode in range(num_episodes):
                print(f"Episode {episode+1}/{num_episodes}")
                
                # Reset environment and data
                observation = self.env.reset()
                self.reset_data()
                
                # Run one episode
                for step in range(max_steps):
                    # Choose action using policy
                    action = policy_fn(observation)
                    
                    # Execute action
                    observation, reward, done, info = self.env.step(action)
                    
                    # Update data and plots
                    self.update_data(observation)
                    
                    # Check if done
                    if done:
                        print(f"Episode finished after {step+1} steps")
                        break
                
                # Save plots at the end of each episode
                self.save_plots(f"gurdy_policy_motion_episode_{episode+1}.png")
                
                # Slight pause between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            self.is_running = False
            self.stop_plotting()


def main():
    """Main function"""
    # Register and create the environment
    import env  # Import to register the environment
    env = gym.make('MyGurdyWalkEnv-v0')
    
    try:
        # Create plotter
        plotter = GurdyGymMotionPlotter(env)
        
        # Run with random actions
        plotter.run_with_random_actions(num_episodes=3, max_steps=200)
        
        # Alternatively, to run with a trained policy:
        # from sarsa import SARSA
        # import pickle
        # 
        # # Load trained model
        # with open('trained_model.pkl', 'rb') as f:
        #     q_table = pickle.load(f)
        # 
        # # Create SARSA agent with loaded Q-table
        # sarsa_agent = SARSA(actions=range(env.action_space.n))
        # sarsa_agent.q = q_table
        # 
        # # Define policy function
        # def sarsa_policy(observation):
        #     state = convert_obs_to_state(observation)  # Discretize state
        #     return sarsa_agent.chooseAction(state)
        # 
        # # Run with policy
        # plotter.run_with_policy(sarsa_policy, num_episodes=3, max_steps=200)
        
    finally:
        env.close()
        plt.close('all')

if __name__ == "__main__":
    main()
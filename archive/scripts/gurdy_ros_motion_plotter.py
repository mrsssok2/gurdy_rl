#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import time
import threading
from matplotlib.animation import FuncAnimation

class GurdyROSMotionPlotter:
    """
    ROS-specific class for creating live plots of Gurdy robot motion data:
    - Velocity vs. Time
    - Velocity vs. Position
    - Velocity vs. Joint Position
    
    This collects data from ROS topics directly rather than from a gym environment.
    """
    def __init__(self, buffer_size=100, update_interval=100):
        """
        Initialize the motion plotter
        
        Args:
            buffer_size (int): Maximum number of data points to keep in history
            update_interval (int): Update interval in milliseconds
        """
        # Initialize ROS node
        rospy.init_node('gurdy_motion_plotter', anonymous=True)
        
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Initialize data buffers
        self.times = []
        self.joint_velocities = []
        self.joint_positions = []
        self.robot_positions = []  # x, y coordinates
        self.robot_linear_velocity = []
        
        # Track the starting time
        self.start_time = time.time()
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation object
        self.ani = None
        
        # Subscribe to joint state, odometry topics
        self.joint_state_sub = rospy.Subscriber('/gurdy/joint_states', JointState, self.joint_state_callback)
        self.odom_sub = rospy.Subscriber('/gurdy/odom', Odometry, self.odom_callback)
        
        # Dictionary to store the latest joint states
        self.latest_joint_positions = [0.0] * 6
        self.latest_joint_velocities = [0.0] * 6
        self.latest_robot_position = [0.0, 0.0]  # x, y
        self.latest_robot_velocity = 0.0
        
        # Mapping from joint name to index (adjust based on your robot's joint names)
        self.joint_name_to_index = {
            'upperlegM1_lowerlegM1_joint': 0,
            'upperlegM2_lowerlegM2_joint': 1,
            'upperlegM3_lowerlegM3_joint': 2,
            'upperlegM4_lowerlegM4_joint': 3,
            'upperlegM5_lowerlegM5_joint': 4,
            'upperlegM6_lowerlegM6_joint': 5
        }
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        
        Args:
            msg (JointState): ROS JointState message
        """
        # Extract joint positions and velocities from message
        for i, name in enumerate(msg.name):
            if name in self.joint_name_to_index:
                idx = self.joint_name_to_index[name]
                self.latest_joint_positions[idx] = msg.position[i]
                self.latest_joint_velocities[idx] = msg.velocity[i]
    
    def odom_callback(self, msg):
        """
        Callback for odometry messages
        
        Args:
            msg (Odometry): ROS Odometry message
        """
        # Extract position and velocity from message
        self.latest_robot_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]
        self.latest_robot_velocity = np.sqrt(
            msg.twist.twist.linear.x**2 + 
            msg.twist.twist.linear.y**2
        )
    
    def update_data_from_ros(self):
        """Update data buffers with the latest data from ROS topics"""
        with self.data_lock:
            current_time = time.time() - self.start_time
            
            # Append new data
            self.times.append(current_time)
            self.joint_positions.append(self.latest_joint_positions.copy())
            self.joint_velocities.append(self.latest_joint_velocities.copy())
            self.robot_positions.append(self.latest_robot_position.copy())
            self.robot_linear_velocity.append(self.latest_robot_velocity)
            
            # Limit buffer size
            if len(self.times) > self.buffer_size:
                self.times = self.times[-self.buffer_size:]
                self.joint_positions = self.joint_positions[-self.buffer_size:]
                self.joint_velocities = self.joint_velocities[-self.buffer_size:]
                self.robot_positions = self.robot_positions[-self.buffer_size:]
                self.robot_linear_velocity = self.robot_linear_velocity[-self.buffer_size:]
    
    def setup_plots(self):
        """Initialize all subplot axes and lines for plotting"""
        # Top row: Velocity vs Time for each joint
        self.ax1 = self.fig.add_subplot(221)
        self.vel_time_lines = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(6):
            line, = self.ax1.plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.vel_time_lines.append(line)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Velocity (rad/s)')
        self.ax1.set_title('Joint Velocity vs Time')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Top right: Robot Linear Velocity vs Position (X-Y trajectory)
        self.ax2 = self.fig.add_subplot(222)
        self.traj_line, = self.ax2.plot([], [], 'b-', label='Trajectory')
        self.traj_scatter = self.ax2.scatter([], [], c=[], cmap='viridis', 
                                            s=50, label='Velocity')
        self.ax2.set_xlabel('X Position (m)')
        self.ax2.set_ylabel('Y Position (m)')
        self.ax2.set_title('Robot Trajectory Colored by Velocity')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Bottom row: 2 subplots for Velocity vs Joint Position
        # First 3 joints
        self.ax3 = self.fig.add_subplot(223)
        self.vel_joint_lines1 = []
        for i in range(3):
            line, = self.ax3.plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.vel_joint_lines1.append(line)
        self.ax3.set_xlabel('Joint Position (rad)')
        self.ax3.set_ylabel('Joint Velocity (rad/s)')
        self.ax3.set_title('Velocity vs Joint Position (Joints 1-3)')
        self.ax3.grid(True)
        self.ax3.legend()
        
        # Last 3 joints
        self.ax4 = self.fig.add_subplot(224)
        self.vel_joint_lines2 = []
        for i in range(3):
            line, = self.ax4.plot([], [], colors[i+3]+'-', label=f'Joint {i+4}')
            self.vel_joint_lines2.append(line)
        self.ax4.set_xlabel('Joint Position (rad)')
        self.ax4.set_ylabel('Joint Velocity (rad/s)')
        self.ax4.set_title('Velocity vs Joint Position (Joints 4-6)')
        self.ax4.grid(True)
        self.ax4.legend()
        
        plt.tight_layout()
    
    def reset_data(self):
        """Clear all data buffers"""
        with self.data_lock:
            self.times = []
            self.joint_velocities = []
            self.joint_positions = []
            self.robot_positions = []
            self.robot_linear_velocity = []
            self.start_time = time.time()
    
    def _update_plots(self, frame):
        """
        Update function for animation
        
        Args:
            frame: Frame number (not used)
            
        Returns:
            list: List of artists to update
        """
        # Update data from ROS
        self.update_data_from_ros()
        
        with self.data_lock:
            if not self.times:  # If no data yet
                return []
                
            # Update velocity vs time plot
            joint_vel_data = np.array(self.joint_velocities)
            for i, line in enumerate(self.vel_time_lines):
                line.set_data(self.times, joint_vel_data[:, i])
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update trajectory plot
            if len(self.robot_positions) > 1:
                # Extract x, y coordinates
                x_pos = [pos[0] for pos in self.robot_positions]
                y_pos = [pos[1] for pos in self.robot_positions]
                
                # Update trajectory line
                self.traj_line.set_data(x_pos, y_pos)
                
                # Update colored scatter points based on velocity
                self.traj_scatter.set_offsets(np.column_stack([x_pos, y_pos]))
                self.traj_scatter.set_array(np.array(self.robot_linear_velocity))
                
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                # Add colorbar if not present
                if not hasattr(self, 'colorbar'):
                    self.colorbar = plt.colorbar(self.traj_scatter, ax=self.ax2)
                    self.colorbar.set_label('Velocity (m/s)')
            
            # Update velocity vs joint position plots
            joint_pos_data = np.array(self.joint_positions)
            
            # First 3 joints
            for i, line in enumerate(self.vel_joint_lines1):
                if joint_pos_data.shape[1] > i:  # Ensure index is valid
                    line.set_data(joint_pos_data[:, i], joint_vel_data[:, i])
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            # Last 3 joints
            for i, line in enumerate(self.vel_joint_lines2):
                if joint_pos_data.shape[1] > i+3:  # Ensure index is valid
                    line.set_data(joint_pos_data[:, i+3], joint_vel_data[:, i+3])
            self.ax4.relim()
            self.ax4.autoscale_view()
            
        # Return artists that were updated
        artists = self.vel_time_lines + [self.traj_line, self.traj_scatter] + self.vel_joint_lines1 + self.vel_joint_lines2
        return artists
    
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
    
    def save_plots(self, filename="gurdy_ros_motion_plots.png"):
        """
        Save the current plots to a file
        
        Args:
            filename (str): Name of the file to save
        """
        plt.figure(self.fig.number)
        plt.savefig(filename)
        print(f"Plots saved to {filename}")
    
    def run(self):
        """Main run loop"""
        self.start_plotting()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Plotting stopped by user")
        finally:
            self.stop_plotting()
            

if __name__ == "__main__":
    try:
        plotter = GurdyROSMotionPlotter()
        plotter.run()
    except rospy.ROSInterruptException:
        pass
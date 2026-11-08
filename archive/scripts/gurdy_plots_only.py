#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import threading
import time
from sensor_msgs.msg import JointState

class SimpleGurdyPlotter:
    """
    Simple standalone plotter for Gurdy robot joint positions and velocities.
    Subscribes to ROS topics without interfering with training.
    """
    def __init__(self, buffer_size=200):
        # Initialize ROS node
        rospy.init_node('simple_gurdy_plotter', anonymous=True)
        
        # Data storage
        self.buffer_size = buffer_size
        self.joint_positions = [[] for _ in range(6)]  # Positions for 6 joints
        self.joint_velocities = [[] for _ in range(6)]  # Velocities for 6 joints
        self.times = []
        self.start_time = time.time()
        
        # Joint names
        self.joint_names = [
            'upperlegM1_lowerlegM1_joint',
            'upperlegM2_lowerlegM2_joint',
            'upperlegM3_lowerlegM3_joint',
            'upperlegM4_lowerlegM4_joint',
            'upperlegM5_lowerlegM5_joint',
            'upperlegM6_lowerlegM6_joint'
        ]
        
        # Thread lock
        self.lock = threading.Lock()
        
        # Create plots
        self.create_plots()
        
        # Subscribe to joint states
        self.sub = rospy.Subscriber('/gurdy/joint_states', JointState, self.joint_callback)
        
        rospy.loginfo("Simple Gurdy plotter initialized")
    
    def joint_callback(self, msg):
        with self.lock:
            current_time = time.time() - self.start_time
            self.times.append(current_time)
            
            # Initialize temporary lists for this timestep
            positions = [0.0] * 6
            velocities = [0.0] * 6
            
            # Extract data for our joints
            for i, name in enumerate(msg.name):
                if name in self.joint_names:
                    joint_idx = self.joint_names.index(name)
                    positions[joint_idx] = msg.position[i]
                    velocities[joint_idx] = msg.velocity[i]
            
            # Add data for each joint
            for i in range(6):
                self.joint_positions[i].append(positions[i])
                self.joint_velocities[i].append(velocities[i])
            
            # Limit buffer size
            if len(self.times) > self.buffer_size:
                self.times.pop(0)
                for i in range(6):
                    self.joint_positions[i].pop(0)
                    self.joint_velocities[i].pop(0)
    
    def create_plots(self):
        # Create figure with 3 subplots
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 16))
        
        # Colors for each joint
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        # Joint positions subplot
        self.axs[0].set_title('Joint Positions')
        self.axs[0].set_xlabel('Time (s)')
        self.axs[0].set_ylabel('Position (rad)')
        self.pos_lines = []
        for i in range(6):
            line, = self.axs[0].plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.pos_lines.append(line)
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Joint velocities subplot
        self.axs[1].set_title('Joint Velocities')
        self.axs[1].set_xlabel('Time (s)')
        self.axs[1].set_ylabel('Velocity (rad/s)')
        self.vel_lines = []
        for i in range(6):
            line, = self.axs[1].plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.vel_lines.append(line)
        self.axs[1].legend()
        self.axs[1].grid(True)
        
        # Phase plot (first 3 joints only for clarity)
        self.axs[2].set_title('Phase Plot (Position vs Velocity)')
        self.axs[2].set_xlabel('Position (rad)')
        self.axs[2].set_ylabel('Velocity (rad/s)')
        self.phase_lines = []
        for i in range(3):  # Just show first 3 joints in phase plot
            line, = self.axs[2].plot([], [], colors[i]+'-', label=f'Joint {i+1}')
            self.phase_lines.append(line)
        self.axs[2].legend()
        self.axs[2].grid(True)
        
        plt.tight_layout()
    
    def update_plots(self):
        with self.lock:
            if not self.times:
                return
            
            # Update position lines
            for i in range(6):
                self.pos_lines[i].set_data(self.times, self.joint_positions[i])
            
            # Update velocity lines
            for i in range(6):
                self.vel_lines[i].set_data(self.times, self.joint_velocities[i])
            
            # Update phase plot (first 3 joints only)
            for i in range(3):
                self.phase_lines[i].set_data(self.joint_positions[i], self.joint_velocities[i])
            
            # Adjust axis limits
            if len(self.times) > 0:
                # Position plot
                self.axs[0].set_xlim(min(self.times), max(self.times))
                all_pos = [pos for joint_pos in self.joint_positions for pos in joint_pos]
                if all_pos:
                    pos_min = min(all_pos)
                    pos_max = max(all_pos)
                    margin = max(0.1, (pos_max - pos_min) * 0.1)  # 10% margin
                    self.axs[0].set_ylim(pos_min - margin, pos_max + margin)
                
                # Velocity plot
                self.axs[1].set_xlim(min(self.times), max(self.times))
                all_vel = [vel for joint_vel in self.joint_velocities for vel in joint_vel]
                if all_vel:
                    vel_min = min(all_vel)
                    vel_max = max(all_vel)
                    margin = max(0.1, (vel_max - vel_min) * 0.1)  # 10% margin
                    self.axs[1].set_ylim(vel_min - margin, vel_max + margin)
                
                # Phase plot
                phase_pos = [pos for i in range(3) for pos in self.joint_positions[i]]
                phase_vel = [vel for i in range(3) for vel in self.joint_velocities[i]]
                if phase_pos and phase_vel:
                    pos_min = min(phase_pos)
                    pos_max = max(phase_pos)
                    vel_min = min(phase_vel)
                    vel_max = max(phase_vel)
                    pos_margin = max(0.1, (pos_max - pos_min) * 0.1)
                    vel_margin = max(0.1, (vel_max - vel_min) * 0.1)
                    self.axs[2].set_xlim(pos_min - pos_margin, pos_max + pos_margin)
                    self.axs[2].set_ylim(vel_min - vel_margin, vel_max + vel_margin)
            
            # Draw plots
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def save_plots(self, filename_prefix="gurdy_plot"):
        # Save each plot with timestamp
        timestamp = int(time.time())
        self.fig.savefig(f"{filename_prefix}_{timestamp}.png")
        rospy.loginfo(f"Saved plots to {filename_prefix}_{timestamp}.png")
    
    def run(self):
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        try:
            # Create a timer for periodic saving
            save_timer = rospy.Timer(rospy.Duration(60), 
                                    lambda event: self.save_plots())
            
            # Update plots periodically
            rate = rospy.Rate(10)  # 10 Hz update rate
            while not rospy.is_shutdown():
                self.update_plots()
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Plotter stopped by user")
        finally:
            # Save final plot
            self.save_plots("gurdy_plot_final")
            plt.close()

if __name__ == "__main__":
    try:
        plotter = SimpleGurdyPlotter(buffer_size=300)
        plotter.run()
    except rospy.ROSInterruptException:
        pass
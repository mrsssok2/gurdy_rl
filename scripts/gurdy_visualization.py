#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import threading
import copy
import math

class GurdyVisualizer:
    """
    Class for visualizing the Gurdy robot in 3D
    """
    def __init__(self, live_update=True):
        """
        Initialize the visualizer
        
        Args:
            live_update (bool): Whether to update the plot in real-time
        """
        self.live_update = live_update
        self.joint_positions = None
        self.head_position = [0, 0, 0.2]  # Initial head position
        
        # Set up the figure and 3D axis for plotting
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        self.body_scatter = None
        self.leg_lines = [None] * 6  # Six legs
        
        # Robot dimensions
        self.leg_length = 0.2
        self.body_radius = 0.15
        
        # Plot lock to prevent concurrent updates
        self.plot_lock = threading.Lock()
        
        # Initialize the plot
        self._init_plot()
        
        # If live update is enabled, show the plot
        if live_update:
            plt.ion()  # Enable interactive mode
            plt.show(block=False)
    
    def _init_plot(self):
        """Initialize the 3D plot"""
        with self.plot_lock:
            # Set axis labels
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title('Gurdy Robot Visualization')
            
            # Set axis limits
            self.ax.set_xlim(-0.5, 0.5)
            self.ax.set_ylim(-0.5, 0.5)
            self.ax.set_zlim(0, 0.5)
            
            # Equal aspect ratio
            self.ax.set_box_aspect([1, 1, 0.5])
            
            # Create scatter plot for body
            body_x, body_y, body_z = self._generate_body_points()
            self.body_scatter = self.ax.scatter(body_x, body_y, body_z, c='b', marker='o', s=50, label='Body')
            
            # Initialize leg lines
            for i in range(6):
                self.leg_lines[i], = self.ax.plot([0, 0], [0, 0], [0, 0], 'r-', linewidth=2)
            
            # Add legend
            self.ax.legend()
            
            # Add ground plane
            x_ground = np.linspace(-0.5, 0.5, 10)
            y_ground = np.linspace(-0.5, 0.5, 10)
            X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
            Z_ground = np.zeros_like(X_ground)
            self.ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='g')
            
            # Tight layout
            plt.tight_layout()
    
    def _generate_body_points(self):
        """
        Generate points for the hexagonal body
        
        Returns:
            tuple: (x points, y points, z points)
        """
        # Generate points for a hexagon
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points for hexagon
        x = self.body_radius * np.cos(angles) + self.head_position[0]
        y = self.body_radius * np.sin(angles) + self.head_position[1]
        z = np.ones_like(x) * self.head_position[2]
        
        # Add central point
        x = np.append(x, self.head_position[0])
        y = np.append(y, self.head_position[1])
        z = np.append(z, self.head_position[2])
        
        return x, y, z
    
    def _calculate_leg_positions(self, joint_angles):
        """
        Calculate leg positions based on joint angles
        
        Args:
            joint_angles (np.array): Array of 6 joint angles
            
        Returns:
            list: List of leg line endpoints as numpy arrays
        """
        # Calculate leg positions
        leg_positions = []
        
        # Hexagon points for leg attachments
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points for hexagon
        
        for i in range(6):
            # Leg base position (where the leg attaches to the body)
            base_x = self.body_radius * np.cos(angles[i]) + self.head_position[0]
            base_y = self.body_radius * np.sin(angles[i]) + self.head_position[1]
            base_z = self.head_position[2]
            
            # Angle from the center to the leg
            leg_angle = angles[i]
            
            # Adjust joint angle (map the joint angle to leg movement)
            # This is a simplified model - in reality, the joint angles would need
            # a more complex mapping to leg positions
            adjusted_angle = joint_angles[i] - 1.0  # Offset to make neutral position reasonable
            
            # Calculate leg endpoint
            leg_x = base_x + self.leg_length * np.cos(leg_angle) * np.sin(adjusted_angle)
            leg_y = base_y + self.leg_length * np.sin(leg_angle) * np.sin(adjusted_angle)
            leg_z = base_z - self.leg_length * np.cos(adjusted_angle)
            
            # Store the leg line as numpy arrays
            leg_positions.append((
                np.array([base_x, leg_x]),
                np.array([base_y, leg_y]),
                np.array([base_z, leg_z])
            ))
        
        return leg_positions
    
    def update_visualization(self, joint_angles, head_height=None):
        """
        Update the robot visualization
        
        Args:
            joint_angles (np.array): Array of 6 joint angles
            head_height (float, optional): Head height. If None, use current height
        """
        # If live update is disabled, do nothing
        if not self.live_update:
            return
        
        # Update head height if provided
        if head_height is not None:
            self.head_position[2] = head_height
        
        # Update joint positions
        self.joint_positions = copy.deepcopy(joint_angles)
        
        with self.plot_lock:
            # Update body position
            body_x, body_y, body_z = self._generate_body_points()
            self.body_scatter._offsets3d = (body_x, body_y, body_z)
            
            # Calculate and update leg positions
            leg_positions = self._calculate_leg_positions(joint_angles)
            for i, leg_line in enumerate(self.leg_lines):
                leg_line.set_data(leg_positions[i][0], leg_positions[i][1])
                leg_line.set_3d_properties(leg_positions[i][2])
            
            # Redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def create_animation(self, joint_angles_sequence, head_heights=None, filename=None, fps=10):
        """
        Create an animation of the robot movement
        
        Args:
            joint_angles_sequence (list): List of joint angle arrays
            head_heights (list, optional): List of head heights. If None, use constant height
            filename (str, optional): Filename to save the animation. If None, don't save
            fps (int): Frames per second
            
        Returns:
            matplotlib.animation.Animation: Animation object
        """
        # Set up the figure for animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Gurdy Robot Movement Animation')
        
        # Set axis limits
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.5)
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 0.5])
        
        # Create initial plots
        body_x, body_y, body_z = self._generate_body_points()
        body_scatter = ax.scatter(body_x, body_y, body_z, c='b', marker='o', s=50, label='Body')
        
        # Initialize leg lines
        leg_lines = [ax.plot([0, 0], [0, 0], [0, 0], 'r-', linewidth=2)[0] for _ in range(6)]
        
        # Add legend
        ax.legend()
        
        # Add ground plane
        x_ground = np.linspace(-0.5, 0.5, 10)
        y_ground = np.linspace(-0.5, 0.5, 10)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.zeros_like(X_ground)
        ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='g')
        
        # Tight layout
        plt.tight_layout()
        
        # Make a copy of the head position
        head_pos = copy.deepcopy(self.head_position)
        
        def update(frame):
            # Update head height if provided
            if head_heights is not None:
                head_pos[2] = head_heights[frame]
            
            # Update body position
            body_x, body_y, body_z = self._generate_body_points()
            body_scatter._offsets3d = (body_x, body_y, body_z)
            
            # Calculate and update leg positions
            leg_positions = self._calculate_leg_positions(joint_angles_sequence[frame])
            for i, leg_line in enumerate(leg_lines):
                leg_line.set_data(leg_positions[i][0], leg_positions[i][1])
                leg_line.set_3d_properties(leg_positions[i][2])
            
            return [body_scatter] + leg_lines
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(joint_angles_sequence), 
            interval=1000/fps, blit=True)
        
        # Save animation if filename is provided
        if filename:
            try:
                # Try with FFMpegWriter explicit configuration
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps)
                anim.save(filename, writer=writer)
            except Exception as e:
                print(f"Error saving animation with FFMpegWriter: {e}")
                # Fallback to default save with no extra args
                try:
                    anim.save(filename, fps=fps)
                except Exception as e:
                    print(f"Error saving animation with default writer: {e}")
                    print("Animation could not be saved, but will still be displayed")
        
        # Show animation
        plt.show()
        
        return anim
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)


# Example usage
if __name__ == '__main__':
    # Create visualizer
    visualizer = GurdyVisualizer()
    
    # Generate some joint angle data for testing
    num_frames = 100
    joint_angles_sequence = []
    head_heights = []
    
    for i in range(num_frames):
        # Generate sinusoidal joint movements (alternating legs)
        phase = i / 10.0
        angles = np.array([
            0.5 * np.sin(phase),
            0.5 * np.sin(phase + np.pi),
            0.5 * np.sin(phase + np.pi/3),
            0.5 * np.sin(phase + 4*np.pi/3),
            0.5 * np.sin(phase + 2*np.pi/3),
            0.5 * np.sin(phase + 5*np.pi/3)
        ])
        joint_angles_sequence.append(angles)
        
        # Also generate varying head height
        head_heights.append(0.2 + 0.05 * np.sin(phase))
    
    # Test real-time visualization
    print("Testing real-time visualization...")
    for i in range(num_frames):
        visualizer.update_visualization(joint_angles_sequence[i], head_heights[i])
        time.sleep(0.1)
    
    # Test animation creation
    print("Creating animation...")
    visualizer.close()
    visualizer = GurdyVisualizer(live_update=False)
    anim = visualizer.create_animation(joint_angles_sequence, head_heights, filename="gurdy_animation.mp4")
    
    print("Done!")
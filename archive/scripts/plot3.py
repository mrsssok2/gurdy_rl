import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time

class HexapodTerrainVisualization:
    def __init__(self):
        # Setup the figure with a grid of subplots
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 4, figure=self.fig)
        
        self.snapshots = []
        for i in range(8):
            row, col = i // 4, i % 4
            ax = self.fig.add_subplot(gs[row, col], projection='3d')
            ax.set_title(f"{i+1}", fontsize=14, fontweight='bold')
            ax.set_box_aspect([1, 1, 0.5])  # Adjust 3D box aspect ratio
            self.snapshots.append(ax)
        
        # Parameters for terrain
        self.x_range = np.linspace(0, 300, 50)
        self.y_range = np.linspace(0, 200, 30)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        
        # Robot parameters
        self.body_radius = 30
        self.leg_segments = [40, 60]  # Upper and lower leg lengths
        
        # Create the terrain once
        self.create_terrain()
        
        # Initial robot position
        self.robot_position = [50, 100, 0]  # [x, y, z]
        
        # Generate walking sequence
        self.generate_walking_sequence()
    
    def terrain_function(self, x, y, timestep=0):
        """Generate a terrain with a hill that changes over time"""
        # Create a gaussian hill
        hill_center_x = 150
        hill_center_y = 100
        hill_height = 40
        hill_width = 40
        
        dist_from_center = np.sqrt((self.X - hill_center_x)**2 + (self.Y - hill_center_y)**2)
        hill = hill_height * np.exp(-(dist_from_center**2) / (2 * hill_width**2))
        
        # Add some random noise for texture
        noise = np.random.rand(self.X.shape[0], self.X.shape[1]) * 5
        
        # Return the combined terrain
        return hill + noise
    
    def create_terrain(self):
        """Create the terrain mesh for all snapshots"""
        self.Z = self.terrain_function(self.X, self.Y)
        
        # Store terrain for each snapshot
        for i, ax in enumerate(self.snapshots):
            # Plot the terrain surface
            terrain = ax.plot_surface(
                self.X, self.Y, self.Z, 
                cmap=cm.terrain, 
                alpha=0.8,
                linewidth=0, 
                antialiased=True
            )
            
            # Set axis labels and limits
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 200)
            ax.set_zlim(0, 80)
            
            # Set view angle
            ax.view_init(elev=30, azim=225)
    
    def leg_kinematics(self, base_pos, angles, leg_number, phase):
        """Calculate leg positions based on joint angles"""
        # Leg base positions around the body (hexapod has 6 legs)
        angle_offsets = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        radial_offset = self.body_radius
        
        # Base of the leg
        base_angle = angle_offsets[leg_number]
        leg_base_x = base_pos[0] + radial_offset * np.cos(base_angle)
        leg_base_y = base_pos[1] + radial_offset * np.sin(base_angle)
        
        # Get terrain height at this position
        # Find nearest point in the terrain mesh
        x_idx = np.abs(self.x_range - leg_base_x).argmin()
        y_idx = np.abs(self.y_range - leg_base_y).argmin()
        terrain_z = self.Z[y_idx, x_idx]
        
        # Adjust base height by terrain height
        leg_base_z = base_pos[2] + terrain_z
        
        # Calculate joint positions
        # Angle 1: hip joint (in xy plane)
        # Angle 2: elevation angle from horizontal
        # Angle 3: knee joint
        
        # Add walking motion based on phase
        step_height = 15
        if phase < 0.5:  # Leg in air (swing phase)
            swing_height = step_height * np.sin(phase * 2 * np.pi)
        else:  # Leg on ground (stance phase)
            swing_height = 0
        
        # Compute leg segment positions
        hip_angle = angles[0] + base_angle
        elevation_angle = angles[1]
        knee_angle = angles[2]
        
        # First segment
        segment1_length = self.leg_segments[0]
        joint1_x = leg_base_x + segment1_length * np.cos(hip_angle) * np.cos(elevation_angle)
        joint1_y = leg_base_y + segment1_length * np.sin(hip_angle) * np.cos(elevation_angle)
        joint1_z = leg_base_z + segment1_length * np.sin(elevation_angle)
        
        # Second segment
        segment2_length = self.leg_segments[1]
        joint2_x = joint1_x + segment2_length * np.cos(hip_angle) * np.cos(elevation_angle + knee_angle)
        joint2_y = joint1_y + segment2_length * np.sin(hip_angle) * np.cos(elevation_angle + knee_angle)
        joint2_z = joint1_z + segment2_length * np.sin(elevation_angle + knee_angle) + swing_height
        
        # Return the 3D coordinates of the leg
        return np.array([
            [leg_base_x, leg_base_y, leg_base_z],
            [joint1_x, joint1_y, joint1_z],
            [joint2_x, joint2_y, joint2_z]
        ])
    
    def get_walking_angles(self, t, leg_number):
        """Generate walking motion for a leg based on time"""
        # Basic tripod gait - alternating legs
        tripod_1 = [0, 2, 4]  # Front-right, middle-left, rear-right
        tripod_2 = [1, 3, 5]  # Front-left, middle-right, rear-left
        
        # Phase offset for each leg
        if leg_number in tripod_1:
            phase = (t % 1.0)
        else:
            phase = (t + 0.5) % 1.0
        
        # Base joint angles
        hip_swing = 0.3 * np.sin(2 * np.pi * phase)
        
        # Elevation angle changes based on phase
        if phase < 0.5:  # Lift leg
            elevation = 0.4 + 0.2 * np.sin(2 * np.pi * phase)
        else:  # Put leg down and move back
            elevation = 0.4
        
        # Knee angle
        knee = -0.8 - 0.2 * np.sin(2 * np.pi * phase)
        
        return [hip_swing, elevation, knee], phase
    
    def generate_walking_sequence(self):
        """Generate a sequence of 8 snapshots of the robot walking"""
        # Generate 8 positions along a path
        path_x = np.linspace(40, 240, 8)
        path_y = np.linspace(100, 100, 8)
        
        for i in range(8):
            # Update robot position along the path
            self.robot_position[0] = path_x[i]
            self.robot_position[1] = path_y[i]
            
            # Get z position from terrain (approximate)
            x_idx = np.abs(self.x_range - self.robot_position[0]).argmin()
            y_idx = np.abs(self.y_range - self.robot_position[1]).argmin()
            self.robot_position[2] = self.Z[y_idx, x_idx] + 30  # Hover above terrain
            
            # Time parameter for walking cycle (adjusted for each snapshot)
            t = i * 0.125
            
            # Draw the robot in this snapshot
            self.draw_robot(i, t)
    
    def draw_robot(self, snapshot_idx, t):
        """Draw the robot at a specific time in a specific snapshot"""
        ax = self.snapshots[snapshot_idx]
        
        # Draw the body
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        body_x = self.robot_position[0] + self.body_radius/2 * np.outer(np.cos(u), np.sin(v))
        body_y = self.robot_position[1] + self.body_radius/2 * np.outer(np.sin(u), np.sin(v))
        body_z = self.robot_position[2] + 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(body_x, body_y, body_z, color='blue', alpha=0.7)
        
        # Draw each leg
        for leg in range(6):
            angles, phase = self.get_walking_angles(t, leg)
            leg_points = self.leg_kinematics(self.robot_position, angles, leg, phase)
            
            # Plot the leg segments
            ax.plot(leg_points[:, 0], leg_points[:, 1], leg_points[:, 2], 'k-', linewidth=2)
            
            # Plot joints
            ax.scatter(leg_points[:, 0], leg_points[:, 1], leg_points[:, 2], color='black', s=30)
    
    def show(self):
        """Display the visualization"""
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Generating hexapod walking sequence...")
    viz = HexapodTerrainVisualization()
    viz.show()
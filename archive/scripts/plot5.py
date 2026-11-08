import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

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
        
        # Parameters for terrain - INCREASED SIZE
        self.x_range = np.linspace(0, 300, 60)  # Wider x range
        self.y_range = np.linspace(0, 300, 60)  # Wider y range
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        
        # Robot parameters
        self.body_radius = 15
        self.leg_length = 40  # Reduced leg length for less broad appearance
        self.body_height = 30  # REDUCED height of body above terrain
        
        # Create the terrain once
        self.create_terrain()
        
        # Initial robot position - adjusted for bigger terrain
        self.robot_position = [75, 150, 0]  # [x, y, z]
        
        # Generate walking sequence
        self.generate_walking_sequence()
    
    def terrain_function(self, x, y):
        """Generate a terrain with a hill"""
        # Create a gaussian hill - positioned relative to larger terrain
        hill_center_x = 150  # Centered in wider terrain
        hill_center_y = 150
        hill_height = 30
        hill_width = 50  # Slightly wider hill
        
        dist_from_center = np.sqrt((self.X - hill_center_x)**2 + (self.Y - hill_center_y)**2)
        hill = hill_height * np.exp(-(dist_from_center**2) / (2 * hill_width**2))
        
        # Add some very slight noise for texture
        np.random.seed(42)  # Set seed for reproducibility
        noise = np.random.rand(self.X.shape[0], self.X.shape[1]) * 2
        
        # Return the combined terrain
        return hill + noise
    
    def create_terrain(self):
        """Create the terrain mesh for all snapshots"""
        self.Z = self.terrain_function(self.X, self.Y)
        
        # Store terrain for each snapshot
        for i, ax in enumerate(self.snapshots):
            # Plot the terrain surface with a blue-green colormap
            terrain = ax.plot_surface(
                self.X, self.Y, self.Z, 
                cmap=plt.cm.viridis, 
                alpha=0.8,
                linewidth=0, 
                antialiased=True
            )
            
            # Set axis labels and limits - UPDATED FOR BIGGER TERRAIN
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 300)
            ax.set_zlim(0, 80)
            
            # Set view angle
            ax.view_init(elev=30, azim=-45)
    
    def get_terrain_height(self, x, y):
        """Get interpolated terrain height at any x,y position"""
        # Ensure x and y are within bounds
        x = max(min(x, self.x_range[-1]), self.x_range[0])
        y = max(min(y, self.y_range[-1]), self.y_range[0])
        
        # Find indices for interpolation
        x_idx_low = np.searchsorted(self.x_range, x, side='right') - 1
        y_idx_low = np.searchsorted(self.y_range, y, side='right') - 1
        
        # Ensure indices are within bounds
        x_idx_low = max(0, min(x_idx_low, len(self.x_range)-2))
        y_idx_low = max(0, min(y_idx_low, len(self.y_range)-2))
        x_idx_high = x_idx_low + 1
        y_idx_high = y_idx_low + 1
        
        # Get terrain heights at corners
        z_ll = self.Z[y_idx_low, x_idx_low]
        z_lh = self.Z[y_idx_low, x_idx_high]
        z_hl = self.Z[y_idx_high, x_idx_low]
        z_hh = self.Z[y_idx_high, x_idx_high]
        
        # Calculate interpolation weights
        x_weight = (x - self.x_range[x_idx_low]) / (self.x_range[x_idx_high] - self.x_range[x_idx_low])
        y_weight = (y - self.y_range[y_idx_low]) / (self.y_range[y_idx_high] - self.y_range[y_idx_low])
        
        # Bilinear interpolation
        z = (1-x_weight) * (1-y_weight) * z_ll + \
            x_weight * (1-y_weight) * z_lh + \
            (1-x_weight) * y_weight * z_hl + \
            x_weight * y_weight * z_hh
            
        return z
    
    def get_leg_position(self, leg_number, phase, robot_pos):
        """Calculate leg positions with simple straight legs"""
        # Leg base positions around the body (hexagon)
        angles = [30, 90, 150, 210, 270, 330]  # angles in degrees
        angles_rad = [np.radians(angle) for angle in angles]
        
        # Calculate base position of leg on body
        leg_base_x = robot_pos[0] + self.body_radius * np.cos(angles_rad[leg_number])
        leg_base_y = robot_pos[1] + self.body_radius * np.sin(angles_rad[leg_number])
        leg_base_z = robot_pos[2]
        
        # Walking parameters - REDUCED stride length for less broad appearance
        stride_length = 20  # Reduced from 30
        step_height = 15    # Reduced from 20
        
        # Determine foot position based on phase
        if phase < 0.5:  # Swing phase (leg in air)
            # Normalize to 0-1 for swing phase
            swing_phase = phase * 2
            
            # Calculate foot position during swing (moving forward in an arc)
            offset_x = stride_length * (swing_phase - 0.5) * np.cos(angles_rad[leg_number])
            offset_y = stride_length * (swing_phase - 0.5) * np.sin(angles_rad[leg_number])
            offset_z = step_height * np.sin(swing_phase * np.pi)
            
            # Foot position with offset from base
            foot_x = leg_base_x + self.leg_length * np.cos(angles_rad[leg_number]) + offset_x
            foot_y = leg_base_y + self.leg_length * np.sin(angles_rad[leg_number]) + offset_y
            
            # Get terrain height and add the step height for swing phase
            terrain_z = self.get_terrain_height(foot_x, foot_y)
            foot_z = terrain_z + offset_z
            
            # Set foot color to red for swing phase
            foot_color = 'red'
            
        else:  # Stance phase (foot on ground)
            # Normalize to 0-1 for stance phase
            stance_phase = (phase - 0.5) * 2
            
            # Calculate foot position during stance (moving backward in contact with ground)
            offset_x = stride_length * (0.5 - stance_phase) * np.cos(angles_rad[leg_number])
            offset_y = stride_length * (0.5 - stance_phase) * np.sin(angles_rad[leg_number])
            
            # Foot position with offset from base
            foot_x = leg_base_x + self.leg_length * np.cos(angles_rad[leg_number]) + offset_x
            foot_y = leg_base_y + self.leg_length * np.sin(angles_rad[leg_number]) + offset_y
            
            # Get terrain height for stance phase (foot on ground)
            foot_z = self.get_terrain_height(foot_x, foot_y)
            
            # Set foot color to green for stance phase
            foot_color = 'green'
        
        return [leg_base_x, leg_base_y, leg_base_z], [foot_x, foot_y, foot_z], foot_color
    
    def get_leg_phase(self, t, leg_number):
        """Determine the phase of a leg at time t"""
        # Tripod gait - alternating legs in two groups
        group1 = [0, 2, 4]  # Right front, right hind, left middle
        group2 = [1, 3, 5]  # Left front, left hind, right middle
        
        # Set phase based on group
        if leg_number in group1:
            phase = (t % 1.0)
        else:
            phase = (t + 0.5) % 1.0
            
        return phase
    
    def generate_walking_sequence(self):
        """Generate a sequence of 8 snapshots of the robot walking"""
        # Generate 8 positions along a path - UPDATED FOR BIGGER TERRAIN
        path_x = np.linspace(75, 225, 8)
        path_y = np.linspace(150, 150, 8)
        
        for i in range(8):
            # Update robot position along the path
            self.robot_position[0] = path_x[i]
            self.robot_position[1] = path_y[i]
            
            # Get z position from terrain
            terrain_z = self.get_terrain_height(self.robot_position[0], self.robot_position[1])
            self.robot_position[2] = terrain_z + self.body_height
            
            # Time parameter for walking cycle
            t = i * 0.125
            
            # Draw the robot in this snapshot
            self.draw_robot(i, t)
    
    def draw_robot(self, snapshot_idx, t):
        """Draw the robot at a specific time in a specific snapshot"""
        ax = self.snapshots[snapshot_idx]
        
        # Draw the body as a simple blue sphere
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 8)
        
        body_radius = self.body_radius
        body_x = self.robot_position[0] + body_radius * np.outer(np.cos(u), np.sin(v))
        body_y = self.robot_position[1] + body_radius * np.outer(np.sin(u), np.sin(v))
        body_z = self.robot_position[2] + body_radius/2 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(body_x, body_y, body_z, color='blue', alpha=0.8)
        
        # Draw each leg - REDUCED LINEWIDTH for thinner legs
        for leg in range(6):
            phase = self.get_leg_phase(t, leg)
            leg_base, foot_pos, foot_color = self.get_leg_position(leg, phase, self.robot_position)
            
            # Draw a simple straight line for the leg with reduced thickness
            leg_x = [leg_base[0], foot_pos[0]]
            leg_y = [leg_base[1], foot_pos[1]]
            leg_z = [leg_base[2], foot_pos[2]]
            
            ax.plot(leg_x, leg_y, leg_z, 'k-', linewidth=1.2)  # Reduced linewidth from 2 to 1.2
            
            # Draw the foot with appropriate color and smaller size
            ax.scatter(foot_pos[0], foot_pos[1], foot_pos[2], 
                      color=foot_color, s=35, zorder=10)  # Reduced size from 50 to 35
            
            # Draw the joint at the body with smaller size
            ax.scatter(leg_base[0], leg_base[1], leg_base[2], 
                     color='black', s=20, zorder=10)  # Reduced size from 30 to 20
    
    def show(self):
        """Display the visualization"""
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    print("Generating hexapod walking sequence...")
    viz = HexapodTerrainVisualization()
    viz.show()
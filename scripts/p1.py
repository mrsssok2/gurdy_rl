#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

def plot_upperlegM3_lowerlegM3_joint(joint_data):
    """
    Plot phase diagram for upperlegM3_lowerlegM3_joint
    """
    joint_name = "upperlegM3_lowerlegM3_joint"
    angles = joint_data[joint_name]["angles"]
    velocities = joint_data[joint_name]["velocities"]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(angles, velocities, s=2, alpha=0.6)
    plt.plot(angles, velocities, 'r-', linewidth=0.5, alpha=0.3)
    plt.title(f'{joint_name}')
    plt.xlabel('Joint Angle (rad)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{joint_name}_phase_diagram.png')
    plt.show()

# Example usage:
if __name__ == "__main__":
    try:
        # Create a gurdyJointPhaseAnalyzer instance and collect data
        from plot import gurdyJointPhaseAnalyzer  # Import your class
        gurdy_analyzer = gurdyJointPhaseAnalyzer()
        
        # Initialize and collect data
        gurdy_analyzer.gurdy_init_pos_sequence()
        gurdy_analyzer.gurdy_moverandomly(num_moves=5)
        gurdy_analyzer.gurdy_hop(num_hops=5)
        
        # Plot only the upperlegM3_lowerlegM3_joint phase diagram
        plot_upperlegM3_lowerlegM3_joint(gurdy_analyzer.joint_data)
        
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3

"""
Multi-algorithm Reinforcement Learning Launch Script for Gurdy Robot

This script launches multiple RL algorithms in parallel, each in their own
namespace, allowing for direct comparison of performance.
"""

import sys
import os
import yaml
import rospy
import rospkg
import threading
import argparse
from std_srvs.srv import Empty
from algorithm_manager import AlgorithmManager
from rl_visualization import RLVisualization
import algorithm_wrapper

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        rospy.logerr(f"Failed to load config from {config_path}: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Launch multiple RL algorithms for Gurdy robot')
    parser.add_argument('--config', type=str, default='rl_comparison_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Number of episodes to train (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization')
    parser.add_argument('--algorithms', type=str, nargs='+',
                      help='List of algorithms to run (default: all enabled in config)')
    
    # Filter out ROS-specific arguments (like __name and __log)
    import sys
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('__')]
    return parser.parse_args(filtered_args)

def main():
    """Main function for launching multiple RL algorithms"""
    # Initialize ROS node
    rospy.init_node('multi_rl_launch', anonymous=True)
    
    # Parse arguments
    args = parse_args()
    
    # Get paths
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_gurdy_description')
    config_path = os.path.join(pkg_path, 'config', args.config)
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        rospy.logerr("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Override episodes if provided via command line
    if args.episodes is not None:
        config['training']['episodes'] = args.episodes
        
    # Filter algorithms if specified
    if args.algorithms:
        # Only keep specified algorithms that are also in the config
        valid_algos = {algo: config['algorithms'][algo] 
                      for algo in args.algorithms 
                      if algo in config['algorithms']}
        
        # Make sure all other algorithms are disabled
        for algo in config['algorithms']:
            if algo not in valid_algos:
                config['algorithms'][algo]['enabled'] = False
                
        # Update config with valid algorithms
        config['algorithms'] = valid_algos
    
    # Initialize visualization if enabled
    visualization = None
    if args.visualize or config.get('visualization', {}).get('live_update', True):
        enabled_algos = [algo for algo, settings in config['algorithms'].items() 
                        if settings.get('enabled', True)]
        visualization = RLVisualization(enabled_algos, 
                                       update_interval=config.get('visualization', {}).get('update_interval', 1.0))
    
    # Initialize algorithm manager
    manager = AlgorithmManager(config['algorithms'], config, visualization)
    
    # Ensure Gazebo is unpaused (for simulation)
    if config['environment']['reset_world_or_sim'] == "SIMULATION":
        try:
            rospy.loginfo("Unpausing Gazebo physics...")
            unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            unpause()
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to unpause Gazebo: {e}")
    
    # Run all algorithms
    rospy.loginfo("Starting training for all enabled algorithms...")
    manager.run_all()
    
    # Start visualization
    if visualization:
        visualization.show()
    
    # Wait for all algorithms to complete or ROS to shutdown
    rate = rospy.Rate(1)  # 1 Hz
    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received, stopping...")
    finally:
        # Stop all algorithms
        manager.stop_all()
        
        # Save results
        manager.save_results()
        
        # Wait for visualization to be closed if active
        if visualization and visualization.is_active():
            rospy.loginfo("Waiting for visualization to be closed...")
            try:
                while visualization.is_active() and not rospy.is_shutdown():
                    rate.sleep()
            except KeyboardInterrupt:
                pass
            finally:
                visualization.close()

if __name__ == "__main__":
    main()
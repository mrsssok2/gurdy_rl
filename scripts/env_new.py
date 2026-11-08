#!/usr/bin/env python3
import gym
import rospy
import numpy as np
from gym import spaces
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import time

class MyGurdyWalkEnv(gym.Env):
    def __init__(self):
        # Ensure ROS is initialized BEFORE calling the parent constructor
        if not rospy.core.is_initialized():
            rospy.init_node('gurdy_env', anonymous=True)
        
        # Call parent constructor AFTER ROS initialization
        super(MyGurdyWalkEnv, self).__init__()
        
        # Joint publishers for all 6 legs
        self.joint_publishers = {
            'head_upperlegM1': rospy.Publisher('/gurdy/head_upperlegM1_joint_position_controller/command', Float64, queue_size=10),
            'head_upperlegM2': rospy.Publisher('/gurdy/head_upperlegM2_joint_position_controller/command', Float64, queue_size=10),
            'head_upperlegM3': rospy.Publisher('/gurdy/head_upperlegM3_joint_position_controller/command', Float64, queue_size=10),
            'head_upperlegM4': rospy.Publisher('/gurdy/head_upperlegM4_joint_position_controller/command', Float64, queue_size=10),
            'head_upperlegM5': rospy.Publisher('/gurdy/head_upperlegM5_joint_position_controller/command', Float64, queue_size=10),
            'head_upperlegM6': rospy.Publisher('/gurdy/head_upperlegM6_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM1_lowerlegM1': rospy.Publisher('/gurdy/upperlegM1_lowerlegM1_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM2_lowerlegM2': rospy.Publisher('/gurdy/upperlegM2_lowerlegM2_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM3_lowerlegM3': rospy.Publisher('/gurdy/upperlegM3_lowerlegM3_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM4_lowerlegM4': rospy.Publisher('/gurdy/upperlegM4_lowerlegM4_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM5_lowerlegM5': rospy.Publisher('/gurdy/upperlegM5_lowerlegM5_joint_position_controller/command', Float64, queue_size=10),
            'upperlegM6_lowerlegM6': rospy.Publisher('/gurdy/upperlegM6_lowerlegM6_joint_position_controller/command', Float64, queue_size=10)
        }
        
        # Subscribers to get robot state
        self.joint_state_sub = rospy.Subscriber('/gurdy/joint_states', JointState, self.joint_state_callback)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Forward, Backward, Stop
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -1.0, 0.0]), 
            high=np.array([np.pi, 1.0, 10.0]), 
            dtype=np.float32
        )
        
        # Internal state tracking
        self.current_state = np.zeros(3)
        self.step_count = 0
        self.walking_phase = 0
        
        # Wait for publishers to be ready
        self._check_publishers_connection()
    
    def _check_publishers_connection(self):
        """Ensure all publishers are connected before proceeding"""
        connections = {}
        for name, publisher in self.joint_publishers.items():
            connections[name] = publisher.get_num_connections()
        
        # Wait until all publishers have at least one subscriber
        while not all(conn > 0 for conn in connections.values()):
            rospy.loginfo("Waiting for publishers to connect...")
            time.sleep(1)
            connections = {name: publisher.get_num_connections() 
                           for name, publisher in self.joint_publishers.items()}
        
        rospy.loginfo("All publishers connected successfully!")
    
    def joint_state_callback(self, msg):
        """Callback to update current robot state"""
        if len(msg.position) > 0:
            self.current_state = [
                msg.position[0] if len(msg.position) > 0 else 0.0,  # Joint angle
                0.0,  # Linear speed (placeholder)
                0.0   # Distance traveled (placeholder)
            ]
    
    def reset(self):
        """Reset the environment to initial state"""
        self.step_count = 0
        self.walking_phase = 0
        
        # Reset joint positions to neutral
        for publisher in self.joint_publishers.values():
            cmd = Float64()
            cmd.data = 0.0  # Neutral position
            publisher.publish(cmd)
        
        # Wait for reset to take effect
        rospy.sleep(1.0)
        
        return self.current_state
    
    def _tripod_gait(self, direction):
        """
        Implement a tripod gait for hexapod walking
        Tripod gait: Alternate groups of 3 legs moving together
        """
        # Walking coefficients based on direction
        sign = 1 if direction == 1 else -1
        
        # Define leg groups (opposite diagonal legs move together)
        leg_groups = [
            # Group 1: Lift and move forward
            {
                'head_upperlegM1': 0.5 * sign, 
                'upperlegM1_lowerlegM1': 0.3 * sign,
                'head_upperlegM3': 0.5 * sign, 
                'upperlegM3_lowerlegM3': 0.3 * sign,
                'head_upperlegM5': 0.5 * sign, 
                'upperlegM5_lowerlegM5': 0.3 * sign
            },
            # Group 2: Opposite group
            {
                'head_upperlegM2': -0.5 * sign, 
                'upperlegM2_lowerlegM2': -0.3 * sign,
                'head_upperlegM4': -0.5 * sign, 
                'upperlegM4_lowerlegM4': -0.3 * sign,
                'head_upperlegM6': -0.5 * sign, 
                'upperlegM6_lowerlegM6': -0.3 * sign
            }
        ]
        
        # Alternate between leg groups
        return leg_groups[self.walking_phase % 2]
    
    def step(self, action):
        """Execute one timestep in the environment"""
        self.step_count += 1
        
        # Translate action to joint commands
        if action == 0:  # Stop
            joint_commands = {pub: 0.0 for pub in self.joint_publishers}
        elif action == 1:  # Move Forward
            joint_commands = self._tripod_gait(1)
            self.walking_phase += 1
        else:  # Move Backward
            joint_commands = self._tripod_gait(-1)
            self.walking_phase += 1
        
        # Send joint commands
        for joint, cmd in joint_commands.items():
            pub = self.joint_publishers[joint]
            msg = Float64()
            msg.data = cmd
            pub.publish(msg)
        
        # Calculate reward (placeholder)
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.step_count >= 100 or self._is_fallen()
        
        return self.current_state, reward, done, {}
    
    def _calculate_reward(self, action):
        """
        Calculate reward based on robot's movement
        """
        if action == 0:  # Stopped
            return -1.0
        elif action == 1:  # Moving Forward
            return 1.0
        else:  # Moving Backward
            return 0.5
    
    def _is_fallen(self):
        """
        Check if robot has fallen
        Placeholder implementation
        """
        return False  # Simplified fallen detection
    
    def render(self, mode='human'):
        """Rendering method (optional)"""
        pass

# Register the environment
gym.register(
    id='MyGurdyWalkEnv-v0',
    entry_point='env:MyGurdyWalkEnv'
)
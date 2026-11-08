#!/usr/bin/env python

import rospy
import numpy
from robot_gazebo_env import RobotGazeboEnv
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import time

class MyGurdySingleLegEnv(RobotGazeboEnv):
    """Superclass for Gurdy robot environments"""

    def __init__(self):
        # Initialize joints and odom attributes
        self.joints = None
        self.odom = None

        # Controllers list
        self.controllers_list = [
            'joint_state_controller',
            'head_upperlegM1_joint_position_controller',
            'head_upperlegM2_joint_position_controller',
            'head_upperlegM3_joint_position_controller',
        ]

        self.robot_name_space = "gurdy"

        # Initialize parent class
        super(MyGurdySingleLegEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=True
        )

        # Episode variables
        self._episode_done = False

        # Unpause simulation and reset controllers
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()

        # ROS Subscribers
        rospy.Subscriber("/gurdy/joint_states", JointState, self._joints_callback)
        rospy.Subscriber("/gurdy/odom", Odometry, self._odom_callback)

        # Joint position publishers
        self._joint_pub_list = [
            rospy.Publisher('/gurdy/head_upperlegM1_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM2_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM3_joint_position_controller/command', Float64, queue_size=1),
        ]

        self._check_publishers_connection()
        
        # Wait for sensors to be ready
        self._check_all_sensors_ready()

        self.gazebo.pauseSim()
        
        # Successful step counter (for tracking progress)
        self.successful_steps = 0

    def _joints_callback(self, data):
        """
        Callback for joint states
        
        Args:
            data (JointState): Joint state message
        """
        self.joints = data

    def _odom_callback(self, data):
        """
        Callback for odometry
        
        Args:
            data (Odometry): Odometry message
        """
        self.odom = data

    def _check_all_sensors_ready(self):
        """
        Check if all sensors are ready
        
        Returns:
            bool: Whether all sensors are ready
        """
        rospy.loginfo("Checking all sensors...")
        
        # Check joint states
        try:
            if self.joints is None:
                rospy.logwarn("Waiting for joint states...")
                rospy.wait_for_message("/gurdy/joint_states", JointState, timeout=5.0)
            rospy.loginfo("Joint states ready")
        except rospy.ROSException as e:
            rospy.logerr("Joint states not ready: " + str(e))
            return False
        
        # Check odometry
        try:
            if self.odom is None:
                rospy.logwarn("Waiting for odometry...")
                rospy.wait_for_message("/gurdy/odom", Odometry, timeout=5.0)
            rospy.loginfo("Odometry ready")
        except rospy.ROSException as e:
            rospy.logerr("Odometry not ready: " + str(e))
            return False
        
        rospy.loginfo("All sensors are ready")
        return True

    def _check_publishers_connection(self):
        """
        Check that all publishers are connected
        """
        rate = rospy.Rate(10)  # 10 Hz
        connected = False
        while not connected and not rospy.is_shutdown():
            connected = True
            for publisher in self._joint_pub_list:
                if publisher.get_num_connections() == 0:
                    connected = False
                    break
                
            if not connected:
                rospy.loginfo("Waiting for subscribers to joint position controllers...")
                rate.sleep()
            
        rospy.loginfo("Joint position publishers connected")

    def reset(self):
        """
        Gym environment reset method
        
        Returns:
            list: Initial observation
        """
        rospy.loginfo("Resetting Gurdy environment")
        
        # Unpause the simulation
        self.gazebo.unpauseSim()
        
        # Reset robot to initial state
        self._set_init_pose()
        
        # Initialize environment variables
        self._init_env_variables()
        
        # Reset episode done flag
        self._episode_done = False
        
        # Get initial observation
        observation = self._get_obs()
        
        # Pause simulation
        self.gazebo.pauseSim()
        
        # Reset successful steps counter
        self.successful_steps = 0
        
        return observation

    def step(self, action):
        """
        Gym environment step method
        
        Args:
            action (int): Action to take
            
        Returns:
            list: Observation
            float: Reward
            bool: Done
            dict: Info
        """
        rospy.loginfo(f"Executing action: {action}")
        
        # Unpause simulation
        self.gazebo.unpauseSim()
        
        # Set the action
        self._set_action(action)
        
        # Wait for physics to stabilize
        time.sleep(0.2)
        
        # Get observation
        observation = self._get_obs()
        
        # Check if episode is done
        done = self._is_done(observation)
        
        # Compute reward
        reward = self._compute_reward(observation, done)
        
        # Update episode done status
        self._episode_done = done
        
        # Increment successful steps if not done
        if not done:
            self.successful_steps += 1
        
        # Additional info
        info = {
            'successful_steps': self.successful_steps
        }
        
        # Pause simulation
        self.gazebo.pauseSim()
        
        return observation, reward, done, info

    def move_joints(self, joint_positions):
        """
        Move robot joints
        
        Args:
            joint_positions (list): List of joint positions
        """
        # Add a check to ensure we have the right number of positions
        if len(joint_positions) != len(self._joint_pub_list):
            rospy.logerr(f"Expected {len(self._joint_pub_list)} joint positions, got {len(joint_positions)}")
            return
            
        for i, pub in enumerate(self._joint_pub_list):
            pub.publish(joint_positions[i])
        
        # Give time for joints to move
        time.sleep(0.1)
        
    def get_joint_angles(self):
        """
        Get joint angles from joint states
        
        Returns:
            list or None: Joint angles or None if not available
        """
        if self.joints is not None and len(self.joints.position) > 0:
            return self.joints.position
        return None
    
    def get_joint_velocities(self):
        """
        Get joint velocities from joint states
        
        Returns:
            list or None: Joint velocities or None if not available
        """
        if self.joints is not None and len(self.joints.velocity) > 0:
            return self.joints.velocity
        return None

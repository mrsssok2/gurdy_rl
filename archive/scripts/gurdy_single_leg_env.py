#!/usr/bin/env python

import numpy
import rospy
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

class GurdySingleLegEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for Gurdy robot environments"""

    def __init__(self):
        # Controllers list for Gurdy robot
        self.controllers_list = [
            'joint_state_controller',
            'head_upperlegM1_joint_position_controller',
            'head_upperlegM2_joint_position_controller',
            'head_upperlegM3_joint_position_controller',
            'head_upperlegM4_joint_position_controller',
            'head_upperlegM5_joint_position_controller',
            'head_upperlegM6_joint_position_controller',
            'upperlegM1_lowerlegM1_joint_position_controller',
            'upperlegM2_lowerlegM2_joint_position_controller',
            'upperlegM3_lowerlegM3_joint_position_controller',
            'upperlegM4_lowerlegM4_joint_position_controller',
            'upperlegM5_lowerlegM5_joint_position_controller',
            'upperlegM6_lowerlegM6_joint_position_controller'
        ]

        self.robot_name_space = "gurdy"

        # Initialize parent class
        super(GurdySingleLegEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=True
        )

        # Unpause simulation and reset controllers
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # ROS Subscribers
        rospy.Subscriber("/gurdy/joint_states", JointState, self._joints_callback)
        rospy.Subscriber("/gurdy/odom", Odometry, self._odom_callback)

        # Joint publishers for each leg joint
        self._leg_joint_publishers = [
            rospy.Publisher('/gurdy/head_upperlegM1_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM2_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM3_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM4_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM5_joint_position_controller/command', Float64, queue_size=1),
            rospy.Publisher('/gurdy/head_upperlegM6_joint_position_controller/command', Float64, queue_size=1)
        ]

        self._check_publishers_connection()

        # Pause simulation
        self.gazebo.pauseSim()

    def _check_all_systems_ready(self):
        """Check that all sensors and systems are operational"""
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        """Check individual sensors"""
        self._check_joint_states_ready()
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        """Wait for joint states message"""
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/gurdy/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current gurdy/joint_states READY=>" + str(self.joints))
            except:
                rospy.logerr("Current gurdy/joint_states not ready yet, retrying")
        return self.joints

    def _check_odom_ready(self):
        """Wait for odometry message"""
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/gurdy/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /gurdy/odom READY=>" + str(self.odom))
            except:
                rospy.logerr("Current /gurdy/odom not ready yet, retrying")
        return self.odom

    def _joints_callback(self, data):
        """Joint states callback"""
        self.joints = data

    def _odom_callback(self, data):
        """Odometry callback"""
        self.odom = data

    def _check_publishers_connection(self):
        """Check that all joint publishers are connected"""
        rate = rospy.Rate(10)
        for publisher in self._leg_joint_publishers:
            while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug(f"No subscribers to {publisher.name} yet, waiting")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    pass
            rospy.logdebug(f"{publisher.name} Publisher Connected")

    # Virtual methods to be implemented by child classes
    def _set_init_pose(self):
        """Sets the Robot in its initial pose"""
        raise NotImplementedError()

    def _init_env_variables(self):
        """Initializes variables at start of episode"""
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates reward based on observations"""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies action to the robot"""
        raise NotImplementedError()

    def _get_obs(self):
        """Gets current observations"""
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode is complete"""
        raise NotImplementedError()

    def move_leg_joints(self, joint_positions):
        """
        Move multiple leg joints
        :param joint_positions: List of joint positions for each leg
        """
        for pub, pos in zip(self._leg_joint_publishers, joint_positions):
            joint_command = Float64()
            joint_command.data = pos
            pub.publish(joint_command)
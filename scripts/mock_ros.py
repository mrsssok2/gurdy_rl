#!/usr/bin/env python
"""
Mock implementations of ROS classes for demonstration purposes
"""

import math
import time
import random
import numpy as np
from dataclasses import dataclass

# Mock ROS messages and services
class Point:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

class Quaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0

class Vector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

class Twist:
    def __init__(self):
        self.linear = Vector3()
        self.angular = Vector3()

class TwistWithCovariance:
    def __init__(self):
        self.twist = Twist()
        self.covariance = [0.0] * 36

class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()

class PoseWithCovariance:
    def __init__(self):
        self.pose = Pose()
        self.covariance = [0.0] * 36

class Odometry:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.pose = PoseWithCovariance()
        self.twist = TwistWithCovariance()

class JointState:
    def __init__(self):
        self.header = Header()
        self.name = ["head_upperlegM1_joint", "head_upperlegM2_joint", "head_upperlegM3_joint"]
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.effort = [0.0, 0.0, 0.0]

class Header:
    def __init__(self):
        self.seq = 0
        self.stamp = Time()
        self.frame_id = ""

class Time:
    def __init__(self):
        self.secs = int(time.time())
        self.nsecs = 0

    def now(self):
        self.secs = int(time.time())
        return self

class Empty:
    def __init__(self):
        pass

class Float64:
    def __init__(self, data=0.0):
        self.data = data

class ODEPhysics:
    def __init__(self):
        self.auto_disable_bodies = False
        self.sor_pgs_precon_iters = 0
        self.sor_pgs_iters = 50
        self.sor_pgs_w = 1.3
        self.sor_pgs_rms_error_tol = 0.0
        self.contact_surface_layer = 0.001
        self.contact_max_correcting_vel = 0.0
        self.cfm = 0.0
        self.erp = 0.2
        self.max_contacts = 20

class SetPhysicsPropertiesRequest:
    def __init__(self):
        self.time_step = 0.0
        self.max_update_rate = 0.0
        self.gravity = Vector3()
        self.ode_config = ODEPhysics()

class SetPhysicsPropertiesResponse:
    def __init__(self):
        self.success = True
        self.status_message = "Success"

class RLExperimentInfo:
    def __init__(self):
        self.episode_number = 0
        self.episode_reward = 0.0

class SwitchControllerRequest:
    def __init__(self):
        self.start_controllers = []
        self.stop_controllers = []
        self.strictness = 1

class SwitchControllerResponse:
    def __init__(self):
        self.ok = True

# Mock ROS functions and classes
class Publisher:
    def __init__(self, topic, msg_type, queue_size=10):
        self.topic = topic
        self.msg_type = msg_type
        self.queue_size = queue_size
        self.connections = 0
        print(f"Created publisher on topic {topic}")

    def publish(self, msg):
        # In a real system, this would publish the message
        pass

    def get_num_connections(self):
        # Simulate connections after a while
        self.connections = 1
        return self.connections

class Subscriber:
    def __init__(self, topic, msg_type, callback):
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        # Initialize the message based on type
        if msg_type == JointState:
            self.msg = JointState()
        elif msg_type == Odometry:
            self.msg = Odometry()
            # Set random values for odometry to simulate movement
            self.msg.twist.twist.linear.x = random.uniform(0.0, 0.5)
            self.msg.pose.pose.position.x = random.uniform(0.0, 1.0)
        print(f"Created subscriber on topic {topic}")
        
        # Call the callback with the mock data
        if callback:
            callback(self.msg)

class Rate:
    def __init__(self, rate):
        self.rate = rate
        self.last_time = time.time()
        self.sleep_time = 1.0 / rate

    def sleep(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed < self.sleep_time:
            time.sleep(self.sleep_time - elapsed)
        self.last_time = time.time()

# Define service types for better type checking
class SetPhysicsProperties:
    pass

class SwitchController:
    pass

class ServiceProxy:
    def __init__(self, service_name, service_type):
        self.service_name = service_name
        self.service_type = service_type
        print(f"Created service proxy for {service_name}")

    def __call__(self, request=None):
        if self.service_type == SetPhysicsProperties:
            return SetPhysicsPropertiesResponse()
        elif self.service_type == SwitchController:
            return SwitchControllerResponse()
        else:
            # Generic empty response
            return Empty()

# Global state
_initialized = False
_param_server = {
    "/gurdy/n_actions": 5,
    "/gurdy/max_joint_angle": 3.14,
    "/gurdy/max_linear_speed": 1.0,
    "/gurdy/max_distance": 10.0,
    "/gurdy/move_distance_reward_weight": 1.5,
    "/gurdy/linear_speed_reward_weight": 1.0,
    "/gurdy/joint_angle_reward_weight": 0.1,
    "/gurdy/energy_penalty_weight": 0.05,
    "/gurdy/end_episode_points": -100,
    "/gurdy/alpha": 0.15,
    "/gurdy/epsilon": 1.0,
    "/gurdy/gamma": 0.95,
    "/gurdy/epsilon_discount": 0.995,
    "/gurdy/min_epsilon": 0.1,
    "/gurdy/nepisodes": 100,  # Reduced for demonstration
    "/gurdy/nsteps": 150,     # Reduced for demonstration
    "/gurdy/load_model_path": "",
    "/gurdy/save_model_freq": 10
}

# ROS simulation functions
def init_node(name, anonymous=False, log_level=None):
    global _initialized
    _initialized = True
    print(f"Initialized ROS node: {name}")

def is_shutdown():
    return False

def on_shutdown(callback):
    pass

def get_param(param_name, default=None):
    return _param_server.get(param_name, default)

def set_param(param_name, value):
    _param_server[param_name] = value

def loginfo(msg):
    print(f"INFO: {msg}")

def logwarn(msg):
    print(f"WARN: {msg}")

def logerr(msg):
    print(f"ERROR: {msg}")

def logdebug(msg):
    pass  # Silenced for cleaner output

def wait_for_service(service_name, timeout=None):
    print(f"Waiting for service {service_name}...")
    time.sleep(0.1)  # Simulate short wait
    print(f"Service {service_name} found")

def wait_for_message(topic, msg_type, timeout=None):
    print(f"Waiting for message on {topic}...")
    
    # Create appropriate message based on type
    if msg_type == JointState:
        msg = JointState()
        # Simulate random joint positions
        msg.position = [random.uniform(-0.5, 0.5) for _ in range(3)]
    elif msg_type == Odometry:
        msg = Odometry()
        # Set random values for odometry to simulate movement
        msg.twist.twist.linear.x = random.uniform(0.0, 0.5)
        msg.pose.pose.position.x = random.uniform(0.0, 1.0)
    else:
        msg = None
    
    time.sleep(0.1)  # Simulate short wait
    print(f"Message received on {topic}")
    return msg

class ROSException(Exception):
    pass

class ServiceException(Exception):
    pass
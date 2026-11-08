#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection():
    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION", max_retry=20):
        """
        Initialize Gazebo connection manager
        
        Args:
            start_init_physics_parameters (bool): Whether to initialize physics parameters
            reset_world_or_sim (str): "SIMULATION", "WORLD", or "NO_RESET_SIM"
            max_retry (int): Maximum number of retries for service calls
        """
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Setup the Physics parameters
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.max_retry = max_retry
        self.reset_count = 0
        
        # Set initial physics parameters
        if start_init_physics_parameters:
            self.init_physics_parameters()
            
    def pauseSim(self):
        """
        Pause the Gazebo simulation
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("Pause physics service call failed: %s"%e)
            
    def unpauseSim(self):
        """
        Unpause the Gazebo simulation
        """
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("Unpause physics service call failed: %s"%e)
            
    def resetSim(self):
        """
        Reset the simulation based on reset_world_or_sim setting
        """
        self.reset_count += 1
        
        if self.reset_world_or_sim == "SIMULATION":
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("NO_RESET_SIM set, not resetting simulation")
        
        # If too many resets, try reconfiguring physics
        if self.reset_count >= self.max_retry and self.start_init_physics_parameters:
            self.reset_count = 0
            self.init_physics_parameters()
    
    def resetSimulation(self):
        """
        Reset the entire simulation
        """
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("Reset simulation service call failed: %s"%e)
    
    def resetWorld(self):
        """
        Reset only the world (object positions)
        """
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("Reset world service call failed: %s"%e)
    
    def init_physics_parameters(self):
        """
        Initialize the physics parameters of the simulation
        """
        rospy.wait_for_service('/gazebo/set_physics_properties')
        
        try:
            # Create the request
            physics_request = SetPhysicsPropertiesRequest()
            
            # Set time step
            physics_request.time_step = Float64(0.001)
            
            # Set max update rate
            physics_request.max_update_rate = Float64(1000.0)
            
            # Set gravity
            gravity = Vector3()
            gravity.x = 0.0
            gravity.y = 0.0
            gravity.z = -9.81
            physics_request.gravity = gravity
            
            # ODE parameters
            ode_config = ODEPhysics()
            ode_config.auto_disable_bodies = False
            ode_config.sor_pgs_precon_iters = 0
            ode_config.sor_pgs_iters = 50
            ode_config.sor_pgs_w = 1.3
            ode_config.sor_pgs_rms_error_tol = 0.0
            ode_config.contact_surface_layer = 0.001
            ode_config.contact_max_correcting_vel = 100.0
            ode_config.cfm = 0.0
            ode_config.erp = 0.2
            ode_config.max_contacts = 20
            physics_request.ode_config = ode_config
            
            # Call the service
            result = self.set_physics(physics_request)
            rospy.logdebug("Physics parameters set: %s", result.success)
            
            return result.success
        except rospy.ServiceException as e:
            rospy.logerr("Set physics properties service call failed: %s"%e)
            return False
    
    def update_gravity_call(self):
        """
        Update gravity settings in the simulator
        """
        # Save current settings
        unpause = self.pause_physics_client
        pause = self.unpause_physics_client
        reset = self.reset_simulation_client
        
        # Call update
        self.change_gravity(0.0, 0.0, -9.81)
        
        # Restore settings
        unpause = self.unpause_physics_client
        pause = self.pause_physics_client
        reset = self.reset_simulation_client
    
    def change_gravity(self, x, y, z):
        """
        Change gravity settings
        
        Args:
            x (float): X component of gravity
            y (float): Y component of gravity
            z (float): Z component of gravity
        """
        rospy.wait_for_service('/gazebo/set_physics_properties')
        
        try:
            # Create request
            physics_request = SetPhysicsPropertiesRequest()
            
            # Set time step
            physics_request.time_step = Float64(0.001)
            
            # Set max update rate
            physics_request.max_update_rate = Float64(1000.0)
            
            # Set gravity
            gravity = Vector3()
            gravity.x = x
            gravity.y = y
            gravity.z = z
            physics_request.gravity = gravity
            
            # ODE parameters
            ode_config = ODEPhysics()
            ode_config.auto_disable_bodies = False
            ode_config.sor_pgs_precon_iters = 0
            ode_config.sor_pgs_iters = 50
            ode_config.sor_pgs_w = 1.3
            ode_config.sor_pgs_rms_error_tol = 0.0
            ode_config.contact_surface_layer = 0.001
            ode_config.contact_max_correcting_vel = 100.0
            ode_config.cfm = 0.0
            ode_config.erp = 0.2
            ode_config.max_contacts = 20
            physics_request.ode_config = ode_config
            
            # Call the service
            result = self.set_physics(physics_request)
            rospy.logdebug("Gravity changed: %s", result.success)
            
            return result.success
        except rospy.ServiceException as e:
            rospy.logerr("Set physics properties service call failed: %s"%e)
            return False
#!/usr/bin/env python

import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse

class ControllersConnection():
    def __init__(self, namespace, controllers_list):
        """
        Initialize the controllers connection manager
        
        Args:
            namespace (str): ROS namespace
            controllers_list (list): List of controller names
        """
        rospy.logwarn("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.switch_service_name = '/'+namespace+'/controller_manager/switch_controller'
        self.switch_service = rospy.ServiceProxy(self.switch_service_name, SwitchController)
        rospy.logwarn("END Init ControllersConnection")

    def switch_controllers(self, controllers_on, controllers_off, strictness=1):
        """
        Give the controllers you want to switch on or off.
        
        Args:
            controllers_on (list): ["name_controller_1", "name_controller2",...,"name_controller_n"]
            controllers_off (list): ["name_controller_1", "name_controller2",...,"name_controller_n"]
            strictness (int): 1 for BEST_EFFORT, 2 for STRICT
            
        Returns:
            bool: True if successful, False otherwise
        """
        rospy.wait_for_service(self.switch_service_name)
        
        try:
            switch_request = SwitchControllerRequest()
            switch_request.start_controllers = controllers_on
            switch_request.stop_controllers = controllers_off
            switch_request.strictness = strictness
            
            # Call the service
            switch_result = self.switch_service(switch_request)
            return switch_result.ok
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)
            return False

    def reset_controllers(self):
        """
        Turn off and on the given controllers
        
        Returns:
            bool: True if successful, False otherwise
        """
        reset_result = False
        
        # Turn off all controllers
        rospy.logdebug("Turning off all controllers...")
        off_result = self.switch_controllers([], self.controllers_list)
        rospy.logdebug("Turn off controllers result=%s", off_result)
        
        # Wait a short time to ensure controllers are off
        rospy.sleep(0.1)
        
        # Turn on all controllers
        rospy.logdebug("Turning on all controllers...")
        on_result = self.switch_controllers(self.controllers_list, [])
        rospy.logdebug("Turn on controllers result=%s", on_result)
        
        reset_result = off_result and on_result
        return reset_result

    def update_controllers_list(self, new_controllers_list):
        """
        Update the list of controllers
        
        Args:
            new_controllers_list (list): New list of controller names
        """
        self.controllers_list = new_controllers_list
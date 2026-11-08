#!/usr/bin/env python3
"""
Put Gurdy back at its spawn x, y and orientation with its legs already in a
given pose, held just above the ground.

/gazebo/reset_world on its own leaves the robot at its spawn height (0.18 m)
and the legs at whatever angles they had, so it falls ~0.1 m while the
controllers are still moving the legs, and a leg can land folded the wrong
way. Here, with physics paused, the joints are set to the target angles and
the base is placed DROP_HEIGHT above the height at which those angles put the
lowest foot on the ground, so the legs are in position before they touch.
"""
import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelConfiguration, SetModelState
from std_srvs.srv import Empty

# Leg geometry from robot/gurdy.xacro
HIP_Z = -0.008          # hip joints below the base_link origin, m
UPPER_LEG_LENGTH = 0.06  # hip to knee, m
LOWER_LEG_LENGTH = 0.06  # knee to foot centre, m
FOOT_RADIUS = 0.008     # m
DROP_HEIGHT = 0.02      # clearance above standing height on reset, m


def foot_depth(upper, lower):
    """
    Height of a foot's centre below its hip, from the joint angles. The
    thigh is `upper` from vertical and the shin `upper - lower` (see the
    joint directions note above NEUTRAL_POSE in env.py).
    """
    return (UPPER_LEG_LENGTH * math.cos(upper) +
            LOWER_LEG_LENGTH * math.cos(upper - lower))


def standing_height(positions):
    """
    base_link height at which the lowest foot just touches flat ground.
    positions: 12 angles, upper legs M1-M6 then lower legs M1-M6.
    """
    depth = max(foot_depth(u, l) for u, l in zip(positions[:6], positions[6:]))
    return -HIP_Z + depth + FOOT_RADIUS


def _call(name, srv_type, **kwargs):
    """Wait for and call a Gazebo service, logging failures. Returns the response or None."""
    rospy.wait_for_service(name)
    try:
        return rospy.ServiceProxy(name, srv_type)(**kwargs)
    except rospy.ServiceException as e:
        rospy.logerr("%s call failed: %s", name, e)
        return None


def reset_to_pose(joint_names, positions, before_unpause=None,
                  model_name="gurdy", urdf_param_name="robot_description"):
    """
    Pause physics, reset the world, set the joints to `positions`, place the
    base DROP_HEIGHT above standing_height(positions) at the spawn x, y and
    orientation with zero velocity, call `before_unpause` (use it to point the
    position controllers at the same angles), then unpause.

    Returns the base height it was placed at.
    """
    z = standing_height(positions) + DROP_HEIGHT
    _call('/gazebo/pause_physics', Empty)
    try:
        # reset_world (not reset_simulation) keeps sim time running
        _call('/gazebo/reset_world', Empty)

        resp = _call('/gazebo/set_model_configuration', SetModelConfiguration,
                     model_name=model_name, urdf_param_name=urdf_param_name,
                     joint_names=joint_names, joint_positions=positions)
        if resp is not None and not resp.success:
            rospy.logerr("/gazebo/set_model_configuration failed: %s",
                         resp.status_message)

        # Keep the spawn x, y and orientation that reset_world restored
        state = ModelState(model_name=model_name, reference_frame="world")
        resp = _call('/gazebo/get_model_state', GetModelState,
                     model_name=model_name, relative_entity_name="world")
        if resp is not None and resp.success:
            state.pose = resp.pose
        else:
            rospy.logerr("/gazebo/get_model_state failed; placing at x=y=0, level")
            state.pose.orientation.w = 1.0
        state.pose.position.z = z  # twist defaults to zero

        resp = _call('/gazebo/set_model_state', SetModelState, model_state=state)
        if resp is not None and not resp.success:
            rospy.logerr("/gazebo/set_model_state failed: %s", resp.status_message)

        if before_unpause is not None:
            before_unpause()
    finally:
        # Always unpause, or any sim-time wait that follows never returns
        _call('/gazebo/unpause_physics', Empty)
    return z

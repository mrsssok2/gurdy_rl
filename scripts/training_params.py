#!/usr/bin/env python3
"""
Training parameters shared by the training scripts.
"""
import rospy


def get_nepisodes(default):
    """
    Number of episodes to train for.

    Reads /gurdy/nepisodes. If /gurdy/smoke_test is true, returns
    /gurdy/smoke_test_episodes (default 5) instead, for a short run that
    checks the whole pipeline (env, agent, saving, plots) before a real run.
    Turn it on in the params yaml, or with smoke_test:=true on the launch
    file.

    Args:
        default (int): Episodes when /gurdy/nepisodes is not set

    Returns:
        int: Number of episodes
    """
    nepisodes = rospy.get_param("/gurdy/nepisodes", default)
    if rospy.get_param("/gurdy/smoke_test", False):
        nepisodes = int(rospy.get_param("/gurdy/smoke_test_episodes", 5))
        rospy.logwarn("SMOKE TEST: running %d episodes only", nepisodes)
    return nepisodes

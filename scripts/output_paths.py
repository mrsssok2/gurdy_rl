#!/usr/bin/env python
"""
Resolve training output directories relative to the my_gurdy_description
package, wherever the workspace lives.
"""
import os

import rospkg
import rospy

PACKAGE_NAME = 'my_gurdy_description'


def get_outdir(subfolder):
    """
    Get the output directory for a training run and make sure it exists.

    The /gurdy/outdir rosparam still overrides the default. An absolute
    value is used as is; a relative one (as in the config yaml files) is
    taken relative to the package directory.

    Args:
        subfolder (str): Default subfolder inside the package, e.g. "output/dqn"

    Returns:
        str: Absolute output directory path
    """
    package_path = rospkg.RosPack().get_path(PACKAGE_NAME)
    outdir = rospy.get_param("/gurdy/outdir", subfolder)
    if not os.path.isabs(outdir):
        outdir = os.path.join(package_path, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

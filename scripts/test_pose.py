#!/usr/bin/env python3
"""
Try a Gurdy stance by hand.

Sends 12 joint angles to the position controllers, waits 2 s (sim time),
then prints commanded vs actual angle for every joint (from
/gurdy/joint_states) and the base roll, pitch and height from /gurdy/odom.

Angles are in radians, ordered upper legs M1-M6, then lower legs M1-M6
(the same order as NEUTRAL_POSE in env.py). Limits: upper -1.55..0,
lower -2.9..1.57; out-of-range values are clipped with a warning.

Usage (with Gazebo and the controllers running):
    rosrun my_gurdy_description test_pose.py -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 \\
        -0.6 -0.6 -0.6 -0.6 -0.6 -0.6
    rosrun my_gurdy_description test_pose.py --reset ...   # reset the world first
"""
import argparse
import math
import sys

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Empty

UPPER_LIMITS = (-1.55, 0.0)
LOWER_LIMITS = (-2.9, 1.5708)
SETTLE_TIME = 2.0  # s


def roll_pitch(q):
    """Roll and pitch (radians) of a geometry_msgs/Quaternion."""
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return roll, math.asin(sin_pitch)


def main():
    parser = argparse.ArgumentParser(
        description="Send 12 joint angles to Gurdy and report the resulting pose.")
    parser.add_argument("angles", nargs=12, type=float, metavar="ANGLE",
                        help="upper legs M1-M6 then lower legs M1-M6, in radians")
    parser.add_argument("--reset", action="store_true",
                        help="call /gazebo/reset_world before sending the angles")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("gurdy_test_pose", anonymous=True)

    joints = (
        ["head_upperlegM%d_joint" % i for i in range(1, 7)] +
        ["upperlegM%d_lowerlegM%d_joint" % (i, i) for i in range(1, 7)]
    )
    topics = ["/gurdy/%s_position_controller/command" % j for j in joints]
    publishers = [rospy.Publisher(t, Float64, queue_size=1) for t in topics]

    angles = []
    for i, angle in enumerate(args.angles):
        lo, hi = UPPER_LIMITS if i < 6 else LOWER_LIMITS
        clipped = min(hi, max(lo, angle))
        if clipped != angle:
            rospy.logwarn("Angle %d (%s) = %.3f is outside [%.2f, %.2f]; using %.3f",
                          i + 1, topics[i].split("/")[2], angle, lo, hi, clipped)
        angles.append(clipped)

    # Publishing before the controllers have connected silently drops messages
    rospy.loginfo("Waiting for the joint controllers to connect...")
    for pub in publishers:
        while pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)

    if args.reset:
        rospy.wait_for_service("/gazebo/reset_world")
        rospy.ServiceProxy("/gazebo/reset_world", Empty)()

    for pub, angle in zip(publishers, angles):
        pub.publish(Float64(data=angle))

    rospy.loginfo("Sent angles, waiting %.0f s...", SETTLE_TIME)
    rospy.sleep(SETTLE_TIME)

    states = rospy.wait_for_message("/gurdy/joint_states", JointState, timeout=5.0)
    odom = rospy.wait_for_message("/gurdy/odom", Odometry, timeout=5.0)
    actual = dict(zip(states.name, states.position))

    print("%-30s %8s %8s %8s" % ("joint", "cmd", "actual", "error"))
    for i, (joint, cmd) in enumerate(zip(joints, angles)):
        if joint not in actual:
            print("%-30s %8.3f %8s %8s  NOT IN joint_states" % (joint, cmd, "-", "-"))
            continue
        pos = actual[joint]
        lo, hi = UPPER_LIMITS if i < 6 else LOWER_LIMITS
        note = "  OUTSIDE [%.2f, %.2f]" % (lo, hi) if not lo - 0.01 <= pos <= hi + 0.01 else ""
        print("%-30s %8.3f %8.3f %+8.3f%s" % (joint, cmd, pos, pos - cmd, note))

    roll, pitch = roll_pitch(odom.pose.pose.orientation)
    print("roll=%.1f deg  pitch=%.1f deg  z=%.4f m" % (
        math.degrees(roll), math.degrees(pitch), odom.pose.pose.position.z))


if __name__ == "__main__":
    main()

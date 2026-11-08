#!/usr/bin/env python3
"""
Try a Gurdy stance by hand.

Sends 12 joint angles to the position controllers, waits 3 s (sim time),
then samples /gurdy/joint_states and /gurdy/odom for 1 s. Prints, per
joint, the commanded angle, mean actual angle, mean error and standard
deviation (a large std means the joint is oscillating), then the mean and
std of base roll, pitch and height and the x, y drift over that second.

Angles are in radians, ordered upper legs M1-M6, then lower legs M1-M6
(the same order as NEUTRAL_POSE in env.py). Limits: upper -1.55..0,
lower -2.9..0; out-of-range values are clipped with a warning.

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
LOWER_LIMITS = (-2.9, 0.0)
SETTLE_TIME = 3.0  # s
SAMPLE_TIME = 1.0  # s


def mean_std(values):
    """Mean and population standard deviation of a non-empty list."""
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


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

    # Record every message that arrives during the sample window
    joint_samples = {j: [] for j in joints}
    odom_samples = []  # (roll, pitch, x, y, z)
    recording = [True]

    def on_joints(msg):
        if recording[0]:
            for name, pos in zip(msg.name, msg.position):
                if name in joint_samples:
                    joint_samples[name].append(pos)

    def on_odom(msg):
        if recording[0]:
            p = msg.pose.pose.position
            roll, pitch = roll_pitch(msg.pose.pose.orientation)
            odom_samples.append((roll, pitch, p.x, p.y, p.z))

    subs = [rospy.Subscriber("/gurdy/joint_states", JointState, on_joints),
            rospy.Subscriber("/gurdy/odom", Odometry, on_odom)]
    rospy.loginfo("Sampling for %.0f s...", SAMPLE_TIME)
    rospy.sleep(SAMPLE_TIME)
    recording[0] = False
    for sub in subs:
        sub.unregister()

    n_joint = max(len(v) for v in joint_samples.values())
    print("%d joint_states samples, %d odom samples over %.1f s" % (
        n_joint, len(odom_samples), SAMPLE_TIME))
    print("%-30s %8s %8s %8s %8s" % ("joint", "cmd", "mean", "error", "std"))
    for i, (joint, cmd) in enumerate(zip(joints, angles)):
        samples = joint_samples[joint]
        if not samples:
            print("%-30s %8.3f %8s %8s %8s  NOT IN joint_states" % (
                joint, cmd, "-", "-", "-"))
            continue
        mean, std = mean_std(samples)
        lo, hi = UPPER_LIMITS if i < 6 else LOWER_LIMITS
        note = ("  OUTSIDE [%.2f, %.2f]" % (lo, hi)
                if not lo - 0.01 <= mean <= hi + 0.01 else "")
        print("%-30s %8.3f %8.3f %+8.3f %8.4f%s" % (
            joint, cmd, mean, mean - cmd, std, note))

    if not odom_samples:
        print("No /gurdy/odom messages received")
        return
    rolls, pitches, xs, ys, zs = zip(*odom_samples)
    roll_m, roll_s = mean_std(rolls)
    pitch_m, pitch_s = mean_std(pitches)
    z_m, z_s = mean_std(zs)
    print("roll  = %6.1f deg  (std %.2f deg)" % (math.degrees(roll_m), math.degrees(roll_s)))
    print("pitch = %6.1f deg  (std %.2f deg)" % (math.degrees(pitch_m), math.degrees(pitch_s)))
    print("z     = %.4f m    (std %.4f m)" % (z_m, z_s))
    print("drift over %.1f s: dx=%+.4f m  dy=%+.4f m  (at x=%.3f y=%.3f)" % (
        SAMPLE_TIME, xs[-1] - xs[0], ys[-1] - ys[0], xs[-1], ys[-1]))

if __name__ == "__main__":
    main()

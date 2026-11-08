#!/usr/bin/env python3
"""
Run a fixed gait through the walk env and report how far it gets.

Prints the foot poses and their IK joint angles, resets the env (every group
GROUND_OUT), then loops a stroke for N cycles. After each cycle it prints
the forward distance from the start, the progress made that cycle, base roll
and pitch, and any fall.

After each action it also prints the joint tracking error, |actual - target|
over all 12 joints, sampled from /gurdy/joint_states from the moment the
action is sent until the end of its hold:
    max  the largest error seen during the action. It includes the jump
         to the new target right at the start, so it is roughly the size
         of the move.
    end  the largest error at the last sample, i.e. how far the worst
         joint still is from its foot pose when the next action starts.
         An end error near max means the pose was never reached.

A stroke moves one group through
    LIFT, <return reach>, LOWER, <push reach>
so the feet travel back in the air and the push happens while they are
planted. A planted foot moving OUT pushes the body away from it along its
spoke; one moving IN pulls the body toward it.

Default (+x): BACK stroke, pushing OUT; FRONT and SIDE stay grounded
    BACK LIFT, BACK REACH_IN, BACK LOWER, BACK REACH_OUT
--alternate (+x): the BACK stroke above, then a FRONT stroke pulling IN
    FRONT LIFT, FRONT REACH_OUT, FRONT LOWER, FRONT REACH_IN
--mirror swaps FRONT and BACK, so the gait should move about as far toward
-x: FRONT pushing OUT, and with --alternate also BACK pulling IN.

The env's own episode-end checks (stuck, goal, step limit) are reported but
ignored; the run stops only on a fall or after N cycles.

Usage (with Gazebo and the controllers running):
    rosrun my_gurdy_description test_gait.py 20
    rosrun my_gurdy_description test_gait.py 20 --mirror
    rosrun my_gurdy_description test_gait.py 20 --alternate --hold 0.3
    rosrun my_gurdy_description test_gait.py --poses   # print the IK angles only
"""
import argparse
import math
import sys

import gym
import rospy
from sensor_msgs.msg import JointState

import env as gurdy_env


def action_index(group, move):
    """Index in env.ACTIONS of the action that applies move to group."""
    return gurdy_env.ACTIONS.index((group, move))


def stroke(group, push):
    """
    Actions for one stroke of group whose planted feet move `push`
    (REACH_OUT or REACH_IN): lift, reach the other way, lower, push.
    """
    back = gurdy_env.REACH_IN if push == gurdy_env.REACH_OUT else gurdy_env.REACH_OUT
    return [(group, gurdy_env.LIFT), (group, back),
            (group, gurdy_env.LOWER), (group, push)]


def gait(mirror, alternate):
    """(group, move) list for one cycle, and the expected direction."""
    pusher, puller = ("FRONT", "BACK") if mirror else ("BACK", "FRONT")
    moves = stroke(pusher, gurdy_env.REACH_OUT)
    if alternate:
        moves += stroke(puller, gurdy_env.REACH_IN)
    return moves, ("-x" if mirror else "+x")


def main():
    parser = argparse.ArgumentParser(
        description="Run a fixed leg-group gait through MyGurdyWalkEnv.")
    parser.add_argument("cycles", type=int, nargs="?", default=0,
                        help="number of gait cycles to run")
    parser.add_argument("--mirror", action="store_true",
                        help="swap FRONT and BACK, to walk toward -x")
    parser.add_argument("--alternate", action="store_true",
                        help="follow each pushing stroke with a pulling stroke "
                             "of the opposite group")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="extra sim seconds to wait after each action, on "
                             "top of the env's own 0.1 s wall-clock wait (default 0)")
    parser.add_argument("--poses", action="store_true",
                        help="print the foot poses and IK angles, then exit")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print(gurdy_env.format_foot_poses())
    if args.poses:
        return
    if args.cycles < 1:
        parser.error("give the number of cycles to run (or --poses)")

    moves, expected = gait(args.mirror, args.alternate)
    cycle_actions = [action_index(group, move) for group, move in moves]
    print("Gait cycle (expect %s): %s" % (
        expected, ", ".join("%s %s" % gm for gm in moves)))

    rospy.init_node("gurdy_test_gait", anonymous=True)
    walk = gym.make("MyGurdyWalkEnv-v0").unwrapped

    # joint_states received while an action runs, as {joint: position}
    joint_samples = []
    recording = [False]

    def on_joints(msg):
        if recording[0]:
            joint_samples.append(dict(zip(msg.name, msg.position)))

    rospy.Subscriber("/gurdy/joint_states", JointState, on_joints)

    def joint_targets():
        """{joint: target angle} for the env's current group foot poses."""
        hips, knees = [0.0] * 6, [0.0] * 6
        for group, legs in gurdy_env.GROUP_LEGS.items():
            hip, knee = gurdy_env.POSE_ANGLES[walk.group_states[group]]
            for leg in legs:
                hips[leg - 1], knees[leg - 1] = hip, knee
        return dict(zip(walk.all_joint_names, hips + knees))

    def tracking_error(targets):
        """
        (max error over the action, worst joint at that moment, end error,
        worst joint at the end) over joint_samples, or None if empty.
        """
        def worst(sample):
            return max((abs(sample[j] - t), j) for j, t in targets.items() if j in sample)
        per_sample = [worst(sample) for sample in joint_samples]
        if not per_sample:
            return None
        peak, end = max(per_sample), per_sample[-1]
        return peak[0], peak[1], end[0], end[1]

    def run(action):
        """Step the env, then hold. Returns (fallen, fall reason, done)."""
        del joint_samples[:]
        recording[0] = True
        _, _, done, info = walk.step(action)
        if args.hold > 0:
            rospy.sleep(args.hold)
        recording[0] = False
        group, move = gurdy_env.ACTIONS[action]
        error = tracking_error(joint_targets())
        if error is None:
            print("  %-6s %-9s no joint_states received" % (group, move))
        else:
            print("  %-6s %-9s tracking error max %.3f rad (%s)  end %.3f rad (%s)"
                  % ((group, move) + error))
        return info["fallen"], info["fall_reason"], done

    def report(label, previous):
        distance = walk.get_forward_position() - walk.start_forward
        roll, pitch = walk.get_roll_pitch()
        print("%-8s forward %+8.4f m  (%+7.4f this cycle)  roll %+6.1f deg  pitch %+6.1f deg"
              % (label, distance, distance - previous,
                 math.degrees(roll), math.degrees(pitch)))
        return distance

    walk.reset()
    distance = report("reset", 0.0)

    env_done_reported = False
    cycle = 0
    for cycle in range(1, args.cycles + 1):
        fallen = False
        for action in cycle_actions:
            fallen, reason, done = run(action)
            if fallen:
                break
            if done and not env_done_reported:
                print("(env would end the episode here; continuing)")
                env_done_reported = True
        distance = report("cycle %d" % cycle, distance)
        if fallen:
            print("FELL in cycle %d: %s" % (cycle, reason))
            break

    print("Total forward distance: %+.4f m over %d cycles (expected %s)"
          % (distance, cycle, expected))


if __name__ == "__main__":
    main()

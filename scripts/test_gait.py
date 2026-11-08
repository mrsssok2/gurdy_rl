#!/usr/bin/env python3
"""
Run a fixed gait through the walk env and report how far it gets.

Prints the foot poses and their IK joint angles, resets the env (every group
GROUND_OUT), then loops a stroke for N cycles. After each cycle it prints
the forward distance from the start, the progress made that cycle, base roll
and pitch, and any fall.

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

    def run(action):
        """Step the env, then hold. Returns (fallen, fall reason, done)."""
        _, _, done, info = walk.step(action)
        if args.hold > 0:
            rospy.sleep(args.hold)
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

#!/usr/bin/env python3
"""
Run a fixed gait through the walk env and report how far it gets.

Resets the env, sets up a stance, then loops a two-action stroke for N
cycles. After each cycle it prints the forward distance from the start, the
progress made that cycle, base roll and pitch, and any fall.

Default (+x): setup FRONT_EXTEND, SIDE_EXTEND, then loop
    BACK_RETRACT (lift the back feet and draw them in)
    BACK_EXTEND  (set them down further out, pushing the body toward +x)
--mirror (-x): setup BACK_EXTEND, SIDE_EXTEND, then loop
    FRONT_RETRACT, FRONT_EXTEND
The mirror should move about as far, in the opposite direction.

The env's own episode-end checks (stuck, goal, step limit) are reported but
ignored; the run stops only on a fall or after N cycles.

Usage (with Gazebo and the controllers running):
    rosrun my_gurdy_description test_gait.py 20
    rosrun my_gurdy_description test_gait.py 20 --mirror
    rosrun my_gurdy_description test_gait.py 20 --hold 0.3
"""
import argparse
import math
import sys

import gym
import rospy

import env as gurdy_env


def action_index(group, state):
    """Index in env.ACTIONS of the action that sets group to state."""
    return gurdy_env.ACTIONS.index((group, state))


def main():
    parser = argparse.ArgumentParser(
        description="Run a fixed leg-group gait through MyGurdyWalkEnv.")
    parser.add_argument("cycles", type=int, help="number of stroke cycles to run")
    parser.add_argument("--mirror", action="store_true",
                        help="drive the FRONT legs instead of the BACK legs (-x)")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="extra sim seconds to wait after each action, on "
                             "top of the env's own 0.1 s wall-clock wait (default 0)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("gurdy_test_gait", anonymous=True)
    walk = gym.make("MyGurdyWalkEnv-v0").unwrapped

    stance, driver = ("BACK", "FRONT") if args.mirror else ("FRONT", "BACK")
    setup = [action_index(stance, gurdy_env.EXTENDED),
             action_index("SIDE", gurdy_env.EXTENDED)]
    stroke = [action_index(driver, gurdy_env.RETRACTED),
              action_index(driver, gurdy_env.EXTENDED)]
    expected = "-x" if args.mirror else "+x"

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
    print("Gait: stance %s + SIDE extended, stroke %s retract/extend, expect %s"
          % (stance, driver, expected))

    distance = report("reset", 0.0)
    for action in setup:
        fallen, reason, _ = run(action)
        if fallen:
            report("setup", distance)
            print("FELL during setup: %s" % reason)
            return
    distance = report("setup", distance)

    env_done_reported = False
    for cycle in range(1, args.cycles + 1):
        for action in stroke:
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
          % (distance, cycle if args.cycles > 0 else 0, expected))


if __name__ == "__main__":
    main()

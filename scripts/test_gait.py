#!/usr/bin/env python3
"""
Run a fixed per-leg gait through the walk env and report how far it gets.

Prints the foot poses and their IK joint angles, resets the env (every leg
GROUND_OUT), then loops a gait cycle for N cycles. After each cycle it prints
the forward distance from the start, the progress made that cycle, base roll
and pitch, and any fall.

After each action it also prints the joint tracking error, |actual - target|
over all 12 joints, sampled from /gurdy/joint_states from the moment the
action is sent until the end of its hold:
    max  the largest error seen during the action, against the action's
         final targets. The env eases the targets there over
         env.MOVE_DURATION, so at the start this is roughly the size of the
         move.
    end  the largest error at the last sample, i.e. how far the worst
         joint still is from its foot pose when the next action starts.
         An end error near max means the pose was never reached.

Every env action moves one leg, so five legs are always on the ground. A
stroke moves one leg through
    LIFT, <return reach>, LOWER, <push reach>
so the foot travels back in the air and the push happens while it is
planted. A planted foot moving OUT pushes the body away from it along its
spoke; one moving IN pulls the body toward it. Spokes: 1 = 210 deg,
2 = 330, 3 = 90, 4 = 150, 5 = 30, 6 = 270 (forward = +x = 0 deg).

--wave (+x): the back legs push OUT, leg 1 then leg 4
    leg 1 pushes the body along 30 deg (+x, +y), leg 4 along -30 deg
    (+x, -y), so the body zig-zags forward
--wave-all (+x): the --wave strokes, then the front legs pull IN, leg 5
    (pulls toward 30 deg) then leg 2 (toward -30 deg)
--mirror mirrors the legs across the y axis (1 <-> 2, 4 <-> 5), so the gait
should move about as far toward -x: legs 2 and 5 push OUT, and with
--wave-all legs 4 and 1 then pull IN.

Sideways (the feet move only along their spokes, so Gurdy cannot turn, but
it can translate along y):
--left (+y): the side legs first, then the diagonal legs in pairs whose x
    components cancel:
    leg 6 (270 deg) pushes OUT          -> body along +y
    leg 3 (90 deg) pulls IN             -> body along +y
    leg 1 (210) OUT, leg 4 (150) IN     -> body along 30 deg, then 150 deg
    leg 2 (330) OUT, leg 5 (30) IN      -> body along 150 deg, then 30 deg
    Legs 3 and 6 move the body their full stroke along y, the diagonal legs
    half of theirs (sin 30 deg).
--right (-y): the --left gait mirrored across the x axis (3 <-> 6,
    1 <-> 4, 2 <-> 5).
For these the report gives the sideways position (world y) instead of the
forward one.

The env's own episode-end checks (stuck, goal, step limit) are reported but
ignored; the run stops only on a fall or after N cycles.

Usage (with Gazebo and the controllers running):
    rosrun my_gurdy_description test_gait.py 20 --wave
    rosrun my_gurdy_description test_gait.py 20 --wave --mirror
    rosrun my_gurdy_description test_gait.py 20 --wave-all --hold 0.3
    rosrun my_gurdy_description test_gait.py 10 --left
    rosrun my_gurdy_description test_gait.py 10 --right
    rosrun my_gurdy_description test_gait.py --poses   # print the IK angles only
"""
import argparse
import math
import sys

import gym
import rospy
from sensor_msgs.msg import JointState

import env as gurdy_env

# Pushing back legs and pulling front legs, in stroke order, for +x
WAVE_PUSH_LEGS = (1, 4)
WAVE_PULL_LEGS = (5, 2)
# Mirror image of each leg across the y axis (spoke angle a -> 180 - a)
MIRROR_LEG = {1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6}
# Side-step strokes for +y, in order: (leg, the planted foot's push)
SIDE_STROKES = ((6, gurdy_env.REACH_OUT), (3, gurdy_env.REACH_IN),
                (1, gurdy_env.REACH_OUT), (4, gurdy_env.REACH_IN),
                (2, gurdy_env.REACH_OUT), (5, gurdy_env.REACH_IN))
# Mirror image of each leg across the x axis (spoke angle a -> -a)
MIRROR_X_LEG = {1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3}


def stroke(leg, push):
    """
    (leg, move) list for one stroke of leg, whose planted foot moves `push`
    (REACH_OUT or REACH_IN): lift, reach the other way, lower, push.
    """
    back = gurdy_env.REACH_IN if push == gurdy_env.REACH_OUT else gurdy_env.REACH_OUT
    return [(leg, gurdy_env.LIFT), (leg, back),
            (leg, gurdy_env.LOWER), (leg, push)]


def gait(wave_all, mirror):
    """(leg, move) list for one cycle, and the expected direction."""
    strokes = [(leg, gurdy_env.REACH_OUT) for leg in WAVE_PUSH_LEGS]
    if wave_all:
        strokes += [(leg, gurdy_env.REACH_IN) for leg in WAVE_PULL_LEGS]
    moves = []
    for leg, push in strokes:
        moves += stroke(MIRROR_LEG[leg] if mirror else leg, push)
    return moves, ("-x" if mirror else "+x")


def side_gait(right):
    """(leg, move) list for one side-step cycle, and the expected direction."""
    moves = []
    for leg, push in SIDE_STROKES:
        moves += stroke(MIRROR_X_LEG[leg] if right else leg, push)
    return moves, ("-y" if right else "+y")


def main():
    parser = argparse.ArgumentParser(
        description="Run a fixed per-leg gait through MyGurdyWalkEnv.")
    parser.add_argument("cycles", type=int, nargs="?", default=0,
                        help="number of gait cycles to run")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wave", action="store_true",
                      help="back legs 1 then 4 push OUT, zig-zagging toward +x")
    mode.add_argument("--wave-all", action="store_true",
                      help="as --wave, then front legs 5 then 2 pull IN")
    mode.add_argument("--left", action="store_true",
                      help="side-step toward +y: legs 6 OUT, 3 IN, then the diagonals")
    mode.add_argument("--right", action="store_true",
                      help="side-step toward -y: the --left gait mirrored")
    parser.add_argument("--mirror", action="store_true",
                        help="mirror the legs across the y axis, to walk toward -x "
                             "(--wave / --wave-all only)")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="extra sim seconds to wait after each action, on "
                             "top of the env's own %.1f s smooth move (default 0)"
                             % gurdy_env.MOVE_DURATION)
    parser.add_argument("--poses", action="store_true",
                        help="print the foot poses and IK angles, then exit")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    print(gurdy_env.format_foot_poses())
    if args.poses:
        return
    if args.cycles < 1:
        parser.error("give the number of cycles to run (or --poses)")
    sideways = args.left or args.right
    if not (args.wave or args.wave_all or sideways):
        parser.error("choose a gait: --wave, --wave-all, --left or --right")
    if sideways and args.mirror:
        parser.error("--mirror applies to --wave / --wave-all only")

    if sideways:
        moves, expected = side_gait(args.right)
    else:
        moves, expected = gait(args.wave_all, args.mirror)
    cycle_actions = [gurdy_env.action_index(leg, move) for leg, move in moves]
    print("Gait cycle (expect %s): %s" % (
        expected, ", ".join("leg %d %s" % lm for lm in moves)))

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
        """{joint: target angle} for the env's current leg foot poses."""
        return dict(zip(walk.all_joint_names,
                        gurdy_env.leg_joint_targets(walk.leg_states)))

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
        leg, move = gurdy_env.ACTIONS[action]
        label = "leg %d" % leg if leg is not None else "all"
        error = tracking_error(joint_targets())
        if error is None:
            print("  %-6s %-9s no joint_states received" % (label, move))
        else:
            print("  %-6s %-9s tracking error max %.3f rad (%s)  end %.3f rad (%s)"
                  % ((label, move) + error))
        return info["fallen"], info["fall_reason"], done

    # Sideways gaits report world y (and the x drift), the others forward
    axis = "sideways" if sideways else "forward"
    position = walk.get_lateral_position if sideways else walk.get_forward_position

    def report(label, previous):
        distance = position() - start[0]
        roll, pitch = walk.get_roll_pitch()
        drift = ("  x drift %+7.4f m" % (walk.get_forward_position() - start[1])
                 if sideways else "")
        print("%-8s %s %+8.4f m  (%+7.4f this cycle)%s  roll %+6.1f deg  pitch %+6.1f deg"
              % (label, axis, distance, distance - previous, drift,
                 math.degrees(roll), math.degrees(pitch)))
        return distance

    walk.reset()
    start = (position(), walk.get_forward_position())
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

    print("Total %s distance: %+.4f m over %d cycles, %d steps (expected %s)"
          % (axis, distance, cycle, cycle * len(cycle_actions), expected))


if __name__ == "__main__":
    main()

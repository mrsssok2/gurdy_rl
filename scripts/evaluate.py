#!/usr/bin/env python3
"""
Evaluate saved DQN checkpoints greedily (epsilon 0) in MyGurdyWalkEnv-v0.

Each episode runs as in training: reset, then step until the env ends the
episode (goal, fall, stuck or /gurdy/nsteps). Per episode it prints the
forward distance, steps, how it ended, falls and invalid lifts; then a
summary. --all evaluates every gurdy_dqn_model_<episode> in the model
directory and prints a ranking, best first (mean forward distance, then
fewest falls, then fewest steps).

Needs Gazebo and the position controllers, and must not run while training
uses the same simulation:
    rosrun my_gurdy_description evaluate.py output/dqn/gurdy_dqn_model_100 --episodes 5
    rosrun my_gurdy_description evaluate.py --all [--episodes 3] [--dir output/dqn]
    rosrun my_gurdy_description evaluate.py --all --csv output/dqn/eval.csv

Relative paths are taken relative to the package.
"""
import argparse
import csv
import os
import re
import sys
import time

MODEL_PREFIX = "gurdy_dqn_model_"
_MODEL_NAME = re.compile(r"^%s(\d+)$" % MODEL_PREFIX)

# How an episode ended
GOAL, FALLEN, STUCK, STEP_LIMIT = "goal", "fallen", "stuck", "step_limit"


def checkpoint_episode(path):
    """Training episode of a checkpoint path, or None if it isn't one."""
    match = _MODEL_NAME.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def list_checkpoints(directory):
    """gurdy_dqn_model_<episode> entries (files or SavedModel dirs) in directory, by episode."""
    if not os.path.isdir(directory):
        return []
    found = [os.path.join(directory, name) for name in os.listdir(directory)
             if checkpoint_episode(name) is not None]
    return sorted(found, key=checkpoint_episode)


def end_reason(fallen, distance, steps, max_steps, target):
    """Why the env ended an episode, in the order env._is_done checks."""
    if fallen:
        return FALLEN
    if distance >= target:
        return GOAL
    if steps >= max_steps:
        return STEP_LIMIT
    return STUCK


def summarize(name, episodes):
    """Summary dict over per-episode dicts (see run_episode)."""
    n = len(episodes)
    distances = [e["distance_m"] for e in episodes]
    return {
        "checkpoint": name,
        "training_episode": checkpoint_episode(name),
        "episodes": n,
        "mean_distance_m": sum(distances) / n if n else 0.0,
        "max_distance_m": max(distances) if n else 0.0,
        "min_distance_m": min(distances) if n else 0.0,
        "mean_steps": sum(e["steps"] for e in episodes) / float(n) if n else 0.0,
        "falls": sum(1 for e in episodes if e["fallen"]),
        "goals": sum(1 for e in episodes if e["end"] == GOAL),
        "mean_invalid_lifts": sum(e["invalid_lifts"] for e in episodes) / float(n) if n else 0.0,
    }


def rank(summaries):
    """Best first: mean forward distance, then fewer falls, then fewer steps."""
    return sorted(summaries, key=lambda s: (-s["mean_distance_m"], s["falls"], s["mean_steps"]))


def format_episode(index, e):
    text = "  episode %2d: forward %+.3f m  steps %3d  end %-10s  falls %d  invalid lifts %d" % (
        index, e["distance_m"], e["steps"], e["end"], int(e["fallen"]), e["invalid_lifts"])
    if e.get("fall_reason"):
        text += "  (%s)" % e["fall_reason"]
    return text


def format_summary(s):
    return ("  summary: mean %+.3f m (min %+.3f, max %+.3f)  mean steps %.0f  "
            "goals %d/%d  falls %d/%d  invalid lifts/episode %.1f" % (
                s["mean_distance_m"], s["min_distance_m"], s["max_distance_m"],
                s["mean_steps"], s["goals"], s["episodes"], s["falls"], s["episodes"],
                s["mean_invalid_lifts"]))


def format_ranking(summaries):
    """Ranking table, best first."""
    header = "%4s  %-24s %8s %9s %9s %6s %6s %6s %8s" % (
        "rank", "checkpoint", "trained", "mean m", "max m", "steps", "goals", "falls", "inv/ep")
    lines = [header, "-" * len(header)]
    for i, s in enumerate(rank(summaries), 1):
        lines.append("%4d  %-24s %8s %+9.3f %+9.3f %6.0f %3d/%-2d %3d/%-2d %8.1f" % (
            i, os.path.basename(os.path.normpath(s["checkpoint"]))[:24],
            "-" if s["training_episode"] is None else s["training_episode"],
            s["mean_distance_m"], s["max_distance_m"], s["mean_steps"],
            s["goals"], s["episodes"], s["falls"], s["episodes"], s["mean_invalid_lifts"]))
    return "\n".join(lines)


def run_episode(walk, policy):
    """One greedy episode. policy(observation) -> action. Returns a stats dict."""
    observation = walk.reset()
    fallen, fall_reason, steps = False, None, 0
    done = False
    while not done:
        observation, _, done, info = walk.step(policy(observation))
        steps = info["steps"]
        if info["fallen"]:
            fallen, fall_reason = True, info["fall_reason"]
    distance = walk.get_forward_position() - walk.start_forward
    return {
        "distance_m": distance,
        "steps": steps,
        "fallen": fallen,
        "fall_reason": fall_reason,
        "invalid_lifts": walk.invalid_lifts,
        "reward": walk.cumulated_reward,
        "end": end_reason(fallen, distance, steps, walk.max_steps,
                          walk.TARGET_FORWARD_DISTANCE),
    }


def evaluate(walk, path, episodes):
    """Evaluate one checkpoint; prints as it goes. Returns (summary, episode dicts)."""
    from dqn_policy import greedy_action, load_model
    print("\n%s" % path, flush=True)
    started = time.time()
    model = load_model(path)
    results = []
    for i in range(1, episodes + 1):
        result = run_episode(walk, lambda obs: greedy_action(model, obs))
        results.append(result)
        print(format_episode(i, result), flush=True)
    summary = summarize(path, results)
    print(format_summary(summary))
    print("  (%.0f s wall time)" % (time.time() - started), flush=True)
    return summary, results


def write_csv(path, rows):
    fields = ["checkpoint", "episode", "distance_m", "steps", "end", "fallen",
              "fall_reason", "invalid_lifts", "reward"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote %s" % path)


def main():
    import rospy
    from dqn_policy import resolve_model_path

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", help="saved model, e.g. output/dqn/gurdy_dqn_model_100")
    parser.add_argument("--all", action="store_true",
                        help="evaluate every %s<episode> in --dir and rank them" % MODEL_PREFIX)
    parser.add_argument("--dir", default="output/dqn", help="model directory for --all")
    parser.add_argument("--episodes", type=int, default=None,
                        help="episodes per checkpoint (default 3 with --all, else 5)")
    parser.add_argument("--csv", help="also write every episode to this CSV file")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.all == bool(args.checkpoint):
        parser.error("give a checkpoint path or --all, not both")
    episodes = args.episodes or (3 if args.all else 5)
    if episodes < 1:
        parser.error("--episodes must be at least 1")

    if args.all:
        directory = resolve_model_path(args.dir)
        paths = list_checkpoints(directory)
        if not paths:
            sys.exit("No %s<episode> checkpoints in %s" % (MODEL_PREFIX, directory))
        print("Evaluating %d checkpoints in %s, %d episodes each" % (len(paths), directory, episodes))
    else:
        paths = [resolve_model_path(args.checkpoint)]
        if not os.path.exists(paths[0]):
            sys.exit("Checkpoint not found: %s" % paths[0])

    rospy.init_node("gurdy_evaluate", anonymous=True)
    import gym
    import env  # noqa: F401  registers MyGurdyWalkEnv-v0
    walk = gym.make("MyGurdyWalkEnv-v0").unwrapped

    summaries, rows = [], []
    for path in paths:
        if rospy.is_shutdown():
            break
        try:
            summary, results = evaluate(walk, path, episodes)
        except Exception as e:  # a broken checkpoint shouldn't end an --all run
            print("  FAILED: %s" % e, flush=True)
            continue
        summaries.append(summary)
        rows += [dict(r, checkpoint=os.path.basename(path), episode=i)
                 for i, r in enumerate(results, 1)]

    if len(summaries) > 1 or args.all:
        print("\nRanking (best first):\n" + format_ranking(summaries))
    if args.csv and rows:
        write_csv(resolve_model_path(args.csv), rows)


if __name__ == "__main__":
    main()

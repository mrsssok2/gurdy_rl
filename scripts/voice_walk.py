"""
Voice command handling for voice_command_node.py, without ROS, so it can be
unit-tested offline (scripts/tests/test_voice_walk.py).

Commands arrive as JSON on /gurdy/voice_command:
    {"action": "walk_forward", "distance_m": 0.2, "mode": "rl" | "scripted", "id": "..."}
    {"action": "stop", "id": "..."}
    {"action": "status", "id": "..."}
"id" is optional and echoed back in the feedback so the client can match it.

run_walk() walks until the robot's forward position has gained the
requested distance. In "scripted" mode it steps the --wave-all gait. In
"rl" mode it acts with the policy, and if the policy makes less than
RL_MIN_PROGRESS of forward progress over RL_WINDOW steps (or there is no
policy) it switches to the scripted gait for the rest of the distance.
"""
import json
import math
from collections import deque

ACTIONS = ("walk_forward", "stop", "status")
MODES = ("rl", "scripted")
DEFAULT_DISTANCE_M = 0.2
MAX_DISTANCE_M = 0.5

# RL fallback: less than this forward progress over the last RL_WINDOW steps
RL_WINDOW = 32
RL_MIN_PROGRESS = 0.005  # m
# Scripted gait stuck check, as in the env (STUCK_WINDOW, STUCK_MIN_PROGRESS)
SCRIPTED_WINDOW = 64
SCRIPTED_MIN_PROGRESS = 0.005  # m
# The scripted --wave-all gait makes about 0.546 m in 320 steps (README
# Baseline), 1.7 mm per step. The step limit allows twice that plus a margin.
SCRIPTED_M_PER_STEP = 0.546 / 320
STEP_MARGIN = 64

# Result statuses
DONE = "done"          # walked the requested distance
STOPPED = "stopped"    # a stop command arrived
FALLEN = "fallen"      # the robot fell
STUCK = "stuck"        # the scripted gait made no progress
STEP_LIMIT = "step_limit"


def parse_command(data):
    """
    Validate a voice_command JSON string. Returns a dict with action, id and,
    for walk_forward, distance_m (clamped to (0, MAX_DISTANCE_M], default
    DEFAULT_DISTANCE_M) and mode (default "scripted"). Raises ValueError.
    """
    try:
        msg = json.loads(data)
    except ValueError as e:
        raise ValueError("not JSON: %s" % e)
    if not isinstance(msg, dict):
        raise ValueError("expected a JSON object")
    action = msg.get("action")
    if action not in ACTIONS:
        raise ValueError("unknown action %r (expected one of %s)" % (action, ", ".join(ACTIONS)))
    cmd = {"action": action, "id": msg.get("id")}
    if action == "walk_forward":
        distance = msg.get("distance_m", DEFAULT_DISTANCE_M)
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            raise ValueError("distance_m must be a number, got %r" % (distance,))
        if not math.isfinite(distance) or distance <= 0:
            distance = DEFAULT_DISTANCE_M
        cmd["distance_m"] = min(distance, MAX_DISTANCE_M)
        mode = msg.get("mode", "scripted")
        if mode not in MODES:
            raise ValueError("unknown mode %r (expected rl or scripted)" % (mode,))
        cmd["mode"] = mode
    return cmd


def max_steps_for(distance_m):
    """Step limit for walking distance_m."""
    return int(math.ceil(2 * distance_m / SCRIPTED_M_PER_STEP)) + STEP_MARGIN


class ProgressWindow(object):
    """Forward positions over the last `window` steps, to spot a stall."""

    def __init__(self, window, min_progress, start):
        self.window = window
        self.min_progress = min_progress
        self.positions = deque([start], maxlen=window + 1)

    def add(self, position):
        self.positions.append(position)

    def stalled(self):
        """True once there are `window` steps and they gained < min_progress."""
        return (len(self.positions) > self.window and
                self.positions[-1] - self.positions[0] < self.min_progress)

    def progress(self):
        return self.positions[-1] - self.positions[0]


class ScriptedGait(object):
    """
    Cycles through a fixed list of actions. The phase carries over between
    walks, so a new walk continues the gait where the last one stopped.
    """

    def __init__(self, cycle_actions):
        self.cycle = list(cycle_actions)
        self.phase = 0

    def next_action(self):
        action = self.cycle[self.phase]
        self.phase = (self.phase + 1) % len(self.cycle)
        return action

    def restart(self):
        self.phase = 0


def run_walk(distance_m, mode, robot, gait, policy=None, policy_error=None,
             should_stop=lambda: False, log=lambda msg: None):
    """
    Walk forward distance_m.

    robot: forward() -> forward position in m; step(action) -> (fallen, reason)
    gait: ScriptedGait for the scripted mode and the RL fallback
    policy: callable() -> action for the robot's current state, or None
    policy_error: why there is no policy, for the result
    should_stop: polled before every step

    Returns a result dict: status (DONE, STOPPED, FALLEN, STUCK, STEP_LIMIT),
    requested_m, distance_m, steps, mode_requested, mode_used ("rl",
    "scripted", or "rl+scripted" after a fallback), fallback, and
    fallback_reason / fallback_at_m / fall_reason where they apply.
    """
    start = robot.forward()
    result = {"requested_m": distance_m, "mode_requested": mode,
              "fallback": False, "steps": 0}
    current = "scripted"
    if mode == "rl":
        if policy is None:
            result.update(fallback=True, fallback_at_m=0.0,
                          fallback_reason=policy_error or "no RL model loaded")
            log("RL mode unavailable (%s); using the scripted gait" % result["fallback_reason"])
        else:
            current = "rl"
    used = {current}
    window = ProgressWindow(RL_WINDOW if current == "rl" else SCRIPTED_WINDOW,
                            RL_MIN_PROGRESS if current == "rl" else SCRIPTED_MIN_PROGRESS,
                            start)
    max_steps = max_steps_for(distance_m)
    status = STEP_LIMIT

    while result["steps"] < max_steps:
        walked = robot.forward() - start
        if walked >= distance_m:
            status = DONE
            break
        if should_stop():
            status = STOPPED
            break
        action = policy() if current == "rl" else gait.next_action()
        fallen, reason = robot.step(action)
        result["steps"] += 1
        position = robot.forward()
        window.add(position)
        if fallen:
            status = FALLEN
            result["fall_reason"] = reason
            break
        if window.stalled():
            if current == "rl":
                result.update(
                    fallback=True, fallback_at_m=round(position - start, 4),
                    fallback_reason="RL made only %.1f mm in %d steps"
                                    % (1000 * window.progress(), RL_WINDOW))
                log("%s; switching to the scripted gait" % result["fallback_reason"])
                current = "scripted"
                used.add(current)
                window = ProgressWindow(SCRIPTED_WINDOW, SCRIPTED_MIN_PROGRESS, position)
            else:
                status = STUCK
                break
    else:
        if robot.forward() - start >= distance_m:
            status = DONE

    result["status"] = status
    result["distance_m"] = round(robot.forward() - start, 4)
    result["mode_used"] = ("rl+scripted" if used == {"rl", "scripted"} else current)
    return result

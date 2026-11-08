#!/usr/bin/env python3
"""
Execute voice commands on Gurdy (in Gazebo, through MyGurdyWalkEnv-v0).

Subscribes /gurdy/voice_command (std_msgs/String, JSON, see voice_walk.py):
    {"action": "walk_forward", "distance_m": 0.2, "mode": "rl" | "scripted", "id": "..."}
    {"action": "walk_backward" | "walk_left" | "walk_right", "distance_m": 0.2, "id": "..."}
    {"action": "stop"}   {"action": "status"}
    {"action": "describe"}
    {"action": "go_to_object", "color": "red" | "green" | "blue", "shape": "ball" | "box" | null}
and publishes exactly one JSON result per command on /gurdy/voice_feedback,
with the command's action and id, so a client can wait for it with
`rostopic echo -n 1`. /gurdy/voice_transcript (std_msgs/String, the raw
STT text) is only logged.

walk_forward steps the --wave-all scripted gait of test_gait.py until the
robot has gained distance_m along the forward axis (at most 0.5 m). With
mode "rl" it acts greedily with the Keras DQN model at ~rl_model_path
(relative paths are under the package), and switches to the scripted gait
if the policy makes less than 5 mm of progress over 32 steps or the model
cannot be loaded; the result says so (fallback, fallback_reason).

walk_backward steps the --wave-all gait mirrored front to back
(test_gait.py --wave-all --mirror) until the robot has moved distance_m
along -forward (world -x). walk_left / walk_right step the scripted
side-step gait of test_gait.py (--left / --right) until the robot has moved
distance_m along world +y / -y. All three go at most 0.5 m, with the same
stop, stuck, fall and step-limit handling, and are scripted only (no RL).
Their distance_m is along their own direction, and it counts toward
total_distance_m like a forward walk's. Walks
start from wherever the robot stands; the sim is reset only at startup
(~reset_on_start) and after a fall (~reset_on_fall).

stop ends a running walk after the current step. The walk's own result
(status "stopped", stopped_by = the stop's id) is then the reply to both, so
a stop during a walk publishes nothing itself; a stop while idle gets status
"idle". status answers right away, also during a walk. A walk command during a walk is answered with
status "busy".

describe answers right away (also during a walk) with the objects in the
latest /gurdy/detections frame from vision_node.py: status "ok" and
objects [{color, shape, side, bearing_deg, distance_m, area_fraction}], or
status "no_camera" if no frame arrives within 2 s.

go_to_object runs like a walk (busy, stop, fall handling): the closed-loop
visual approach of visual_approach.py, side-stepping until the target is
within ~bearing_tol_deg of the image centre and walking forward until it
covers ~close_area_ball / ~close_area_box of the image. Its result has
status reached / not_found / lost / stopped / fallen / stuck / gave_up /
no_camera, color, shape, moves, steps, forward_m, left_m and target (the
last sighting).

Run with the position controllers loaded, e.g.
    roslaunch my_gurdy_description voice_demo.launch [rl_model_path:=...]
"""
import json
import os
import queue
import threading
import time

import gym
import numpy as np
import rospy
from std_msgs.msg import String

import env as gurdy_env
import visual_approach
import voice_walk
from dqn_policy import greedy_action, load_model, resolve_model_path
from test_gait import gait as gait_moves, side_gait

COMMAND_TOPIC = "/gurdy/voice_command"
FEEDBACK_TOPIC = "/gurdy/voice_feedback"
TRANSCRIPT_TOPIC = "/gurdy/voice_transcript"
DETECTIONS_TOPIC = "/gurdy/detections"
# Wall-clock seconds to wait for a camera frame
DESCRIBE_TIMEOUT_S = 2.0
LOOK_TIMEOUT_S = 3.0


class EnvRobot(object):
    """voice_walk's robot interface on top of the walk env."""

    def __init__(self, walk):
        self.walk = walk
        self.observation = None

    def forward(self):
        return self.walk.get_forward_position()

    def backward(self):
        """Position along -forward."""
        return -self.walk.get_forward_position()

    def lateral(self):
        """Sideways position, world y (positive to the left)."""
        return self.walk.get_lateral_position()

    def right(self):
        """Position toward the right, -y."""
        return -self.walk.get_lateral_position()

    def begin(self):
        """
        Start a walk from the current pose without resetting the sim: the
        episode bookkeeping (start point, step count, last action) restarts,
        so the RL observation looks like the start of an episode, and the
        leg poses stay as they are.
        """
        w = self.walk
        w.cumulated_steps = 0
        w.cumulated_reward = 0
        w.last_action = None
        w.invalid_lift = False
        w.invalid_lifts = 0
        w._has_fallen = False
        w.start_point = w._odom.pose.pose.position
        w.start_forward = w.get_forward_position()
        w.prev_forward = w.start_forward
        w.forward_history.clear()
        w.forward_history.append(w.start_forward)
        self.observation = w._get_obs()

    def step(self, action):
        self.observation, _, _, info = self.walk.step(int(action))
        return info["fallen"], info["fall_reason"]

    def lower_lifted_legs(self):
        """Put any lifted leg down, e.g. after a stop mid-stroke."""
        for leg in gurdy_env.LEGS:
            if gurdy_env.POSE_FLAGS[self.walk.leg_states[leg]][0]:
                self.step(gurdy_env.action_index(leg, gurdy_env.LOWER))


class VoiceCommandNode(object):
    def __init__(self):
        rospy.init_node("gurdy_voice_command")
        self.reset_on_fall = rospy.get_param("~reset_on_fall", True)
        self.model_path = resolve_model_path(rospy.get_param("~rl_model_path", ""))
        self.model = None
        self.model_error = None

        self.walk = gym.make("MyGurdyWalkEnv-v0").unwrapped
        self.robot = EnvRobot(self.walk)
        def scripted(moves_and_direction):
            moves, _ = moves_and_direction
            return voice_walk.ScriptedGait(
                [gurdy_env.action_index(leg, move) for leg, move in moves])
        # One gait per walk action, each keeping its own phase
        self.gaits = {"walk_forward": scripted(gait_moves(wave_all=True, mirror=False)),
                      "walk_backward": scripted(gait_moves(wave_all=True, mirror=True)),
                      "walk_left": scripted(side_gait(right=False)),
                      "walk_right": scripted(side_gait(right=True))}
        # Position along each walk's direction, and its scripted gait's speed
        self.positions = {"walk_forward": self.robot.forward,
                          "walk_backward": self.robot.backward,
                          "walk_left": self.robot.lateral,
                          "walk_right": self.robot.right}
        self.m_per_step = {"walk_forward": voice_walk.SCRIPTED_M_PER_STEP,
                           "walk_backward": voice_walk.BACKWARD_M_PER_STEP,
                           "walk_left": voice_walk.SIDE_M_PER_STEP,
                           "walk_right": voice_walk.SIDE_M_PER_STEP}
        self.approach_config = visual_approach.Config(
            bearing_tol_deg=rospy.get_param("~bearing_tol_deg", visual_approach.BEARING_TOL_DEG),
            close_area={"ball": rospy.get_param("~close_area_ball",
                                                visual_approach.CLOSE_AREA["ball"]),
                        "box": rospy.get_param("~close_area_box",
                                               visual_approach.CLOSE_AREA["box"])},
            lost_checks=rospy.get_param("~lost_checks", visual_approach.LOST_CHECKS))
        if rospy.get_param("~reset_on_start", True):
            rospy.loginfo("Resetting the robot")
            self.walk.reset()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.commands = queue.Queue()
        self.busy = False
        self.stop_id = None          # id of the stop that ended the walk
        self.current = None          # the walk command running
        self.current_start = None    # its position along its direction at the start
        self.current_position = None  # () -> that position now
        self.total_m = 0.0           # distance walked since startup, all directions
        self.last_result = None
        # Latest /gurdy/detections message (dict), guarded by detection_cond
        self.detection = None
        self.detection_cond = threading.Condition()

        self.feedback = rospy.Publisher(FEEDBACK_TOPIC, String, queue_size=10)
        rospy.Subscriber(COMMAND_TOPIC, String, self.on_command)
        rospy.Subscriber(TRANSCRIPT_TOPIC, String, self.on_transcript)
        rospy.Subscriber(DETECTIONS_TOPIC, String, self.on_detections)
        if self.model_path:
            # Load now so the first RL walk doesn't wait for TensorFlow
            self.load_model()
        rospy.loginfo("Voice command node ready on %s", COMMAND_TOPIC)

    def load_model(self):
        """Load the Keras model once. Returns it, or None with model_error set."""
        if self.model is not None or self.model_error is not None:
            return self.model
        if not self.model_path:
            self.model_error = "rl_model_path is not set"
        elif not os.path.exists(self.model_path):
            self.model_error = "RL model not found: %s" % self.model_path
        else:
            try:
                self.model = load_model(self.model_path)
                rospy.loginfo("Loaded RL model %s", self.model_path)
            except Exception as e:  # any load failure means scripted fallback
                self.model_error = "could not load the RL model: %s" % e
        if self.model_error:
            rospy.logwarn(self.model_error)
        return self.model

    def greedy_action(self):
        return greedy_action(self.model, self.robot.observation)

    def publish(self, cmd, **fields):
        fields["action"] = cmd.get("action")
        if cmd.get("id") is not None:
            fields["id"] = cmd["id"]
        data = json.dumps(fields)
        rospy.loginfo("Feedback: %s", data)
        self.feedback.publish(String(data=data))

    def on_transcript(self, msg):
        rospy.loginfo("Transcript: %s", msg.data)

    def on_detections(self, msg):
        try:
            detection = json.loads(msg.data)
        except ValueError:
            rospy.logwarn_throttle(10.0, "Bad detections message: %r" % msg.data[:200])
            return
        with self.detection_cond:
            self.detection = detection
            self.detection_cond.notify_all()

    def look(self, after, timeout=LOOK_TIMEOUT_S):
        """
        The first detections message from a frame taken at or after sim time
        `after` (s), waiting up to `timeout` wall-clock s; None if none came.
        """
        deadline = time.time() + timeout
        with self.detection_cond:
            while not rospy.is_shutdown():
                if self.detection is not None and self.detection.get("stamp", 0.0) >= after:
                    return self.detection
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.detection_cond.wait(min(0.1, remaining))
        return None

    def describe_view(self):
        """Feedback fields for describe: the objects in a recent frame."""
        detection = self.look(rospy.Time.now().to_sec() - 1.0, DESCRIBE_TIMEOUT_S)
        if detection is None:
            return {"status": "no_camera",
                    "message": "no frame on %s; is vision_node.py running?" % DETECTIONS_TOPIC}
        keys = ("color", "shape", "side", "bearing_deg", "distance_m", "area_fraction")
        return {"status": "ok",
                "objects": [{k: o.get(k) for k in keys} for o in detection.get("objects", [])]}

    def on_command(self, msg):
        rospy.loginfo("Command: %s", msg.data)
        try:
            cmd = voice_walk.parse_command(msg.data)
        except ValueError as e:
            try:
                raw = json.loads(msg.data)
                cmd = raw if isinstance(raw, dict) else {}
            except ValueError:
                cmd = {}
            self.publish(cmd, status="error", message=str(e))
            return

        if cmd["action"] == "stop":
            with self.lock:
                busy = self.busy
                if busy:
                    self.stop_id = cmd.get("id")
                    self.stop_event.set()
            if not busy:
                self.publish(cmd, status="idle", message="not walking")
        elif cmd["action"] == "status":
            self.publish(cmd, **self.status())
        elif cmd["action"] == "describe":
            self.publish(cmd, **self.describe_view())
        else:
            with self.lock:
                if self.busy:
                    self.publish(cmd, status="busy", message="already walking",
                                 walking=self.current)
                    return
                self.busy = True
            self.commands.put(cmd)

    def status(self):
        with self.lock:
            status = {"status": "walking" if self.busy else "idle",
                      "total_distance_m": round(self.total_m, 4),
                      "last_result": self.last_result}
            if self.busy and self.current_start is not None:
                status["walking"] = self.current
                status["distance_m"] = round(self.current_position() - self.current_start, 4)
        roll, pitch = self.walk.get_roll_pitch()
        status["roll_deg"] = round(float(np.degrees(roll)), 1)
        status["pitch_deg"] = round(float(np.degrees(pitch)), 1)
        return status

    def execute_walk(self, cmd):
        self.stop_event.clear()
        self.stop_id = None
        action = cmd["action"]
        position = self.positions[action]
        policy = None
        # Backward and sideways walks are always scripted (parse_command sets
        # their mode)
        if cmd["mode"] == "rl" and self.load_model() is not None:
            policy = self.greedy_action
        self.robot.begin()
        with self.lock:
            self.current = {"action": action, "distance_m": cmd["distance_m"],
                            "mode": cmd["mode"]}
            self.current_position = position
            self.current_start = position()
        rospy.loginfo("Walking %.2f m %s, mode %s", cmd["distance_m"],
                      self.DIRECTIONS[action], cmd["mode"])
        result = voice_walk.run_walk(
            cmd["distance_m"], cmd["mode"], self.robot, self.gaits[action],
            policy=policy, policy_error=self.model_error,
            should_stop=lambda: self.stop_event.is_set() or rospy.is_shutdown(),
            log=rospy.logwarn, position=position, m_per_step=self.m_per_step[action])

        if result["status"] == voice_walk.FALLEN:
            if self.reset_on_fall:
                rospy.logwarn("Fell (%s); resetting the robot", result.get("fall_reason"))
                self.walk.reset()
                for gait in self.gaits.values():
                    gait.restart()
                result["reset"] = True
        else:
            self.robot.lower_lifted_legs()
        if result["status"] == voice_walk.STOPPED and self.stop_id is not None:
            result["stopped_by"] = self.stop_id
        result["message"] = self.describe(result, action)
        with self.lock:
            self.total_m += result["distance_m"]
            result["total_distance_m"] = round(self.total_m, 4)
            self.last_result = {k: result[k] for k in
                                ("status", "distance_m", "mode_used", "requested_m")}
            self.last_result["action"] = action
            self.current = None
            self.current_start = None
            self.current_position = None
            self.busy = False
        self.publish(cmd, **result)

    def execute_approach(self, cmd):
        """go_to_object: the visual approach (visual_approach.run_approach)."""
        self.stop_event.clear()
        self.stop_id = None
        self.robot.begin()
        self.robot.lower_lifted_legs()
        with self.lock:
            self.current = {"action": "go_to_object", "color": cmd["color"],
                            "shape": cmd["shape"]}
            self.current_position = self.robot.forward
            self.current_start = self.robot.forward()
        rospy.loginfo("Going to the %s %s", cmd["color"], cmd["shape"] or "object")
        result = visual_approach.run_approach(
            self.robot, self.gaits,
            look=lambda: self.look(rospy.Time.now().to_sec()),
            color=cmd["color"], shape=cmd["shape"], config=self.approach_config,
            should_stop=lambda: self.stop_event.is_set() or rospy.is_shutdown(),
            log=rospy.loginfo)

        if result["status"] == visual_approach.FALLEN:
            if self.reset_on_fall:
                rospy.logwarn("Fell (%s); resetting the robot", result.get("fall_reason"))
                self.walk.reset()
                for gait in self.gaits.values():
                    gait.restart()
                result["reset"] = True
        else:
            self.robot.lower_lifted_legs()
        if result["status"] == visual_approach.STOPPED and self.stop_id is not None:
            result["stopped_by"] = self.stop_id
        result["message"] = "%s: %s %s, %d moves, %d steps, forward %.3f m, left %.3f m" % (
            result["status"], result["color"], result["shape"] or "object", result["moves"],
            result["steps"], result["forward_m"], result["left_m"])
        with self.lock:
            self.total_m += result["distance_m"]
            result["total_distance_m"] = round(self.total_m, 4)
            self.last_result = {k: result[k] for k in
                                ("status", "distance_m", "color", "shape")}
            self.last_result["action"] = "go_to_object"
            self.current = None
            self.current_start = None
            self.current_position = None
            self.busy = False
        self.publish(cmd, **result)

    DIRECTIONS = {"walk_forward": "forward", "walk_backward": "backward",
                  "walk_left": "left", "walk_right": "right"}

    @classmethod
    def describe(cls, result, action="walk_forward"):
        side = "" if action == "walk_forward" else " " + cls.DIRECTIONS[action]
        text = "%s: walked %.3f m of %.3f m%s in %d steps, mode %s" % (
            result["status"], result["distance_m"], result["requested_m"], side,
            result["steps"], result["mode_used"])
        if result["fallback"]:
            text += "; switched to scripted (%s)" % result["fallback_reason"]
        if result["status"] == voice_walk.FALLEN:
            text += "; fell (%s)" % result.get("fall_reason")
        return text

    def spin(self):
        while not rospy.is_shutdown():
            try:
                cmd = self.commands.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if cmd["action"] == "go_to_object":
                    self.execute_approach(cmd)
                else:
                    self.execute_walk(cmd)
            except Exception as e:
                rospy.logerr("Walk failed: %s", e)
                with self.lock:
                    self.busy = False
                    self.current = None
                    self.current_start = None
                    self.current_position = None
                self.publish(cmd, status="error", message="walk failed: %s" % e)


if __name__ == "__main__":
    VoiceCommandNode().spin()

#!/usr/bin/env python3
"""
Execute voice commands on Gurdy (in Gazebo, through MyGurdyWalkEnv-v0).

Subscribes /gurdy/voice_command (std_msgs/String, JSON, see voice_walk.py):
    {"action": "walk_forward", "distance_m": 0.2, "mode": "rl" | "scripted", "id": "..."}
    {"action": "stop"}   {"action": "status"}
and publishes exactly one JSON result per command on /gurdy/voice_feedback,
with the command's action and id, so a client can wait for it with
`rostopic echo -n 1`. /gurdy/voice_transcript (std_msgs/String, the raw
STT text) is only logged.

walk_forward steps the --wave-all scripted gait of test_gait.py until the
robot has gained distance_m along the forward axis (at most 0.5 m). With
mode "rl" it acts greedily with the Keras DQN model at ~rl_model_path
(relative paths are under the package), and switches to the scripted gait
if the policy makes less than 5 mm of progress over 32 steps or the model
cannot be loaded; the result says so (fallback, fallback_reason). Walks
start from wherever the robot stands; the sim is reset only at startup
(~reset_on_start) and after a fall (~reset_on_fall).

stop ends a running walk after the current step. The walk's own result
(status "stopped", stopped_by = the stop's id) is then the reply to both, so
a stop during a walk publishes nothing itself; a stop while idle gets status
"idle". status answers right away, also during a walk. A walk command during a walk is answered with
status "busy".

Run with the position controllers loaded, e.g.
    roslaunch my_gurdy_description voice_demo.launch [rl_model_path:=...]
"""
import json
import os
import queue
import threading

import gym
import numpy as np
import rospkg
import rospy
from std_msgs.msg import String

import env as gurdy_env
import voice_walk
from test_gait import gait as gait_moves

COMMAND_TOPIC = "/gurdy/voice_command"
FEEDBACK_TOPIC = "/gurdy/voice_feedback"
TRANSCRIPT_TOPIC = "/gurdy/voice_transcript"


def dqn_features(observation):
    """
    DQN input for an env observation. Must match extract_state_features in
    train_gurdy_dqn.py, which can't be imported here because it opens the
    training plots on import.
    """
    obs = np.asarray(observation, dtype=np.float32)
    return np.concatenate([
        obs[gurdy_env.OBS_LEG_CODES],
        [obs[gurdy_env.OBS_FORWARD_VELOCITY] / 2.0],
        [obs[gurdy_env.OBS_FORWARD_DISTANCE] / 10.0],
        [obs[gurdy_env.OBS_HEAD_HEIGHT] / 0.3],
        obs[gurdy_env.OBS_LAST_ACTION.start:gurdy_env.OBS_SIZE],
    ]).astype(np.float32)


class EnvRobot(object):
    """voice_walk's robot interface on top of the walk env."""

    def __init__(self, walk):
        self.walk = walk
        self.observation = None

    def forward(self):
        return self.walk.get_forward_position()

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
        self.model_path = self._resolve(rospy.get_param("~rl_model_path", ""))
        self.model = None
        self.model_error = None

        self.walk = gym.make("MyGurdyWalkEnv-v0").unwrapped
        self.robot = EnvRobot(self.walk)
        moves, _ = gait_moves(wave_all=True, mirror=False)
        self.gait = voice_walk.ScriptedGait(
            [gurdy_env.action_index(leg, move) for leg, move in moves])
        if rospy.get_param("~reset_on_start", True):
            rospy.loginfo("Resetting the robot")
            self.walk.reset()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.commands = queue.Queue()
        self.busy = False
        self.stop_id = None          # id of the stop that ended the walk
        self.current = None          # the walk command running
        self.current_start = None    # forward position at its start
        self.total_m = 0.0           # forward distance walked since startup
        self.last_result = None

        self.feedback = rospy.Publisher(FEEDBACK_TOPIC, String, queue_size=10)
        rospy.Subscriber(COMMAND_TOPIC, String, self.on_command)
        rospy.Subscriber(TRANSCRIPT_TOPIC, String, self.on_transcript)
        if self.model_path:
            # Load now so the first RL walk doesn't wait for TensorFlow
            self.load_model()
        rospy.loginfo("Voice command node ready on %s", COMMAND_TOPIC)

    @staticmethod
    def _resolve(path):
        if path and not os.path.isabs(path):
            path = os.path.join(rospkg.RosPack().get_path("my_gurdy_description"), path)
        return path

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
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_path)
                rospy.loginfo("Loaded RL model %s", self.model_path)
            except Exception as e:  # any load failure means scripted fallback
                self.model_error = "could not load the RL model: %s" % e
        if self.model_error:
            rospy.logwarn(self.model_error)
        return self.model

    def greedy_action(self):
        state = dqn_features(self.robot.observation).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        return int(np.argmax(q_values))

    def publish(self, cmd, **fields):
        fields["action"] = cmd.get("action")
        if cmd.get("id") is not None:
            fields["id"] = cmd["id"]
        data = json.dumps(fields)
        rospy.loginfo("Feedback: %s", data)
        self.feedback.publish(String(data=data))

    def on_transcript(self, msg):
        rospy.loginfo("Transcript: %s", msg.data)

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
                status["distance_m"] = round(self.robot.forward() - self.current_start, 4)
        roll, pitch = self.walk.get_roll_pitch()
        status["roll_deg"] = round(float(np.degrees(roll)), 1)
        status["pitch_deg"] = round(float(np.degrees(pitch)), 1)
        return status

    def execute_walk(self, cmd):
        self.stop_event.clear()
        self.stop_id = None
        policy = None
        if cmd["mode"] == "rl" and self.load_model() is not None:
            policy = self.greedy_action
        self.robot.begin()
        with self.lock:
            self.current = {"distance_m": cmd["distance_m"], "mode": cmd["mode"]}
            self.current_start = self.robot.forward()
        rospy.loginfo("Walking %.2f m forward, mode %s", cmd["distance_m"], cmd["mode"])
        result = voice_walk.run_walk(
            cmd["distance_m"], cmd["mode"], self.robot, self.gait,
            policy=policy, policy_error=self.model_error,
            should_stop=lambda: self.stop_event.is_set() or rospy.is_shutdown(),
            log=rospy.logwarn)

        if result["status"] == voice_walk.FALLEN:
            if self.reset_on_fall:
                rospy.logwarn("Fell (%s); resetting the robot", result.get("fall_reason"))
                self.walk.reset()
                self.gait.restart()
                result["reset"] = True
        else:
            self.robot.lower_lifted_legs()
        if result["status"] == voice_walk.STOPPED and self.stop_id is not None:
            result["stopped_by"] = self.stop_id
        result["message"] = self.describe(result)
        with self.lock:
            self.total_m += result["distance_m"]
            result["total_distance_m"] = round(self.total_m, 4)
            self.last_result = {k: result[k] for k in
                                ("status", "distance_m", "mode_used", "requested_m")}
            self.current = None
            self.current_start = None
            self.busy = False
        self.publish(cmd, **result)

    @staticmethod
    def describe(result):
        text = "%s: walked %.3f m of %.3f m in %d steps, mode %s" % (
            result["status"], result["distance_m"], result["requested_m"],
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
                self.execute_walk(cmd)
            except Exception as e:
                rospy.logerr("Walk failed: %s", e)
                with self.lock:
                    self.busy = False
                    self.current = None
                    self.current_start = None
                self.publish(cmd, status="error", message="walk failed: %s" % e)


if __name__ == "__main__":
    VoiceCommandNode().spin()

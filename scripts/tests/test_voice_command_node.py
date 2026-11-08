"""Offline smoke test of voice_command_node.py, with stub rospy, gym, env
and test_gait modules (no ROS needed). Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import json
import os
import sys
import threading
import time
import types
import unittest
from collections import deque

import numpy as np

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)


# --- stub modules ---------------------------------------------------------

class FakeWalk(object):
    """Walk env stand-in: every action moves the base 2 mm forward."""
    GAIN = 0.002

    def __init__(self):
        self.x = 0.0
        self.leg_states = dict.fromkeys(range(1, 7), "GROUND_OUT")
        self.forward_history = deque()
        self.resets = 0
        self.actions = []
        self.step_delay = 0.0
        self.fall_at = None
        self._odom = types.SimpleNamespace(
            pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=None)))

    def get_forward_position(self):
        return self.x

    def get_roll_pitch(self):
        return 0.0, 0.0

    def reset(self):
        self.resets += 1
        self.x = 0.0
        self.leg_states = dict.fromkeys(range(1, 7), "GROUND_OUT")

    def _get_obs(self):
        return np.zeros(35, dtype=np.float32)

    def step(self, action):
        time.sleep(self.step_delay)
        self.actions.append(action)
        leg, move = ACTIONS[action]
        if move == "LIFT":
            self.leg_states[leg] = "LIFT_OUT"
        elif move == "LOWER":
            self.leg_states[leg] = "GROUND_OUT"
        self.x += self.GAIN
        fallen = self.fall_at is not None and len(self.actions) >= self.fall_at
        return self._get_obs(), 0.0, False, {"fallen": fallen,
                                             "fall_reason": "tilt" if fallen else None}


LEGS = (1, 2, 3, 4, 5, 6)
MOVES = ("LIFT", "LOWER", "REACH_IN", "REACH_OUT")
ACTIONS = [(leg, move) for leg in LEGS for move in MOVES] + [(None, "STAND")]
PARAMS = {}
PUBLISHED = []
SUBSCRIBERS = {}


def install_stubs():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: PARAMS.get(name, default)
    rospy.is_shutdown = lambda: False
    for level in ("loginfo", "logwarn", "logerr"):
        setattr(rospy, level, lambda *a, **k: None)

    class Publisher(object):
        def __init__(self, topic, *a, **k):
            self.topic = topic

        def publish(self, msg):
            PUBLISHED.append((self.topic, msg.data))

    rospy.Publisher = Publisher
    rospy.Subscriber = lambda topic, cls, cb: SUBSCRIBERS.__setitem__(topic, cb)

    rospkg = types.ModuleType("rospkg")
    rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda pkg: "/pkg")

    gym = types.ModuleType("gym")
    gym.make = lambda name: types.SimpleNamespace(unwrapped=FakeWalk())

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = lambda data="": types.SimpleNamespace(data=data)
    std_msgs.msg = std_msgs_msg

    env = types.ModuleType("env")
    env.LEGS = LEGS
    env.LIFT, env.LOWER, env.REACH_IN, env.REACH_OUT = MOVES
    env.ACTIONS = ACTIONS
    env.action_index = lambda leg, move: ACTIONS.index((leg, move))
    env.POSE_FLAGS = {"GROUND_OUT": (False, True), "LIFT_OUT": (True, True)}
    env.OBS_LEG_CODES = slice(0, 6)
    env.OBS_FORWARD_VELOCITY, env.OBS_FORWARD_DISTANCE, env.OBS_HEAD_HEIGHT = 6, 7, 8
    env.OBS_LAST_ACTION = slice(9, 34)
    env.OBS_SIZE = 35

    test_gait = types.ModuleType("test_gait")

    def gait(wave_all, mirror):
        moves = []
        for leg in (1, 4, 5, 2):
            moves += [(leg, "LIFT"), (leg, "REACH_IN"), (leg, "LOWER"), (leg, "REACH_OUT")]
        return moves, "+x"

    test_gait.gait = gait

    sys.modules.update({"rospy": rospy, "rospkg": rospkg, "gym": gym,
                        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
                        "env": env, "test_gait": test_gait})


install_stubs()
import voice_command_node as node_mod  # noqa: E402


# --- tests ----------------------------------------------------------------

class NodeTest(unittest.TestCase):
    def setUp(self):
        PARAMS.clear()
        del PUBLISHED[:]
        SUBSCRIBERS.clear()
        self.node = node_mod.VoiceCommandNode()
        self.walk = self.node.walk

    def send(self, **cmd):
        SUBSCRIBERS[node_mod.COMMAND_TOPIC](types.SimpleNamespace(data=json.dumps(cmd)))

    def feedback(self):
        return [json.loads(d) for t, d in PUBLISHED if t == node_mod.FEEDBACK_TOPIC]

    def run_queued(self):
        self.node.execute_walk(self.node.commands.get_nowait())

    def test_resets_on_start(self):
        self.assertEqual(self.walk.resets, 1)

    def test_scripted_walk(self):
        self.send(action="walk_forward", distance_m=0.1, mode="scripted", id="w1")
        self.run_queued()
        [result] = self.feedback()
        self.assertEqual((result["action"], result["id"], result["status"]),
                         ("walk_forward", "w1", "done"))
        self.assertGreaterEqual(result["distance_m"], 0.1)
        self.assertEqual(result["mode_used"], "scripted")
        self.assertIn("message", result)
        # Ends with every leg down
        self.assertTrue(all(s == "GROUND_OUT" for s in self.walk.leg_states.values()))
        # The gait starts with leg 1 LIFT
        self.assertEqual(self.walk.actions[0], 0)

    def test_rl_without_model_falls_back(self):
        self.send(action="walk_forward", distance_m=0.05, mode="rl", id="r1")
        self.run_queued()
        [result] = self.feedback()
        self.assertEqual(result["status"], "done")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["fallback_reason"], "rl_model_path is not set")
        self.assertIn("switched to scripted", result["message"])

    def test_rl_with_model(self):
        class Model(object):
            def predict(self, x, verbose=0):
                assert x.shape == (1, 35)
                q = np.zeros((1, 25))
                q[0, 3] = 1.0  # leg 1 REACH_OUT
                return q
        self.node.model = Model()
        self.send(action="walk_forward", distance_m=0.02, mode="rl")
        self.run_queued()
        [result] = self.feedback()
        self.assertEqual((result["status"], result["mode_used"]), ("done", "rl"))
        self.assertTrue(all(a == 3 for a in self.walk.actions))

    def test_status_and_total(self):
        self.send(action="walk_forward", distance_m=0.05)
        self.run_queued()
        self.send(action="status", id="s1")
        status = self.feedback()[-1]
        self.assertEqual((status["action"], status["id"], status["status"]),
                         ("status", "s1", "idle"))
        self.assertGreaterEqual(status["total_distance_m"], 0.05)
        self.assertEqual(status["last_result"]["status"], "done")

    def test_stop_when_idle(self):
        self.send(action="stop", id="x")
        self.assertEqual(self.feedback()[-1]["status"], "idle")

    def test_bad_command(self):
        SUBSCRIBERS[node_mod.COMMAND_TOPIC](types.SimpleNamespace(data='{"action": "dance", "id": 3}'))
        result = self.feedback()[-1]
        self.assertEqual((result["status"], result["id"]), ("error", 3))
        SUBSCRIBERS[node_mod.COMMAND_TOPIC](types.SimpleNamespace(data="not json"))
        self.assertEqual(self.feedback()[-1]["status"], "error")

    def test_fall_resets(self):
        self.walk.fall_at = 3
        self.send(action="walk_forward", distance_m=0.2)
        self.run_queued()
        result = self.feedback()[-1]
        self.assertEqual(result["status"], "fallen")
        self.assertTrue(result["reset"])
        self.assertEqual(self.walk.resets, 2)

    def test_stop_busy_and_status_during_walk(self):
        """Commands sent from another thread while the main loop walks."""
        self.walk.step_delay = 0.005
        worker = threading.Thread(target=self.node.spin)
        worker.daemon = True
        worker.start()
        self.send(action="walk_forward", distance_m=0.5, id="long")
        time.sleep(0.15)
        self.send(action="walk_forward", distance_m=0.1, id="second")
        self.send(action="status", id="st")
        self.send(action="stop", id="halt")
        deadline = time.time() + 5
        while time.time() < deadline and not any(
                f.get("id") == "long" for f in self.feedback()):
            time.sleep(0.01)
        by_id = {f.get("id"): f for f in self.feedback()}
        self.assertEqual(by_id["second"]["status"], "busy")
        self.assertEqual(by_id["st"]["status"], "walking")
        self.assertGreater(by_id["st"]["distance_m"], 0)
        # The stop gets no reply of its own; the walk's result is it
        self.assertNotIn("halt", by_id)
        self.assertEqual(by_id["long"]["status"], "stopped")
        self.assertEqual(by_id["long"]["stopped_by"], "halt")
        self.assertEqual(len(self.feedback()), 3)
        self.assertLess(by_id["long"]["distance_m"], 0.5)
        self.assertFalse(self.node.busy)
        # A lifted leg was put back down after the stop
        self.assertTrue(all(s == "GROUND_OUT" for s in self.walk.leg_states.values()))


class FeatureTest(unittest.TestCase):
    def test_matches_training_features(self):
        obs = np.arange(35, dtype=np.float32) / 35.0
        features = node_mod.dqn_features(obs)
        expected = np.concatenate([obs[0:6], [obs[6] / 2.0], [obs[7] / 10.0],
                                   [obs[8] / 0.3], obs[9:35]])
        self.assertEqual(features.shape, (35,))
        np.testing.assert_allclose(features, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

"""Offline tests for voice_walk (no ROS). Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_walk as vw  # noqa: E402


class FakeRobot(object):
    """Moves `gain(action, step)` metres per step; can fall at a step."""

    def __init__(self, gain, fall_at=None):
        self.position = 1.0  # not 0, so the walk must measure from its start
        self.gain = gain
        self.fall_at = fall_at
        self.actions = []

    def forward(self):
        return self.position

    def step(self, action):
        self.actions.append(action)
        self.position += self.gain(action, len(self.actions))
        if self.fall_at is not None and len(self.actions) >= self.fall_at:
            return True, "tilt"
        return False, None


SCRIPTED = "S"  # the fake gait's actions are strings starting with S
RL = "R"


def gait():
    return vw.ScriptedGait([SCRIPTED + str(i) for i in range(4)])


def by_kind(scripted_gain, rl_gain):
    return lambda action, step: scripted_gain if action.startswith(SCRIPTED) else rl_gain


class ParseCommandTest(unittest.TestCase):
    def test_walk(self):
        cmd = vw.parse_command(json.dumps(
            {"action": "walk_forward", "distance_m": 0.3, "mode": "rl", "id": "a1"}))
        self.assertEqual(cmd, {"action": "walk_forward", "distance_m": 0.3,
                               "mode": "rl", "id": "a1"})

    def test_walk_defaults_and_cap(self):
        cmd = vw.parse_command('{"action": "walk_forward"}')
        self.assertEqual((cmd["distance_m"], cmd["mode"], cmd["id"]), (0.2, "scripted", None))
        self.assertEqual(vw.parse_command(
            '{"action": "walk_forward", "distance_m": 2}')["distance_m"], 0.5)
        self.assertEqual(vw.parse_command(
            '{"action": "walk_forward", "distance_m": -1}')["distance_m"], 0.2)

    def test_side_walks_are_scripted(self):
        for action in ("walk_left", "walk_right"):
            cmd = vw.parse_command(json.dumps(
                {"action": action, "distance_m": 0.9, "mode": "rl", "id": "s"}))
            self.assertEqual(cmd, {"action": action, "distance_m": 0.5,
                                   "mode": "scripted", "id": "s"})
            self.assertEqual(vw.parse_command(json.dumps({"action": action}))["distance_m"], 0.2)

    def test_stop_status(self):
        self.assertEqual(vw.parse_command('{"action": "stop"}')["action"], "stop")
        self.assertEqual(vw.parse_command('{"action": "status", "id": 7}')["id"], 7)

    def test_errors(self):
        for data in ("nope", "[1]", '{"action": "dance"}',
                     '{"action": "walk_forward", "mode": "fast"}',
                     '{"action": "walk_forward", "distance_m": "far"}'):
            with self.assertRaises(ValueError, msg=data):
                vw.parse_command(data)


class ProgressWindowTest(unittest.TestCase):
    def test_needs_full_window(self):
        w = vw.ProgressWindow(3, 0.005, 0.0)
        for _ in range(2):
            w.add(0.0)
            self.assertFalse(w.stalled())
        w.add(0.0)
        self.assertTrue(w.stalled())

    def test_slides(self):
        w = vw.ProgressWindow(2, 0.005, 0.0)
        for p in (0.01, 0.02, 0.022, 0.023):
            w.add(p)
        self.assertTrue(w.stalled())  # 0.020 -> 0.023 is 3 mm


class ScriptedGaitTest(unittest.TestCase):
    def test_cycles_and_keeps_phase(self):
        g = gait()
        self.assertEqual([g.next_action() for _ in range(6)],
                         ["S0", "S1", "S2", "S3", "S0", "S1"])
        self.assertEqual(g.next_action(), "S2")
        g.restart()
        self.assertEqual(g.next_action(), "S0")


class RunWalkTest(unittest.TestCase):
    def test_scripted_reaches_distance(self):
        robot = FakeRobot(lambda a, s: 0.002)
        result = vw.run_walk(0.2, "scripted", robot, gait())
        self.assertEqual(result["status"], vw.DONE)
        self.assertGreaterEqual(result["distance_m"], 0.2)
        self.assertLess(result["distance_m"], 0.2021)
        self.assertEqual(result["steps"], 100)
        self.assertEqual(result["mode_used"], "scripted")
        self.assertFalse(result["fallback"])
        self.assertTrue(all(a.startswith(SCRIPTED) for a in robot.actions))

    def test_rl_good_policy_stays_rl(self):
        robot = FakeRobot(by_kind(0.002, 0.004))
        result = vw.run_walk(0.2, "rl", robot, gait(), policy=lambda: "R")
        self.assertEqual((result["status"], result["mode_used"]), (vw.DONE, "rl"))
        self.assertFalse(result["fallback"])
        self.assertEqual(result["steps"], 50)

    def test_rl_stall_falls_back_after_32_steps(self):
        robot = FakeRobot(by_kind(0.002, 0.0001))  # RL: 3.2 mm per 32 steps
        result = vw.run_walk(0.1, "rl", robot, gait(), policy=lambda: "R")
        self.assertEqual(result["status"], vw.DONE)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["mode_used"], "rl+scripted")
        self.assertIn("32 steps", result["fallback_reason"])
        self.assertAlmostEqual(result["fallback_at_m"], 0.0032)
        self.assertEqual(robot.actions[:32], ["R"] * 32)
        self.assertTrue(all(a.startswith(SCRIPTED) for a in robot.actions[32:]))
        self.assertGreaterEqual(result["distance_m"], 0.1)

    def test_rl_slow_but_enough_progress_keeps_rl(self):
        robot = FakeRobot(by_kind(0.002, 0.0002))  # 6.4 mm per 32 steps
        result = vw.run_walk(0.01, "rl", robot, gait(), policy=lambda: "R")
        self.assertEqual((result["status"], result["mode_used"]), (vw.DONE, "rl"))

    def test_rl_backwards_falls_back(self):
        robot = FakeRobot(by_kind(0.002, -0.001))
        result = vw.run_walk(0.05, "rl", robot, gait(), policy=lambda: "R")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["status"], vw.DONE)

    def test_rl_without_policy_uses_scripted(self):
        robot = FakeRobot(lambda a, s: 0.002)
        result = vw.run_walk(0.05, "rl", robot, gait(), policy=None,
                             policy_error="rl_model_path is not set")
        self.assertEqual(result["mode_used"], "scripted")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["fallback_reason"], "rl_model_path is not set")
        self.assertEqual(result["fallback_at_m"], 0.0)

    def test_stop(self):
        robot = FakeRobot(lambda a, s: 0.002)
        result = vw.run_walk(0.2, "scripted", robot, gait(),
                             should_stop=lambda: len(robot.actions) >= 10)
        self.assertEqual(result["status"], vw.STOPPED)
        self.assertEqual(result["steps"], 10)
        self.assertAlmostEqual(result["distance_m"], 0.02)

    def test_fall(self):
        robot = FakeRobot(lambda a, s: 0.002, fall_at=5)
        result = vw.run_walk(0.2, "scripted", robot, gait())
        self.assertEqual((result["status"], result["fall_reason"]), (vw.FALLEN, "tilt"))
        self.assertEqual(result["steps"], 5)

    def test_scripted_stuck(self):
        robot = FakeRobot(lambda a, s: 0.0)
        result = vw.run_walk(0.2, "scripted", robot, gait())
        self.assertEqual(result["status"], vw.STUCK)
        self.assertEqual(result["steps"], vw.SCRIPTED_WINDOW)

    def test_step_limit(self):
        # Slow enough to never stall (6 mm per 64 steps) or arrive
        robot = FakeRobot(lambda a, s: 0.0001)
        result = vw.run_walk(0.2, "scripted", robot, gait())
        self.assertEqual(result["status"], vw.STEP_LIMIT)
        self.assertEqual(result["steps"], vw.max_steps_for(0.2))

    def test_already_there(self):
        robot = FakeRobot(lambda a, s: 0.002)
        result = vw.run_walk(0.0, "scripted", robot, gait())
        self.assertEqual((result["status"], result["steps"]), (vw.DONE, 0))

    def test_step_limit_covers_scripted_gait(self):
        # The real gait (1.7 mm per step) must fit in the limit for 50 cm
        self.assertGreater(vw.max_steps_for(0.5), 0.5 / vw.SCRIPTED_M_PER_STEP)

    def test_sideways_uses_position(self):
        """A side-step measures position(), not forward(), with its own step limit."""
        robot = FakeRobot(lambda a, s: 0.002)
        lateral = {"y": 0.3}

        def step(action):
            lateral["y"] -= 0.0015  # walking right: y falls
            robot.actions.append(action)
            return False, None
        robot.step = step
        result = vw.run_walk(0.1, "scripted", robot, gait(),
                             position=lambda: -lateral["y"], m_per_step=vw.SIDE_M_PER_STEP)
        self.assertEqual(result["status"], vw.DONE)
        self.assertGreaterEqual(result["distance_m"], 0.1)
        self.assertEqual(result["steps"], 67)  # ceil(0.1 / 1.5 mm)

    def test_step_limit_covers_side_gait(self):
        self.assertLess(vw.SIDE_M_PER_STEP, vw.SCRIPTED_M_PER_STEP)
        self.assertGreater(vw.max_steps_for(0.5, vw.SIDE_M_PER_STEP),
                           vw.max_steps_for(0.5))


if __name__ == "__main__":
    unittest.main()

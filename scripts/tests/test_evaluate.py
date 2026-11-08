"""Offline tests for evaluate.py (no ROS). Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import os
import shutil
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate as ev  # noqa: E402


def episode(distance, steps=100, fallen=False, invalid=0, end=None):
    return {"distance_m": distance, "steps": steps, "fallen": fallen, "fall_reason": None,
            "invalid_lifts": invalid, "reward": 0.0,
            "end": end or (ev.FALLEN if fallen else ev.STUCK)}


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_lists_models_by_episode(self):
        for name in ("gurdy_dqn_model_100", "gurdy_dqn_model_25", "gurdy_dqn_model_25_state.json",
                     "gurdy_dqn_model_100_state.json", "dqn_learning_curve.png", "gurdy_dqn_model_x"):
            open(os.path.join(self.dir, name), "w").close()
        os.mkdir(os.path.join(self.dir, "gurdy_dqn_model_50"))  # SavedModel directory
        names = [os.path.basename(p) for p in ev.list_checkpoints(self.dir)]
        self.assertEqual(names, ["gurdy_dqn_model_25", "gurdy_dqn_model_50", "gurdy_dqn_model_100"])

    def test_missing_dir(self):
        self.assertEqual(ev.list_checkpoints(os.path.join(self.dir, "nope")), [])

    def test_episode_number(self):
        self.assertEqual(ev.checkpoint_episode("output/dqn/gurdy_dqn_model_125/"), 125)
        self.assertIsNone(ev.checkpoint_episode("gurdy_dqn_model_125_state.json"))


class StatsTest(unittest.TestCase):
    def test_end_reason(self):
        self.assertEqual(ev.end_reason(True, 0.6, 10, 400, 0.5), ev.FALLEN)
        self.assertEqual(ev.end_reason(False, 0.5, 290, 400, 0.5), ev.GOAL)
        self.assertEqual(ev.end_reason(False, 0.2, 400, 400, 0.5), ev.STEP_LIMIT)
        self.assertEqual(ev.end_reason(False, 0.0, 64, 400, 0.5), ev.STUCK)

    def test_summarize(self):
        s = ev.summarize("d/gurdy_dqn_model_75", [
            episode(0.5, 290, end=ev.GOAL), episode(0.1, 64, invalid=4), episode(0.0, 20, fallen=True)])
        self.assertAlmostEqual(s["mean_distance_m"], 0.2)
        self.assertEqual((s["max_distance_m"], s["min_distance_m"]), (0.5, 0.0))
        self.assertEqual((s["falls"], s["goals"], s["episodes"]), (1, 1, 3))
        self.assertAlmostEqual(s["mean_steps"], 124.666, places=2)
        self.assertAlmostEqual(s["mean_invalid_lifts"], 4 / 3.0)
        self.assertEqual(s["training_episode"], 75)

    def test_rank_best_first(self):
        a = ev.summarize("gurdy_dqn_model_25", [episode(0.10)])
        b = ev.summarize("gurdy_dqn_model_50", [episode(0.30)])
        c = ev.summarize("gurdy_dqn_model_75", [episode(0.30, fallen=True)])
        d = ev.summarize("gurdy_dqn_model_100", [episode(0.30, steps=50)])
        self.assertEqual([s["checkpoint"] for s in ev.rank([a, b, c, d])],
                         ["gurdy_dqn_model_100", "gurdy_dqn_model_50", "gurdy_dqn_model_75",
                          "gurdy_dqn_model_25"])

    def test_ranking_table(self):
        table = ev.format_ranking([ev.summarize("x/gurdy_dqn_model_25", [episode(0.1)]),
                                   ev.summarize("x/gurdy_dqn_model_50", [episode(0.3)])])
        lines = table.splitlines()
        self.assertIn("rank", lines[0])
        self.assertTrue(lines[2].lstrip().startswith("1  gurdy_dqn_model_50"))
        self.assertTrue(lines[3].lstrip().startswith("2  gurdy_dqn_model_25"))

    def test_episode_lines(self):
        self.assertIn("forward +0.123 m", ev.format_episode(1, episode(0.123)))
        self.assertIn("goals 0/1", ev.format_summary(ev.summarize("m", [episode(0.1)])))


class RunEpisodeTest(unittest.TestCase):
    def test_runs_until_done(self):
        class Walk(object):
            TARGET_FORWARD_DISTANCE = 0.5
            max_steps = 400

            def reset(self):
                self.x, self.start_forward, self.n = 1.0, 1.0, 0
                self.invalid_lifts, self.cumulated_reward = 0, 0.0
                return "obs0"

            def get_forward_position(self):
                return self.x

            def step(self, action):
                assert action == 7
                self.n += 1
                self.x += 0.01
                self.invalid_lifts += self.n % 2
                return "obs", 1.0, self.n >= 50, {"steps": self.n, "fallen": False,
                                                  "fall_reason": None}

        seen = []
        result = ev.run_episode(Walk(), lambda obs: seen.append(obs) or 7)
        self.assertEqual(result["steps"], 50)
        self.assertAlmostEqual(result["distance_m"], 0.5)
        self.assertEqual(result["end"], ev.GOAL)
        self.assertEqual(result["invalid_lifts"], 25)
        self.assertEqual(seen[0], "obs0")


if __name__ == "__main__":
    unittest.main()

"""Offline tests for dashboard_state, dashboard (headless, Agg) and plot_path.
Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import dashboard  # noqa: E402
import dashboard_state as ds  # noqa: E402
import plot_path  # noqa: E402

POSES = {"GROUND_IN": (-1.40, -2.30), "GROUND_OUT": (-1.45, -2.10),
         "LIFT_IN": (-1.30, -2.50), "LIFT_OUT": (-1.50, -2.40)}
LIFTED = ("LIFT_IN", "LIFT_OUT")


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def roll_quat(roll):
    return math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)


def make_state():
    s = ds.DashboardState(pose_angles=POSES, lifted_poses=LIFTED)
    for leg in range(1, 7):
        s.joint_command("head_upperlegM%d_joint" % leg, POSES["GROUND_OUT"][0])
        s.joint_command("upperlegM%d_lowerlegM%d_joint" % (leg, leg), POSES["GROUND_OUT"][1])
    return s


def at(s, t, x, y=0.0, z=0.08, quat=(0, 0, 0, 1)):
    s.odom(t, x, y, z, *quat)


class GeometryTest(unittest.TestCase):
    def test_rpy(self):
        r, p, y = ds.quaternion_to_rpy(*yaw_quat(math.radians(30)))
        self.assertAlmostEqual(math.degrees(y), 30)
        self.assertAlmostEqual(r, 0)
        r, p, y = ds.quaternion_to_rpy(*roll_quat(math.radians(70)))
        self.assertAlmostEqual(math.degrees(r), 70)

    def test_leg_joint(self):
        self.assertEqual(ds.leg_joint("head_upperlegM3_joint"), (3, "hip"))
        self.assertEqual(ds.leg_joint("upperlegM5_lowerlegM5_joint"), (5, "knee"))
        self.assertIsNone(ds.leg_joint("other"))


class StateTest(unittest.TestCase):
    def test_walk_lifecycle(self):
        s = make_state()
        at(s, 0.0, 1.0, 0.5)
        self.assertEqual(s.status(), ds.STANDING)
        self.assertEqual(s.start, (1.0, 0.5))
        s.transcript("गरदी आगे बीस सेंटीमीटर चलो।")
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.2, "mode": "rl", "id": "a"}))
        self.assertEqual(s.status(), ds.WALKING)
        self.assertEqual(s.target_point(), (1.2, 0.5))
        self.assertEqual(s.mode_label(), "RL")
        at(s, 1.0, 1.05, 0.51)
        done, target = s.walk_progress()
        self.assertAlmostEqual(done, 0.05)
        self.assertEqual(target, 0.2)
        self.assertAlmostEqual(s.total_distance(), 0.05)
        # A second walk while walking is answered busy and must not replace the first
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.5, "id": "b"}))
        s.feedback(json.dumps({"action": "walk_forward", "id": "b", "status": "busy"}))
        self.assertEqual(s.walk["id"], "a")
        s.feedback(json.dumps({"action": "walk_forward", "id": "a", "status": "done",
                               "distance_m": 0.201, "requested_m": 0.2, "mode_used": "rl+scripted",
                               "fallback": True, "fallback_reason": "RL made only 1.0 mm in 32 steps",
                               "total_distance_m": 0.201}))
        self.assertEqual(s.status(), ds.STANDING)
        self.assertIsNone(s.target_point())
        self.assertEqual(s.walk_progress(), (0.201, 0.2))
        self.assertEqual(s.mode_label(), "RL -> scripted")
        self.assertEqual(s.total_distance(), 0.201)
        rows = dict(s.panel())
        self.assertEqual(rows["this walk"], "0.201 / 0.200 m")
        self.assertIn("fallback", rows["result"])
        self.assertEqual(rows["heard"], "गरदी आगे बीस सेंटीमीटर चलो।")

    def test_backward_target(self):
        s = make_state()
        at(s, 0.0, 1.0, 0.5)
        s.command(json.dumps({"action": "walk_backward", "distance_m": 0.2, "mode": "rl",
                              "id": "b"}))
        self.assertEqual(s.status(), ds.WALKING)
        # Behind the robot, and scripted whatever mode was sent
        self.assertEqual(s.target_point(), (0.8, 0.5))
        self.assertEqual(s.mode_label(), "scripted")
        self.assertEqual(s.command_text, "walk_backward 0.20 m (scripted)")
        at(s, 1.0, 0.95, 0.53)
        self.assertAlmostEqual(s.walk_progress()[0], 0.05)
        self.assertAlmostEqual(s.total_distance(), 0.05)
        s.feedback(json.dumps({"action": "walk_backward", "id": "b", "status": "stopped",
                               "distance_m": 0.07, "requested_m": 0.2,
                               "mode_used": "scripted", "total_distance_m": 0.07}))
        self.assertEqual(s.status(), ds.STOPPED)
        self.assertIsNone(s.target_point())
        # forward_sign -1: forward is world -x, so backward is +x
        s = ds.DashboardState(forward_sign=-1)
        at(s, 0.0, 1.0, 0.5)
        s.command(json.dumps({"action": "walk_backward", "distance_m": 0.3, "id": "c"}))
        self.assertEqual(s.target_point(), (1.3, 0.5))

    def test_sideways_target(self):
        s = make_state()
        at(s, 0.0, 1.0, 0.5)
        s.command(json.dumps({"action": "walk_left", "distance_m": 0.2, "id": "l"}))
        self.assertEqual(s.status(), ds.WALKING)
        self.assertEqual(s.target_point(), (1.0, 0.7))
        self.assertEqual(s.mode_label(), "scripted")
        self.assertEqual(s.command_text, "walk_left 0.20 m (scripted)")
        # Progress is along +y only; x drift does not count
        at(s, 1.0, 1.02, 0.56)
        self.assertAlmostEqual(s.walk_progress()[0], 0.06)
        self.assertAlmostEqual(s.total_distance(), 0.06)
        self.assertAlmostEqual(s.csv_row()["target_y"], 0.7)
        s.feedback(json.dumps({"action": "walk_left", "id": "l", "status": "done",
                               "distance_m": 0.2004, "requested_m": 0.2,
                               "mode_used": "scripted", "total_distance_m": 0.2004}))
        self.assertIsNone(s.target_point())
        self.assertEqual(s.walk_progress(), (0.2004, 0.2))
        self.assertIn("done: 0.200 m of 0.200 m", s.result_text)
        # Right is toward -y, and forward_sign does not flip it
        s = ds.DashboardState(forward_sign=-1)
        at(s, 0.0, 1.0, 0.5)
        s.command(json.dumps({"action": "walk_right", "distance_m": 0.3, "id": "r"}))
        self.assertEqual(s.target_point(), (1.0, 0.2))
        at(s, 1.0, 1.0, 0.4)
        self.assertAlmostEqual(s.walk_progress()[0], 0.1)

    def test_stopped_and_fallen(self):
        s = make_state()
        at(s, 0.0, 0.0)
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.3}))
        s.feedback(json.dumps({"action": "walk_forward", "status": "stopped", "distance_m": 0.1,
                               "requested_m": 0.3}))
        self.assertEqual(s.status(), ds.STOPPED)
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.3, "id": "f"}))
        s.feedback(json.dumps({"action": "walk_forward", "id": "f", "status": "fallen",
                               "distance_m": 0.02, "requested_m": 0.3, "reset": True}))
        self.assertEqual(s.status(), ds.FALLEN)

    def test_tilt_is_fallen(self):
        s = make_state()
        at(s, 0.0, 0.0, quat=roll_quat(math.radians(75)))
        self.assertEqual(s.status(), ds.FALLEN)

    def test_status_feedback_does_not_end_walk(self):
        s = make_state()
        at(s, 0.0, 0.0)
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.2, "id": "w"}))
        s.feedback(json.dumps({"action": "status", "status": "walking", "total_distance_m": 0.4}))
        self.assertEqual(s.status(), ds.WALKING)
        self.assertIn("status: walking", s.result_text)

    def test_lifted_leg(self):
        s = make_state()
        self.assertEqual(s.lifted_legs(), [])
        s.joint_command("head_upperlegM4_joint", -1.31)  # near LIFT_IN
        s.joint_command("upperlegM4_lowerlegM4_joint", -2.48)
        self.assertEqual(s.lifted_legs(), [4])
        self.assertEqual(dict(s.panel())["lifted leg"], "4")

    def test_trail_breaks_on_reset(self):
        s = make_state()
        at(s, 0.0, 0.0)
        at(s, 0.1, 0.0005)   # under TRAIL_MIN_STEP_M: skipped
        at(s, 0.2, 0.01)
        at(s, 0.3, 0.5)      # teleport
        self.assertEqual(s.trail_x, [0.0, 0.01, None, 0.5])

    def test_csv_rows_at_10hz(self):
        s = make_state()
        logged = []
        for i in range(25):
            at(s, i * 0.02, i * 0.001)
            if s.should_log():
                logged.append(s.csv_row())
        self.assertEqual(len(logged), 5)  # 0.0, 0.1, ..., 0.4 s
        self.assertEqual(set(logged[0]), set(ds.CSV_FIELDS))
        self.assertEqual(logged[0]["status"], ds.STANDING)
        at(s, 0.0, 0.0)  # sim time reset
        self.assertTrue(s.should_log())

    def test_bad_messages(self):
        s = make_state()
        s.command("nope")
        s.feedback("[1, 2]")
        s.feedback("nope")
        self.assertIn("not JSON", s.command_text)
        self.assertEqual(s.status(), ds.STANDING)


class DashboardFigureTest(unittest.TestCase):
    def test_renders(self):
        import matplotlib.pyplot as plt
        s = make_state()
        fig = plt.figure(figsize=(15, 8))
        board = dashboard.Dashboard(fig, s, threading.Lock())
        board.update()  # before any odom
        at(s, 0.0, 0.0, 0.0, quat=yaw_quat(0.3))
        s.transcript("गरदी आगे बीस सेंटीमीटर चलो। " * 4)
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.2, "id": "a"}))
        for i in range(1, 20):
            at(s, i * 0.1, i * 0.005, 0.001 * i, quat=yaw_quat(0.3))
            board.update(i)
        self.assertEqual(board.status_text.get_text(), ds.WALKING)
        self.assertEqual(board.target.get_data()[0][0], 0.2)
        lo, hi = board.ax.get_xlim()
        self.assertLessEqual(lo, -0.1)
        self.assertGreaterEqual(hi, 0.3)
        heard = [v.get_text() for v in board.values if "गरदी" in v.get_text()][0]
        self.assertLessEqual(len(heard.split("\n")), 2)
        fig.canvas.draw()
        plt.close(fig)

    def test_devanagari_font(self):
        path = dashboard.find_devanagari_font()
        if path is None:
            self.skipTest("no Devanagari font installed here")
        from matplotlib.ft2font import FT2Font
        self.assertIn(0x0917, FT2Font(path).get_charmap())


class PlotPathTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write_log(self):
        s = make_state()
        path = os.path.join(self.dir, "path_test.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ds.CSV_FIELDS)
            writer.writeheader()
            t = 0.0
            for i in range(10):
                at(s, t, 0.0)
                t += 0.1
                if s.should_log():
                    writer.writerow(s.csv_row())
            s.command(json.dumps({"action": "walk_forward", "distance_m": 0.2, "id": "a"}))
            for i in range(30):
                at(s, t, i * 0.007, 0.002 * i)
                t += 0.1
                if s.should_log():
                    writer.writerow(s.csv_row())
            s.feedback(json.dumps({"action": "walk_forward", "id": "a", "status": "fallen",
                                   "distance_m": 0.2, "requested_m": 0.2}))
            for i in range(5):
                at(s, t, 0.0)  # reset: back to the start
                t += 0.1
                if s.should_log():
                    writer.writerow(s.csv_row())
        return path

    def test_segments(self):
        rows = plot_path.read_path(self.write_log())
        self.assertEqual(len(rows), 45)
        statuses = [status for status, _ in plot_path.segments(rows)]
        self.assertEqual(statuses, ["STANDING", "WALKING", "FALLEN"])
        self.assertEqual(rows[15]["target_x"], 0.2)

    def test_writes_png(self):
        csv_path = self.write_log()
        plot_path.main([csv_path])
        png = os.path.splitext(csv_path)[0] + ".png"
        with open(png, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        out = os.path.join(self.dir, "other.png")
        plot_path.main([csv_path, "-o", out])
        self.assertTrue(os.path.getsize(out) > 10000)


if __name__ == "__main__":
    unittest.main()

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


class ImageTest(unittest.TestCase):
    def msg(self, encoding, img, pad=0):
        import types
        import numpy as np
        h, w = img.shape[:2]
        rows = img.reshape(h, -1)
        rows = np.hstack([rows, np.full((h, pad), 7, np.uint8)])
        return types.SimpleNamespace(encoding=encoding, height=h, width=w,
                                     step=rows.shape[1], data=rows.tobytes())

    def test_bgr_with_padding(self):
        import numpy as np
        img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        rgb = dashboard.image_to_rgb(self.msg("bgr8", img, pad=4))
        np.testing.assert_array_equal(rgb, img[:, :, ::-1])

    def test_rgb_mono_and_decimate(self):
        import numpy as np
        img = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        np.testing.assert_array_equal(dashboard.image_to_rgb(self.msg("rgb8", img), 2),
                                      img[::2, ::2])
        mono = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rgb = dashboard.image_to_rgb(self.msg("mono8", mono))
        self.assertEqual(rgb.shape, (3, 4, 3))
        np.testing.assert_array_equal(rgb[:, :, 1], mono)

    def test_bad_encoding(self):
        import numpy as np
        with self.assertRaises(ValueError):
            dashboard.image_to_rgb(self.msg("32FC1", np.zeros((2, 2), np.uint8)))


class VisionStateTest(unittest.TestCase):
    def test_go_to_object_walks_then_stands(self):
        s = make_state()
        at(s, 0.0, 0.0, 0.0)
        s.command(json.dumps({"action": "go_to_object", "color": "red", "shape": "ball",
                              "id": "g"}))
        self.assertEqual(s.command_text, "go_to_object red ball")
        self.assertEqual(s.status(), ds.WALKING)
        self.assertIsNone(s.target_point())
        # A walk meanwhile is busy: no target drawn for it
        s.command(json.dumps({"action": "walk_forward", "distance_m": 0.2, "id": "w"}))
        self.assertIsNone(s.walk)
        s.feedback(json.dumps({"action": "go_to_object", "id": "g", "status": "reached",
                               "color": "red", "shape": "ball", "distance_m": 0.155,
                               "total_distance_m": 0.155}))
        self.assertEqual(s.status(), ds.STANDING)
        self.assertEqual(s.result_text, "reached: red ball, 0.155 m")
        self.assertAlmostEqual(s.total_distance(), 0.155)
        rows = dict(s.panel())
        self.assertEqual(rows["this walk"], "-")

    def test_go_to_object_stopped(self):
        s = make_state()
        at(s, 0.0, 0.0, 0.0)
        s.command(json.dumps({"action": "go_to_object", "color": "blue", "id": "g"}))
        self.assertEqual(s.command_text, "go_to_object blue object")
        s.feedback(json.dumps({"action": "go_to_object", "id": "g", "status": "stopped",
                               "color": "blue", "shape": "box", "distance_m": 0.05}))
        self.assertEqual(s.status(), ds.STOPPED)

    def test_describe_result(self):
        s = make_state()
        s.feedback(json.dumps({"action": "describe", "status": "ok", "objects": [
            {"color": "red", "shape": "ball", "side": "center"},
            {"color": "blue", "shape": "box", "side": "left"}]}))
        self.assertEqual(s.result_text, "sees: red ball (center), blue box (left)")
        s.feedback(json.dumps({"action": "describe", "status": "ok", "objects": []}))
        self.assertEqual(s.result_text, "sees: nothing")


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

    def test_renders_with_camera(self):
        import time
        import types
        import matplotlib.pyplot as plt
        import numpy as np
        s = make_state()
        feed = dashboard.CameraFeed()
        fig = plt.figure(figsize=(20, 8))
        board = dashboard.Dashboard(fig, s, threading.Lock(), feed)
        board.update()  # no frame yet: the placeholder stays
        self.assertIn(dashboard.CAMERA_TOPIC, board.cam_text.get_text())
        bgr = np.zeros((600, 800, 3), np.uint8)
        bgr[:, :, 2] = 200  # red in BGR
        msg = types.SimpleNamespace(encoding="bgr8", height=600, width=800, step=2400,
                                    data=bgr.tobytes())
        feed.set(dashboard.image_to_rgb(msg, dashboard.CAMERA_DECIMATE), time.time())
        board.update()
        shown = board.cam_image.get_array()
        n = dashboard.CAMERA_DECIMATE
        self.assertEqual(shown.shape, (600 // n, 800 // n, 3))
        self.assertEqual(tuple(shown[0, 0]), (200, 0, 0))  # shown as RGB red
        self.assertEqual(board.cam_text.get_text(), "")
        board.update_camera(time.time() + 10)
        self.assertEqual(board.cam_text.get_text(), "no new frames")
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


def ui_msg(state, seq, session="s1", **kw):
    return json.dumps(dict(kw, state=state, session=session, seq=seq))


class VoiceUiStateTest(unittest.TestCase):
    def test_states_and_reply(self):
        s = ds.DashboardState()
        self.assertEqual(s.voice_display(0.0)[0], ds.V_UNKNOWN)
        self.assertTrue(s.ui_state(ui_msg("LISTENING", 1, detail="speak now")))
        self.assertEqual(s.voice_display(0.0), (ds.V_LISTENING, "speak now"))
        s.ui_state(ui_msg("SPEAKING", 4, reply="Theek hai"))
        s.ui_state(ui_msg("IDLE", 5))
        self.assertEqual(s.voice_display(0.0), (ds.V_IDLE, ""))
        self.assertEqual(s.voice_reply, "Theek hai")  # kept after IDLE

    def test_out_of_order_and_bad(self):
        s = ds.DashboardState()
        s.ui_state(ui_msg("THINKING", 3))
        self.assertFalse(s.ui_state(ui_msg("LISTENING", 2)))   # late
        self.assertFalse(s.ui_state("not json"))
        self.assertFalse(s.ui_state(json.dumps({"state": "DANCING"})))
        self.assertFalse(s.ui_state(json.dumps(["IDLE"])))
        self.assertEqual(s.voice_state, ds.V_THINKING)
        self.assertTrue(s.ui_state(ui_msg("LISTENING", 1, session="s2")))  # client restarted

    def test_speak_waiting_then_offline(self):
        s = ds.DashboardState()
        s.talk_request("r1", 100.0)
        self.assertEqual(s.voice_display(101.0)[0], ds.V_WAITING)
        self.assertEqual(s.voice_display(100.0 + ds.UI_ANSWER_TIMEOUT_S + 1)[0], ds.V_OFFLINE)
        s.ui_state(ui_msg("LISTENING", 1, id="r1"))  # the client answers
        self.assertEqual(s.voice_display(200.0)[0], ds.V_LISTENING)

    def test_speak_while_busy(self):
        s = ds.DashboardState()
        s.ui_state(ui_msg("THINKING", 1))
        s.talk_request("r2", 100.0)
        label, detail = s.voice_display(100.0 + ds.UI_ANSWER_TIMEOUT_S + 1)
        self.assertEqual(label, ds.V_THINKING)
        self.assertIn("busy", detail)

    def test_stop_note(self):
        s = ds.DashboardState()
        s.ui_state(ui_msg("ROBOT ACTING", 1, detail="walking"))
        s.stop_pressed("ui-1")
        self.assertEqual(s.voice_display(0.0)[1], "stop sent (ui-1)")
        s.ui_state(ui_msg("SPEAKING", 2, reply="Gurdy ruk gaya"))
        self.assertEqual(s.voice_display(0.0)[1], "")


class VoiceButtonsTest(unittest.TestCase):
    def click(self, fig, button):
        from matplotlib.backend_bases import MouseEvent
        fig.canvas.draw()
        x, y = button.ax.transAxes.transform((0.5, 0.5))
        for name in ("button_press_event", "button_release_event"):
            fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=1))

    def test_buttons_publish(self):
        import matplotlib.pyplot as plt
        s = make_state()
        talks, stops = [], []
        fig = plt.figure(figsize=(20, 9.5))
        board = dashboard.Dashboard(fig, s, threading.Lock(), dashboard.CameraFeed(),
                                    on_speak=talks.append, on_stop=stops.append)
        board.update()
        self.assertEqual(board.voice_text.get_text(), ds.V_UNKNOWN)
        self.click(fig, board.speak_button)
        self.assertEqual(len(talks), 1)
        req = json.loads(talks[0])
        self.assertEqual(req["source"], "dashboard")
        self.assertEqual(s.talk_pending[0], req["id"])
        board.update()
        self.assertEqual(board.voice_text.get_text(), ds.V_WAITING)
        s.ui_state(ui_msg("ROBOT ACTING", 3, id=req["id"], detail="walking"))
        board.update()
        self.assertEqual(board.voice_text.get_text(), "ROBOT ACTING")
        self.click(fig, board.stop_button)
        cmd = json.loads(stops[0])
        self.assertEqual(cmd["action"], "stop")
        self.assertTrue(cmd["id"].startswith("ui-"))
        s.ui_state(ui_msg("SPEAKING", 4, reply="Main ek AI assistant hoon. Gurdy ruk gaya. " * 6))
        board.update()
        self.assertEqual(board.voice_text.get_bbox_patch().get_facecolor()[:3],
                         matplotlib.colors.to_rgb(dashboard.VOICE_COLORS["SPEAKING"]))
        self.assertLessEqual(len(board.reply_text.get_text().split("\n")), 3)
        self.assertTrue(board.reply_text.get_text().endswith("..."))
        fig.canvas.draw()
        plt.close(fig)

    def test_no_buttons_without_callbacks(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 8))
        board = dashboard.Dashboard(fig, make_state(), threading.Lock())
        board.update()
        self.assertIsNone(board.voice_text)
        self.assertFalse(hasattr(board, "speak_button"))
        plt.close(fig)

    def test_button_label_fallback(self):
        self.assertEqual(dashboard.button_label("", "*", "Speak"), "* Speak")
        self.assertEqual(dashboard.button_label("A", "*", "Speak"), "A Speak")


if __name__ == "__main__":
    unittest.main()

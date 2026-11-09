"""Offline tests for web_dashboard (no ROS). The Flask tests are skipped
without Flask, the JPEG test without OpenCV and Pillow. Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import json
import math
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashboard_state as ds  # noqa: E402
import web_dashboard as web  # noqa: E402

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False

try:
    import numpy as np
except ImportError:
    np = None

POSES = {"GROUND_OUT": (-1.45, -2.10), "LIFT_OUT": (-1.50, -2.40)}


def make_state():
    s = ds.DashboardState(pose_angles=POSES, lifted_poses=["LIFT_OUT"])
    for leg in range(1, 7):
        s.joint_command("head_upperlegM%d_joint" % leg, POSES["GROUND_OUT"][0])
        s.joint_command("upperlegM%d_lowerlegM%d_joint" % (leg, leg), POSES["GROUND_OUT"][1])
    return s


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class FakeClock(object):
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestSnapshot(unittest.TestCase):
    def test_empty_state_is_json(self):
        snap = web.snapshot(make_state(), 0.0)
        json.dumps(snap, allow_nan=False)
        self.assertEqual(snap["status"], ds.STANDING)
        self.assertIsNone(snap["x"])
        self.assertIsNone(snap["start"])
        self.assertIsNone(snap["walk"])
        self.assertIsNone(snap["progress"])
        self.assertEqual(snap["lifted"], [])
        self.assertEqual(snap["voice"]["label"], ds.V_UNKNOWN)
        self.assertTrue(snap["camera_stale"])

    def test_walk_pose_and_texts(self):
        s = make_state()
        s.odom(1.0, 0.1, 0.2, 0.05, *yaw_quat(math.radians(30)))
        s.transcript("आगे चलो")
        s.command(json.dumps({"id": "a", "action": "walk_forward", "distance_m": 0.2,
                              "mode": "scripted"}))
        s.odom(2.0, 0.15, 0.2, 0.05, *yaw_quat(math.radians(30)))
        s.joint_command("head_upperlegM3_joint", POSES["LIFT_OUT"][0])
        s.joint_command("upperlegM3_lowerlegM3_joint", POSES["LIFT_OUT"][1])
        snap = web.snapshot(s, 0.0, camera_age=0.5)
        json.dumps(snap, ensure_ascii=False, allow_nan=False)
        self.assertEqual(snap["status"], ds.WALKING)
        self.assertAlmostEqual(snap["yaw_deg"], 30.0, places=1)
        self.assertEqual(snap["start"], [0.1, 0.2])
        self.assertEqual(snap["walk"]["start"], [0.1, 0.2])
        self.assertEqual(snap["walk"]["target"], [0.3, 0.2])
        self.assertAlmostEqual(snap["progress"][0], 0.05)
        self.assertEqual(snap["progress"][1], 0.2)
        self.assertEqual(snap["lifted"], [3])
        self.assertEqual(snap["mode"], "scripted")
        self.assertEqual(snap["heard"], "आगे चलो")
        self.assertIn("walk_forward", snap["command"])
        self.assertFalse(snap["camera_stale"])

    def test_no_pose_table_means_lifted_unknown(self):
        self.assertIsNone(web.snapshot(ds.DashboardState(), 0.0)["lifted"])

    def test_nan_becomes_null(self):
        self.assertIsNone(web._num(float("nan")))
        self.assertIsNone(web._num(float("inf")))
        self.assertEqual(web._num(1.23456, 2), 1.23)


class TestTrail(unittest.TestCase):
    def test_incremental_with_reset_break(self):
        s = make_state()
        s.odom(0.0, 0.0, 0.0, 0.05, *yaw_quat(0))
        s.odom(0.1, 0.01, 0.0, 0.05, *yaw_quat(0))
        points, offset = web.trail_since(s, 0)
        self.assertEqual(points, [[0.0, 0.0], [0.01, 0.0]])
        self.assertEqual(offset, 2)
        s.odom(0.2, 1.0, 0.0, 0.05, *yaw_quat(0))  # teleport: segment break
        points, offset = web.trail_since(s, offset)
        self.assertEqual(points, [None, [1.0, 0.0]])
        self.assertEqual(offset, 4)
        self.assertEqual(web.trail_since(s, offset), ([], 4))


class TestHub(unittest.TestCase):
    def test_speak_and_stop_publish_like_the_tk_dashboard(self):
        talks, commands = [], []
        clock = FakeClock()
        hub = web.Hub(make_state(), publish_talk=talks.append,
                      publish_command=commands.append, clock=clock)
        request_id = hub.speak()
        self.assertEqual(json.loads(talks[0]), {"id": request_id, "source": "dashboard"})
        self.assertEqual(hub.snapshot()["voice"]["label"], ds.V_WAITING)
        clock.t += ds.UI_ANSWER_TIMEOUT_S + 1
        self.assertEqual(hub.snapshot()["voice"]["label"], ds.V_OFFLINE)

        stop_id = hub.stop()
        self.assertTrue(stop_id.startswith("ui-"))
        self.assertEqual(json.loads(commands[0]),
                         {"id": stop_id, "action": "stop", "source": "dashboard"})

    def test_update_returns_new_trail_points(self):
        hub = web.Hub(make_state())
        hub.state.odom(0.0, 0.0, 0.0, 0.05, *yaw_quat(0))
        snap, offset = hub.update(0)
        self.assertEqual((snap["trail_from"], snap["trail"], offset), (0, [[0.0, 0.0]], 1))
        snap, offset = hub.update(offset)
        self.assertEqual((snap["trail_from"], snap["trail"], offset), (1, [], 1))

    def test_camera_age(self):
        clock = FakeClock()
        hub = web.Hub(make_state(), clock=clock)
        self.assertIsNone(hub.snapshot()["camera_age"])
        hub._frame_time = clock.t
        clock.t += web.CAMERA_STALE_S + 0.5
        snap = hub.snapshot()
        self.assertAlmostEqual(snap["camera_age"], web.CAMERA_STALE_S + 0.5)
        self.assertTrue(snap["camera_stale"])

    def test_wait_frame_times_out_and_close_wakes(self):
        hub = web.Hub(make_state())
        self.assertIsNone(hub.wait_frame(0, 0.05))
        self.assertIsNone(hub.latest_frame())
        result = []
        waiter = threading.Thread(target=lambda: result.append(hub.wait_frame(0, 5.0)))
        waiter.start()
        time.sleep(0.05)
        hub.close()
        waiter.join(1.0)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [None])


def _have_jpeg():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        try:
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False


@unittest.skipUnless(np is not None and _have_jpeg(), "needs numpy and OpenCV or Pillow")
class TestCamera(unittest.TestCase):
    def test_frames_encode_once_and_stream(self):
        hub = web.Hub(make_state())
        rgb = np.zeros((60, 80, 3), np.uint8)
        rgb[:, :, 0] = 255
        hub.set_frame(rgb)
        seq, jpg = hub.wait_frame(0, 1.0)
        self.assertEqual(seq, 1)
        self.assertTrue(jpg.startswith(b"\xff\xd8"))
        self.assertIs(hub.latest_frame()[1], jpg)  # cached, not re-encoded

        stream = web.mjpeg_stream(hub, max_fps=1000, keepalive=0.05)
        first = next(stream)
        self.assertTrue(first.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(jpg, first)
        self.assertIn(jpg, next(stream))  # no new frame: keepalive resend
        hub.close()


@unittest.skipUnless(HAVE_FLASK, "needs Flask")
class TestApp(unittest.TestCase):
    def setUp(self):
        self.talks, self.commands = [], []
        self.hub = web.Hub(make_state(), publish_talk=self.talks.append,
                           publish_command=self.commands.append)
        self.client = web.create_app(self.hub).test_client()

    def tearDown(self):
        self.hub.close()

    def test_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"EventSource", r.data)
        r.close()

    def test_state_and_buttons(self):
        self.assertEqual(self.client.get("/api/state").get_json()["status"], ds.STANDING)
        r = self.client.post("/api/speak").get_json()
        self.assertTrue(r["ok"])
        self.assertEqual(json.loads(self.talks[0])["id"], r["id"])
        r = self.client.post("/api/stop").get_json()
        self.assertEqual(json.loads(self.commands[0])["action"], "stop")
        self.assertEqual(self.client.get("/api/speak").status_code, 405)

    def test_events_stream(self):
        self.hub.state.transcript("रुको")
        r = self.client.get("/events", buffered=False)
        self.assertTrue(r.mimetype.startswith("text/event-stream"))
        chunks = r.response
        self.assertEqual(next(chunks), b"retry: 1000\n\n")
        data = next(chunks).decode("utf-8")
        self.assertTrue(data.startswith("data: "))
        snap = json.loads(data[len("data: "):])
        self.assertEqual(snap["heard"], "रुको")
        self.assertEqual(snap["trail_from"], 0)
        r.close()


if __name__ == "__main__":
    unittest.main()

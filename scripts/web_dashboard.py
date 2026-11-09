#!/usr/bin/env python3
"""
Browser version of dashboard.py: the same live demo view, served by a small
Flask server in the container and opened on Windows at http://localhost:8080
(the container publishes port 8080, see README "Web dashboard").

    rosrun my_gurdy_description web_dashboard.py
    rosrun my_gurdy_description web_dashboard.py _port:=8081 _log:=false

One page (web/index.html):
  - top-down map (canvas): start, path trail, Gurdy with its heading, and the
    running walk's start and target point
  - the live camera, /gurdy/camera/annotated as an MJPEG stream (/camera.mjpg)
  - status (STANDING / WALKING / STOPPED / FALLEN), x / y / heading, this
    walk's distance and the total, roll / pitch / height, the lifted leg, mode
  - the last transcript (Devanagari), command and result, and the voice
    client's state (/gurdy/ui/state) and last reply
  - a big Speak button (a talk request on /gurdy/ui/talk, for
    voice_client/main.py --ui) and a red Stop button (a stop straight on
    /gurdy/voice_command), as in dashboard.py

The page gets the state as Server-Sent Events (/events) at 10 Hz. The state
logic is dashboard_state.DashboardState, shared with dashboard.py, and the
path is logged to output/demo/path_<timestamp>.csv as there (_log:=false: no
log). Needs Flask (pip3 install -r requirements-web.txt, in the container).

Endpoints: GET / (page), /events (SSE), /camera.mjpg, /api/state (one JSON
snapshot); POST /api/speak, /api/stop.
"""
import json
import math
import os
import threading
import time

from dashboard_state import (UI_STATE_TOPIC, UI_TALK_TOPIC, VOICE_COMMAND_TOPIC,
                             new_request_id, stop_command_data, talk_request_data)

DEFAULT_HOST, DEFAULT_PORT = "0.0.0.0", 8080
CAMERA_TOPIC = "/gurdy/camera/annotated"
# State updates to the page per second
SSE_PERIOD_S = 0.1
# Most camera frames per second sent to one viewer
CAMERA_MAX_FPS = 15.0
JPEG_QUALITY = 80
# With no new frame, the last one is resent this often (wall-clock s), so a
# closed browser tab is noticed (the write fails) and its thread ends
CAMERA_KEEPALIVE_S = 2.0
# A frame older than this (wall-clock s) is shown as stale, as dashboard.py
CAMERA_STALE_S = 2.0
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")


def _num(value, digits=4):
    """value rounded, or None if it is missing or not finite (JSON has no NaN)."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _point(xy, digits=4):
    return None if xy is None else [_num(xy[0], digits), _num(xy[1], digits)]


def snapshot(state, now, camera_age=None):
    """
    The page's view of a DashboardState at wall time now, as a JSON-able
    dict. Call with the state's lock held. camera_age: wall-clock s since
    the last camera frame, or None if there was none.
    """
    progress = state.walk_progress()
    voice_label, voice_detail = state.voice_display(now)
    walk = state.walk
    return {
        "time": _num(state.time, 3),
        "status": state.status(),
        "x": _num(state.x), "y": _num(state.y), "z": _num(state.z),
        "yaw_deg": _num(math.degrees(state.yaw), 2),
        "roll_deg": _num(math.degrees(state.roll), 2),
        "pitch_deg": _num(math.degrees(state.pitch), 2),
        "start": _point(state.start),
        "walk": None if walk is None else {
            "action": walk["action"], "start": _point(walk["start"]),
            "target": _point(walk["target_xy"])},
        "progress": None if progress is None else [_num(progress[0]), _num(progress[1])],
        "total_m": _num(state.total_distance()),
        # None: lifted legs unknown (no pose table)
        "lifted": state.lifted_legs() if state.pose_angles else None,
        "mode": state.mode_label(),
        "heard": state.transcript_text,
        "command": state.command_text,
        "result": state.result_text,
        "voice": {"label": voice_label, "detail": voice_detail, "reply": state.voice_reply},
        "camera_age": _num(camera_age, 2),
        "camera_stale": camera_age is None or camera_age > CAMERA_STALE_S,
    }


def trail_since(state, offset):
    """
    (points, new offset): the trail points from index offset on, as [x, y]
    or None (a segment break after a reset). Call with the lock held.
    The trail only grows, so a page asks for what it hasn't got yet.
    """
    points = [None if x is None else [round(x, 4), round(y, 4)]
              for x, y in zip(state.trail_x[offset:], state.trail_y[offset:])]
    return points, len(state.trail_x)


def encode_jpeg(rgb, quality=JPEG_QUALITY):
    """HxWx3 RGB uint8 array -> JPEG bytes (OpenCV, else Pillow)."""
    try:
        import cv2
    except ImportError:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return jpg.tobytes()


class Hub(object):
    """
    What the web server and the ROS callbacks share: the DashboardState
    (under lock), the latest camera frame, and the Speak / Stop publishers.
    """

    def __init__(self, state, lock=None, publish_talk=None, publish_command=None,
                 clock=time.time):
        """publish_talk(data) / publish_command(data): send String data on
        /gurdy/ui/talk / /gurdy/voice_command (None: don't)."""
        self.state = state
        self.lock = lock or threading.Lock()
        self.publish_talk, self.publish_command = publish_talk, publish_command
        self.clock = clock
        self.closed = False
        # Camera: the latest RGB frame, its number and wall arrival time;
        # its JPEG is made on demand (only while someone watches), once
        self._frame_cond = threading.Condition()
        self._frame, self._frame_seq, self._frame_time = None, 0, None
        self._jpeg, self._jpeg_seq = None, 0

    # --- state --------------------------------------------------------------

    def snapshot(self):
        now = self.clock()
        with self._frame_cond:
            frame_time = self._frame_time
        with self.lock:
            return snapshot(self.state, now,
                            None if frame_time is None else now - frame_time)

    def update(self, offset):
        """(snapshot with the trail points after offset, new offset)."""
        now = self.clock()
        with self._frame_cond:
            frame_time = self._frame_time
        with self.lock:
            snap = snapshot(self.state, now, None if frame_time is None else now - frame_time)
            points, new_offset = trail_since(self.state, offset)
        snap["trail_from"], snap["trail"] = offset, points
        return snap, new_offset

    # --- buttons ------------------------------------------------------------

    def speak(self):
        """Speak button: ask the voice client to listen. Returns the request id."""
        request_id = new_request_id()
        with self.lock:
            self.state.talk_request(request_id, self.clock())
        if self.publish_talk is not None:
            self.publish_talk(talk_request_data(request_id))
        return request_id

    def stop(self):
        """Stop button: a stop command straight to the robot. Returns its id."""
        request_id = "ui-" + new_request_id()
        with self.lock:
            self.state.stop_pressed(request_id)
        if self.publish_command is not None:
            self.publish_command(stop_command_data(request_id))
        return request_id

    # --- camera -------------------------------------------------------------

    def set_frame(self, rgb):
        with self._frame_cond:
            self._frame = rgb
            self._frame_seq += 1
            self._frame_time = self.clock()
            self._frame_cond.notify_all()

    def wait_frame(self, after_seq, timeout):
        """
        (seq, jpeg) of the first frame newer than after_seq, waiting up to
        timeout s; None if none came (or the hub closed).
        """
        deadline = time.time() + timeout
        with self._frame_cond:
            while self._frame_seq <= after_seq and not self.closed:
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._frame_cond.wait(left)
            if self.closed:
                return None
            return self._frame_seq, self._jpeg_of_latest()

    def latest_frame(self):
        """(seq, jpeg) of the latest frame, or None before the first."""
        with self._frame_cond:
            if self._frame is None:
                return None
            return self._frame_seq, self._jpeg_of_latest()

    def _jpeg_of_latest(self):
        # Called with _frame_cond held; one encode per frame for all viewers
        if self._jpeg_seq != self._frame_seq:
            self._jpeg, self._jpeg_seq = encode_jpeg(self._frame), self._frame_seq
        return self._jpeg

    def close(self):
        with self._frame_cond:
            self.closed = True
            self._frame_cond.notify_all()


def sse_stream(hub, period=SSE_PERIOD_S):
    """The /events body: a state message every period s."""
    yield "retry: 1000\n\n"
    offset = 0
    while not hub.closed:
        snap, offset = hub.update(offset)
        yield "data: %s\n\n" % json.dumps(snap, ensure_ascii=False, allow_nan=False)
        time.sleep(period)


def mjpeg_stream(hub, max_fps=CAMERA_MAX_FPS, keepalive=CAMERA_KEEPALIVE_S):
    """The /camera.mjpg body: multipart JPEG frames as they arrive."""
    def part(jpg):
        return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                % len(jpg)) + jpg + b"\r\n"
    seq, sent = 0, 0.0
    latest = hub.latest_frame()
    if latest is not None:
        seq, jpg = latest
        yield part(jpg)
        sent = time.time()
    while not hub.closed:
        got = hub.wait_frame(seq, keepalive)
        if got is None:
            latest = hub.latest_frame()
            if latest is not None:  # resend: notices a closed tab
                yield part(latest[1])
            continue
        seq, jpg = got
        wait = 1.0 / max_fps - (time.time() - sent)
        if wait > 0:
            time.sleep(wait)
        yield part(jpg)
        sent = time.time()


def create_app(hub, page=PAGE):
    from flask import Flask, Response, jsonify, send_file

    app = Flask(__name__)

    @app.route("/")
    def index():
        response = send_file(page, mimetype="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/events")
    def events():
        return Response(sse_stream(hub), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.route("/camera.mjpg")
    def camera():
        return Response(mjpeg_stream(hub), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.route("/api/state")
    def api_state():
        return jsonify(hub.snapshot())

    @app.route("/api/speak", methods=["POST"])
    def api_speak():
        return jsonify({"ok": True, "id": hub.speak()})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        return jsonify({"ok": True, "id": hub.stop()})

    return app


def main():
    import logging

    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from dashboard import PathLog, image_to_rgb, make_state, subscribe_state

    # disable_signals: Ctrl+C stops the web server, which then shuts ROS down
    rospy.init_node("gurdy_web_dashboard", anonymous=True, disable_signals=True)
    host = rospy.get_param("~host", DEFAULT_HOST)
    port = int(rospy.get_param("~port", DEFAULT_PORT))

    state = make_state()
    lock = threading.Lock()
    log = PathLog() if rospy.get_param("~log", True) else None
    subscribe_state(state, lock, log)
    # Not latched: a talk request must not replay to a restarted voice client
    talk_pub = rospy.Publisher(UI_TALK_TOPIC, String, queue_size=5)
    command_pub = rospy.Publisher(VOICE_COMMAND_TOPIC, String, queue_size=5)

    def publish_talk(data):
        rospy.loginfo("Speak pressed: %s on %s (%d subscriber(s))", data, UI_TALK_TOPIC,
                      talk_pub.get_num_connections())
        talk_pub.publish(String(data=data))

    def publish_command(data):
        rospy.loginfo("Stop pressed: %s on %s", data, VOICE_COMMAND_TOPIC)
        command_pub.publish(String(data=data))

    hub = Hub(state, lock, publish_talk, publish_command)

    def on_image(msg):
        try:
            rgb = image_to_rgb(msg)
        except ValueError as e:
            rospy.logwarn_throttle(10.0, "Camera: %s" % e)
            return
        hub.set_frame(rgb)
    rospy.Subscriber(CAMERA_TOPIC, Image, on_image, queue_size=1, buff_size=2 ** 24)

    app = create_app(hub)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # no per-request lines
    rospy.loginfo("Web dashboard on http://%s:%d (from Windows: http://localhost:%d); "
                  "voice state from %s", host, port, port, UI_STATE_TOPIC)
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        hub.close()
        if log is not None:
            with lock:
                log.close()
            print("Path saved to %s" % log.path)
        rospy.signal_shutdown("web dashboard stopped")


if __name__ == "__main__":
    main()

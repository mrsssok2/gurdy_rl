#!/usr/bin/env python3
"""
Live demo dashboard for Gurdy (matplotlib/Tk window, shown through WSLg like
the training plots).

Left: top-down map in metres with a grid, the start point, the path trail,
the current position with a heading arrow (from /gurdy/odom), and during a
voice walk the walk's start and target point.
Middle: the live camera view, /gurdy/camera/annotated from vision_node.py
(detections boxed and labelled); ~camera:=false hides it.
Right, refreshed at ~10 Hz: x, y, heading, this walk's distance (done /
target) and the total, roll / pitch / height, the lifted leg (from the
position controller commands), status (STANDING / WALKING / STOPPED /
FALLEN), mode (RL / scripted), and the last transcript, command and result
from /gurdy/voice_transcript, /gurdy/voice_command and /gurdy/voice_feedback.
Bottom: a big Speak button (publishes a talk request on /gurdy/ui/talk for
voice_client/main.py --ui, which has the mic on Windows), a red Stop button
(publishes a stop straight on /gurdy/voice_command, no voice needed), the
voice client's state from /gurdy/ui/state (IDLE / LISTENING / THINKING /
ROBOT ACTING / SPEAKING) and its last spoken reply.

The path is logged at 10 Hz to output/demo/path_<timestamp>.csv (in the
package; ~outdir overrides). Plot a saved log with plot_path.py.

    rosrun my_gurdy_description dashboard.py
    rosrun my_gurdy_description dashboard.py _camera:=false   # no camera panel
    rosrun my_gurdy_description dashboard.py _buttons:=false  # no Speak / Stop
"""
import csv
import os
import threading
import time

from dashboard_state import (CSV_FIELDS, FALLEN, STANDING, STOPPED, UI_STATE_TOPIC,
                             UI_TALK_TOPIC, V_ACTING, V_IDLE, V_LISTENING, V_OFFLINE,
                             V_SPEAKING, V_THINKING, V_UNKNOWN, V_WAITING,
                             VOICE_COMMAND_TOPIC, WALKING, DashboardState,
                             new_request_id, stop_command_data, talk_request_data)

STATUS_COLORS = {STANDING: "#2e7d32", WALKING: "#1565c0", STOPPED: "#ef6c00", FALLEN: "#c62828"}
VOICE_COLORS = {V_IDLE: "#546e7a", V_LISTENING: "#c62828", V_THINKING: "#f9a825",
                V_ACTING: "#1565c0", V_SPEAKING: "#2e7d32", V_WAITING: "#8e24aa",
                V_OFFLINE: "#9e9e9e", V_UNKNOWN: "#9e9e9e"}
SPEAK_COLOR, SPEAK_HOVER = "#1565c0", "#1e88e5"
STOP_COLOR, STOP_HOVER = "#c62828", "#e53935"
REPLY_WRAP = 70       # characters per line for the last reply
# Height (figure fraction) of the bottom strip with the buttons
BUTTON_STRIP = 0.17
REFRESH_MS = 100
MIN_SPAN_M = 0.6      # smallest map extent
MAP_MARGIN_M = 0.15
ARROW_M = 0.06        # heading arrow length
TEXT_WRAP = 34        # characters per line for transcript / command / result
# Fonts that cover Devanagari, for the transcript, best first. The default
# DejaVu fonts have no Devanagari and draw boxes. Debian/Ubuntu:
# apt-get install fonts-noto-core (or fonts-lohit-deva)
DEVANAGARI_FONTS = ("Noto Sans Devanagari", "Lohit Devanagari", "Nirmala UI", "Mangal",
                    "Kohinoor Devanagari", "Gargi", "FreeSans")
DEVANAGARI_PROBE = "ग"  # GA
CAMERA_TOPIC = "/gurdy/camera/annotated"
# Show every CAMERA_DECIMATE-th pixel (2: 800x600 -> 400x300, cheaper
# redraws but the detection labels get hard to read)
CAMERA_DECIMATE = 1
# A frame older than this (wall-clock s) is marked stale
CAMERA_STALE_S = 2.0


def image_to_rgb(msg, decimate=1):
    """
    sensor_msgs/Image (rgb8, bgr8 or mono8) -> HxWx3 uint8 RGB numpy array,
    without cv_bridge. Rows may be padded (step > width * channels).
    """
    import numpy as np
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError("unsupported image encoding %r" % msg.encoding)
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1]
    elif channels == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img[::decimate, ::decimate])


class CameraFeed(object):
    """Latest camera frame (RGB array) and its arrival time, set from a callback."""

    def __init__(self):
        self.frame = None
        self.received = None

    def set(self, rgb, now):
        self.frame, self.received = rgb, now


def find_devanagari_font():
    """Path of an installed font with Devanagari glyphs, or None."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    def covers(path):
        try:
            return ord(DEVANAGARI_PROBE) in FT2Font(path).get_charmap()
        except Exception:
            return False

    fonts = font_manager.fontManager.ttflist
    for name in DEVANAGARI_FONTS:
        for font in fonts:
            if font.name == name and covers(font.fname):
                return font.fname
    for font in fonts:  # anything else that has it
        if "devanagari" in font.name.lower() and covers(font.fname):
            return font.fname
    return None


def has_glyphs(text):
    """True if matplotlib's default font draws every character of text."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    try:
        charmap = FT2Font(font_manager.findfont(font_manager.FontProperties())).get_charmap()
    except Exception:
        return False
    return all(ord(c) in charmap for c in text if not c.isspace())


def button_label(icon, fallback, text):
    """'icon text' if the font has the icon, else 'fallback text' (no boxes)."""
    return "%s %s" % (icon if has_glyphs(icon) else fallback, text)


def wrap(text, width=TEXT_WRAP, lines=2):
    """Wrap text to at most `lines` lines of `width`, with an ellipsis if cut."""
    import textwrap
    out = textwrap.wrap(text, width) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][:width - 3] + "..."
    return "\n".join(out)


class Dashboard(object):
    """The figure; update() redraws it from a DashboardState under lock."""

    def __init__(self, fig, state, lock, camera=None, on_speak=None, on_stop=None):
        """
        camera: a CameraFeed for the middle camera panel, or None for none.
        on_speak(data) / on_stop(data): publish the talk request / the stop
        command (String data). With both None there are no buttons.
        """
        self.fig, self.state, self.lock = fig, state, lock
        self.camera = camera
        self.on_speak, self.on_stop = on_speak, on_stop
        buttons = on_speak is not None or on_stop is not None
        bottom = 0.07 + (BUTTON_STRIP if buttons else 0.0)
        if camera is None:
            grid = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.08,
                                    left=0.06, right=0.98, top=0.93, bottom=bottom)
            map_cell, panel_cell = grid[0, 0], grid[0, 1]
        else:
            grid = fig.add_gridspec(1, 3, width_ratios=[3, 3, 2.4], wspace=0.1,
                                    left=0.04, right=0.99, top=0.93, bottom=bottom)
            map_cell, panel_cell = grid[0, 0], grid[0, 2]
            cam = self.cam_ax = fig.add_subplot(grid[0, 1])
            cam.set_title("Camera - what Gurdy sees", fontsize=18)
            cam.set_xticks([])
            cam.set_yticks([])
            self.cam_image = None
            self.cam_text = cam.text(0.5, 0.5, "waiting for\n" + CAMERA_TOPIC, ha="center",
                                     va="center", fontsize=14, color="#555555",
                                     transform=cam.transAxes)
        ax = self.ax = fig.add_subplot(map_cell)
        ax.set_title("Gurdy - top view", fontsize=18)
        ax.set_xlabel("x (m)", fontsize=14)
        ax.set_ylabel("y (m)", fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="both", alpha=0.4)
        ax.tick_params(labelsize=12)
        self.trail, = ax.plot([], [], "-", color="#1565c0", lw=2, label="path")
        self.start, = ax.plot([], [], "o", color="#2e7d32", ms=12, label="start")
        self.walk_start, = ax.plot([], [], "s", color="#6a1b9a", ms=8, mfc="none", mew=2)
        self.target_line, = ax.plot([], [], "--", color="#6a1b9a", lw=1.5)
        self.target, = ax.plot([], [], "X", color="#6a1b9a", ms=16, label="target")
        self.robot, = ax.plot([], [], "o", color="#c62828", ms=14, label="Gurdy")
        self.heading = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1,
                                 color="#c62828", width=0.006, zorder=5)
        ax.legend(loc="upper left", fontsize=12)
        self.limits = None

        panel = self.panel = fig.add_subplot(panel_cell)
        panel.axis("off")
        self.status_text = panel.text(0.5, 0.97, STANDING, ha="center", va="top", fontsize=30,
                                      fontweight="bold", color="white",
                                      bbox=dict(boxstyle="round,pad=0.4", fc=STATUS_COLORS[STANDING]))
        n = len(state.panel())
        self.labels, self.values = [], []
        top, bottom = 0.84, 0.0
        step = (top - bottom) / n
        for i in range(n):
            y = top - i * step
            self.labels.append(panel.text(0.0, y, "", ha="left", va="top", fontsize=15,
                                          color="#555555"))
            self.values.append(panel.text(0.36, y, "", ha="left", va="top", fontsize=17,
                                          family="monospace"))
        # The transcript is usually Devanagari
        self.devanagari_font = find_devanagari_font()
        if self.devanagari_font:
            from matplotlib.font_manager import FontProperties
            heard = [label for label, _ in state.panel()].index("heard")
            self.values[heard].set_fontproperties(
                FontProperties(fname=self.devanagari_font, size=17))
        self.voice_text = None
        if buttons:
            self._add_voice_strip()

    def _add_voice_strip(self):
        """Bottom strip: Speak and Stop buttons, voice state, last reply."""
        from matplotlib.widgets import Button
        fig = self.fig
        h = BUTTON_STRIP - 0.03
        speak_ax = fig.add_axes([0.02, 0.02, 0.15, h])
        stop_ax = fig.add_axes([0.18, 0.02, 0.11, h])
        self.speak_button = Button(speak_ax, button_label("\U0001F3A4", "\u25CF", "Speak"),
                                   color=SPEAK_COLOR, hovercolor=SPEAK_HOVER)
        self.stop_button = Button(stop_ax, button_label("\u23F9", "\u25A0", "Stop"),
                                  color=STOP_COLOR, hovercolor=STOP_HOVER)
        for button in (self.speak_button, self.stop_button):
            button.label.set_fontsize(26)
            button.label.set_fontweight("bold")
            button.label.set_color("white")
        self.speak_button.on_clicked(self.speak)
        self.stop_button.on_clicked(self.stop)
        y = 0.02 + h
        self.voice_text = fig.text(0.31, y - 0.012, V_UNKNOWN, ha="left", va="top", fontsize=24,
                                   fontweight="bold", color="white",
                                   bbox=dict(boxstyle="round,pad=0.35",
                                             fc=VOICE_COLORS[V_UNKNOWN]))
        self.voice_detail = fig.text(0.31, 0.03, "", ha="left", va="bottom", fontsize=13,
                                     color="#555555")
        fig.text(0.52, y - 0.005, "last reply", ha="left", va="top", fontsize=14,
                 color="#555555")
        self.reply_text = fig.text(0.52, y - 0.045, "-", ha="left", va="top", fontsize=17)

    def speak(self, _event=None):
        """Speak button: ask the voice client to listen."""
        request_id = new_request_id()
        with self.lock:
            self.state.talk_request(request_id, time.time())
        if self.on_speak is not None:
            self.on_speak(talk_request_data(request_id))

    def stop(self, _event=None):
        """Stop button: a stop command straight to the robot."""
        request_id = "ui-" + new_request_id()
        with self.lock:
            self.state.stop_pressed(request_id)
        if self.on_stop is not None:
            self.on_stop(stop_command_data(request_id))

    def update_voice(self, now):
        with self.lock:
            label, detail = self.state.voice_display(now)
            reply = self.state.voice_reply
        self.voice_text.set_text(label)
        self.voice_text.get_bbox_patch().set_facecolor(VOICE_COLORS.get(label, "#9e9e9e"))
        self.voice_detail.set_text(wrap(detail, 60, 1) if detail else "")
        self.reply_text.set_text(wrap(reply, REPLY_WRAP, 3) if reply else "-")

    def _fit(self, xs, ys):
        """Grow the map limits to cover xs, ys (never shrink, so it doesn't jitter)."""
        if not xs:
            return
        lo_x, hi_x = min(xs) - MAP_MARGIN_M, max(xs) + MAP_MARGIN_M
        lo_y, hi_y = min(ys) - MAP_MARGIN_M, max(ys) + MAP_MARGIN_M
        if self.limits is not None:
            lo_x, hi_x = min(lo_x, self.limits[0]), max(hi_x, self.limits[1])
            lo_y, hi_y = min(lo_y, self.limits[2]), max(hi_y, self.limits[3])
        for lo, hi, axis in ((lo_x, hi_x, "x"), (lo_y, hi_y, "y")):
            if hi - lo < MIN_SPAN_M:
                mid = (lo + hi) / 2.0
                lo, hi = mid - MIN_SPAN_M / 2.0, mid + MIN_SPAN_M / 2.0
            if axis == "x":
                lo_x, hi_x = lo, hi
            else:
                lo_y, hi_y = lo, hi
        limits = (lo_x, hi_x, lo_y, hi_y)
        if limits != self.limits:
            self.limits = limits
            self.ax.set_xlim(lo_x, hi_x)
            self.ax.set_ylim(lo_y, hi_y)

    def update_camera(self, now):
        """Show the latest camera frame; grey out the title if it is stale."""
        with self.lock:
            frame, received = self.camera.frame, self.camera.received
        if frame is None:
            return
        if self.cam_image is None:
            self.cam_image = self.cam_ax.imshow(frame, interpolation="bilinear")
            self.cam_text.set_text("")
        elif frame.shape == self.cam_image.get_array().shape:
            self.cam_image.set_data(frame)
        else:
            self.cam_image.set_data(frame)
            self.cam_image.set_extent((-0.5, frame.shape[1] - 0.5, frame.shape[0] - 0.5, -0.5))
        stale = received is None or now - received > CAMERA_STALE_S
        self.cam_text.set_text("no new frames" if stale else "")
        self.cam_text.set_color("#c62828")

    def update(self, _frame=None):
        import math
        if self.camera is not None:
            self.update_camera(time.time())
        if self.voice_text is not None:
            self.update_voice(time.time())
        with self.lock:
            s = self.state
            trail_x, trail_y = list(s.trail_x), list(s.trail_y)
            x, y, yaw = s.x, s.y, s.yaw
            start = s.start
            walk = dict(s.walk) if s.walk else None
            status = s.status()
            rows = s.panel()

        nan = float("nan")
        self.trail.set_data([nan if v is None else v for v in trail_x],
                            [nan if v is None else v for v in trail_y])
        xs = [v for v in trail_x if v is not None]
        ys = [v for v in trail_y if v is not None]
        if start:
            self.start.set_data([start[0]], [start[1]])
        if x is not None:
            self.robot.set_data([x], [y])
            self.heading.set_offsets([[x, y]])
            self.heading.set_UVC([ARROW_M * math.cos(yaw)], [ARROW_M * math.sin(yaw)])
            xs.append(x)
            ys.append(y)
        if walk:
            (sx, sy), (tx, ty) = walk["start"], walk["target_xy"]
            self.walk_start.set_data([sx], [sy])
            self.target.set_data([tx], [ty])
            self.target_line.set_data([sx, tx], [sy, ty])
            xs.append(tx)
            ys.append(ty)
        else:
            for artist in (self.walk_start, self.target, self.target_line):
                artist.set_data([], [])
        self._fit(xs, ys)

        self.status_text.set_text(status)
        self.status_text.get_bbox_patch().set_facecolor(STATUS_COLORS[status])
        for (label, value), label_text, value_text in zip(rows, self.labels, self.values):
            label_text.set_text(label)
            value_text.set_text(wrap(value) if label in ("heard", "command", "result") else value)
        return []


def default_log_path(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return os.path.join(outdir, "path_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))


def make_state():
    """A DashboardState with Gurdy's poses and forward direction (from env.py)."""
    import env as gurdy_env
    return DashboardState(
        pose_angles=gurdy_env.POSE_ANGLES,
        lifted_poses=[p for p, (lifted, _) in gurdy_env.POSE_FLAGS.items() if lifted],
        forward_sign=gurdy_env.MyGurdyWalkEnv.FORWARD_SIGN)


class PathLog(object):
    """The 10 Hz CSV path log in ~outdir (default output/demo in the package)."""

    def __init__(self):
        import rospkg
        import rospy
        outdir = rospy.get_param("~outdir", "output/demo")
        if not os.path.isabs(outdir):
            outdir = os.path.join(rospkg.RosPack().get_path("my_gurdy_description"), outdir)
        self.path = default_log_path(outdir)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        rospy.loginfo("Logging the path to %s", self.path)

    def write(self, state):
        """Write a row if one is due. Call with the state's lock held."""
        if state.should_log() and not self.file.closed:
            self.writer.writerow(state.csv_row())
            self.file.flush()

    def close(self):
        self.file.close()


def subscribe_state(state, lock, log=None):
    """
    Feed state from /gurdy/odom, the 12 position controller commands and the
    voice topics (transcript, command, feedback, /gurdy/ui/state). log: a
    PathLog to write at 10 Hz, or None. Needs rospy.init_node first.
    """
    import rospy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64, String

    def on_odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with lock:
            state.odom(msg.header.stamp.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            if log is not None:
                log.write(state)

    def joint_callback(joint):
        def on_command(msg):
            with lock:
                state.joint_command(joint, msg.data)
        return on_command

    def text_callback(method):
        def on_text(msg):
            with lock:
                method(msg.data)
        return on_text

    rospy.Subscriber("/gurdy/odom", Odometry, on_odom)
    joints = (["head_upperlegM%d_joint" % i for i in range(1, 7)] +
              ["upperlegM%d_lowerlegM%d_joint" % (i, i) for i in range(1, 7)])
    for joint in joints:
        rospy.Subscriber("/gurdy/%s_position_controller/command" % joint, Float64,
                         joint_callback(joint))
    rospy.Subscriber("/gurdy/voice_transcript", String, text_callback(state.transcript))
    rospy.Subscriber("/gurdy/voice_command", String, text_callback(state.command))
    rospy.Subscriber("/gurdy/voice_feedback", String, text_callback(state.feedback))
    rospy.Subscriber(UI_STATE_TOPIC, String, text_callback(state.ui_state))


def main():
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    rospy.init_node("gurdy_dashboard", anonymous=True)
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    state = make_state()
    lock = threading.Lock()
    log = PathLog()
    subscribe_state(state, lock, log)
    # Not latched: a talk request must not replay to a restarted voice client
    talk_pub = rospy.Publisher(UI_TALK_TOPIC, String, queue_size=5)
    command_pub = rospy.Publisher(VOICE_COMMAND_TOPIC, String, queue_size=5)

    def on_speak(data):
        rospy.loginfo("Speak pressed: %s on %s (%d subscriber(s))", data, UI_TALK_TOPIC,
                      talk_pub.get_num_connections())
        talk_pub.publish(String(data=data))

    def on_stop(data):
        rospy.loginfo("Stop pressed: %s on %s", data, VOICE_COMMAND_TOPIC)
        command_pub.publish(String(data=data))

    camera = CameraFeed() if rospy.get_param("~camera", True) else None
    if camera is not None:
        def on_image(msg):
            try:
                rgb = image_to_rgb(msg, CAMERA_DECIMATE)
            except ValueError as e:
                rospy.logwarn_throttle(10.0, "Camera panel: %s" % e)
                return
            with lock:
                camera.set(rgb, time.time())
        rospy.Subscriber(CAMERA_TOPIC, Image, on_image, queue_size=1, buff_size=2 ** 24)

    buttons = rospy.get_param("~buttons", True)
    height = 9.5 if buttons else 8
    fig = plt.figure("Gurdy dashboard", figsize=(15 if camera is None else 20, height))
    dashboard = Dashboard(fig, state, lock, camera, on_speak=on_speak if buttons else None,
                          on_stop=on_stop if buttons else None)
    if dashboard.devanagari_font:
        rospy.loginfo("Transcript font: %s", dashboard.devanagari_font)
    else:
        rospy.logwarn("No Devanagari font found; the transcript will show as boxes. "
                      "Install one, e.g. apt-get install fonts-noto-core")
    animation = FuncAnimation(fig, dashboard.update, interval=REFRESH_MS, cache_frame_data=False)
    try:
        plt.show()  # returns when the window is closed
    finally:
        del animation
        with lock:
            log.close()
        print("Path saved to %s" % log.path)
        rospy.signal_shutdown("dashboard closed")


if __name__ == "__main__":
    main()

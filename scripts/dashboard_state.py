#!/usr/bin/env python3
"""
State behind dashboard.py, without ROS or matplotlib so it can be tested
offline (scripts/tests/test_dashboard.py).

DashboardState is fed the messages dashboard.py receives:
    odom()           /gurdy/odom pose
    joint_command()  the 12 position controller commands, for the lifted leg
    transcript()     /gurdy/voice_transcript
    command()        /gurdy/voice_command (JSON)
    feedback()       /gurdy/voice_feedback (JSON)
and works out the status, the current walk's progress and target point,
the path trail and the CSV log rows.
"""
import json
import math

STANDING, WALKING, STOPPED, FALLEN = "STANDING", "WALKING", "STOPPED", "FALLEN"

# Fallen if roll or pitch exceeds this, as env.MAX_TILT_DEG
FALL_TILT_DEG = 60.0
# A jump this large between two odom messages is a reset (teleport): the
# trail starts a new segment instead of drawing a line across the map
RESET_JUMP_M = 0.10
# Trail points closer than this to the last one are skipped
TRAIL_MIN_STEP_M = 0.002
# CSV log period, s of ROS time
LOG_PERIOD_S = 0.1

CSV_FIELDS = ["time_s", "x", "y", "z", "yaw_deg", "roll_deg", "pitch_deg", "status",
              "mode", "walk_m", "target_m", "target_x", "target_y", "lifted_legs"]

MODE_LABELS = {"rl": "RL", "scripted": "scripted", "rl+scripted": "RL -> scripted"}

# Walk actions and their direction in the world, as a unit (x, y) vector;
# forward is multiplied by forward_sign. Sideways is world +y / -y, as the
# node measures it (env.get_lateral_position).
WALK_DIRECTIONS = {"walk_forward": (1.0, 0.0), "walk_left": (0.0, 1.0),
                   "walk_right": (0.0, -1.0)}


def quaternion_to_rpy(qx, qy, qz, qw):
    """(roll, pitch, yaw) in radians, ZYX convention (as env.euler_from_quaternion)."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sin_pitch = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def leg_joint(joint_name):
    """(leg, "hip" | "knee") for a controller joint name, or None."""
    if joint_name.startswith("head_upperlegM"):
        return int(joint_name[len("head_upperlegM")]), "hip"
    if joint_name.startswith("upperlegM"):
        return int(joint_name[len("upperlegM")]), "knee"
    return None


def nearest_pose(hip, knee, pose_angles):
    """Name of the pose in {pose: (hip, knee)} closest to (hip, knee)."""
    return min(pose_angles, key=lambda p: (pose_angles[p][0] - hip) ** 2 +
               (pose_angles[p][1] - knee) ** 2)


def describe_command(cmd):
    action = cmd.get("action")
    if action == "walk_forward":
        return "walk_forward %.2f m (%s)" % (float(cmd.get("distance_m", 0.0)),
                                            MODE_LABELS.get(cmd.get("mode"), cmd.get("mode", "?")))
    if action in WALK_DIRECTIONS:
        return "%s %.2f m (scripted)" % (action, float(cmd.get("distance_m", 0.0)))
    return str(action)


def describe_feedback(fb):
    action, status = fb.get("action"), fb.get("status")
    if action in WALK_DIRECTIONS and "distance_m" in fb and status not in ("busy", "error"):
        text = "%s: %.3f m of %.3f m, %s" % (status, fb["distance_m"], fb.get("requested_m", 0.0),
                                             MODE_LABELS.get(fb.get("mode_used"), fb.get("mode_used")))
        if fb.get("fallback"):
            text += " (fallback: %s)" % fb.get("fallback_reason")
        return text
    if action == "status" and "total_distance_m" in fb:
        return "status: %s, total %.3f m" % (status, fb["total_distance_m"])
    return "%s: %s%s" % (action, status, (" - " + fb["message"]) if fb.get("message") else "")


class DashboardState(object):
    def __init__(self, pose_angles=None, lifted_poses=(), forward_sign=1):
        self.pose_angles = pose_angles or {}
        self.lifted_poses = set(lifted_poses)
        self.forward_sign = forward_sign

        self.time = None
        self.x = self.y = self.z = None
        self.roll = self.pitch = self.yaw = 0.0
        self.start = None              # first odom position (x, y)
        self.trail_x, self.trail_y = [], []  # None separates segments

        self.joint_targets = {}        # (leg, "hip"|"knee") -> last command
        self.transcript_text = ""
        self.command_text = ""
        self.result_text = ""
        self.mode = None               # last requested / used walk mode

        self.walk = None               # the running walk (see command())
        self.last_walk = None          # (status, distance_m, requested_m) of the last one
        self.total_m = 0.0
        self._last_log = None

    # --- inputs -----------------------------------------------------------

    def odom(self, t, x, y, z, qx, qy, qz, qw):
        jumped = (self.x is not None and
                  math.hypot(x - self.x, y - self.y) > RESET_JUMP_M)
        self.time = t
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = quaternion_to_rpy(qx, qy, qz, qw)
        if self.start is None:
            self.start = (x, y)
        if jumped and self.trail_x:
            self.trail_x.append(None)
            self.trail_y.append(None)
        if (not self.trail_x or self.trail_x[-1] is None or
                math.hypot(x - self.trail_x[-1], y - self.trail_y[-1]) >= TRAIL_MIN_STEP_M):
            self.trail_x.append(x)
            self.trail_y.append(y)

    def joint_command(self, joint_name, value):
        key = leg_joint(joint_name)
        if key is not None:
            self.joint_targets[key] = value

    def transcript(self, text):
        self.transcript_text = text

    def command(self, data):
        try:
            cmd = json.loads(data)
        except ValueError:
            self.command_text = "(not JSON) " + data[:60]
            return
        if not isinstance(cmd, dict):
            return
        self.command_text = describe_command(cmd)
        # A walk while one runs is answered "busy" by the node; keep the first
        action = cmd.get("action")
        if action in WALK_DIRECTIONS and self.walk is None and self.x is not None:
            target = float(cmd.get("distance_m", 0.2))
            dx, dy = WALK_DIRECTIONS[action]
            if action == "walk_forward":
                dx *= self.forward_sign
                mode = cmd.get("mode", "scripted")
            else:
                mode = "scripted"  # sideways walks have no RL
            self.walk = {
                "id": cmd.get("id"), "action": action, "target_m": target, "mode": mode,
                "start": (self.x, self.y), "direction": (dx, dy),
                "target_xy": (self.x + dx * target, self.y + dy * target),
            }
            self.mode = self.walk["mode"]

    def feedback(self, data):
        try:
            fb = json.loads(data)
        except ValueError:
            self.result_text = "(not JSON) " + data[:60]
            return
        if not isinstance(fb, dict):
            return
        self.result_text = describe_feedback(fb)
        if "total_distance_m" in fb:
            self.total_m = fb["total_distance_m"]
        if fb.get("action") not in WALK_DIRECTIONS or self.walk is None:
            return
        if fb.get("id") != self.walk["id"]:
            return  # e.g. "busy" for a second walk
        if fb.get("status") in ("error",):
            self.walk = None
            return
        self.last_walk = (fb.get("status"), fb.get("distance_m", 0.0), fb.get("requested_m"))
        if fb.get("mode_used"):
            self.mode = fb["mode_used"]
        self.walk = None

    # --- derived ----------------------------------------------------------

    def forward(self):
        return self.forward_sign * (self.x or 0.0)

    def walk_done(self):
        """Distance the running walk has made along its direction."""
        (sx, sy), (dx, dy) = self.walk["start"], self.walk["direction"]
        return ((self.x or 0.0) - sx) * dx + ((self.y or 0.0) - sy) * dy

    def walk_progress(self):
        """(done_m, target_m) of the running walk, else of the last one, else None."""
        if self.walk is not None:
            return self.walk_done(), self.walk["target_m"]
        if self.last_walk is not None:
            return self.last_walk[1], self.last_walk[2]
        return None

    def target_point(self):
        return self.walk["target_xy"] if self.walk is not None else None

    def tilted(self):
        return max(abs(math.degrees(self.roll)), abs(math.degrees(self.pitch))) > FALL_TILT_DEG

    def status(self):
        if self.tilted():
            return FALLEN
        if self.walk is not None:
            return WALKING
        if self.last_walk is not None:
            if self.last_walk[0] == "fallen":
                return FALLEN
            if self.last_walk[0] == "stopped":
                return STOPPED
        return STANDING

    def total_distance(self):
        """Total walked, including the running walk so far."""
        total = self.total_m
        if self.walk is not None:
            total += max(0.0, self.walk_done())
        return total

    def lifted_legs(self):
        """Legs whose commanded hip/knee are nearest a lifted foot pose."""
        if not self.pose_angles:
            return []
        lifted = []
        for leg in range(1, 7):
            hip, knee = self.joint_targets.get((leg, "hip")), self.joint_targets.get((leg, "knee"))
            if hip is None or knee is None:
                continue
            if nearest_pose(hip, knee, self.pose_angles) in self.lifted_poses:
                lifted.append(leg)
        return lifted

    def mode_label(self):
        return MODE_LABELS.get(self.mode, "-") if self.mode else "-"

    def panel(self):
        """(label, value) rows for the right-hand panel."""
        def m(v):
            return "-" if v is None else "%+.3f m" % v
        progress = self.walk_progress()
        lifted = self.lifted_legs()
        rows = [
            ("x / y", "%s  %s" % (m(self.x), m(self.y))),
            ("heading", "%+.1f deg" % math.degrees(self.yaw)),
            ("this walk", "-" if progress is None or progress[1] is None else
             "%.3f / %.3f m" % (progress[0], progress[1])),
            ("total", "%.3f m" % self.total_distance()),
            ("roll / pitch", "%+.1f / %+.1f deg" % (math.degrees(self.roll), math.degrees(self.pitch))),
            ("height", m(self.z).replace("+", "")),
            ("lifted leg", ", ".join(str(leg) for leg in lifted) if lifted else
             ("none" if self.pose_angles else "?")),
            ("mode", self.mode_label()),
            ("heard", self.transcript_text or "-"),
            ("command", self.command_text or "-"),
            ("result", self.result_text or "-"),
        ]
        return rows

    # --- CSV log ------------------------------------------------------------

    def should_log(self):
        """True if a row is due (every LOG_PERIOD_S of odom time)."""
        if self.time is None:
            return False
        if self._last_log is None or self.time - self._last_log >= LOG_PERIOD_S - 1e-9 \
                or self.time < self._last_log:  # sim time went back (reset_world)
            self._last_log = self.time
            return True
        return False

    def csv_row(self):
        progress = self.walk_progress() if self.walk is not None else None
        target = self.target_point()

        def f(v, digits=4):
            return "" if v is None else round(v, digits)
        return {
            "time_s": f(self.time, 3), "x": f(self.x), "y": f(self.y), "z": f(self.z),
            "yaw_deg": f(math.degrees(self.yaw), 2), "roll_deg": f(math.degrees(self.roll), 2),
            "pitch_deg": f(math.degrees(self.pitch), 2), "status": self.status(),
            "mode": self.mode or "", "walk_m": f(progress[0]) if progress else "",
            "target_m": f(progress[1]) if progress else "",
            "target_x": f(target[0]) if target else "", "target_y": f(target[1]) if target else "",
            "lifted_legs": " ".join(str(leg) for leg in self.lifted_legs()),
        }

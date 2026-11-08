"""The dashboard's Speak button <-> the voice client (main.py --ui).

The dashboard (scripts/dashboard.py, in the container) publishes a talk
request on /gurdy/ui/talk when Speak is pressed. The client keeps one
`rostopic echo` on that topic open (robot_link.TopicStream, restarted if it
dies), records on each request, and reports what it is doing on
/gurdy/ui/state. Both are std_msgs/String with JSON data:

    talk:  {"id": "<request id>", "source": "dashboard"}
    state: {"state": "LISTENING", "session": "<this client run>", "seq": 3,
            "id": "<request id>", "reply": "...", "heard": "...", "detail": "..."}

Each state goes out in its own `docker exec ... rostopic pub`, so two can
arrive out of order; the dashboard keeps only the highest seq of a session.
The topic names and states match scripts/dashboard_state.py (checked by
tests/test_ui.py).
"""
import json
import subprocess
import uuid

from robot_link import parse_echo_data

UI_TALK_TOPIC = "/gurdy/ui/talk"
UI_STATE_TOPIC = "/gurdy/ui/state"

IDLE, LISTENING, THINKING, ACTING, SPEAKING = (
    "IDLE", "LISTENING", "THINKING", "ROBOT ACTING", "SPEAKING")
STATES = (IDLE, LISTENING, THINKING, ACTING, SPEAKING)


def parse_talk(block):
    """
    A talk request dict from one `rostopic echo` message, or None. JSON
    objects are used as they are; any other non-empty text (e.g. a hand-typed
    `rostopic pub /gurdy/ui/talk std_msgs/String "data: go"`) is a request
    without an id.
    """
    data = parse_echo_data(block)
    if data is None or not data.strip():
        return None
    try:
        msg = json.loads(data)
    except ValueError:
        msg = None
    if isinstance(msg, dict):
        return msg
    return {"id": None, "text": data.strip()}


class UiLink(object):
    """Publishes the client's state on /gurdy/ui/state through a RobotLink."""

    def __init__(self, robot, log=print):
        self.robot = robot
        self.log = log
        self.session = uuid.uuid4().hex[:6]
        self.seq = 0
        self.last = None      # the last state message sent
        self._procs = []

    def message(self, state, request_id=None, reply=None, heard=None, detail=None):
        """The next state message (a dict), with seq advanced."""
        if state not in STATES:
            raise ValueError("unknown UI state %r" % (state,))
        self.seq += 1
        msg = {"state": state, "session": self.session, "seq": self.seq}
        for key, value in (("id", request_id), ("reply", reply), ("heard", heard),
                           ("detail", detail)):
            if value:
                msg[key] = value
        return msg

    def state(self, state, request_id=None, reply=None, heard=None, detail=None):
        """Publish a state (without waiting). Returns the message sent."""
        msg = self.message(state, request_id, reply, heard, detail)
        self.last = msg
        self.log("[UI] %s%s" % (state, " - " + detail if detail else ""))
        self._reap()
        proc = self.robot.publish_text(UI_STATE_TOPIC, json.dumps(msg))
        if proc is not None:
            self._procs.append(proc)
        return msg

    def _reap(self):
        """Forget finished publishers; log the ones that failed."""
        running = []
        for proc in self._procs:
            if proc.poll() is None:
                running.append(proc)
                continue
            try:
                out, err = proc.communicate(timeout=1)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                out, err = "", ""
            if proc.returncode != 0:
                self.log("[UI] state publish failed (exit %s): %s"
                         % (proc.returncode, ((err or out) or "").strip()[:300]))
        self._procs = running

    def close(self, timeout=10.0):
        """Wait (up to timeout s each) for the state publishers still running."""
        for proc in self._procs:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._reap()

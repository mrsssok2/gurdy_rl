"""Talk to the ROS side in the Docker container with `docker exec ... rostopic`.

rostopic pub parses its message argument as YAML, so a bare JSON object would
be read as a mapping of message fields. The String is sent as
    data: "<the text, JSON-escaped>"
JSON string escapes (ensure_ascii, so Devanagari becomes \\uXXXX) are valid
YAML double-quoted escapes, and the whole argument is shell-quoted for bash.
"""
import json
import re
import shlex
import subprocess
import time

COMMAND_TOPIC = "/gurdy/voice_command"
TRANSCRIPT_TOPIC = "/gurdy/voice_transcript"
FEEDBACK_TOPIC = "/gurdy/voice_feedback"
DEFAULT_CONTAINER = "noetic"
DEFAULT_SETUP = "/root/catkin_ws/devel/setup.bash"


def yaml_string_arg(text):
    """rostopic pub argument for a std_msgs/String with data = text."""
    return "data: " + json.dumps(text, ensure_ascii=True)


def pub_script(topic, text):
    """bash snippet that publishes text once on topic."""
    return "rostopic pub -1 %s std_msgs/String %s" % (topic, shlex.quote(yaml_string_arg(text)))


def publish_script(transcript=None, command=None):
    """
    bash snippet publishing the transcript and/or the command (a dict, sent
    as JSON), or None. Each pub latches for 3 s, so they run side by side.
    """
    scripts = []
    if transcript is not None:
        scripts.append(pub_script(TRANSCRIPT_TOPIC, transcript))
    if command is not None:
        scripts.append(pub_script(COMMAND_TOPIC, json.dumps(command)))
    if not scripts:
        return None
    return " & ".join(scripts) + " ; wait"


def echo_script(topic, count=1):
    return "rostopic echo -n %d %s" % (count, topic)


def docker_argv(container, setup, script):
    # The braces keep `source` applying to all of script: without them
    # "source x && a & b" would background "source x && a" and run b unsourced
    return ["docker", "exec", container, "bash", "-c",
            "source %s && { %s; }" % (setup, script)]


def _unquote_double(body):
    """Value of a YAML double-quoted scalar's body (may span folded lines)."""
    lines = body.split("\n")
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("\\") and not text.endswith("\\\\"):
            text = text[:-1] + line.lstrip()  # escaped line break: no space
        else:
            text += " " + line.strip()
    return json.loads('"' + text + '"')


def parse_echo_data(output):
    """
    The `data` string of the first message in `rostopic echo` output for a
    std_msgs/String, or None. Handles double-quoted, single-quoted and plain
    YAML scalars, including values folded over several lines.
    """
    match = re.search(r"^data:[ \t]*(.*?)(?:^---|^\.\.\.\s*$|\Z)", output, re.S | re.M)
    if not match:
        return None
    value = match.group(1).strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        try:
            return _unquote_double(value[1:-1])
        except ValueError:
            pass
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return " ".join(l.strip() for l in value[1:-1].split("\n")).replace("''", "'")
    return " ".join(l.strip() for l in value.split("\n"))


def parse_feedback(output):
    """Feedback dict from `rostopic echo` output, or None."""
    data = parse_echo_data(output)
    if data is not None:
        try:
            msg = json.loads(data)
            if isinstance(msg, dict):
                return msg
        except ValueError:
            pass
    # Last resort: the outermost {...}, with \" unescaped
    match = re.search(r"\{.*\}", output, re.S)
    if match:
        try:
            msg = json.loads(match.group(0).replace('\\"', '"'))
            return msg if isinstance(msg, dict) else None
        except ValueError:
            return None
    return None


class RobotLink(object):
    def __init__(self, container=DEFAULT_CONTAINER, setup=DEFAULT_SETUP,
                 echo_delay=2.0, dry_run=False, log=print):
        self.container = container
        self.setup = setup
        # Time for `rostopic echo` to subscribe before the command goes out
        self.echo_delay = echo_delay
        self.dry_run = dry_run
        self.log = log

    def _argv(self, script):
        return docker_argv(self.container, self.setup, script)

    def _popen(self, script):
        argv = self._argv(script)
        if self.dry_run:
            self.log("[dry-run] " + subprocess.list2cmdline(argv))
            return None
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace")

    def publish(self, transcript=None, command=None):
        """
        Publish the transcript and/or command (dict, sent as JSON) in one
        docker exec, without waiting for it. Returns the process (or None).
        """
        script = publish_script(transcript, command)
        return self._popen(script) if script else None

    def send_and_wait(self, command, transcript=None, timeout=30.0, poll=None):
        """
        Publish command (and transcript), then wait for the feedback with the
        command's id. poll() is called about every 0.1 s while waiting (e.g.
        to listen for a stop). Returns the feedback dict, or None on timeout.
        """
        deadline = time.time() + timeout
        echo = self._popen(echo_script(FEEDBACK_TOPIC))
        if echo is not None:
            self._sleep(self.echo_delay, poll)
        publisher = self.publish(transcript, command)
        if echo is None:
            return None
        try:
            while True:
                while echo.poll() is None:
                    if time.time() > deadline:
                        return None
                    self._sleep(0.1, poll)
                out, err = echo.communicate()
                if echo.returncode != 0:
                    self.log("[robot] rostopic echo failed (exit %d): %s"
                             % (echo.returncode, (err or out).strip()[:500]))
                    return None
                feedback = parse_feedback(out)
                if feedback is None:
                    self.log("[robot] could not parse feedback: %r" % out[:500])
                elif feedback.get("id") == command.get("id"):
                    return feedback
                else:
                    self.log("[robot] ignoring feedback for another command: %s" % feedback)
                if time.time() > deadline:
                    return None
                echo = self._popen(echo_script(FEEDBACK_TOPIC))
        finally:
            if echo is not None and echo.poll() is None:
                echo.kill()
            self._check_publisher(publisher)

    def _check_publisher(self, publisher):
        if publisher is None:
            return
        try:
            out, err = publisher.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            publisher.kill()
            return
        if publisher.returncode != 0:
            self.log("[robot] rostopic pub failed (exit %d): %s"
                     % (publisher.returncode, (err or out).strip()[:500]))

    @staticmethod
    def _sleep(seconds, poll):
        end = time.time() + seconds
        while time.time() < end:
            if poll is not None:
                poll()
            time.sleep(min(0.1, max(0.0, end - time.time())))

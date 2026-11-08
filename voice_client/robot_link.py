"""Talk to the ROS side in the Docker container with `docker exec ... rostopic`.

rostopic pub parses its message argument as YAML, so a bare JSON object would
be read as a mapping of message fields. The String is sent as
    data: "<the text, JSON-escaped>"
JSON string escapes (ensure_ascii, so Devanagari becomes \\uXXXX) are valid
YAML double-quoted escapes, and the whole argument is shell-quoted for bash.

Feedback comes from one long-running `rostopic echo` (started before the
command is published), whose output is split on rostopic's "---" separator and
matched to the command by id.

Test the link alone (prints the raw echo stdout/stderr):
    python voice_client/robot_link.py --test-status
"""
import json
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

try:
    import yaml
except ImportError:  # the hand-written parser below is the fallback
    yaml = None

COMMAND_TOPIC = "/gurdy/voice_command"
TRANSCRIPT_TOPIC = "/gurdy/voice_transcript"
FEEDBACK_TOPIC = "/gurdy/voice_feedback"
DEFAULT_CONTAINER = "noetic"
DEFAULT_SETUP = "/root/catkin_ws/devel/setup.bash"
DEFAULT_ECHO_DELAY = 4.0


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


def echo_script(topic, count=None):
    """
    bash snippet echoing topic (count messages, or until killed). Python
    output to a pipe is block-buffered, so PYTHONUNBUFFERED makes each
    message reach us as soon as it is received.
    """
    n = "" if count is None else "-n %d " % count
    return "PYTHONUNBUFFERED=1 rostopic echo %s%s" % (n, topic)


def docker_argv(container, setup, script):
    # The braces keep `source` applying to all of script: without them
    # "source x && a & b" would background "source x && a" and run b unsourced
    return ["docker", "exec", container, "bash", "-c",
            "source %s && { %s; }" % (setup, script)]


def _yaml_escapes_to_json(text):
    """Turn the YAML-only escapes in a double-quoted body into JSON ones."""
    yaml_only = {" ": " ", "\t": "\\t", "/": "/", "0": "\\u0000",
                 "e": "\\u001b", "a": "\\u0007", "v": "\\u000b"}

    def sub(m):
        return yaml_only.get(m.group(1), m.group(0))
    return re.sub(r"\\(.)", sub, text)


def _unquote_double(body):
    """Value of a YAML double-quoted scalar's body (may span folded lines)."""
    lines = body.split("\n")
    text = lines[0]
    for line in lines[1:]:
        if re.search(r"(?<!\\)(?:\\\\)*\\$", text):
            text = text[:-1] + line.lstrip()  # escaped line break: no space
        else:
            text += " " + line.strip()
    return json.loads('"' + _yaml_escapes_to_json(text) + '"')


def _message_text(output):
    """The first message in `rostopic echo` output: its lines up to "---"."""
    lines = []
    for line in output.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip() == "---":
            if any(l.strip() for l in lines):
                break
            lines = []
            continue
        lines.append(line)
    return "\n".join(lines)


def _parse_echo_data_yaml(text):
    """data via PyYAML, or None if text isn't a mapping with a data key."""
    try:
        msg = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if isinstance(msg, dict) and "data" in msg:
        data = msg["data"]
        return "" if data is None else str(data)
    return None


def _parse_echo_data_manual(text):
    """Fallback without PyYAML: double-, single-quoted and plain scalars."""
    match = re.search(r"^data:[ \t]*(.*?)(?:^\.\.\.\s*$|\Z)", text, re.S | re.M)
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


def parse_echo_data(output):
    """
    The `data` string of the first message in `rostopic echo` output for a
    std_msgs/String, or None. rostopic prints the message as YAML and wraps
    long strings over several lines (a trailing backslash is an escaped line
    break, backslash-space an escaped space), so the message's lines (up to
    "---") are joined and parsed with yaml.safe_load. A hand-written parser
    is the fallback if PyYAML is missing or fails.
    """
    text = _message_text(output)
    if yaml is not None:
        data = _parse_echo_data_yaml(text)
        if data is not None:
            return data
    return _parse_echo_data_manual(text)


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


def split_echo_messages(lines):
    """
    Group `rostopic echo` output lines into messages: yields one text block
    per message, each ending at a "---" line (rostopic's separator).
    """
    block = []
    for line in lines:
        if line.strip() == "---":
            if any(l.strip() for l in block):
                yield "".join(block)
            block = []
        else:
            block.append(line)
    if any(l.strip() for l in block):
        yield "".join(block)


def walk_timeout(distance_m, speed=0.005, margin=30.0):
    """Seconds to wait for a walk's result: the distance at 5 mm/s, plus margin."""
    return float(distance_m) / speed + margin


class FeedbackListener(object):
    """
    One long-running `rostopic echo` on the feedback topic. Reader threads
    split its stdout into messages, parse them and queue the dicts. Every raw
    stdout/stderr line is logged when verbose, and kept (last 50) for
    report_exit().
    """

    def __init__(self, argv, log=print, verbose=False):
        self.log = log
        self.verbose = verbose
        self.messages = queue.Queue()
        self.stdout_tail = []
        self.stderr_tail = []
        self.started = time.time()
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     encoding="utf-8", errors="replace", bufsize=1)
        self._threads = []
        for target in (self._read_stdout, self._read_stderr):
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            self._threads.append(t)

    def _stdout_lines(self):
        for line in self.proc.stdout:
            if self.verbose:
                self.log("[echo stdout] " + line.rstrip("\r\n"))
            self.stdout_tail = (self.stdout_tail + [line])[-50:]
            yield line

    def _read_stdout(self):
        try:
            for block in split_echo_messages(self._stdout_lines()):
                feedback = parse_feedback(block)
                if feedback is None:
                    self.log("[robot] could not parse feedback message: %r" % block[:500])
                else:
                    self.messages.put(feedback)
        except (ValueError, OSError):
            pass  # pipe closed by close()

    def _read_stderr(self):
        try:
            for line in self.proc.stderr:
                line = line.rstrip("\r\n")
                self.stderr_tail = (self.stderr_tail + [line])[-50:]
                if self.verbose:
                    self.log("[echo stderr] " + line)
        except (ValueError, OSError):
            pass

    def finished(self):
        """True once the process has exited and all its output is parsed."""
        return self.proc.poll() is not None and not any(t.is_alive() for t in self._threads)

    def get(self, timeout):
        """Next feedback dict, or None if none arrived within timeout."""
        try:
            return self.messages.get(timeout=timeout)
        except queue.Empty:
            return None

    def report_exit(self):
        """Log that the echo process ended, with its raw output."""
        self.log("[robot] rostopic echo exited (code %s) after %.1f s"
                 % (self.proc.returncode, time.time() - self.started))
        self.log("[robot] echo stdout: %r" % "".join(self.stdout_tail)[-1000:])
        self.log("[robot] echo stderr: %r" % "\n".join(self.stderr_tail)[-1000:])

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return
        for t in self._threads:
            t.join(timeout=1.0)
        for pipe in (self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


class RobotLink(object):
    def __init__(self, container=DEFAULT_CONTAINER, setup=DEFAULT_SETUP,
                 echo_delay=DEFAULT_ECHO_DELAY, dry_run=False, verbose=False, log=print):
        self.container = container
        self.setup = setup
        # Time for `rostopic echo` to subscribe before the command goes out
        self.echo_delay = echo_delay
        self.dry_run = dry_run
        self.verbose = verbose
        self.log = log
        # Why the last send_and_wait ended: "ok", "timeout", "cancelled", "dry-run"
        self.last_outcome = None

    def _argv(self, script):
        return docker_argv(self.container, self.setup, script)

    def _show(self, argv):
        if self.dry_run:
            self.log("[dry-run] " + subprocess.list2cmdline(argv))
        elif self.verbose:
            self.log("[robot] run: " + subprocess.list2cmdline(argv))

    def _popen(self, script):
        argv = self._argv(script)
        self._show(argv)
        if self.dry_run:
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

    def listen(self):
        """Start a FeedbackListener on the feedback topic (None in a dry run)."""
        argv = self._argv(echo_script(FEEDBACK_TOPIC))
        self._show(argv)
        if self.dry_run:
            return None
        return FeedbackListener(argv, log=self.log, verbose=self.verbose)

    def send_and_wait(self, command, transcript=None, timeout=30.0, poll=None):
        """
        Start the feedback subscriber, give it echo_delay s to subscribe,
        publish command (and transcript), then wait for the feedback whose id
        matches the command's. poll() is called about every 0.1 s while
        waiting (e.g. to listen for a stop); if it returns True the wait is
        cancelled. If the echo process dies it is reported and restarted.
        Returns the feedback dict, or None (last_outcome says why).
        """
        self.last_outcome = None
        listener = self.listen()
        if listener is not None and self._sleep(self.echo_delay, poll):
            listener.close()
            self.last_outcome = "cancelled"
            return None
        publisher = self.publish(transcript, command)
        if listener is None:
            self.last_outcome = "dry-run"
            return None
        deadline = time.time() + timeout
        try:
            while True:
                feedback = listener.get(0.1)
                if feedback is not None:
                    if self.verbose:
                        self.log("[robot] feedback: %s" % json.dumps(feedback))
                    if feedback.get("id") == command.get("id"):
                        self.last_outcome = "ok"
                        return feedback
                    self.log("[robot] ignoring feedback for another command: %s" % feedback)
                    continue
                if poll is not None and poll():
                    self.last_outcome = "cancelled"
                    return None
                if time.time() > deadline:
                    self.last_outcome = "timeout"
                    return None
                if listener.finished() and listener.messages.empty():
                    listener.report_exit()
                    listener.close()
                    self.log("[robot] restarting rostopic echo")
                    listener = self.listen()
                    self._sleep(1.0, None)  # don't spin if docker fails at once
        finally:
            listener.close()
            self._check_publisher(publisher)

    def _check_publisher(self, publisher):
        if publisher is None:
            return
        try:
            out, err = publisher.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            publisher.kill()
            return
        if self.verbose:
            self.log("[pub stdout] %r" % out)
            self.log("[pub stderr] %r" % err)
        if publisher.returncode != 0:
            self.log("[robot] rostopic pub failed (exit %d): %s"
                     % (publisher.returncode, (err or out).strip()[:500]))

    def status(self, timeout=20.0):
        """Send a status command; returns the robot's status dict or None."""
        cmd = {"id": uuid.uuid4().hex[:8], "action": "status"}
        return self.send_and_wait(cmd, timeout=timeout)

    @staticmethod
    def _sleep(seconds, poll):
        """Sleep, calling poll(); returns True if poll() asked to cancel."""
        end = time.time() + seconds
        while time.time() < end:
            if poll is not None and poll():
                return True
            time.sleep(min(0.1, max(0.0, end - time.time())))
        return False


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Test the Docker/ROS link on its own.")
    ap.add_argument("--test-status", action="store_true",
                    help="send a status command; print the raw echo output and the reply")
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--setup", default=DEFAULT_SETUP, help="setup.bash in the container")
    ap.add_argument("--echo-delay", type=float, default=DEFAULT_ECHO_DELAY)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args(argv)
    if not args.test_status:
        ap.print_help()
        return 0
    link = RobotLink(container=args.container, setup=args.setup,
                     echo_delay=args.echo_delay, verbose=True)
    feedback = link.status(timeout=args.timeout)
    print("outcome: %s" % link.last_outcome)
    print("feedback: %s" % (json.dumps(feedback) if feedback is not None else None))
    return 0 if feedback is not None else 1


if __name__ == "__main__":
    sys.exit(main())

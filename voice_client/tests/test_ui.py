"""Offline tests for the dashboard Speak button link (ui_link, main.py --ui).
No ROS or docker: echo processes are local python stand-ins, the robot and
the talk stream are fakes. Run from the repo root:
    python -m unittest discover -s voice_client/tests
"""
import json
import os
import queue
import random
import sys
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# scripts/dashboard_state.py has no ROS imports: used to check both ends agree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))

import dashboard_state as ds  # noqa: E402
import main  # noqa: E402
import robot_link as rl  # noqa: E402
import ui_link  # noqa: E402


def echo_text(*datas):
    """`rostopic echo` output for std_msgs/String messages with these data."""
    return "".join("data: %s\n---\n" % json.dumps(d) for d in datas)


def talk(request_id):
    return json.dumps({"id": request_id, "source": "dashboard"})


class ProtocolTest(unittest.TestCase):
    def test_both_ends_agree(self):
        self.assertEqual(ui_link.UI_TALK_TOPIC, ds.UI_TALK_TOPIC)
        self.assertEqual(ui_link.UI_STATE_TOPIC, ds.UI_STATE_TOPIC)
        self.assertEqual(ui_link.STATES, ds.VOICE_STATES)
        self.assertEqual(ds.VOICE_COMMAND_TOPIC, rl.COMMAND_TOPIC)

    def test_parse_talk_json(self):
        req = ui_link.parse_talk(echo_text(ds.talk_request_data("ab12cd34")))
        self.assertEqual(req, {"id": "ab12cd34", "source": "dashboard"})

    def test_parse_talk_plain_text_and_empty(self):
        self.assertEqual(ui_link.parse_talk("data: \"go\"\n"), {"id": None, "text": "go"})
        self.assertEqual(ui_link.parse_talk("data: go\n"), {"id": None, "text": "go"})
        self.assertIsNone(ui_link.parse_talk("data: ''\n"))
        self.assertIsNone(ui_link.parse_talk("WARNING: no messages received\n"))

    def test_parse_talk_folded(self):
        # rostopic wraps long strings; a trailing backslash continues the line
        out = 'data: "{\\"id\\": \\"ab12\\", \\"source\\": \\"dash\\\n  board\\"}"\n---\n'
        self.assertEqual(ui_link.parse_talk(out), {"id": "ab12", "source": "dashboard"})

    def test_stop_button_command_is_valid(self):
        cmd = json.loads(ds.stop_command_data("ui-1"))
        self.assertEqual((cmd["action"], cmd["id"]), ("stop", "ui-1"))


class FakeProc(object):
    def __init__(self, code=0, err=""):
        self.returncode, self.err = code, err

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return "", self.err

    def wait(self, timeout=None):
        return self.returncode


class RecordingRobot(object):
    """RobotLink stand-in: records publishes; send_and_wait is scripted."""

    def __init__(self, feedback=None, during_wait=None):
        self.published = []          # (topic, data)
        self.commands = []           # (transcript, command) via publish()
        self.feedback = feedback     # f(cmd) -> feedback dict
        self.during_wait = during_wait or 0   # poll() calls while "walking"
        self.last_outcome = None
        self.proc = FakeProc()

    def publish_text(self, topic, text):
        self.published.append((topic, text))
        return self.proc

    def publish(self, transcript=None, command=None):
        self.commands.append((transcript, command))

    def send_and_wait(self, command, transcript=None, timeout=30.0, poll=None):
        self.commands.append((transcript, command))
        if poll is not None:
            for _ in range(self.during_wait):
                if poll():
                    self.last_outcome = "cancelled"
                    return None
        self.last_outcome = "ok"
        return self.feedback(command) if self.feedback else None

    def states(self):
        return [json.loads(d) for t, d in self.published if t == ui_link.UI_STATE_TOPIC]


class UiLinkTest(unittest.TestCase):
    def test_messages_and_seq(self):
        robot = RecordingRobot()
        link = ui_link.UiLink(robot, log=lambda s: None)
        link.state(ui_link.LISTENING, request_id="r1", detail="speak now")
        link.state(ui_link.SPEAKING, request_id="r1", reply="Theek hai")
        a, b = robot.states()
        self.assertEqual((a["state"], a["seq"], a["id"]), ("LISTENING", 1, "r1"))
        self.assertEqual((b["seq"], b["reply"]), (2, "Theek hai"))
        self.assertNotIn("reply", a)
        self.assertEqual(a["session"], b["session"])
        with self.assertRaises(ValueError):
            link.state("DANCING")

    def test_failed_publish_is_logged(self):
        robot = RecordingRobot()
        robot.proc = FakeProc(1, "ERROR: Unable to communicate with master!")
        logged = []
        link = ui_link.UiLink(robot, log=logged.append)
        link.state(ui_link.IDLE)
        link.state(ui_link.IDLE)  # reaps the first
        self.assertTrue(any("Unable to communicate" in l for l in logged))

    def test_dashboard_keeps_newest_when_out_of_order(self):
        robot = RecordingRobot()
        link = ui_link.UiLink(robot, log=lambda s: None)
        for state in (ui_link.LISTENING, ui_link.THINKING, ui_link.ACTING,
                      ui_link.SPEAKING, ui_link.IDLE):
            link.state(state, request_id="r1", reply="Main ruk gaya"
                       if state in (ui_link.SPEAKING, ui_link.IDLE) else None)
        datas = [d for _, d in robot.published]
        random.Random(3).shuffle(datas)
        s = ds.DashboardState()
        for d in datas:
            s.ui_state(d)
        self.assertEqual(s.voice_state, ds.V_IDLE)
        self.assertEqual(s.voice_reply, "Main ruk gaya")
        # A new client run (new session) starts over at seq 1
        link2 = ui_link.UiLink(RecordingRobot(), log=lambda s: None)
        self.assertTrue(s.ui_state(json.dumps(link2.message(ui_link.LISTENING))))
        self.assertEqual(s.voice_state, ds.V_LISTENING)


class FakeEchoLink(rl.RobotLink):
    """RobotLink whose rostopic echo runs a python stand-in printing outputs[i]."""

    def __init__(self, outputs, linger=30):
        rl.RobotLink.__init__(self, log=self.collect)
        self.logged = []
        self.outputs = list(outputs)
        self.linger = linger

    def collect(self, text):
        self.logged.append(text)

    def _argv(self, script):
        assert "rostopic echo" in script and ui_link.UI_TALK_TOPIC in script, script
        out = self.outputs.pop(0) if self.outputs else ""
        last = not self.outputs
        code = ("import sys,time\n"
                "for l in %r.splitlines(True):\n"
                "    sys.stdout.write(l); sys.stdout.flush(); time.sleep(0.01)\n"
                "sys.stderr.write('echo-err\\n')\n"
                "time.sleep(%s)\n" % (out, self.linger if last else 0))
        return [sys.executable, "-c", code]


class TopicStreamTest(unittest.TestCase):
    def collect(self, stream, n, timeout=10.0):
        got, end = [], time.time() + timeout
        while len(got) < n and time.time() < end:
            msg = stream.get(0.1)
            if msg is not None:
                got.append(msg)
        return got

    def test_restarts_dead_echo(self):
        link = FakeEchoLink([
            echo_text(talk("first")),                           # then dies
            "ERROR: Unable to communicate with master!\n",      # dies at once
            echo_text(talk("second"), "go"),                    # stays up
        ])
        stream = link.stream(ui_link.UI_TALK_TOPIC, parse=ui_link.parse_talk,
                             restart_delay=0.05)
        try:
            got = self.collect(stream, 3)
        finally:
            stream.close()
        self.assertEqual([g.get("id") for g in got], ["first", "second", None])
        self.assertEqual(got[2]["text"], "go")
        self.assertEqual(stream.restarts, 2)
        self.assertTrue(any("rostopic echo exited" in l for l in link.logged))
        self.assertTrue(any("echo-err" in l for l in link.logged))
        self.assertTrue(any("could not parse" in l for l in link.logged))

    def test_close_kills_echo_and_drain(self):
        link = FakeEchoLink([echo_text(talk("a"), talk("b"))])
        stream = link.stream(ui_link.UI_TALK_TOPIC, parse=ui_link.parse_talk)
        end = time.time() + 10
        while stream.messages.qsize() < 2 and time.time() < end:
            time.sleep(0.05)
        self.assertEqual([m["id"] for m in stream.drain()], ["a", "b"])
        self.assertIsNone(stream.get())
        listener = stream.listener
        stream.close()
        self.assertIsNotNone(listener.proc.poll())  # the lingering echo was killed

    def test_dry_run_has_no_stream(self):
        link = rl.RobotLink(dry_run=True, log=lambda s: None)
        stream = link.stream(ui_link.UI_TALK_TOPIC)
        stream.close()
        self.assertIsNone(stream.get())


class FakeKeyboard(object):
    def __init__(self, lines=()):
        self.lines = list(lines)   # None entries: no key this poll

    def poll(self):
        return self.lines.pop(0) if self.lines else None


class FakeStream(object):
    def __init__(self, requests=()):
        self.q = queue.Queue()
        for r in requests:
            self.q.put(r)
        self.closed = False

    def get(self, timeout=0.0):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def drain(self):
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    def close(self):
        self.closed = True


def make_demo(robot, transcripts, keyboard=(), requests=()):
    """VoiceDemo --ui --type --no-tts with fakes; transcripts are what is 'heard'."""
    args = types.SimpleNamespace(
        type=True, no_tts=True, mode="scripted", container="noetic", setup="x",
        echo_delay=0.0, dry_run=True, verbose=False, ui=False, seconds=4.0,
        language="hi-IN", voice="Poorvi", tts_language="hi-en")
    demo = main.VoiceDemo(args)
    demo.robot = robot
    demo.ui = ui_link.UiLink(robot, log=lambda s: None)
    demo.talk = FakeStream(requests)
    demo.keyboard = FakeKeyboard(keyboard)
    heard = list(transcripts)

    def fake_input(prompt=""):
        item = heard.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    demo.wait_trigger = lambda idle_sleep=0.0: _wait(demo)
    return demo, fake_input


def _wait(demo):
    """wait_trigger without sleeping; quits when nothing is left."""
    for _ in range(1000):
        trigger = demo.talk_trigger()
        if trigger is not None:
            return trigger
        if not demo.keyboard.lines and demo.talk.q.empty():
            return "quit"
    return "quit"


class UiFlowTest(unittest.TestCase):
    def run_demo(self, demo, fake_input):
        real_input, main.input = getattr(main, "input", None), fake_input
        try:
            demo.run_ui()
        finally:
            if real_input is None:
                del main.input
            else:
                main.input = real_input

    def test_speak_status(self):
        robot = RecordingRobot(feedback=lambda c: {"id": c["id"], "action": "status",
                                                   "status": "idle"})
        demo, inp = make_demo(robot, ["Gurdy status"], requests=[{"id": "req1"}])
        self.run_demo(demo, inp)
        states = robot.states()
        self.assertEqual([s["state"] for s in states],
                         ["IDLE", "LISTENING", "THINKING", "ROBOT ACTING", "SPEAKING",
                          "IDLE", "IDLE"])
        self.assertEqual({s.get("id") for s in states[1:6]}, {"req1"})
        self.assertTrue(states[4]["reply"])
        self.assertEqual(states[5]["reply"], states[4]["reply"])
        self.assertEqual(states[-1]["detail"], "voice client stopped")
        self.assertEqual(robot.commands[0][1]["action"], "status")
        self.assertTrue(demo.talk.closed)
        # The dashboard shows the last reply, IDLE
        s = ds.DashboardState()
        for st in states:
            s.ui_state(json.dumps(st))
        self.assertEqual(s.voice_display(0.0)[0], ds.V_IDLE)
        self.assertEqual(s.voice_reply, states[4]["reply"])

    def test_enter_still_works(self):
        robot = RecordingRobot()
        demo, inp = make_demo(robot, ["RL mode"], keyboard=[""])
        self.run_demo(demo, inp)
        states = robot.states()
        self.assertTrue(states[1]["id"].startswith("key-"))
        self.assertIn("SPEAKING", [s["state"] for s in states])
        self.assertEqual(demo.mode, "rl")

    def test_stop_during_walk(self):
        req = []

        def feedback(cmd):
            return {"id": cmd["id"], "action": "walk_forward", "status": "stopped",
                    "distance_m": 0.05}
        robot = RecordingRobot(feedback=feedback, during_wait=3)
        demo, inp = make_demo(robot, ["aage bees centimeter chalo", "gurdy ruko"],
                              requests=[{"id": "walk1"}])
        # The second Speak press comes while walking
        orig = robot.send_and_wait

        def send_and_wait(command, **kw):
            demo.talk.q.put({"id": "stop1"})
            req.append(command)
            return orig(command, **kw)
        robot.send_and_wait = send_and_wait
        self.run_demo(demo, inp)
        walk = req[0]
        self.assertEqual(walk["action"], "walk_forward")
        stops = [c for t, c in robot.commands if c and c["action"] == "stop"]
        self.assertEqual(len(stops), 1)
        names = [(s["state"], s.get("id")) for s in robot.states()]
        self.assertIn(("ROBOT ACTING", "walk1"), names)
        acting = [s for s in robot.states() if s["state"] == "ROBOT ACTING"]
        self.assertEqual(acting[-1]["detail"], "stop sent")
        self.assertEqual(names[-2][0], "IDLE")  # replied to the walk's result

    def test_non_stop_during_walk_is_refused(self):
        robot = RecordingRobot(feedback=lambda c: {"id": c["id"], "action": "walk_forward",
                                                   "status": "done", "distance_m": 0.2},
                               during_wait=2)
        demo, inp = make_demo(robot, ["aage bees centimeter chalo", "peeche chalo"],
                              requests=[{"id": "walk1"}])
        orig = robot.send_and_wait

        def send_and_wait(command, **kw):
            demo.talk.q.put({"id": "other"})
            return orig(command, **kw)
        robot.send_and_wait = send_and_wait
        self.run_demo(demo, inp)
        actions = [c["action"] for t, c in robot.commands if c]
        self.assertEqual(actions, ["walk_forward"])
        details = [s.get("detail", "") for s in robot.states()]
        self.assertTrue(any("only ruko" in d for d in details))

    def test_presses_while_busy_are_dropped(self):
        robot = RecordingRobot(feedback=lambda c: {"id": c["id"], "action": "status",
                                                   "status": "idle"})
        demo, inp = make_demo(robot, ["Gurdy status"], requests=[{"id": "a"}])
        orig = robot.send_and_wait

        def send_and_wait(command, **kw):  # pressed twice during a non-walk
            demo.talk.q.put({"id": "b"})
            demo.talk.q.put({"id": "c"})
            return orig(command, **kw)
        robot.send_and_wait = send_and_wait
        self.run_demo(demo, inp)
        self.assertEqual([s["state"] for s in robot.states()].count("LISTENING"), 1)

    def test_error_and_nothing_heard_go_idle(self):
        robot = RecordingRobot()
        demo, inp = make_demo(robot, [RuntimeError("mic unplugged"), "", "RL mode"],
                              requests=[{"id": "a"}, {"id": "b"}, {"id": "c"}])
        # The three requests are queued at once; keep them from being drained
        demo.talk.drain = lambda: []
        self.run_demo(demo, inp)
        idle = [s.get("detail") for s in robot.states() if s["state"] == "IDLE"]
        self.assertIn("error: mic unplugged", idle)
        self.assertIn("nothing heard", idle)
        self.assertEqual(demo.mode, "rl")  # still served the third press

    def test_stt_failure_goes_idle(self):
        robot = RecordingRobot()
        demo, inp = make_demo(robot, [], requests=[{"id": "a"}])
        demo.listen = lambda: None
        self.run_demo(demo, inp)
        self.assertIn("speech to text failed",
                      [s.get("detail") for s in robot.states()])

    def test_without_ui_publishes_nothing(self):
        robot = RecordingRobot()
        demo, _ = make_demo(robot, [])
        demo.ui = None
        demo.ui_state(ui_link.LISTENING)
        self.assertEqual(robot.published, [])


class KeyboardTest(unittest.TestCase):
    def test_windows_line_editing(self):
        keys = list("qx\bz\r")
        fake = types.SimpleNamespace(kbhit=lambda: bool(keys), getwch=lambda: keys.pop(0))
        real, main.msvcrt = main.msvcrt, fake
        try:
            kb = main.Keyboard()
            self.assertEqual(kb.poll(), "qz")
            self.assertIsNone(kb.poll())
        finally:
            main.msvcrt = real

    def test_quit_and_key(self):
        robot = RecordingRobot()
        demo, _ = make_demo(robot, [], keyboard=["Q", "", None])
        self.assertEqual(demo.talk_trigger(), "quit")
        self.assertEqual(demo.talk_trigger(), "key")
        self.assertIsNone(demo.talk_trigger())


class ArgsTest(unittest.TestCase):
    def test_ui_flag_builds_stream(self):
        sys_argv = sys.argv
        sys.argv = ["main.py", "--ui", "--type", "--no-tts", "--dry-run"]
        built = []
        real = main.VoiceDemo.run
        main.VoiceDemo.run = lambda self: built.append(self)
        try:
            main.main()
        finally:
            main.VoiceDemo.run = real
            sys.argv = sys_argv
        demo = built[0]
        self.assertIsNotNone(demo.ui)
        self.assertIsNotNone(demo.talk)
        demo.talk.close()


if __name__ == "__main__":
    unittest.main()

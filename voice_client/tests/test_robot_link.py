"""Tests for robot_link and replies. Run from the repo root:
    python -m unittest discover -s voice_client/tests
"""
import json
import os
import shlex
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import replies  # noqa: E402
import robot_link as rl  # noqa: E402
from command_parser import parse  # noqa: E402

FEEDBACK = {"action": "walk_forward", "id": "ab12", "status": "done", "distance_m": 0.2013}


class PublishTest(unittest.TestCase):
    def test_command_arg_is_yaml_string(self):
        cmd = json.dumps({"action": "walk_forward", "distance_m": 0.2, "mode": "rl"})
        script = rl.pub_script(rl.COMMAND_TOPIC, cmd)
        argv = shlex.split(script)  # what bash passes to rostopic
        self.assertEqual(argv[:5], ["rostopic", "pub", "-1", rl.COMMAND_TOPIC, "std_msgs/String"])
        self.assertEqual(len(argv), 6)
        key, value = argv[5].split(": ", 1)
        self.assertEqual(key, "data")
        # JSON-quoted text is a YAML double-quoted scalar; JSON decodes it the same
        self.assertEqual(json.loads(value), cmd)

    def test_devanagari_transcript_is_ascii(self):
        text = "गरदी आगे बीस सेंटीमीटर चलो।"
        script = rl.pub_script(rl.TRANSCRIPT_TOPIC, text)
        self.assertTrue(script.isascii())
        value = shlex.split(script)[5].split(": ", 1)[1]
        self.assertEqual(json.loads(value), text)

    def test_single_quote_survives_shell(self):
        script = rl.pub_script(rl.TRANSCRIPT_TOPIC, "it's")
        self.assertEqual(json.loads(shlex.split(script)[5][6:]), "it's")

    def test_docker_argv(self):
        argv = rl.docker_argv("noetic", "/root/catkin_ws/devel/setup.bash", "rostopic list")
        self.assertEqual(argv, ["docker", "exec", "noetic", "bash", "-c",
                                "source /root/catkin_ws/devel/setup.bash && { rostopic list; }"])

    def test_setup_applies_to_every_publisher(self):
        # Run the generated bash with stand-ins for source and rostopic
        import shutil
        import subprocess
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("no bash")
        script = rl.publish_script("x", {"action": "stop"})
        argv = rl.docker_argv("c", "/dev/null", script)
        prelude = ("source() { export SOURCED=1; }; "
                   "rostopic() { echo \"$SOURCED $3\"; }; ")
        out = subprocess.run([bash, "-c", prelude + argv[5]], capture_output=True,
                             text=True, timeout=10).stdout.splitlines()
        self.assertEqual(sorted(line for line in out if line),
                         ["1 " + rl.COMMAND_TOPIC, "1 " + rl.TRANSCRIPT_TOPIC])


class ParseEchoTest(unittest.TestCase):
    def test_double_quoted(self):
        out = 'data: "%s"\n---\n' % json.dumps(FEEDBACK).replace('"', '\\"')
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)

    def test_single_quoted(self):
        out = "data: '%s'\n---\n" % json.dumps(FEEDBACK)
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)

    def test_folded_double_quoted(self):
        # yaml.dump wraps long double-quoted scalars at ~80 columns
        import textwrap
        body = json.dumps(FEEDBACK).replace('"', '\\"')
        lines = textwrap.wrap(body, 40, break_on_hyphens=False)
        out = 'data: "' + "\n  ".join(lines) + '"\n---\n'
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)

    def test_escaped_line_break(self):
        body = json.dumps(FEEDBACK).replace('"', '\\"')
        out = 'data: "%s\\\n  %s"\n---\n' % (body[:30], body[30:])
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)

    def test_document_end_marker(self):
        out = "data: '%s'\n...\n" % json.dumps(FEEDBACK)
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)

    def test_garbage(self):
        self.assertIsNone(rl.parse_feedback("ERROR: Unable to communicate with master!\n"))
        self.assertIsNone(rl.parse_feedback(""))

    def test_crlf(self):
        out = 'data: "%s"\r\n---\r\n' % json.dumps(FEEDBACK).replace('"', '\\"')
        self.assertEqual(rl.parse_feedback(out), FEEDBACK)


def echo_text(*msgs):
    """rostopic echo output for these feedback dicts (escaped JSON, '---')."""
    return "".join('data: "%s"\n---\n' % json.dumps(m).replace('"', '\\"') for m in msgs)


class SplitEchoTest(unittest.TestCase):
    def test_two_messages(self):
        other = dict(FEEDBACK, id="zz99")
        lines = echo_text(other, FEEDBACK).splitlines(True)
        blocks = list(rl.split_echo_messages(lines))
        self.assertEqual([rl.parse_feedback(b) for b in blocks], [other, FEEDBACK])

    def test_unterminated_last_message(self):
        lines = echo_text(FEEDBACK)[:-4].splitlines(True)  # no trailing ---
        self.assertEqual([rl.parse_feedback(b) for b in rl.split_echo_messages(lines)],
                         [FEEDBACK])

    def test_walk_timeout(self):
        self.assertAlmostEqual(rl.walk_timeout(0.2), 70.0)
        self.assertAlmostEqual(rl.walk_timeout(0.5), 130.0)


class FakeLink(rl.RobotLink):
    """RobotLink whose docker commands run a local python stand-in instead."""

    def __init__(self, echo_outputs, **kw):
        kw.setdefault("echo_delay", 0.0)
        rl.RobotLink.__init__(self, log=self.collect, **kw)
        self.logged = []
        self.echo_outputs = list(echo_outputs)  # one per echo process started

    def collect(self, text):
        self.logged.append(text)

    def _argv(self, script):
        if "rostopic echo" in script:
            out = self.echo_outputs.pop(0) if self.echo_outputs else ""
            # print the output slowly, then stay alive like a real echo
            code = ("import sys,time\n"
                    "for l in %r.splitlines(True):\n"
                    "    sys.stdout.write(l); sys.stdout.flush(); time.sleep(0.01)\n"
                    "sys.stderr.write('echo-err\\n')\n"
                    "time.sleep(%s)\n" % (out, 0 if self.echo_outputs else 30))
            return [sys.executable, "-c", code]
        return [sys.executable, "-c", "pass"]


class SendAndWaitTest(unittest.TestCase):
    def test_matches_by_id(self):
        cmd = {"id": "ab12", "action": "walk_forward"}
        link = FakeLink([echo_text(dict(FEEDBACK, id="old1"), FEEDBACK)])
        self.assertEqual(link.send_and_wait(cmd, timeout=10), FEEDBACK)
        self.assertEqual(link.last_outcome, "ok")
        self.assertTrue(any("another command" in l for l in link.logged))

    def test_restarts_dead_echo_and_reports_output(self):
        cmd = {"id": "ab12", "action": "walk_forward"}
        link = FakeLink(["ERROR: Unable to communicate with master!\n", echo_text(FEEDBACK)])
        self.assertEqual(link.send_and_wait(cmd, timeout=10), FEEDBACK)
        self.assertTrue(any("echo stderr" in l and "echo-err" in l for l in link.logged))

    def test_verbose_logs_raw_lines(self):
        cmd = {"id": "ab12", "action": "status"}
        link = FakeLink([echo_text(FEEDBACK)], verbose=True)
        link.send_and_wait(cmd, timeout=10)
        self.assertTrue(any(l.startswith("[echo stdout] data:") for l in link.logged))

    def test_timeout(self):
        link = FakeLink([""])
        self.assertIsNone(link.send_and_wait({"id": "x", "action": "status"}, timeout=0.5))
        self.assertEqual(link.last_outcome, "timeout")

    def test_poll_cancels(self):
        link = FakeLink([""])
        self.assertIsNone(link.send_and_wait({"id": "x", "action": "walk_forward"},
                                             timeout=10, poll=lambda: True))
        self.assertEqual(link.last_outcome, "cancelled")

    def test_status_command(self):
        link = FakeLink([""])
        sent = []
        link.publish = lambda transcript=None, command=None: sent.append(command)
        link.status(timeout=0.3)
        self.assertEqual(sent[0]["action"], "status")
        self.assertTrue(sent[0]["id"])


class ReplyTest(unittest.TestCase):
    def reply(self, text, feedback=None, mode="scripted"):
        r = replies.build_reply(parse(text), feedback, mode)
        self.assertTrue(r.startswith("Main ek AI assistant hoon. "), r)
        self.assertFalse(any(c.isdigit() for c in r), r)  # numbers as words
        return r

    def test_number_words(self):
        self.assertEqual(replies.number_words(20), "bees")
        self.assertEqual(replies.number_words(50), "pachaas")
        self.assertEqual(replies.number_words(100), "sau")
        self.assertEqual(replies.distance_words(0.2013), "bees centimeter")
        self.assertEqual(replies.distance_words(0.187), "unnees centimeter")
        self.assertEqual(replies.distance_words(1.5), "ek point paanch meter")
        self.assertEqual(replies.distance_words(2.0), "do meter")

    def test_walk_done(self):
        r = self.reply("गरदी आगे बीस सेंटीमीटर चलो।",
                       dict(FEEDBACK, mode_used="scripted", fallback=False))
        self.assertIn("Gurdy bees centimeter aage chala, normal mode mein.", r)

    def test_walk_rl_fallback(self):
        r = self.reply("aage tees cm", dict(FEEDBACK, distance_m=0.3, mode_used="rl+scripted",
                                            fallback=True), mode="rl")
        self.assertIn("tees centimeter", r)
        self.assertIn("normal mode par switch", r)

    def test_walk_rl_no_model(self):
        r = self.reply("aage chalo", dict(FEEDBACK, mode_used="scripted", fallback=True,
                                          mode_requested="rl"), mode="rl")
        self.assertIn("RL model nahi mila", r)

    def test_capped(self):
        r = self.reply("aage sau cm chalo", dict(FEEDBACK, distance_m=0.5, mode_used="scripted"))
        self.assertIn("pachaas centimeter tak", r)

    def test_stopped_fallen_stuck(self):
        self.assertIn("ruk gaya", self.reply("ruko", dict(FEEDBACK, status="stopped", distance_m=0.1)))
        r = self.reply("aage chalo", dict(FEEDBACK, status="fallen", reset=True, distance_m=0.05))
        self.assertIn("gir gaya", r)
        self.assertIn("wapas khada", r)
        self.assertIn("aage nahi badh", self.reply("aage chalo", dict(FEEDBACK, status="stuck", distance_m=0.002)))

    def test_stop_idle(self):
        self.assertIn("pehle se ruka", self.reply("ruko", {"action": "stop", "status": "idle"}))

    def test_status(self):
        r = self.reply("kitna chala", {"action": "status", "status": "idle", "total_distance_m": 0.45})
        self.assertIn("kul paintaalees centimeter", r)

    def test_mode_unknown_no_reply(self):
        self.assertIn("RL mode on", self.reply("RL mode"))
        self.assertIn("normal mode on", self.reply("normal mode"))
        self.assertIn("samajh nahi", self.reply("namaste"))
        self.assertIn("jawab nahi", self.reply("aage chalo", None))
        self.assertIn("error", self.reply("aage chalo", {"status": "error"}))


    def test_reset(self):
        self.assertIn("reset", self.reply("Gurdy wapas"))


if __name__ == "__main__":
    unittest.main()

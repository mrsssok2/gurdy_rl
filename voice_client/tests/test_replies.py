"""Tests for replies and main.make_command. Run from the repo root:
    python -m unittest discover -s voice_client/tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from command_parser import parse  # noqa: E402
from main import make_command  # noqa: E402
from replies import build_reply  # noqa: E402


def walk(action, status, distance_m, **extra):
    return dict(action=action, status=status, distance_m=distance_m,
                requested_m=0.2, mode_used="scripted", fallback=False, **extra)


class SideReplyTest(unittest.TestCase):
    def test_done(self):
        self.assertEqual(
            build_reply(parse("baayen bees cm chalo"), walk("walk_left", "done", 0.2012)),
            "Main ek AI assistant hoon. Main bees centimeter baayen chala.")
        self.assertEqual(
            build_reply(parse("daayen tees cm chalo"), walk("walk_right", "done", 0.3)),
            "Main ek AI assistant hoon. Main tees centimeter daayen chala.")

    def test_capped(self):
        reply = build_reply(parse("baayen ek meter chalo"), walk("walk_left", "done", 0.5))
        self.assertIn("pachaas centimeter tak hi", reply)
        self.assertTrue(reply.endswith("Main pachaas centimeter baayen chala."))

    def test_other_statuses(self):
        parsed = parse("daayen chalo")
        self.assertIn("Main ruk gaya. Main das centimeter daayen chala.",
                      build_reply(parsed, walk("walk_right", "stopped", 0.1)))
        fallen = build_reply(parsed, walk("walk_right", "fallen", 0.05, reset=True))
        self.assertIn("Main paanch centimeter daayen chalne ke baad gir gaya.", fallen)
        self.assertIn("wapas khada", fallen)
        self.assertIn("ulti taraf", build_reply(parsed, walk("walk_right", "stuck", -0.01)))
        self.assertIn("Main daayen nahi badh paaya, sirf do centimeter chala.",
                      build_reply(parsed, walk("walk_right", "stuck", 0.02)))
        self.assertIn("dheere", build_reply(parsed, walk("walk_right", "step_limit", 0.1)))
        self.assertIn("abhi chal raha hai",
                      build_reply(parsed, {"action": "walk_right", "status": "busy"}))

    def test_stop_during_side_walk(self):
        reply = build_reply(parse("ruko"), walk("walk_left", "stopped", 0.08))
        self.assertIn("Main aath centimeter baayen chala.", reply)

    def test_forward_unchanged(self):
        self.assertEqual(
            build_reply(parse("aage bees cm chalo"), walk("walk_forward", "done", 0.2)),
            "Main ek AI assistant hoon. Gurdy bees centimeter aage chala, normal mode mein.")

    def test_unsupported_direction(self):
        self.assertIn("sirf aage, baayen ya daayen", build_reply(parse("peeche chalo")))
        self.assertIn("samajh nahi aaya", build_reply(parse("namaste")))


class MakeCommandTest(unittest.TestCase):
    def test_side_walk_has_no_mode(self):
        cmd = make_command(parse("baayen tees cm chalo"), "rl")
        self.assertEqual((cmd["action"], cmd["distance_m"]), ("walk_left", 0.3))
        self.assertNotIn("mode", cmd)
        self.assertEqual(make_command(parse("daayen chalo"), "rl")["action"], "walk_right")

    def test_forward_and_unknown(self):
        self.assertEqual(make_command(parse("aage chalo"), "rl")["mode"], "rl")
        self.assertIsNone(make_command(parse("peeche chalo"), "rl"))


if __name__ == "__main__":
    unittest.main()

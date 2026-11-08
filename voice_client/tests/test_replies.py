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


class BackwardReplyTest(unittest.TestCase):
    def test_done(self):
        self.assertEqual(
            build_reply(parse("peeche bees cm chalo"), walk("walk_backward", "done", 0.2031)),
            "Main ek AI assistant hoon. Main bees centimeter peeche chala.")

    def test_other_statuses(self):
        parsed = parse("peeche chalo")
        self.assertEqual(build_reply(parsed, walk("walk_backward", "stopped", 0.1)),
                         "Main ek AI assistant hoon. Main ruk gaya. Main das centimeter "
                         "peeche chala.")
        fallen = build_reply(parsed, walk("walk_backward", "fallen", 0.05, reset=True))
        self.assertIn("Main paanch centimeter peeche chalne ke baad gir gaya.", fallen)
        self.assertIn("Main peeche nahi badh paaya, sirf teen centimeter chala.",
                      build_reply(parsed, walk("walk_backward", "stuck", 0.03)))
        self.assertIn("ulti taraf", build_reply(parsed, walk("walk_backward", "stuck", -0.02)))

    def test_stop_during_backward_walk(self):
        self.assertIn("Main aath centimeter peeche chala.",
                      build_reply(parse("ruko"), walk("walk_backward", "stopped", 0.08)))

    def test_capped(self):
        reply = build_reply(parse("peeche ek meter"), walk("walk_backward", "done", 0.5))
        self.assertIn("pachaas centimeter tak hi", reply)


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
        self.assertIn("Gurdy sirf aage, peeche, baayen ya daayen chal sakta hai.",
                      build_reply(parse("upar chalo")))
        self.assertIn("samajh nahi aaya", build_reply(parse("namaste")))


class MakeCommandTest(unittest.TestCase):
    def test_side_walk_has_no_mode(self):
        cmd = make_command(parse("baayen tees cm chalo"), "rl")
        self.assertEqual((cmd["action"], cmd["distance_m"]), ("walk_left", 0.3))
        self.assertNotIn("mode", cmd)
        self.assertEqual(make_command(parse("daayen chalo"), "rl")["action"], "walk_right")

    def test_forward_and_unknown(self):
        self.assertEqual(make_command(parse("aage chalo"), "rl")["mode"], "rl")
        self.assertIsNone(make_command(parse("upar chalo"), "rl"))

    def test_backward_has_no_mode(self):
        cmd = make_command(parse("peeche das cm chalo"), "rl")
        self.assertEqual((cmd["action"], cmd["distance_m"]), ("walk_backward", 0.1))
        self.assertNotIn("mode", cmd)


P = "Main ek AI assistant hoon. "
RED_BALL = {"color": "red", "shape": "ball", "side": "center", "bearing_deg": 4.0}
BLUE_BOX = {"color": "blue", "shape": "box", "side": "left", "bearing_deg": 29.0}
GREEN_BOX = {"color": "green", "shape": "box", "side": "right", "bearing_deg": -29.0}


def seen(*objects):
    return {"action": "describe", "status": "ok", "objects": list(objects)}


def went(status, color="red", shape="ball", **extra):
    return dict(action="go_to_object", status=status, color=color, shape=shape, **extra)


class DescribeReplyTest(unittest.TestCase):
    def reply(self, feedback):
        return build_reply(parse("kya dikh raha hai"), feedback)

    def test_two_objects(self):
        self.assertEqual(self.reply(seen(BLUE_BOX, RED_BALL)), P +
                         "Mujhe saamne laal gend aur baayen taraf neela dabba dikh raha hai.")

    def test_three_objects_in_side_order(self):
        self.assertEqual(self.reply(seen(GREEN_BOX, BLUE_BOX, RED_BALL)), P +
                         "Mujhe saamne laal gend, baayen taraf neela dabba aur daayen taraf "
                         "hara dabba dikh raha hai.")

    def test_one_feminine_object(self):
        self.assertEqual(self.reply(seen(dict(RED_BALL, color="green", side="right"))),
                         P + "Mujhe daayen taraf hari gend dikh rahi hai.")

    def test_duplicates_mentioned_once(self):
        self.assertEqual(self.reply(seen(RED_BALL, dict(RED_BALL, side="left"))),
                         P + "Mujhe saamne laal gend dikh rahi hai.")

    def test_nothing(self):
        self.assertEqual(self.reply(seen()),
                         P + "Mujhe abhi saamne koi laal, neeli ya hari cheez nahi dikh rahi.")

    def test_no_camera(self):
        self.assertIn("camera se tasveer nahi", self.reply({"action": "describe",
                                                            "status": "no_camera"}))

    def test_command(self):
        self.assertEqual(make_command(parse("kya dikh raha hai"), "rl")["action"], "describe")


class GoToReplyTest(unittest.TestCase):
    def reply(self, feedback, text="laal gend ke paas jao"):
        return build_reply(parse(text), feedback)

    def test_reached(self):
        self.assertEqual(self.reply(went("reached")), P + "Main laal gend ke paas pahunch gaya.")
        self.assertEqual(self.reply(went("reached", "blue", "box")),
                         P + "Main neele dabbe ke paas pahunch gaya.")

    def test_not_found(self):
        self.assertEqual(self.reply(went("not_found")), P + "Mujhe laal gend nahi dikh rahi.")
        self.assertEqual(self.reply(went("not_found", "green", "box")),
                         P + "Mujhe hara dabba nahi dikh raha.")
        self.assertEqual(self.reply(went("not_found", "blue", None)),
                         P + "Mujhe neeli cheez nahi dikh rahi.")

    def test_lost_and_failures(self):
        self.assertEqual(self.reply(went("lost")),
                         P + "Laal gend nazar se kho gayi, isliye main ruk gaya.")
        self.assertEqual(self.reply(went("fallen", reset=True)), P +
                         "Main laal gend ki taraf jaate hue gir gaya. "
                         "Mujhe wapas khada kar diya gaya.")
        self.assertEqual(self.reply(went("stuck", "green", "box")),
                         P + "Main hare dabbe ki taraf aage nahi badh paaya.")
        self.assertEqual(self.reply(went("gave_up")),
                         P + "Main laal gend tak nahi pahunch paaya.")
        self.assertIn("Pehle use rukne", self.reply(went("busy")))
        self.assertIn("camera se tasveer nahi", self.reply(went("no_camera")))
        self.assertEqual(self.reply({"action": "go_to_object", "status": "error"}),
                         P + "Robot ne error diya.")

    def test_stopped_by_voice(self):
        self.assertEqual(self.reply(went("stopped", distance_m=0.1), text="ruko"),
                         P + "Main ruk gaya. Main laal gend ki taraf das centimeter chala.")

    def test_command(self):
        cmd = make_command(parse("neele dabbe tak jao"), "rl")
        self.assertEqual({k: cmd[k] for k in ("action", "color", "shape")},
                         {"action": "go_to_object", "color": "blue", "shape": "box"})
        self.assertNotIn("mode", cmd)
        self.assertIsNone(make_command(parse("laal hara"), "rl"))


if __name__ == "__main__":
    unittest.main()

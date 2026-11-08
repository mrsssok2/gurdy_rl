"""Tests for command_parser. Run from the repo root:
    python -m unittest discover -s voice_client/tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from command_parser import parse, tokenize  # noqa: E402


class WalkForwardTest(unittest.TestCase):
    def assertWalk(self, text, cm, defaulted=False, capped=False):
        result = parse(text)
        self.assertEqual(result["intent"], "walk_forward", (text, result))
        self.assertAlmostEqual(result["distance_cm"], cm, msg=text)
        self.assertAlmostEqual(result["distance_m"], cm / 100.0, msg=text)
        self.assertEqual(result["defaulted"], defaulted, text)
        self.assertEqual(result["capped"], capped, text)
        return result

    def test_real_transcript(self):
        # hi-IN transcript of "Gurdy, aage bees centimeter chalo"
        result = self.assertWalk("गरदी आगे बीस सेंटीमीटर चलो।", 20)
        self.assertTrue(result["wake_word"])

    def test_wake_word_variants(self):
        for wake in ("गरदी", "गर्डी", "गुर्डी", "Gurdy,", "gurdi"):
            result = self.assertWalk(wake + " आगे बीस सेंटीमीटर चलो।", 20)
            self.assertTrue(result["wake_word"], wake)

    def test_wake_word_optional(self):
        self.assertFalse(self.assertWalk("आगे बीस सेंटीमीटर चलो", 20)["wake_word"])

    def test_devanagari_numbers(self):
        self.assertWalk("आगे दस सेंटीमीटर चलो", 10)
        self.assertWalk("आगे तीस सेंटीमीटर चलो", 30)
        self.assertWalk("आगे पचास सेंटीमीटर चलो", 50)
        self.assertWalk("आगे २५ सेमी चलो", 25)

    def test_romanized(self):
        self.assertWalk("Gurdy, aage bees centimeter chalo", 20)
        self.assertWalk("aage das cm chalo", 10)
        self.assertWalk("aage tees centimeter", 30)
        self.assertWalk("gurdy aage pachaas cm chalo", 50)

    def test_english_and_digits(self):
        self.assertWalk("Gurdy move forward 30 cm", 30)
        self.assertWalk("forward twenty five centimeters", 25)
        self.assertWalk("aage 15cm chalo", 15)
        self.assertWalk("गरदी आगे 20 सेंटीमीटर चलो।", 20)

    def test_metres(self):
        # Converted, then capped at 50 cm
        self.assertWalk("aage aadha meter chalo", 50)
        self.assertWalk("आगे एक मीटर चलो", 50, capped=True)
        self.assertWalk("aage 0.3 m", 30)

    def test_default_and_cap(self):
        self.assertWalk("Gurdy aage chalo", 20, defaulted=True)
        self.assertWalk("आगे चलो", 20, defaulted=True)
        self.assertWalk("aage sau centimeter chalo", 50, capped=True)
        self.assertWalk("aage 80 cm", 50, capped=True)

    def test_number_without_forward_word(self):
        self.assertWalk("बीस सेंटीमीटर चलो", 20)

    def test_do_is_not_two_without_unit(self):
        self.assertWalk("aage chalo do", 20, defaulted=True)
        self.assertWalk("aage do meter chalo", 50, capped=True)


class OtherIntentTest(unittest.TestCase):
    def test_stop(self):
        for text in ("रुको", "गरदी रुक जाओ।", "ruko", "Gurdy stop!", "ruk jao"):
            self.assertEqual(parse(text)["intent"], "stop", text)

    def test_stop_beats_walk(self):
        self.assertEqual(parse("ruko, aage mat chalo")["intent"], "stop")

    def test_status(self):
        for text in ("गरदी कितना चला?", "kitna chala", "Gurdy status", "kitna aage chala",
                     "कितनी दूर चले"):
            self.assertEqual(parse(text)["intent"], "status", text)

    def test_mode(self):
        cases = {
            "RL mode": "rl", "Gurdy, R.L. mode": "rl", "आर एल मोड": "rl", "आरएल मोड": "rl",
            "normal mode": "scripted", "नॉर्मल मोड": "scripted", "गरदी नार्मल मोड।": "scripted",
        }
        for text, mode in cases.items():
            result = parse(text)
            self.assertEqual(result["intent"], "mode", text)
            self.assertEqual(result["mode"], mode, text)

    def test_unknown(self):
        for text in ("", "नमस्ते", "Gurdy", "hello how are you", "normal", "बीस"):
            self.assertEqual(parse(text)["intent"], "unknown", text)

    def test_text_is_kept(self):
        self.assertEqual(parse("रुको।")["text"], "रुको।")


class SidewaysTest(unittest.TestCase):
    def assertSide(self, text, intent, cm, defaulted=False, capped=False):
        result = parse(text)
        self.assertEqual(result["intent"], intent, (text, result))
        self.assertAlmostEqual(result["distance_cm"], cm, msg=text)
        self.assertAlmostEqual(result["distance_m"], cm / 100.0, msg=text)
        self.assertEqual((result["defaulted"], result["capped"]), (defaulted, capped), text)
        return result

    def test_left_words(self):
        for word in ("बाएं", "बाएँ", "baayen", "left", "Left"):
            self.assertSide("Gurdy %s bees centimeter chalo" % word, "walk_left", 20)

    def test_right_words(self):
        for word in ("दाएं", "दाएँ", "दायें", "daayen", "right"):
            self.assertSide("गरदी %s तीस सेंटीमीटर चलो।" % word, "walk_right", 30)

    def test_devanagari_sentence(self):
        result = self.assertSide("गरदी बाएं बीस सेंटीमीटर चलो।", "walk_left", 20)
        self.assertTrue(result["wake_word"])

    def test_distance_parsing_and_cap(self):
        self.assertSide("baayen twenty five cm chalo", "walk_left", 25)
        self.assertSide("दाएँ २० सेमी", "walk_right", 20)
        self.assertSide("daayen ek meter chalo", "walk_right", 50, capped=True)
        self.assertSide("baayen 0.3 m", "walk_left", 30)
        self.assertSide("baayen chalo", "walk_left", 20, defaulted=True)
        self.assertSide("right", "walk_right", 20, defaulted=True)

    def test_stop_and_status_win(self):
        self.assertEqual(parse("ruko, baayen mat jao")["intent"], "stop")
        self.assertEqual(parse("kitna baayen chala")["intent"], "status")


class BackwardTest(unittest.TestCase):
    def assertBack(self, text, cm, defaulted=False, capped=False):
        result = parse(text)
        self.assertEqual(result["intent"], "walk_backward", (text, result))
        self.assertAlmostEqual(result["distance_cm"], cm, msg=text)
        self.assertAlmostEqual(result["distance_m"], cm / 100.0, msg=text)
        self.assertEqual((result["defaulted"], result["capped"]), (defaulted, capped), text)
        return result

    def test_words(self):
        for word in ("पीछे", "peeche", "piche", "back", "backward", "reverse", "Back"):
            self.assertBack("Gurdy %s bees centimeter chalo" % word, 20)

    def test_devanagari_sentence(self):
        result = self.assertBack("गरदी पीछे बीस सेंटीमीटर चलो।", 20)
        self.assertTrue(result["wake_word"])

    def test_distance_parsing_and_cap(self):
        self.assertBack("go back twenty five cm", 25)
        self.assertBack("पीछे ३० सेमी", 30)
        self.assertBack("peeche ek meter chalo", 50, capped=True)
        self.assertBack("reverse 0.1 m", 10)
        self.assertBack("peeche chalo", 20, defaulted=True)

    def test_stop_reset_and_status_win(self):
        self.assertEqual(parse("ruko, peeche mat jao")["intent"], "stop")
        self.assertEqual(parse("kitna peeche chala")["intent"], "status")
        # "wapas" stays a reset, not a backward walk
        self.assertEqual(parse("Gurdy wapas")["intent"], "reset")


class UnsupportedDirectionTest(unittest.TestCase):
    def assertUnknown(self, text, word):
        result = parse(text)
        self.assertEqual(result["intent"], "unknown", (text, result))
        self.assertEqual(result["unsupported"], word, text)

    def test_turns(self):
        self.assertUnknown("baayen mudo", "mudo")
        self.assertUnknown("turn right", "turn")
        self.assertUnknown("गरदी घूमो", "घूमो")

    def test_up_down(self):
        self.assertUnknown("upar chalo", "upar")
        self.assertUnknown("neeche jao bees cm", "neeche")

    def test_two_directions(self):
        self.assertUnknown("aage baayen chalo", "aage baayen")
        self.assertUnknown("left right", "left right")
        self.assertUnknown("aage peeche chalo", "aage peeche")
        self.assertUnknown("peeche baayen", "peeche baayen")

    def test_plain_unknown_has_no_unsupported(self):
        self.assertNotIn("unsupported", parse("namaste"))

    def test_forward_without_direction_still_works(self):
        self.assertEqual(parse("bees cm chalo")["intent"], "walk_forward")


class TokenizeTest(unittest.TestCase):
    def test_keeps_vowel_signs(self):
        self.assertEqual(tokenize("गरदी आगे बीस सेंटीमीटर चलो।"),
                         ["गरदी", "आगे", "बीस", "सेंटीमीटर", "चलो"])

    def test_splits_glued_units(self):
        self.assertEqual(tokenize("20cm 0.5m २०सेमी"), ["20", "cm", "0.5", "m", "20", "सेमी"])



class ResetTest(unittest.TestCase):
    def test_reset_words(self):
        for text in ("Gurdy wapas", "reset", "वापस", "vapas aao", "Gurdy reset karo"):
            self.assertEqual(parse(text)["intent"], "reset", text)

    def test_reset_wins_over_stop(self):
        self.assertEqual(parse("ruko, reset")["intent"], "reset")


class VisionCommandTest(unittest.TestCase):
    def test_describe(self):
        for text in ("क्या दिख रहा है", "kya dikh raha hai?", "Gurdy, kya dikhai de raha hai",
                     "what do you see", "Gurdy what can you see", "तुम्हें क्या दिखता है",
                     "aage dekho", "laal gend dikh rahi hai kya"):
            self.assertEqual(parse(text)["intent"], "describe", text)

    def assertGoTo(self, text, color, shape):
        result = parse(text)
        self.assertEqual((result["intent"], result.get("color"), result.get("shape")),
                         ("go_to_object", color, shape), text)

    def test_go_to_object(self):
        self.assertGoTo("Gurdy, laal gend ke paas jao", "red", "ball")
        self.assertGoTo("लाल गेंद के पास जाओ", "red", "ball")
        self.assertGoTo("नीले डिब्बे तक जाओ", "blue", "box")
        self.assertGoTo("neele dabbe ke paas chalo", "blue", "box")
        self.assertGoTo("hare dabbe tak pahuncho", "green", "box")
        self.assertGoTo("हरी गेंद", "green", "ball")
        self.assertGoTo("go to the green box", "green", "box")
        self.assertGoTo("red ball", "red", "ball")
        self.assertGoTo("neeli wali cheez ke paas jao", "blue", None)
        self.assertGoTo("लाल के पास जाओ", "red", None)

    def test_go_word_wins_over_describe(self):
        self.assertGoTo("dekho aur laal gend ke paas jao", "red", "ball")

    def test_colour_alone_is_unknown(self):
        for text in ("laal", "नीला", "hara rang"):
            self.assertEqual(parse(text)["intent"], "unknown", text)

    def test_stop_and_status_still_win(self):
        self.assertEqual(parse("ruko, laal gend ke paas mat jao")["intent"], "stop")
        self.assertEqual(parse("kitna chala")["intent"], "status")

    def test_walks_unchanged(self):
        self.assertEqual(parse("aage bees cm chalo")["intent"], "walk_forward")
        self.assertEqual(parse("baayen das cm chalo")["intent"], "walk_left")
        self.assertEqual(parse("upar chalo")["intent"], "unknown")


if __name__ == "__main__":
    unittest.main()

"""Offline tests for vision.py: synthetic images and a saved Gazebo frame.
Needs OpenCV (cv2) and numpy; skipped without cv2. Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import vision
except ImportError:  # no OpenCV here
    cv2 = None

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GREY = (150, 150, 150)
RED, GREEN, BLUE = (0, 0, 220), (0, 200, 0), (200, 0, 0)  # BGR


def canvas(width=800, height=600):
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = GREY
    return img


def noisy(img, sigma=4.0, seed=0):
    rng = np.random.RandomState(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def only(objects, color):
    found = [o for o in objects if o["color"] == color]
    assert len(found) == 1, found
    return found[0]


@unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed")
class TestCamera(unittest.TestCase):
    def test_focal_length(self):
        cam = vision.Camera()
        self.assertAlmostEqual(cam.f, 400 / math.tan(1.3962634 / 2), places=6)

    def test_bearing_sign(self):
        cam = vision.Camera()
        self.assertAlmostEqual(cam.bearing_deg(400), 0.0)
        self.assertGreater(cam.bearing_deg(100), 0)   # left of centre is +
        self.assertLess(cam.bearing_deg(700), 0)
        self.assertAlmostEqual(cam.bearing_deg(0), 40.0, places=3)  # half the FOV

    def test_ground_distance(self):
        cam = vision.Camera(tilt_deg=12.0, height_m=0.128)
        # The centre pixel looks 12 deg down: hits the ground at h / tan(12)
        self.assertAlmostEqual(cam.ground_distance(400, 300), 0.128 / math.tan(math.radians(12)),
                               places=6)
        # At and above the horizon: no ground hit
        self.assertIsNone(cam.ground_distance(400, 0))
        horizon_v = 300 - cam.f * math.tan(math.radians(12))
        self.assertIsNone(cam.ground_distance(400, horizon_v))
        # Lower in the image is nearer
        self.assertLess(cam.ground_distance(400, 550), cam.ground_distance(400, 400))

    def test_size_distance(self):
        cam = vision.Camera()
        self.assertAlmostEqual(cam.size_distance(cam.f * 0.06 / 0.3, 0.06), 0.3)
        self.assertIsNone(cam.size_distance(0, 0.06))


@unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed")
class TestDetectSynthetic(unittest.TestCase):
    def test_empty_image(self):
        self.assertEqual(vision.detect(noisy(canvas())), [])

    def test_three_objects(self):
        img = canvas()
        cv2.circle(img, (400, 350), 45, RED, -1)
        cv2.rectangle(img, (60, 250), (200, 360), BLUE, -1)
        cv2.rectangle(img, (600, 250), (740, 350), GREEN, -1)
        objects = vision.detect(noisy(img))
        self.assertEqual(len(objects), 3)
        red, blue, green = only(objects, "red"), only(objects, "blue"), only(objects, "green")
        self.assertEqual(red["shape"], "ball")
        self.assertEqual(blue["shape"], "box")
        self.assertEqual(green["shape"], "box")
        self.assertEqual(red["side"], "center")
        self.assertFalse(red["clipped"] or red["clipped_bottom"])
        self.assertEqual(blue["side"], "left")
        self.assertEqual(green["side"], "right")
        self.assertAlmostEqual(red["bearing_deg"], 0.0, delta=0.5)
        self.assertGreater(blue["bearing_deg"], 20)
        self.assertLess(green["bearing_deg"], -20)
        self.assertEqual(red["centroid"], [400, 350])
        self.assertAlmostEqual(red["area_fraction"], math.pi * 45 ** 2 / 480000.0, delta=0.0005)
        # Largest first
        areas = [o["area_fraction"] for o in objects]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_dark_and_light_shades(self):
        """A shaded ball (bright top, dark bottom, as Gazebo renders it) is one blob."""
        img = canvas()
        for r, shade in ((60, 90), (45, 160), (25, 240)):
            cv2.circle(img, (300, 300 + (60 - r) // 2), r, (0, 0, shade), -1)
        objects = vision.detect(noisy(img))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["shape"], "ball")

    def test_red_hue_wraps(self):
        """Red on both sides of hue 0 (orange-red and pink-red) is red."""
        img = canvas()
        cv2.circle(img, (200, 300), 40, (0, 30, 220), -1)   # hue ~4
        cv2.circle(img, (600, 300), 40, (40, 0, 220), -1)   # hue ~175
        objects = vision.detect(img)
        self.assertEqual([o["color"] for o in objects], ["red", "red"])

    def test_thin_line_and_specks_ignored(self):
        """Gazebo's red world-x line and small specks are not objects."""
        img = canvas()
        cv2.line(img, (400, 200), (420, 599), RED, 2)
        cv2.circle(img, (100, 500), 3, BLUE, -1)
        self.assertEqual(vision.detect(img), [])

    def test_grey_and_white_ignored(self):
        img = canvas()
        cv2.rectangle(img, (100, 100), (300, 300), (250, 250, 250), -1)
        cv2.rectangle(img, (400, 100), (600, 300), (30, 30, 30), -1)
        self.assertEqual(vision.detect(img), [])

    def test_clipped(self):
        img = canvas()
        cv2.circle(img, (400, 560), 80, RED, -1)
        obj = vision.detect(noisy(img))[0]
        self.assertTrue(obj["clipped"])
        self.assertTrue(obj["clipped_bottom"])
        self.assertEqual(obj["shape"], "ball")
        # Cut off at the bottom: no ground point, so the size-based distance
        self.assertAlmostEqual(obj["distance_m"],
                               vision.Camera().size_distance(obj["bbox"][2], 0.06), places=3)

    def test_clipped_shapes(self):
        """Close objects cut off by the border: half a ball is still a ball."""
        for (center, radius) in (((400, 600), 120), ((790, 300), 100), ((400, 640), 150)):
            img = canvas()
            cv2.circle(img, center, radius, RED, -1)
            obj = vision.detect(noisy(img))[0]
            self.assertTrue(obj["clipped"])
            self.assertEqual(obj["shape"], "ball", (center, radius))
        for pts in ([(250, 380), (560, 380), (600, 599), (210, 599)],     # trapezoid
                    [(300, 300), (500, 280), (520, 599), (280, 599)],
                    [(600, 200), (799, 200), (799, 420), (600, 420)]):    # cut at the side
            img = canvas()
            cv2.fillPoly(img, [np.array(pts, np.int32)], BLUE)
            obj = vision.detect(noisy(img))[0]
            self.assertTrue(obj["clipped"])
            self.assertEqual(obj["shape"], "box", pts)

    def test_ground_distance_from_bottom(self):
        img = canvas()
        cv2.circle(img, (400, 350), 45, RED, -1)
        obj = vision.detect(img)[0]
        cam = vision.Camera()
        x, y, w, h = obj["bbox"]
        self.assertAlmostEqual(obj["distance_m"], cam.ground_distance(x + w / 2.0, y + h),
                               places=3)

    def test_custom_ranges(self):
        """A yellow range finds a yellow blob the defaults don't."""
        img = canvas()
        cv2.circle(img, (400, 300), 40, (0, 220, 220), -1)  # yellow, hue 30
        self.assertEqual(vision.detect(img), [])
        ranges = {"yellow": [((20, 100, 40), (35, 255, 255))]}
        objects = vision.detect(img, ranges=ranges)
        self.assertEqual([o["color"] for o in objects], ["yellow"])

    def test_other_image_size(self):
        img = canvas(400, 300)
        cv2.circle(img, (100, 150), 20, RED, -1)
        obj = vision.detect(img)[0]
        # Same FOV, so the same angle as at (200, 300) in 800x600
        self.assertAlmostEqual(obj["bearing_deg"], vision.Camera().bearing_deg(200), delta=0.2)

    def test_annotate(self):
        img = canvas()
        cv2.circle(img, (400, 350), 45, RED, -1)
        objects = vision.detect(img)
        out = vision.annotate(img, objects)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))
        self.assertEqual(vision.label(objects[0])[:9], "red ball ")


@unittest.skipIf(cv2 is None, "OpenCV (cv2) is not installed")
class TestGazeboFrame(unittest.TestCase):
    """
    A real frame from the forward camera in worlds/gurdy_demo.world, robot
    standing at the spawn point (0.5, 0) facing +x. Camera at x 0.555 m.
    """

    def setUp(self):
        self.img = cv2.imread(os.path.join(DATA, "gazebo_demo_frame.jpg"))
        self.assertIsNotNone(self.img)
        self.objects = vision.detect(self.img)

    def test_finds_the_three_objects(self):
        self.assertEqual(sorted((o["color"], o["shape"]) for o in self.objects),
                         [("blue", "box"), ("green", "box"), ("red", "ball")])

    def test_ball(self):
        ball = only(self.objects, "red")
        # Ball at (0.85, 0.03): 0.295 m ahead of the camera, 3 cm left
        self.assertAlmostEqual(ball["bearing_deg"], math.degrees(math.atan2(0.03, 0.295)), delta=1.0)
        self.assertAlmostEqual(ball["distance_m"], 0.296, delta=0.03)
        self.assertEqual(ball["side"], "center")

    def test_boxes(self):
        blue, green = only(self.objects, "blue"), only(self.objects, "green")
        # Box centres (0.95, 0.22) and (1.0, -0.25)
        self.assertAlmostEqual(blue["bearing_deg"], math.degrees(math.atan2(0.22, 0.395)), delta=2.0)
        self.assertAlmostEqual(green["bearing_deg"], -math.degrees(math.atan2(0.25, 0.445)),
                               delta=2.0)
        self.assertEqual((blue["side"], green["side"]), ("left", "right"))
        # Nearest bottom corners: (0.91, 0.18) and (0.96, -0.21)
        self.assertAlmostEqual(blue["distance_m"], math.hypot(0.355, 0.18), delta=0.04)
        self.assertAlmostEqual(green["distance_m"], math.hypot(0.405, 0.21), delta=0.04)


if __name__ == "__main__":
    unittest.main()

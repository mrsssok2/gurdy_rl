"""Offline tests for visual_approach.py, with a fake robot and a fake camera
that projects world objects the way the head camera would (no ROS, no
OpenCV). Run from the repo root:
    python -m unittest discover -s scripts/tests
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visual_approach as va  # noqa: E402
from voice_walk import ScriptedGait  # noqa: E402

CAMERA_X = 0.055         # camera ahead of the base, m
FOCAL_PX = 400 / math.tan(1.3962634 / 2)
IMAGE_PX = 800 * 600
HALF_FOV_DEG = 40.0
# Per-step progress of each gait in the fake, m (as measured in Gazebo)
STEP_M = {"walk_forward": (0.0017, 0.0), "walk_left": (0.0, 0.0011),
          "walk_right": (0.0, -0.0011)}
CYCLE = {"walk_forward": 16, "walk_left": 24, "walk_right": 24}


def make_gaits():
    return {name: ScriptedGait([(name, i) for i in range(n)]) for name, n in CYCLE.items()}


class FakeRobot(object):
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y
        self.actions = []
        self.fall_at = None
        self.frozen = False   # True: steps make no progress (stuck)

    def forward(self):
        return self.x

    def lateral(self):
        return self.y

    def step(self, action):
        self.actions.append(action)
        if not self.frozen:
            dx, dy = STEP_M[action[0]]
            self.x += dx
            self.y += dy
        fallen = self.fall_at is not None and len(self.actions) >= self.fall_at
        return fallen, ("roll 70 deg" if fallen else None)


class FakeCamera(object):
    """
    look(): detections of the world objects [(color, shape, x, y, size_m)]
    seen from the robot. hide_after: object indices invisible after that
    many looks.
    """

    def __init__(self, robot, objects, hide_after=None):
        self.robot, self.objects = robot, objects
        self.hide_after = hide_after
        self.looks = 0

    def __call__(self):
        self.looks += 1
        found = []
        for color, shape, ox, oy, size in self.objects:
            if self.hide_after is not None and self.looks > self.hide_after:
                continue
            dx, dy = ox - (self.robot.x + CAMERA_X), oy - self.robot.y
            d = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dy, dx))
            if dx <= 0.02 or abs(bearing) > HALF_FOV_DEG:
                continue
            width_px = FOCAL_PX * size / d
            area = (math.pi / 4 if shape == "ball" else 1.3) * width_px ** 2 / IMAGE_PX
            # The bottom image row sees the ground 0.132 m ahead
            clipped_bottom = dx < 0.132
            found.append({"color": color, "shape": shape, "bearing_deg": round(bearing, 1),
                          "area_fraction": round(area, 4), "distance_m": round(d, 3),
                          "clipped": clipped_bottom, "clipped_bottom": clipped_bottom,
                          "side": "center"})
        return {"stamp": 0.0, "objects": found}


BALL = ("red", "ball", 0.35, 0.03, 0.06)
BLUE_BOX = ("blue", "box", 0.45, 0.22, 0.08)
GREEN_BOX = ("green", "box", 0.50, -0.25, 0.08)


def approach(objects, color, shape=None, robot=None, camera=None, **kwargs):
    robot = robot or FakeRobot()
    camera = camera or FakeCamera(robot, objects)
    log = []
    result = va.run_approach(robot, make_gaits(), camera, color, shape,
                             log=log.append, **kwargs)
    return result, robot, log


class TestFindTarget(unittest.TestCase):
    OBJECTS = [
        {"color": "red", "shape": "ball", "area_fraction": 0.01, "clipped": False},
        {"color": "red", "shape": "box", "area_fraction": 0.03, "clipped": False},
        {"color": "blue", "shape": "box", "area_fraction": 0.05, "clipped": True,
         "clipped_bottom": True},
        {"color": "green", "shape": "box", "area_fraction": 0.02, "clipped": True,
         "clipped_bottom": False},
    ]

    def test_largest_of_colour(self):
        self.assertEqual(va.find_target(self.OBJECTS, "red")["shape"], "box")

    def test_shape(self):
        self.assertEqual(va.find_target(self.OBJECTS, "red", "ball")["shape"], "ball")
        self.assertIsNone(va.find_target(self.OBJECTS, "purple"))
        self.assertIsNone(va.find_target([], "red"))
        self.assertIsNone(va.find_target(None, "red"))

    def test_clipped_at_bottom_counts_for_either_shape(self):
        self.assertEqual(va.find_target(self.OBJECTS, "blue", "ball")["color"], "blue")
        unclipped = [dict(self.OBJECTS[2], clipped=False, clipped_bottom=False)]
        self.assertIsNone(va.find_target(unclipped, "blue", "ball"))

    def test_clipped_at_side_keeps_its_shape(self):
        """A green box cut off at the edge of the view is not a green ball."""
        self.assertIsNone(va.find_target(self.OBJECTS, "green", "ball"))
        self.assertEqual(va.find_target(self.OBJECTS, "green", "box")["color"], "green")

    def test_side_cycles(self):
        self.assertEqual(va.side_cycles({"bearing_deg": 10, "distance_m": None}), 1)
        self.assertEqual(va.side_cycles({"bearing_deg": 10, "distance_m": 0.1}), 1)
        # 0.5 m at 30 deg: 25 cm sideways, capped at 3 cycles
        self.assertEqual(va.side_cycles({"bearing_deg": -30, "distance_m": 0.5}), 3)
        # ~5.2 cm: 2 cycles
        self.assertEqual(va.side_cycles({"bearing_deg": 15, "distance_m": 0.2}), 2)


class TestApproach(unittest.TestCase):
    def assert_whole_cycles(self, robot):
        """Every move is whole cycles from phase 0 of one gait."""
        i = 0
        while i < len(robot.actions):
            name = robot.actions[i][0]
            n = CYCLE[name]
            block = robot.actions[i:i + n]
            self.assertEqual(block, [(name, k) for k in range(n)])
            i += n

    def test_ball_straight_ahead(self):
        result, robot, _ = approach([BALL, BLUE_BOX, GREEN_BOX], "red", "ball")
        self.assertEqual(result["status"], va.REACHED)
        self.assertEqual((result["color"], result["shape"]), ("red", "ball"))
        # Stopped short of the ball, at the close area
        d = math.hypot(BALL[2] - robot.x - CAMERA_X, BALL[3] - robot.y)
        self.assertTrue(0.10 < d < 0.2, d)
        self.assertGreaterEqual(result["target"]["area_fraction"], va.CLOSE_AREA["ball"])
        self.assertAlmostEqual(result["forward_m"], robot.x, places=4)
        self.assertGreater(result["forward_m"], 0.1)
        self.assertEqual(result["steps"], len(robot.actions))
        self.assert_whole_cycles(robot)

    def test_box_to_the_left(self):
        result, robot, log = approach([BALL, BLUE_BOX, GREEN_BOX], "blue", "box")
        self.assertEqual(result["status"], va.REACHED)
        # Lined up: stepped left, then walked forward
        self.assertEqual(robot.actions[0][0], "walk_left")
        self.assertGreater(result["left_m"], 0.1)
        self.assertGreater(result["forward_m"], 0.05)
        self.assertLessEqual(abs(result["target"]["bearing_deg"]), 40)
        self.assertTrue(any("walk_forward" in line for line in log))
        self.assert_whole_cycles(robot)

    def test_box_to_the_right_no_shape_given(self):
        result, robot, _ = approach([BALL, BLUE_BOX, GREEN_BOX], "green")
        self.assertEqual(result["status"], va.REACHED)
        self.assertEqual(robot.actions[0][0], "walk_right")
        self.assertLess(result["left_m"], -0.1)
        # The shape comes from what was seen
        self.assertEqual(result["shape"], "box")
        self.assertEqual(result["distance_m"],
                         round(abs(result["forward_m"]) + abs(result["left_m"]), 4))

    def test_already_close(self):
        robot = FakeRobot(x=0.2)
        result, robot, _ = approach([BALL], "red", robot=robot)
        self.assertEqual((result["status"], result["moves"], result["steps"]),
                         (va.REACHED, 0, 0))

    def test_not_found(self):
        camera_robot = FakeRobot()
        camera = FakeCamera(camera_robot, [BALL])
        result, robot, _ = approach(None, "green", robot=camera_robot, camera=camera)
        self.assertEqual(result["status"], va.NOT_FOUND)
        self.assertEqual((result["steps"], camera.looks), (0, va.LOST_CHECKS))
        self.assertIsNone(result["target"])
        self.assertIsNone(result["shape"])

    def test_wrong_shape_not_found(self):
        result, _, _ = approach([BALL], "red", "box")
        self.assertEqual(result["status"], va.NOT_FOUND)
        self.assertEqual(result["shape"], "box")

    def test_lost(self):
        robot = FakeRobot()
        camera = FakeCamera(robot, [BALL], hide_after=2)
        result, _, _ = approach(None, "red", robot=robot, camera=camera)
        self.assertEqual(result["status"], va.LOST)
        self.assertEqual(result["moves"], 2)
        self.assertEqual(camera.looks, 2 + va.LOST_CHECKS)
        self.assertEqual(result["target"]["color"], "red")

    def test_lost_checks_config(self):
        robot = FakeRobot()
        camera = FakeCamera(robot, [BALL], hide_after=1)
        result, _, _ = approach(None, "red", robot=robot, camera=camera,
                                config=va.Config(lost_checks=5))
        self.assertEqual((result["status"], camera.looks), (va.LOST, 6))

    def test_passes_out_of_view_sideways(self):
        """An object beside the path leaves the 80 deg view: lost, not a crash."""
        robot = FakeRobot()
        camera = FakeCamera(robot, [("red", "ball", 0.06, 0.08, 0.06)])
        result, _, _ = approach(None, "red", robot=robot, camera=camera)
        self.assertIn(result["status"], (va.NOT_FOUND, va.LOST))

    def test_stop(self):
        robot = FakeRobot()
        result, robot, _ = approach([BALL], "red", robot=robot,
                                    should_stop=lambda: len(robot.actions) >= 20)
        self.assertEqual(result["status"], va.STOPPED)
        self.assertEqual(result["steps"], 20)

    def test_stop_before_start(self):
        result, robot, _ = approach([BALL], "red", should_stop=lambda: True)
        self.assertEqual((result["status"], result["steps"]), (va.STOPPED, 0))

    def test_fall(self):
        robot = FakeRobot()
        robot.fall_at = 7
        result, _, _ = approach([BALL], "red", robot=robot)
        self.assertEqual((result["status"], result["steps"]), (va.FALLEN, 7))
        self.assertEqual(result["fall_reason"], "roll 70 deg")

    def test_stuck(self):
        robot = FakeRobot()
        robot.frozen = True
        result, _, _ = approach([BALL], "red", robot=robot)
        self.assertEqual((result["status"], result["moves"]), (va.STUCK, va.STUCK_MOVES))

    def test_no_camera(self):
        result = va.run_approach(FakeRobot(), make_gaits(), lambda: None, "red")
        self.assertEqual((result["status"], result["steps"]), (va.NO_CAMERA, 0))

    def test_gave_up(self):
        result, _, _ = approach([BLUE_BOX], "blue", config=va.Config(max_moves=2))
        self.assertEqual((result["status"], result["moves"]), (va.GAVE_UP, 2))
        result, _, _ = approach([BLUE_BOX], "blue", config=va.Config(max_steps=30))
        self.assertEqual(result["status"], va.GAVE_UP)
        self.assertLess(result["steps"], 30 + 3 * 24)

    def test_close_area_config(self):
        near, _, _ = approach([BALL], "red", config=va.Config(close_area={"ball": 0.12}))
        far, _, _ = approach([BALL], "red", config=va.Config(close_area={"ball": 0.03}))
        self.assertGreater(near["forward_m"], far["forward_m"])

    def test_close_but_off_centre_side_steps_first(self):
        """Close enough at the start but ~17 deg left: not reached until centred."""
        ball = ("red", "ball", 0.35, 0.05, 0.06)
        robot = FakeRobot(x=0.13)
        first = FakeCamera(robot, [ball])()["objects"][0]
        self.assertGreaterEqual(first["area_fraction"], va.CLOSE_AREA["ball"])
        self.assertGreater(first["bearing_deg"], va.BEARING_TOL_DEG)
        result, robot, _ = approach([ball], "red", "ball", robot=robot)
        self.assertEqual(result["status"], va.REACHED)
        self.assertEqual(robot.actions[0][0], "walk_left")
        self.assertGreater(result["moves"], 0)
        self.assertLessEqual(abs(result["target"]["bearing_deg"]), va.BEARING_TOL_DEG)

    def test_reached_is_centred_unless_cut_off(self):
        """With a threshold met from the start, every reached result is centred."""
        for ball in (("red", "ball", 0.4, 0.15, 0.06), ("red", "ball", 0.4, -0.12, 0.06)):
            result, robot, _ = approach([ball], "red",
                                        config=va.Config(close_area={"ball": 0.001}))
            self.assertEqual(result["status"], va.REACHED, ball)
            self.assertGreater(result["moves"], 0, ball)
            self.assertTrue(all(a[0] != "walk_forward" for a in robot.actions), ball)
            self.assertLessEqual(abs(result["target"]["bearing_deg"]), va.BEARING_TOL_DEG)

    def test_cut_off_at_the_bottom_is_reached(self):
        """Even with an area threshold it can't reach, a target cut off by the
        bottom of the image (closer than ~0.13 m) is reached."""
        result, robot, _ = approach([BALL], "red", config=va.Config(close_area={"ball": 0.9}))
        self.assertEqual(result["status"], va.REACHED)
        self.assertTrue(result["target"]["clipped_bottom"])
        self.assertGreater(BALL[2] - robot.x - CAMERA_X, 0.1)

    def test_bearing_tolerance(self):
        """
        The ball 3 cm left is 6 deg off centre at the start and ~11 deg when
        close: a 20 deg tolerance never side-steps, a 3 deg one does.
        """
        _, robot, _ = approach([("red", "ball", 0.35, 0.03, 0.06)], "red",
                               config=va.Config(bearing_tol_deg=20))
        self.assertTrue(all(a[0] == "walk_forward" for a in robot.actions))
        _, robot, _ = approach([("red", "ball", 0.35, 0.03, 0.06)], "red",
                               config=va.Config(bearing_tol_deg=3))
        self.assertIn("walk_left", {a[0] for a in robot.actions})

    def test_far_target_takes_bigger_steps(self):
        _, robot, log = approach([("red", "ball", 0.6, 0.0, 0.06)], "red")
        self.assertIn("walk_forward x2", log[0])
        self.assertIn("walk_forward x1", log[-1])


if __name__ == "__main__":
    unittest.main()

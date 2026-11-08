"""
Closed-loop visual approach for the voice command "go_to_object", without
ROS, so it can be unit-tested offline (scripts/tests/test_visual_approach.py).
voice_command_node.py runs it with the walk env and vision_node.py's
/gurdy/detections.

Gurdy cannot turn, so it lines up with the target by stepping sideways:

    loop:
        look      wait for a camera frame taken after the last move
        target    the largest detection of the colour (and shape, if asked)
                  not seen: count a miss; after LOST_CHECKS misses in a row
                  stop (NOT_FOUND if it was never seen, else LOST)
        reached?  close (area_fraction >= close_area[shape]) AND centred
                  (|bearing| <= bearing_tol_deg), or cut off by the bottom
                  of the image (clipped_bottom: the bottom row sees the
                  ground ~0.13 m ahead, so it is closer than that)
                  -> REACHED. Close but off-centre falls through to a
                  side-step.
        |bearing| > bearing_tol_deg: side-step toward it (left if the
                  bearing is positive), 1-3 side-step cycles (~2.6 cm each)
                  depending on the estimated sideways offset
        else      walk forward 1 cycle (~2.7 cm), 2 while the target is
                  still small (far)

Each move is whole gait cycles started from phase 0, so all legs are down
when the next move (maybe another gait) starts. Before every step it checks
should_stop() (STOPPED). A fall ends it (FALLEN), STUCK_MOVES moves in a row
that each gain less than MOVE_MIN_PROGRESS end it (STUCK), and MAX_MOVES
moves or MAX_STEPS steps end it (GAVE_UP).
"""
import math

# Result statuses (STOPPED and FALLEN as in voice_walk)
REACHED = "reached"
NOT_FOUND = "not_found"    # never seen
LOST = "lost"              # seen, then missing for LOST_CHECKS frames in a row
STOPPED = "stopped"
FALLEN = "fallen"
STUCK = "stuck"
GAVE_UP = "gave_up"        # MAX_MOVES / MAX_STEPS without getting close
NO_CAMERA = "no_camera"    # no detections arrive (vision_node not running?)

COLORS = ("red", "green", "blue")
SHAPES = ("ball", "box")

# Off-centre beyond this: side-step instead of walking forward
BEARING_TOL_DEG = 8.0
# Close enough when the target covers this fraction of the image. The box
# (8 cm) is bigger than the ball (6 cm), so it needs a larger fraction for
# the same distance. ~0.04 is a 6 cm ball ~0.18 m from the camera. Keep it
# below what the object can reach before the bottom of the image cuts it
# off (from ~0.15 m): in Gazebo the ball peaked at ~0.056, so 0.06 was
# never reached and Gurdy walked into it.
CLOSE_AREA = {"ball": 0.04, "box": 0.09}
# Misses in a row before giving up on the target
LOST_CHECKS = 3
# Gait progress per cycle, m: --wave-all ~16 x 1.7 mm, side-step ~24 x 1.1 mm
FORWARD_M_PER_CYCLE = 16 * 0.0017
SIDE_M_PER_CYCLE = 24 * 0.0011
MAX_SIDE_CYCLES = 3
# Forward cycles per move while the target covers less than
# FAR_AREA_RATIO x its close area (still far away), else 1
FAR_FORWARD_CYCLES = 2
FAR_AREA_RATIO = 0.25
# A move gaining less than this along its direction made no progress
MOVE_MIN_PROGRESS = 0.003
STUCK_MOVES = 3
MAX_MOVES = 40
MAX_STEPS = 800


def find_target(objects, color, shape=None):
    """
    The largest detection of color (and shape, if given), or None. A blob
    cut off by the bottom of the image also counts for either shape: the
    object is right in front, and its shape guess is less reliable. One cut
    off at the side does not ("hari gend" must not pick a green box at the
    edge of the view).
    """
    candidates = [o for o in objects or () if o.get("color") == color]
    if shape:
        matching = [o for o in candidates if o.get("shape") == shape]
        candidates = matching or [o for o in candidates if o.get("clipped_bottom")]
    if not candidates:
        return None
    return max(candidates, key=lambda o: o.get("area_fraction", 0.0))


def side_cycles(target):
    """Side-step cycles toward target: its sideways offset / ~2.6 cm, 1..3."""
    distance = target.get("distance_m")
    if distance is None:
        return 1
    offset = abs(distance * math.sin(math.radians(target["bearing_deg"])))
    return max(1, min(MAX_SIDE_CYCLES, int(round(offset / SIDE_M_PER_CYCLE))))


class Config(object):
    def __init__(self, bearing_tol_deg=BEARING_TOL_DEG, close_area=None,
                 lost_checks=LOST_CHECKS, max_moves=MAX_MOVES, max_steps=MAX_STEPS):
        self.bearing_tol_deg = bearing_tol_deg
        self.close_area = dict(CLOSE_AREA)
        self.close_area.update(close_area or {})
        self.lost_checks = lost_checks
        self.max_moves = max_moves
        self.max_steps = max_steps

    def close_area_for(self, shape):
        return self.close_area.get(shape, min(self.close_area.values()))


def _summary(target):
    keys = ("color", "shape", "bearing_deg", "distance_m", "area_fraction", "side",
            "clipped_bottom")
    return {k: target.get(k) for k in keys} if target else None


def run_approach(robot, gaits, look, color, shape=None, config=None,
                 should_stop=lambda: False, log=lambda msg: None):
    """
    Walk up to the color (shape) object. See the module doc.

    robot: step(action) -> (fallen, fall_reason); forward() and lateral()
        -> position in m along +x and +y (left)
    gaits: {"walk_forward" | "walk_left" | "walk_right": voice_walk.ScriptedGait}
    look: () -> detections dict ({"objects": [...]}) from a frame taken after
        the last move, or None if none arrived in time
    Returns a result dict: status, color, shape (asked for, else as seen),
    moves, steps, forward_m, left_m, distance_m (path length), target (the
    last sighting), and fall_reason after a fall.
    """
    config = config or Config()
    start_fwd, start_lat = robot.forward(), robot.lateral()
    result = {"color": color, "shape": shape, "moves": 0, "steps": 0}
    seen = False
    misses = 0
    no_progress = 0
    last = None
    status = GAVE_UP

    while result["moves"] < config.max_moves and result["steps"] < config.max_steps:
        if should_stop():
            status = STOPPED
            break
        detections = look()
        if detections is None:
            status = NO_CAMERA
            break
        target = find_target(detections.get("objects"), color, shape)
        if target is None:
            misses += 1
            log("no %s %s in view (%d/%d)" % (color, shape or "object", misses,
                                               config.lost_checks))
            if misses >= config.lost_checks:
                status = LOST if seen else NOT_FOUND
                break
            continue
        seen, misses, last = True, 0, target
        close_area = config.close_area_for(shape or target["shape"])
        bearing = target["bearing_deg"]
        centred = abs(bearing) <= config.bearing_tol_deg
        close = target["area_fraction"] >= close_area
        if (close and centred) or target.get("clipped_bottom"):
            status = REACHED
            break

        # Off-centre (close or not): side-step toward it first
        if not centred:
            action = "walk_left" if bearing > 0 else "walk_right"
            cycles = side_cycles(target)
            position = robot.lateral if action == "walk_left" else (lambda: -robot.lateral())
        else:
            action = "walk_forward"
            far = target["area_fraction"] < FAR_AREA_RATIO * close_area
            cycles = FAR_FORWARD_CYCLES if far else 1
            position = robot.forward
        log("%s %s: bearing %+.1f deg, area %.3f -> %s x%d" % (
            color, target["shape"], bearing, target["area_fraction"], action, cycles))

        gait = gaits[action]
        gait.restart()
        before = position()
        fallen = stopped = False
        for _ in range(cycles * len(gait.cycle)):
            if should_stop():
                stopped = True
                break
            fallen, reason = robot.step(gait.next_action())
            result["steps"] += 1
            if fallen:
                result["fall_reason"] = reason
                break
        result["moves"] += 1
        if fallen:
            status = FALLEN
            break
        if stopped:
            status = STOPPED
            break
        if position() - before < MOVE_MIN_PROGRESS:
            no_progress += 1
            if no_progress >= STUCK_MOVES:
                status = STUCK
                break
        else:
            no_progress = 0

    forward_m = robot.forward() - start_fwd
    left_m = robot.lateral() - start_lat
    result.update(status=status, forward_m=round(forward_m, 4), left_m=round(left_m, 4),
                  distance_m=round(abs(forward_m) + abs(left_m), 4), target=_summary(last))
    if result["shape"] is None and last is not None:
        result["shape"] = last["shape"]
    return result

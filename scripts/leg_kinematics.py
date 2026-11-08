#!/usr/bin/env python3
"""
Planar 2-link kinematics for one Gurdy leg.

Each leg's hip and knee both turn about the leg's local y axis, so the foot
moves only in the leg's radial vertical plane (its "spoke"). In that plane:
    r  reach: horizontal distance of the foot centre from the hip, positive
       outward along the spoke, in metres
    h  height of the foot centre relative to the hip, positive up (a foot
       below the hip has negative h), in metres

Joint conventions, as in env.py (worked out from robot/gurdy.xacro):
    hip  (upper leg, limits -1.55..0): 0 points the thigh straight down;
         negative swings it outward and up. Thigh angle from vertical,
         outward positive: a = -hip.
    knee (lower leg, limits -2.9..0): shin angle from vertical, outward
         positive: b = knee - hip.
So the foot is at
    r = THIGH * sin(a) + SHIN * sin(b)
    h = -(THIGH * cos(a) + SHIN * cos(b))

IK picks the knee-out branch (thigh further outward than the hip-foot line,
shin folded back inward), which is how the robot stands. The other branch
would need hip > 0 for any foot below and outside the hip.

Run this file to print the neutral reference and a few sample solutions.
"""
import math

# Leg geometry from robot/gurdy.xacro (hip to knee, knee to foot centre)
THIGH_LENGTH = 0.06
SHIN_LENGTH = 0.06

# Joint limits from robot/gurdy.xacro, radians
HIP_LIMITS = (-1.55, 0.0)
KNEE_LIMITS = (-2.9, 0.0)

# Reference pose: the crouch the robot holds in scripts/test_pose.py (see the
# note above NEUTRAL_POSE in env.py). Its foot position is the reference the
# gait foot poses are placed around.
NEUTRAL_HIP = -1.50
NEUTRAL_KNEE = -2.25


def foot_position(hip, knee):
    """
    Forward kinematics: foot (r, h) relative to the hip for the given joint
    angles. See the module docstring for the conventions.
    """
    a = -hip
    b = knee - hip
    r = THIGH_LENGTH * math.sin(a) + SHIN_LENGTH * math.sin(b)
    h = -(THIGH_LENGTH * math.cos(a) + SHIN_LENGTH * math.cos(b))
    return r, h


# Foot position of the neutral pose, (r, h) in metres
NEUTRAL_FOOT = foot_position(NEUTRAL_HIP, NEUTRAL_KNEE)


def within_limits(hip, knee):
    """True if both angles are inside their joint limits."""
    return (HIP_LIMITS[0] <= hip <= HIP_LIMITS[1] and
            KNEE_LIMITS[0] <= knee <= KNEE_LIMITS[1])


def leg_ik(r, h):
    """
    Inverse kinematics: joint angles that put the foot at reach r and height
    h relative to the hip (knee-out branch).

    Args:
        r (float): reach along the spoke, outward positive, m
        h (float): height relative to the hip, up positive, m

    Returns:
        tuple: (hip, knee) in radians

    Raises:
        ValueError: if the point is out of reach or the solution is outside
            the joint limits. The message gives the point and the angles.
    """
    depth = -h
    dist = math.hypot(r, depth)
    max_reach = THIGH_LENGTH + SHIN_LENGTH
    min_reach = abs(THIGH_LENGTH - SHIN_LENGTH)
    if dist > max_reach or dist <= min_reach:
        raise ValueError(
            "foot (r=%+.4f, h=%+.4f) m is %.4f m from the hip, outside the "
            "reachable %.4f..%.4f m" % (r, h, dist, min_reach, max_reach))

    # Direction of the hip-to-foot line from straight down, outward positive
    phi = math.atan2(r, depth)
    # Law of cosines: angle between that line and the thigh, and between that
    # line and the shin
    thigh_off = math.acos((THIGH_LENGTH ** 2 + dist ** 2 - SHIN_LENGTH ** 2)
                          / (2 * THIGH_LENGTH * dist))
    shin_off = math.acos((SHIN_LENGTH ** 2 + dist ** 2 - THIGH_LENGTH ** 2)
                         / (2 * SHIN_LENGTH * dist))
    a = phi + thigh_off  # thigh outward of the line
    b = phi - shin_off   # shin back inward of it
    hip = -a
    knee = b + hip

    if not within_limits(hip, knee):
        raise ValueError(
            "foot (r=%+.4f, h=%+.4f) m needs hip %+.3f, knee %+.3f rad, outside "
            "the limits hip %s, knee %s" % (r, h, hip, knee, HIP_LIMITS, KNEE_LIMITS))
    return hip, knee


def main():
    r0, h0 = NEUTRAL_FOOT
    print("Neutral pose hip %+.3f knee %+.3f -> foot r %+.1f mm, h %+.1f mm"
          % (NEUTRAL_HIP, NEUTRAL_KNEE, 1000 * r0, 1000 * h0))
    hip, knee = leg_ik(r0, h0)
    print("IK of that foot position:  hip %+.3f knee %+.3f" % (hip, knee))
    print()
    print("Sample points around the neutral foot (dr, dh in mm):")
    for dr in (-15, 0, 15):
        for dh in (-30, -10, 0, 20):
            r, h = r0 + dr / 1000.0, h0 + dh / 1000.0
            try:
                hip, knee = leg_ik(r, h)
                result = "hip %+.3f knee %+.3f" % (hip, knee)
            except ValueError as e:
                result = "FAILS: %s" % e
            print("  dr %+3d dh %+3d  %s" % (dr, dh, result))


if __name__ == "__main__":
    main()

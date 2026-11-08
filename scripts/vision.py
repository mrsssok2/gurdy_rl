"""
Colour-blob detection for Gurdy's forward camera, without ROS, so it can be
tested offline on saved or synthetic images (scripts/tests/test_vision.py).
vision_node.py wraps it in a ROS node.

detect(bgr, camera) finds red, green and blue blobs by HSV thresholds and
returns one dict per blob:
    color          "red" | "green" | "blue"
    shape          "ball" | "box", a guess from how round the blob is
    centroid       [u, v] in pixels
    bbox           [x, y, w, h] in pixels
    bearing_deg    angle from the image centre, positive to the LEFT (as a
                   ROS yaw), from the camera's horizontal field of view
    area_fraction  blob area / image area
    distance_m     rough ground distance from the camera, or None
    clipped        True if the blob touches the image border
    clipped_bottom True if it touches the bottom edge: the object reaches
                   closer than the ground seen there (~0.13 m)
    side           "left" | "center" | "right" (|bearing| <= center_deg is
                   center)

Distance: the lowest pixel of a blob is where the object meets the ground,
so with the camera's height and downward tilt the ray through that pixel
hits the ground at the object's distance. If the blob is cut off by the
bottom of the image, or the ray doesn't reach the ground, the distance comes
from the blob's width and the object's known size instead (6 cm ball, 8 cm
box, as in worlds/gurdy_demo.world).

Tune HSV_RANGES (OpenCV hue is 0..179) and the thresholds below; see the
README's computer-vision section.
"""
import math

import cv2
import numpy as np

# HSV ranges per colour, as lists of (low, high) because red wraps around
# hue 0. Saturation >= 100 keeps out the grey ground and sky; value >= 40
# keeps the shaded side of an object.
HSV_RANGES = {
    "red": [((0, 100, 40), (10, 255, 255)), ((170, 100, 40), (179, 255, 255))],
    "green": [((40, 100, 40), (85, 255, 255))],
    "blue": [((100, 100, 40), (130, 255, 255))],
}
# Blobs smaller than this fraction of the image are ignored (noise, and the
# thin red line Gazebo draws along the world x axis)
MIN_AREA_FRACTION = 0.0005
# Morphological opening kernel, px: removes lines and specks up to ~this wide
OPEN_KERNEL_PX = 5
# Blob area / area of its minimum enclosing circle: ~0.9+ for a ball, ~0.5
# to ~0.7 for a box seen from slightly above
BALL_MIN_CIRCLE_FILL = 0.75
# A blob cut off by the image border is a ball if its free edge fits a
# circle with an RMS error under this fraction of the radius
BALL_MAX_FIT_ERROR = 0.04
# Known object sizes for the size-based distance, m
OBJECT_SIZE_M = {"ball": 0.06, "box": 0.08}
# |bearing| up to this is "center" for describing
CENTER_DEG = 10.0

COLOR_BGR = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 80, 0)}


class Camera(object):
    """
    Pinhole model of the head camera (robot/gurdy.xacro): 800x600, 1.396 rad
    horizontal FOV, square pixels, tilted tilt_deg down, height_m above the
    ground (base standing height ~0.093 m + 0.035 m mount).
    """

    def __init__(self, width=800, height=600, hfov_rad=1.3962634, tilt_deg=12.0,
                 height_m=0.128):
        self.width, self.height = width, height
        self.hfov_rad = hfov_rad
        self.tilt_deg = tilt_deg
        self.height_m = height_m
        self.f = (width / 2.0) / math.tan(hfov_rad / 2.0)
        self.cx, self.cy = width / 2.0, height / 2.0

    def resized(self, width, height):
        """The same camera at another image size (e.g. from the first frame)."""
        return Camera(width, height, self.hfov_rad, self.tilt_deg, self.height_m)

    def bearing_deg(self, u):
        """Horizontal angle to pixel column u, positive left."""
        return math.degrees(math.atan2(self.cx - u, self.f))

    def ground_distance(self, u, v):
        """
        Distance along the ground from the camera to where the ray through
        pixel (u, v) hits the ground, or None if it points at or above the
        horizon (within 1 degree).
        """
        below = math.radians(self.tilt_deg) + math.atan2(v - self.cy, self.f)
        if below < math.radians(1.0):
            return None
        forward = self.height_m / math.tan(below)
        lateral = forward * (self.cx - u) / self.f
        return math.hypot(forward, lateral)

    def size_distance(self, width_px, size_m):
        """Distance at which an object size_m wide spans width_px."""
        if width_px <= 0:
            return None
        return self.f * size_m / width_px


def color_mask(hsv, color, ranges=None):
    """Binary mask (uint8 0/255) of the pixels in color's HSV ranges, opened."""
    ranges = (ranges or HSV_RANGES)[color]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    if OPEN_KERNEL_PX > 1:
        kernel = np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _contours(mask):
    # OpenCV 3 returns (image, contours, hierarchy), 4 returns (contours, hierarchy)
    # Every boundary pixel (APPROX_NONE), for the circle fit in guess_shape
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return found[-2]


def fit_circle(points):
    """
    Least-squares (Kasa) circle through points, an (N, 2) array. Returns
    (cx, cy, r, rms) with rms the RMS radial error, or None.
    """
    pts = np.asarray(points, np.float64)
    if len(pts) < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        (c0, c1, c2), _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = c0 / 2.0, c1 / 2.0
    r2 = c2 + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return cx, cy, r, rms


def guess_shape(contour, image_size=None):
    """
    'ball' if the contour fills most of its enclosing circle, else 'box'.

    A blob cut off by the image border (a close ball is cut off at the
    bottom) is instead a ball if its edge points away from the border lie
    on a circle (RMS error under BALL_MAX_FIT_ERROR of the radius).
    """
    area = cv2.contourArea(contour)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return "box"
    fill = area / (math.pi * radius * radius)
    if fill >= BALL_MIN_CIRCLE_FILL:
        return "ball"
    if image_size is not None:
        width, height = image_size
        pts = contour.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x + w >= width - 1 or y + h >= height - 1:
            inner = pts[(pts[:, 0] > 2) & (pts[:, 0] < width - 3) &
                        (pts[:, 1] > 2) & (pts[:, 1] < height - 3)]
            fit = fit_circle(inner)
            if fit is not None and fit[3] < BALL_MAX_FIT_ERROR * fit[2]:
                return "ball"
    return "box"


def side_of(bearing_deg, center_deg=CENTER_DEG):
    if bearing_deg > center_deg:
        return "left"
    if bearing_deg < -center_deg:
        return "right"
    return "center"


def estimate_distance(camera, bbox, shape, clipped_bottom, clipped_sides):
    """Rough distance in m (see the module doc), or None."""
    x, y, w, h = bbox
    if not clipped_bottom:
        d = camera.ground_distance(x + w / 2.0, y + h)
        if d is not None:
            return d
    if not clipped_sides:
        return camera.size_distance(w, OBJECT_SIZE_M[shape])
    return None


def detect(bgr, camera=None, ranges=None, min_area_fraction=MIN_AREA_FRACTION,
           center_deg=CENTER_DEG):
    """Detections in a BGR image, largest first (see the module doc)."""
    height, width = bgr.shape[:2]
    if camera is None:
        camera = Camera(width, height)
    elif (camera.width, camera.height) != (width, height):
        camera = camera.resized(width, height)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    image_area = float(width * height)
    objects = []
    for color in (ranges or HSV_RANGES):
        mask = color_mask(hsv, color, ranges)
        for contour in _contours(mask):
            area = cv2.contourArea(contour)
            if area / image_area < min_area_fraction:
                continue
            x, y, w, h = [int(v) for v in cv2.boundingRect(contour)]
            m = cv2.moments(contour)
            if m["m00"] > 0:
                u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                u, v = x + w / 2.0, y + h / 2.0
            shape = guess_shape(contour, (width, height))
            clipped_bottom = y + h >= height - 1
            clipped_sides = x <= 0 or x + w >= width - 1
            clipped = clipped_bottom or clipped_sides or y <= 0
            bearing = camera.bearing_deg(u)
            distance = estimate_distance(camera, (x, y, w, h), shape,
                                         clipped_bottom, clipped_sides)
            objects.append({
                "color": color,
                "shape": shape,
                "centroid": [int(round(u)), int(round(v))],
                "bbox": [x, y, w, h],
                "bearing_deg": round(bearing, 1),
                "area_fraction": round(area / image_area, 4),
                "distance_m": None if distance is None else round(distance, 3),
                "clipped": clipped,
                "clipped_bottom": clipped_bottom,
                "side": side_of(bearing, center_deg),
            })
    objects.sort(key=lambda o: -o["area_fraction"])
    return objects


def label(obj):
    """Short text label, e.g. 'red ball 0.30m +9deg'."""
    text = "%s %s" % (obj["color"], obj["shape"])
    if obj.get("distance_m") is not None:
        text += " %.2fm" % obj["distance_m"]
    # + 0.0 turns -0.0 into 0.0, so a centred object reads "+0deg", not "-0deg"
    return text + " %+.0fdeg" % (round(obj["bearing_deg"]) + 0.0)


def annotate(bgr, objects, center_deg=CENTER_DEG, camera=None):
    """
    Copy of bgr with each detection's box, centroid and label, the image
    centre line, and dashed lines at +-center_deg.
    """
    out = bgr.copy()
    height, width = out.shape[:2]
    if camera is None or (camera.width, camera.height) != (width, height):
        camera = (camera or Camera()).resized(width, height)
    cx = int(round(camera.cx))
    cv2.line(out, (cx, 0), (cx, height - 1), (255, 255, 255), 1)
    offset = int(round(camera.f * math.tan(math.radians(center_deg))))
    for x in (cx - offset, cx + offset):
        for y0 in range(0, height, 20):
            cv2.line(out, (x, y0), (x, min(y0 + 10, height - 1)), (200, 200, 200), 1)
    for obj in objects:
        color = COLOR_BGR.get(obj["color"], (255, 255, 255))
        x, y, w, h = obj["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 4)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, tuple(obj["centroid"]), 4, (255, 255, 255), -1)
        text = label(obj)
        ty = y - 8 if y > 24 else y + h + 22
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 4)
        tx = max(0, min(x, width - text_w - 2))
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA)
    return out

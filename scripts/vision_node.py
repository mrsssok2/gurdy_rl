#!/usr/bin/env python3
"""
Colour-object detection on Gurdy's forward camera (see vision.py).

Subscribes /gurdy/camera/image_raw (sensor_msgs/Image) and, for every
processed frame (at most ~rate Hz), publishes
    /gurdy/detections         std_msgs/String, JSON:
        {"stamp": <image time, s>, "seq": n, "width": 800, "height": 600,
         "objects": [{"color", "shape", "centroid", "bbox", "bearing_deg",
                      "area_fraction", "distance_m", "clipped", "side"}, ...]}
        objects are largest first; the list is empty when nothing is seen,
        so the message also says the camera is alive.
    /gurdy/camera/annotated   sensor_msgs/Image (bgr8): the frame with boxes
                              and labels

Params (defaults match robot/gurdy.xacro and worlds/gurdy_demo.world):
    ~rate               max frames processed per second (10)
    ~camera_tilt_deg    downward tilt of the camera (12)
    ~camera_height_m    camera height above the ground when standing (0.128)
    ~hfov_rad           horizontal field of view (1.3962634)
    ~min_area_fraction  smaller blobs are ignored (0.0005)
    ~center_deg         |bearing| up to this is "center" (10)
    ~hsv_ranges         override vision.HSV_RANGES, e.g.
                        {"red": [[[0, 100, 40], [10, 255, 255]], [[170, 100, 40], [179, 255, 255]]]}

    rosrun my_gurdy_description vision_node.py
"""
import json

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

import vision

IMAGE_TOPIC = "/gurdy/camera/image_raw"
DETECTIONS_TOPIC = "/gurdy/detections"
ANNOTATED_TOPIC = "/gurdy/camera/annotated"


def parse_ranges(value):
    """~hsv_ranges param (dict, or a JSON string) -> {color: [(low, high), ...]}."""
    if isinstance(value, str):
        value = json.loads(value)
    return {color: [(tuple(int(c) for c in low), tuple(int(c) for c in high))
                    for low, high in ranges]
            for color, ranges in value.items()}


class VisionNode(object):
    def __init__(self):
        rospy.init_node("gurdy_vision")
        self.camera = vision.Camera(
            hfov_rad=rospy.get_param("~hfov_rad", 1.3962634),
            tilt_deg=rospy.get_param("~camera_tilt_deg", 12.0),
            height_m=rospy.get_param("~camera_height_m", 0.128))
        self.min_area = rospy.get_param("~min_area_fraction", vision.MIN_AREA_FRACTION)
        self.center_deg = rospy.get_param("~center_deg", vision.CENTER_DEG)
        self.ranges = vision.HSV_RANGES
        if rospy.has_param("~hsv_ranges"):
            self.ranges = dict(vision.HSV_RANGES)
            self.ranges.update(parse_ranges(rospy.get_param("~hsv_ranges")))
            rospy.loginfo("HSV ranges: %s", self.ranges)
        rate = float(rospy.get_param("~rate", 10.0))
        self.min_period = rospy.Duration(1.0 / rate) if rate > 0 else rospy.Duration(0)
        self.last_stamp = None
        self.seq = 0
        self.bridge = CvBridge()
        self.detections = rospy.Publisher(DETECTIONS_TOPIC, String, queue_size=1)
        self.annotated = rospy.Publisher(ANNOTATED_TOPIC, Image, queue_size=1)
        # buff_size: a raw 800x600 frame is 1.4 MB; the default buffer would
        # queue stale frames behind it
        rospy.Subscriber(IMAGE_TOPIC, Image, self.on_image, queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("Vision node: %s -> %s, %s", IMAGE_TOPIC, DETECTIONS_TOPIC, ANNOTATED_TOPIC)

    def on_image(self, msg):
        stamp = msg.header.stamp
        if self.last_stamp is not None and stamp >= self.last_stamp and \
                stamp - self.last_stamp < self.min_period:
            return
        self.last_stamp = stamp
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "cv_bridge: %s" % e)
            return
        objects = vision.detect(bgr, self.camera, self.ranges, self.min_area, self.center_deg)
        self.seq += 1
        self.detections.publish(String(data=json.dumps({
            "stamp": stamp.to_sec(), "seq": self.seq,
            "width": bgr.shape[1], "height": bgr.shape[0], "objects": objects})))
        if self.annotated.get_num_connections() > 0:
            out = self.bridge.cv2_to_imgmsg(
                vision.annotate(bgr, objects, self.center_deg, self.camera), "bgr8")
            out.header = msg.header
            self.annotated.publish(out)


if __name__ == "__main__":
    VisionNode()
    rospy.spin()

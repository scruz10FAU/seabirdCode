#!/usr/bin/env python3
"""
seabird/buoy_detector_v2.py
============================
ROS2 node — improved buoy detector with depth filtering, 3D position
estimation, and temporal tracking.

Improvements over v1:
  - Depth-aware: rejects false positives above/below water surface
  - 3D position: publishes buoy position in camera frame using depth + intrinsics
  - Temporal tracking: requires N detections in M frames before confirming
  - Tuned for 320x240 resolution
  - Richer detection messages (depth, 3D pos, confidence)
  - Median depth sampling (robust to noisy depth edges)

Subscribes:
  /iris_0/front_cam/rgb         (sensor_msgs/Image)
  /iris_0/front_cam/depth       (sensor_msgs/Image)
  /iris_0/front_cam/camera_info (sensor_msgs/CameraInfo)

Publishes:
  /seabird/buoy_detections  (std_msgs/String)  — JSON per detection (v2 format)
  /seabird/debug_image      (sensor_msgs/Image) — annotated camera feed

Run:
  python3 buoy_detector_v2.py [--hsv-debug] [--no-depth]

Detection JSON format (v2):
  {
    "color": "red",
    "centroid": [cx, cy],
    "bbox": [x, y, w, h],
    "area_px": 320.0,
    "depth_m": 12.4,
    "pos_cam": [x, y, z],       // 3D in camera frame (meters)
    "confidence": 0.85,          // temporal confidence 0-1
    "img_wh": [320, 240],
    "stamp": 1709654321.123
  }
"""

import json
import sys
import argparse
import collections
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import message_filters
import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    raise SystemExit("[ERROR] cv_bridge not found. apt install ros-humble-cv-bridge")


# ── HSV Color Ranges (OpenCV: H 0-179, S 0-255, V 0-255) ────────────────────
COLOR_RANGES: dict[str, list[tuple]] = {
    "red": [
        (np.array([0,   100,  60]), np.array([14,  255, 255])),
        (np.array([160, 100,  60]), np.array([179, 255, 255])),
    ],
    "green": [
        (np.array([35,  60,  45]), np.array([90,  255, 255])),
    ],
    "blue": [
        (np.array([95,  60,  45]), np.array([140, 255, 255])),
    ],
}

DEBUG_COLORS_BGR = {
    "red":   (0,   0,   255),
    "green": (0,   200, 0),
    "blue":  (255, 130, 0),
}

# ── Detection Thresholds (tuned for 320x240) ─────────────────────────────────
MIN_BLOB_AREA_PX   = 40       # smaller at lower res
MAX_BLOB_AREA_PX   = 20000
MIN_ASPECT_RATIO   = 0.2
MAX_ASPECT_RATIO   = 5.0

# ── Depth filtering ──────────────────────────────────────────────────────────
DEPTH_MIN_M        = 1.5      # closer than this = drone body / artifact
DEPTH_MAX_M        = 60.0     # farther than this = sky / water reflection
DEPTH_SAMPLE_RADIUS = 3       # pixels around centroid to sample median depth

# ── Temporal tracking ─────────────────────────────────────────────────────────
TRACK_WINDOW       = 10       # frames to look back
TRACK_MIN_HITS     = 3        # need this many detections in window to confirm
PUBLISH_COOLDOWN   = 1.0      # seconds between publishing same color


class TrackedDetection:
    """Sliding-window tracker for a single buoy color."""
    __slots__ = ("history", "window_size")

    def __init__(self, window_size: int = TRACK_WINDOW):
        self.window_size = window_size
        # deque of (frame_idx, det_dict_or_None)
        self.history: collections.deque = collections.deque(maxlen=window_size)

    def push(self, frame_idx: int, det: dict | None):
        self.history.append((frame_idx, det))

    @property
    def hit_count(self) -> int:
        return sum(1 for _, d in self.history if d is not None)

    @property
    def confidence(self) -> float:
        if not self.history:
            return 0.0
        return self.hit_count / min(len(self.history), self.window_size)

    @property
    def latest(self) -> dict | None:
        for _, d in reversed(self.history):
            if d is not None:
                return d
        return None


class BuoyDetectorV2(Node):

    def __init__(self, hsv_debug: bool = False, use_depth: bool = True):
        super().__init__("buoy_detector_v2")
        self.bridge = CvBridge()
        self.hsv_debug = hsv_debug
        self.use_depth = use_depth
        self.frame_count = 0
        self.last_pub: dict[str, float] = {}

        # Camera intrinsics (filled from CameraInfo)
        self.fx = self.fy = self.cx = self.cy = None
        self.intrinsics_received = False

        # Temporal trackers per color
        self.trackers: dict[str, TrackedDetection] = {
            c: TrackedDetection() for c in COLOR_RANGES
        }

        # ── Subscribers ──────────────────────────────────────────────────
        qos = rclpy.qos.qos_profile_sensor_data

        if use_depth:
            self.sub_rgb = message_filters.Subscriber(
                self, Image, "/iris_0/front_cam/rgb", qos_profile=qos)
            self.sub_depth = message_filters.Subscriber(
                self, Image, "/iris_0/front_cam/depth", qos_profile=qos)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth], queue_size=10, slop=0.1)
            self.sync.registerCallback(self.synced_callback)
        else:
            self.sub_rgb_only = self.create_subscription(
                Image, "/iris_0/front_cam/rgb", self.rgb_only_callback, qos)

        self.sub_info = self.create_subscription(
            CameraInfo, "/iris_0/front_cam/camera_info",
            self.info_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────
        self.pub_det   = self.create_publisher(String, "/seabird/buoy_detections", 10)
        self.pub_debug = self.create_publisher(Image,  "/seabird/debug_image", 10)

        mode = "RGB+Depth" if use_depth else "RGB only"
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[buoy_detector_v2] Mode: {mode}")
        self.get_logger().info(f"  Tracking window: {TRACK_WINDOW} frames, min hits: {TRACK_MIN_HITS}")
        self.get_logger().info(f"  Depth filter: {DEPTH_MIN_M}-{DEPTH_MAX_M}m")
        self.get_logger().info(f"  HSV debug: {'ON' if hsv_debug else 'off'}")
        self.get_logger().info("=" * 60)

    # ── Camera intrinsics ─────────────────────────────────────────────────

    def info_callback(self, msg: CameraInfo):
        if self.intrinsics_received:
            return
        K = msg.k  # 3x3 row-major
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.intrinsics_received = True
        self.get_logger().info(
            f"[intrinsics] fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}")

    # ── Callback wrappers ─────────────────────────────────────────────────

    def rgb_only_callback(self, rgb_msg: Image):
        self._process(rgb_msg, None)

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        self._process(rgb_msg, depth_msg)

    # ── Core detection ────────────────────────────────────────────────────

    def _process(self, rgb_msg: Image, depth_msg: Image | None):
        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB convert: {e}")
            return

        depth = None
        if depth_msg is not None:
            try:
                depth = self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding="passthrough").astype(np.float32)
            except Exception as e:
                self.get_logger().warn(f"Depth convert: {e}")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        debug = frame.copy()
        h_img, w_img = frame.shape[:2]
        now = self.get_clock().now().nanoseconds * 1e-9

        for color, ranges in COLOR_RANGES.items():

            # ── HSV mask ──────────────────────────────────────────────
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)

            k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_small, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big,   iterations=2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_BLOB_AREA_PX <= area <= MAX_BLOB_AREA_PX):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(h, 1)
                if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                    continue

                cx, cy_px = x + w // 2, y + h // 2

                # ── Depth check ───────────────────────────────────
                d_m = None
                if depth is not None and self.use_depth:
                    d_m = self._sample_depth(depth, cx, cy_px, h_img, w_img)
                    if d_m is not None and not (DEPTH_MIN_M <= d_m <= DEPTH_MAX_M):
                        if self.hsv_debug:
                            self.get_logger().info(
                                f"  [depth-reject] {color} depth={d_m:.1f}m "
                                f"outside [{DEPTH_MIN_M}-{DEPTH_MAX_M}]")
                        continue

                if area > best_area:
                    best_area = area
                    best = {
                        "cnt": cnt, "x": x, "y": y, "w": w, "h": h,
                        "cx": cx, "cy": cy_px, "area": area, "depth_m": d_m,
                    }

            # ── Update tracker ────────────────────────────────────
            tracker = self.trackers[color]
            if best is not None:
                tracker.push(self.frame_count, best)
            else:
                tracker.push(self.frame_count, None)

            # ── Draw debug (even unconfirmed) ─────────────────────
            if best is not None:
                bgr = DEBUG_COLORS_BGR[color]
                x, y, w, h = best["x"], best["y"], best["w"], best["h"]
                conf = tracker.confidence

                # Solid box if confirmed, dashed if tentative
                if tracker.hit_count >= TRACK_MIN_HITS:
                    cv2.rectangle(debug, (x, y), (x+w, y+h), bgr, 2)
                else:
                    # Draw dashed rectangle
                    for i in range(x, x+w, 8):
                        cv2.line(debug, (i, y), (min(i+4, x+w), y), bgr, 1)
                        cv2.line(debug, (i, y+h), (min(i+4, x+w), y+h), bgr, 1)

                cv2.circle(debug, (best["cx"], best["cy"]), 4, bgr, -1)
                d_str = f" {best['depth_m']:.1f}m" if best["depth_m"] else ""
                label = f"{color} {conf:.0%}{d_str}"
                cv2.putText(debug, label, (x, max(y-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)

                # ── HSV debug ─────────────────────────────────────
                if self.hsv_debug:
                    roi_hsv = hsv[y:y+h, x:x+w]
                    roi_mask = mask[y:y+h, x:x+w]
                    if roi_mask.any():
                        mean = cv2.mean(roi_hsv, mask=roi_mask)
                        self.get_logger().info(
                            f"  [hsv] {color:6s} area={best['area']:5.0f} "
                            f"HSV=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f}) "
                            f"depth={best['depth_m']}")

            # ── Publish confirmed detection ───────────────────────
            if tracker.hit_count < TRACK_MIN_HITS:
                continue
            latest = tracker.latest
            if latest is None:
                continue

            elapsed = now - self.last_pub.get(color, 0.0)
            if elapsed < PUBLISH_COOLDOWN:
                continue

            # 3D position in camera frame
            pos_cam = None
            if latest["depth_m"] is not None and self.intrinsics_received:
                pos_cam = self._pixel_to_3d(
                    latest["cx"], latest["cy"], latest["depth_m"])

            self.last_pub[color] = now
            payload = json.dumps({
                "color":      color,
                "centroid":   [latest["cx"], latest["cy"]],
                "bbox":       [latest["x"], latest["y"], latest["w"], latest["h"]],
                "area_px":    float(latest["area"]),
                "depth_m":    latest["depth_m"],
                "pos_cam":    pos_cam,
                "confidence": round(tracker.confidence, 3),
                "img_wh":     [w_img, h_img],
                "stamp":      now,
            })
            self.pub_det.publish(String(data=payload))

            d_str = f"  depth={latest['depth_m']:.1f}m" if latest["depth_m"] else ""
            p_str = ""
            if pos_cam:
                p_str = f"  pos=({pos_cam[0]:.1f},{pos_cam[1]:.1f},{pos_cam[2]:.1f})"
            self.get_logger().info(
                f"  ● {color:<6} conf={tracker.confidence:.0%} "
                f"area={latest['area']:.0f}px²{d_str}{p_str}")

        # ── Frame info on debug image ─────────────────────────────────
        cv2.putText(debug, f"f{self.frame_count}", (4, h_img - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))

    # ── Depth sampling ────────────────────────────────────────────────────

    def _sample_depth(self, depth: np.ndarray, cx: int, cy: int,
                      h_img: int, w_img: int) -> float | None:
        """Sample median depth in a small window around (cx, cy).
        Returns None if no valid depth pixels found."""
        r = DEPTH_SAMPLE_RADIUS
        y0 = max(0, cy - r)
        y1 = min(h_img, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w_img, cx + r + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    # ── Pixel → 3D (camera frame) ────────────────────────────────────────

    def _pixel_to_3d(self, u: int, v: int, z: float) -> list[float]:
        """Pinhole projection: pixel (u,v) + depth z → 3D in camera frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return [round(x, 3), round(y, 3), round(z, 3)]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seabird buoy detector v2")
    parser.add_argument("--hsv-debug", action="store_true",
                        help="Print mean HSV + depth of each blob")
    parser.add_argument("--no-depth", action="store_true",
                        help="Run without depth (RGB only, no 3D position)")
    known, _ = parser.parse_known_args()

    rclpy.init()
    node = BuoyDetectorV2(
        hsv_debug=known.hsv_debug,
        use_depth=not known.no_depth,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"\n[detector] Stopped after {node.frame_count} frames")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
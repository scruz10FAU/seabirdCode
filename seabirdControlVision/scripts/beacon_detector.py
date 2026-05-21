#!/usr/bin/env python3
"""
beacon_detector.py — Detects a single beacon class with one_beacon.pt,
then isolates the light area inside the bounding box and classifies its color
using HSV thresholding on the brightest pixels.

Usage:
    python3 beacon_detector.py
    python3 beacon_detector.py -m models/one_beacon.pt -d
    python3 beacon_detector.py -td 0.5

Color classification pipeline (post-detection):
  1. Crop the detected bounding box from the BGR frame.
  2. Convert crop to HSV.
  3. Mask pixels with high Value (bright / lit) and moderate Saturation.
  4. Compute the circular mean hue of those pixels.
  5. Map hue angle to a color name: red, yellow, green, cyan, blue, magenta, or white.
"""

import sys
import argparse
import os
sys.path.insert(0, os.path.expanduser("~/seabird/scripts"))

import threading
import numpy as np
from typing import List, Optional, Tuple
import json
import time
import cv2

DEFAULT_TOPIC_PREFIX = "/zed/zed_node"
DRONE_POSE_TOPIC     = "/mavros/local_position/pose"
GPS_TOPIC            = "/mavros/global_position/gp_origin"
EARTH_RADIUS_M       = 6378137.0

# Lazily imported only when ROS mode is used
def _import_ros():
    global rclpy, Node, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    global Image, CameraInfo, PoseStamped, GeoPointStamped, String, message_filters
    global CameraInterface, CameraConfig, Detection, Intrinsics
    global IMG_W, IMG_H, FX, FY, CX, CY, camera_to_world, YoloDetector
    import rclpy as _rclpy; rclpy = _rclpy
    from rclpy.node import Node as _Node; Node = _Node
    from rclpy.qos import (QoSProfile as _QP, ReliabilityPolicy as _RP,
                            HistoryPolicy as _HP, DurabilityPolicy as _DP)
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy = _QP, _RP, _HP, _DP
    from sensor_msgs.msg import Image as _Im, CameraInfo as _CI
    Image, CameraInfo = _Im, _CI
    from geometry_msgs.msg import PoseStamped as _PS; PoseStamped = _PS
    from geographic_msgs.msg import GeoPointStamped as _GPS; GeoPointStamped = _GPS
    from std_msgs.msg import String as _Str; String = _Str
    import message_filters as _mf; message_filters = _mf
    from camera_interface import CameraInterface as _CAM, CameraConfig as _CC, Detection as _D, Intrinsics as _I
    CameraInterface, CameraConfig, Detection, Intrinsics = _CAM, _CC, _D, _I
    from seabird_config import IMG_W as _W, IMG_H as _H, FX as _FX, FY as _FY, CX as _CX, CY as _CY, camera_to_world as _c2w
    IMG_W, IMG_H, FX, FY, CX, CY, camera_to_world = _W, _H, _FX, _FY, _CX, _CY, _c2w
    from yolo_detector import YoloDetector as _YD; YoloDetector = _YD

# ── Color classification ───────────────────────────────────────────────────────

# HSV saturation/value thresholds for "bright, lit" pixels
_SAT_MIN = 60    # ignore nearly-grey pixels
_VAL_MIN = 160   # only consider bright pixels (the light itself)

# Hue boundary table (degrees, 0-180 in OpenCV).
# Each entry: (hue_center, half_width, label)
_HUE_BANDS = [
    (  0, 10, "red"),
    ( 15, 12, "orange"),
    ( 30, 15, "yellow"),
    ( 60, 20, "green"),
    ( 90, 15, "cyan"),
    (105, 20, "blue"),
    (135, 15, "magenta"),
    (165, 15, "red"),   # wraps back to red
]


def _circular_mean_hue(hues: np.ndarray) -> float:
    """Circular mean of hue values (0-179 → 0-π radians × 2)."""
    angles = hues.astype(np.float32) * (2 * np.pi / 180.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(mean_angle * 180.0 / (2 * np.pi))   # back to 0-180


def classify_beacon_color(bgr_crop: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Given a BGR crop of a beacon bounding box, return:
        (color_name, confidence, light_mask)

    confidence: fraction of crop pixels that are "lit" (higher = cleaner read).
    light_mask: uint8 mask of the bright pixels used (same H×W as crop).
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0, np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Bright, saturated pixels → the light source
    light_mask = ((s >= _SAT_MIN) & (v >= _VAL_MIN)).astype(np.uint8) * 255

    lit_pixels = np.count_nonzero(light_mask)
    total_pixels = bgr_crop.shape[0] * bgr_crop.shape[1]
    confidence = lit_pixels / max(total_pixels, 1)

    if lit_pixels < 5:
        # Fallback: not enough saturated pixels — check if it's very bright white
        very_bright = (v >= 220)
        if np.count_nonzero(very_bright) > total_pixels * 0.1:
            return "white", float(np.count_nonzero(very_bright) / total_pixels), light_mask
        return "unknown", 0.0, light_mask

    hues = h[light_mask > 0]
    mean_hue = _circular_mean_hue(hues)

    # Map mean hue to the closest band
    best_label = "unknown"
    best_dist  = 180.0
    for center, half, label in _HUE_BANDS:
        dist = abs(mean_hue - center)
        dist = min(dist, 180.0 - dist)   # circular distance
        if dist < best_dist:
            best_dist  = dist
            best_label = label

    return best_label, confidence, light_mask


# ── ROS2 Camera Node ───────────────────────────────────────────────────────────

def local_enu_to_gps(world_pos: np.ndarray,
                     origin_lat: float,
                     origin_lon: float,
                     origin_alt: float) -> Tuple[float, float, float]:
    east, north, up = world_pos[0], world_pos[1], world_pos[2]
    dlat = np.degrees(north / EARTH_RADIUS_M)
    dlon = np.degrees(east / (EARTH_RADIUS_M * np.cos(np.radians(origin_lat))))
    return (origin_lat + dlat, origin_lon + dlon, origin_alt + up)


class BeaconCamera(Node):
    """
    Subscribes to ZED camera topics and runs beacon detection + color classification.

    Publishes:
        /seabird/beacon_detections  — JSON with label "beacon", detected color, position
    """

    def __init__(self, topic_prefix: str = DEFAULT_TOPIC_PREFIX):
        super().__init__("beacon_camera")
        self._topic_prefix = topic_prefix

        self._rgb:        Optional[np.ndarray] = None
        self._depth:      Optional[np.ndarray] = None
        self._intrinsics: Optional[Intrinsics]  = None
        self._new_frame:  bool = False
        self._frame_lock  = threading.Lock()

        self._drone_pos:       Optional[np.ndarray] = None
        self._drone_quat_wxyz: Optional[np.ndarray] = None
        self._pose_lock        = threading.Lock()

        self._gps_origin:      Optional[Tuple[float, float, float]] = None
        self._gps_origin_lock  = threading.Lock()

        self._is_open: bool = False
        self._detector: Optional[YoloDetector] = None
        self.detection_pub = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if self._is_open:
            return True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._info_sub = self.create_subscription(
            CameraInfo,
            f"{self._topic_prefix}/rgb/color/rect/camera_info",
            self._on_camera_info,
            qos,
        )

        rgb_sub   = message_filters.Subscriber(
            self, Image, f"{self._topic_prefix}/rgb/color/rect/image", qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            self, Image, f"{self._topic_prefix}/depth/depth_registered", qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._on_synced_frame)

        self.detection_pub = self.create_publisher(String, "/seabird/beacon_detections", 10)

        self._pose_sub = self.create_subscription(
            PoseStamped, DRONE_POSE_TOPIC, self._on_drone_pose, qos
        )

        origin_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._origin_sub = self.create_subscription(
            GeoPointStamped, GPS_TOPIC, self._on_gps_origin, origin_qos
        )

        self._is_open = True
        self.get_logger().info("BeaconCamera open — waiting for frames…")
        return True

    def open_for_video(self) -> bool:
        """
        Minimal ROS setup for video-file input mode.
        Creates the detection publisher and subscribes to drone pose + GPS,
        but skips all camera image subscriptions (frames come from cv2.VideoCapture).
        """
        if self._is_open:
            return True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.detection_pub = self.create_publisher(String, "/seabird/beacon_detections", 10)

        self._pose_sub = self.create_subscription(
            PoseStamped, DRONE_POSE_TOPIC, self._on_drone_pose, qos
        )

        origin_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._origin_sub = self.create_subscription(
            GeoPointStamped, GPS_TOPIC, self._on_gps_origin, origin_qos
        )

        self._is_open = True
        self.get_logger().info("BeaconCamera open (video-file mode) — pose + GPS only")
        return True

    def close(self) -> None:
        self._is_open = False

    def grab(self) -> bool:
        if not self._is_open:
            return False
        rclpy.spin_once(self, timeout_sec=0.05)
        with self._frame_lock:
            if self._new_frame:
                self._new_frame = False
                return True
        return False

    def enable_detection(self, model_path: str) -> bool:
        self._detector = YoloDetector(
            weights=model_path,
            class_names=["beacon"],   # single class — no color in the model
            imgsz=320,
            conf_thresh=0.5,
        )
        ok = self._detector.start(enable_tracking=True)
        if not ok:
            self._detector = None
            self.get_logger().error("YoloDetector failed to start")
        return ok

    def get_rgb(self)   -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._rgb.copy() if self._rgb is not None else None

    def get_depth(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._depth.copy() if self._depth is not None else None

    def get_drone_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._pose_lock:
            if self._drone_pos is None:
                return None, None
            return self._drone_pos.copy(), self._drone_quat_wxyz.copy()

    def get_gps_origin(self) -> Optional[Tuple[float, float, float]]:
        with self._gps_origin_lock:
            return self._gps_origin

    def get_detections(self) -> List[Detection]:
        if self._detector is None:
            return []
        with self._frame_lock:
            rgb   = self._rgb.copy()   if self._rgb   is not None else None
            depth = self._depth.copy() if self._depth is not None else None
        if rgb is None:
            return []
        return self._detector.detect(rgb, depth, self._intrinsics)

    # ── ROS2 Callbacks ─────────────────────────────────────────────────────────

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._intrinsics is not None:
            return
        self._intrinsics = Intrinsics(
            fx=FX, fy=FY, cx=CX, cy=CY, width=IMG_W, height=IMG_H
        )
        self.get_logger().info(
            f"Intrinsics set from config: fx={FX:.1f} fy={FY:.1f} "
            f"cx={CX:.1f} cy={CY:.1f} {IMG_W}x{IMG_H}"
        )
        self.destroy_subscription(self._info_sub)

    def _on_synced_frame(self, rgb_msg: Image, depth_msg: Image) -> None:
        channels = len(rgb_msg.data) // (rgb_msg.height * rgb_msg.width)
        rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, channels
        )
        bgr   = rgb[:, :, :3][:, :, ::-1].copy()
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width
        ).copy()
        with self._frame_lock:
            self._rgb      = bgr
            self._depth    = depth
            self._new_frame = True

    def _on_drone_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        with self._pose_lock:
            self._drone_pos       = np.array([p.x, p.y, p.z])
            self._drone_quat_wxyz = np.array([q.w, q.x, q.y, q.z])

    def _on_gps_origin(self, msg: GeoPointStamped) -> None:
        with self._gps_origin_lock:
            self._gps_origin = (
                msg.position.latitude,
                msg.position.longitude,
                msg.position.altitude,
            )
        self.get_logger().info(
            f"GPS origin: lat={msg.position.latitude:.7f} "
            f"lon={msg.position.longitude:.7f}"
        )


# ── Video test mode (no ROS) ───────────────────────────────────────────────────

_COLOR_BGR = {
    "red":     (0,   0,   255),
    "orange":  (0,   128, 255),
    "yellow":  (0,   255, 255),
    "green":   (0,   255,   0),
    "cyan":    (255, 255,   0),
    "blue":    (255,   0,   0),
    "magenta": (255,   0, 255),
    "white":   (255, 255, 255),
    "unknown": (180, 180, 180),
}


def _annotate_frame(frame: np.ndarray, boxes, names: dict) -> np.ndarray:
    """Draw detections + color classification on a single BGR frame."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = names.get(cls, str(cls))

        crop = frame[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
        beacon_color, color_conf, light_mask = classify_beacon_color(crop)
        draw_color = _COLOR_BGR.get(beacon_color, (180, 180, 180))

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

        # Tint the lit pixels
        if light_mask is not None and light_mask.any():
            lm_full = np.zeros(frame.shape[:2], dtype=np.uint8)
            lm_h = min(light_mask.shape[0], y2 - y1)
            lm_w = min(light_mask.shape[1], x2 - x1)
            lm_full[y1:y1+lm_h, x1:x1+lm_w] = light_mask[:lm_h, :lm_w]
            tint = np.zeros_like(frame)
            tint[:] = draw_color
            frame[lm_full > 0] = cv2.addWeighted(frame, 0.5, tint, 0.5, 0)[lm_full > 0]

        txt = f"{label} [{beacon_color}] det={conf:.2f} col={color_conf:.2f}"
        cv2.putText(frame, txt, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

        print(f"  {txt}  bbox=({x1},{y1},{x2},{y2})")

    return frame


def run_video(video_path: str,
              model_path: str = "models/one_beacon.pt",
              save_output: bool = False,
              conf: float = 0.5) -> None:
    """
    Run beacon detection + color classification on a local video file.
    No ROS required — uses ultralytics.YOLO directly.

    Press 'q' to quit, SPACE to pause/resume.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[beacon-video] ultralytics not installed: pip install ultralytics")
        return

    if not os.path.exists(model_path):
        print(f"[beacon-video] Model not found: {model_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[beacon-video] Cannot open video: {video_path}")
        return

    model  = YOLO(model_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[beacon-video] {video_path}  {width}x{height} @ {fps:.1f}fps  {total} frames")
    print(f"[beacon-video] Model: {model_path}  conf≥{conf}")
    print("[beacon-video] Press 'q' to quit, SPACE to pause")

    writer = None
    if save_output:
        out_path = os.path.splitext(video_path)[0] + "_beacon_out.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[beacon-video] Saving output → {out_path}")

    frame_idx = 0
    paused    = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[beacon-video] End of video")
                    break
                frame_idx += 1

                results = model(frame, conf=conf, verbose=False)
                boxes   = results[0].boxes
                names   = results[0].names

                print(f"Frame {frame_idx}/{total} — {len(boxes)} detection(s)")
                frame = _annotate_frame(frame, boxes, names)

                if writer:
                    writer.write(frame)

            cv2.imshow("Beacon Detector — Video Test", frame)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    except KeyboardInterrupt:
        print("\n[beacon-video] Interrupted")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[beacon-video] Done — {frame_idx} frames processed")


# ── Video + ROS mode ──────────────────────────────────────────────────────────

def run_video_ros(video_path: str,
                  model_path: str = "models/one_beacon.pt",
                  save_output: bool = False,
                  conf: float = 0.5) -> None:
    """
    Read frames from a local video file, publish detections to ROS, and show
    a live display window.

    Initializes a ROS2 node for:
      - Publishing to /seabird/beacon_detections
      - Receiving drone pose from /mavros/local_position/pose
      - Receiving GPS origin from /mavros/global_position/gp_origin

    Frames come from cv2.VideoCapture — no camera topic subscriptions needed.
    Press 'q' to quit, SPACE to pause/resume.
    """
    _import_ros()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[beacon-ros-video] ultralytics not installed: pip install ultralytics")
        return

    if not os.path.exists(model_path):
        print(f"[beacon-ros-video] Model not found: {model_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[beacon-ros-video] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[beacon-ros-video] {video_path}  {width}x{height} @ {fps:.1f}fps  {total} frames")
    print(f"[beacon-ros-video] Model: {model_path}  conf≥{conf}")
    print("[beacon-ros-video] Publishing → /seabird/beacon_detections")
    print("[beacon-ros-video] Press 'q' to quit, SPACE to pause")

    writer = None
    if save_output:
        out_path = os.path.splitext(video_path)[0] + "_beacon_ros_out.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[beacon-ros-video] Saving output → {out_path}")

    rclpy.init()
    cam   = BeaconCamera()
    model = YOLO(model_path)

    if not cam.open_for_video():
        print("[beacon-ros-video] Failed to open ROS node")
        rclpy.shutdown()
        cap.release()
        return

    frame_idx = 0
    paused    = False

    try:
        while rclpy.ok():
            # Spin once to receive latest pose / GPS
            rclpy.spin_once(cam, timeout_sec=0.0)

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[beacon-ros-video] End of video")
                    break
                frame_idx += 1

                drone_pos, _ = cam.get_drone_pose()

                results = model(frame, conf=conf, verbose=False)
                boxes   = results[0].boxes

                print(f"Frame {frame_idx}/{total} — {len(boxes)} detection(s)")

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    det_conf = float(box.conf[0])

                    crop = frame[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
                    beacon_color, color_conf, light_mask = classify_beacon_color(crop)
                    draw_color = _COLOR_BGR.get(beacon_color, (180, 180, 180))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

                    if light_mask is not None and light_mask.any():
                        lm_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                        lm_h = min(light_mask.shape[0], y2 - y1)
                        lm_w = min(light_mask.shape[1], x2 - x1)
                        lm_full[y1:y1+lm_h, x1:x1+lm_w] = light_mask[:lm_h, :lm_w]
                        tint = np.zeros_like(frame)
                        tint[:] = draw_color
                        frame[lm_full > 0] = cv2.addWeighted(
                            frame, 0.5, tint, 0.5, 0
                        )[lm_full > 0]

                    label_txt = (
                        f"beacon [{beacon_color}] "
                        f"det={det_conf:.2f} col={color_conf:.2f}"
                    )
                    cv2.putText(frame, label_txt, (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

                    # Publish to ROS
                    msg = String()
                    msg.data = json.dumps({
                        "label":            "beacon",
                        "color":            beacon_color,
                        "color_confidence": color_conf,
                        "confidence":       det_conf,
                        "bbox":             [x1, y1, x2, y2],
                        "position_3d":      None,
                        "world_position":   None,
                        "gps_position":     None,
                        "drone_position":   drone_pos.tolist() if drone_pos is not None else None,
                        "tracking_id":      -1,
                        "timestamp":        time.time(),
                    })
                    cam.detection_pub.publish(msg)
                    print(f"  {label_txt}")

                if writer:
                    writer.write(frame)

            cv2.imshow("Beacon Detector — Video + ROS", frame)
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    except KeyboardInterrupt:
        print("\n[beacon-ros-video] Interrupted")
    finally:
        cap.release()
        if writer:
            writer.release()
        cam.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        print(f"[beacon-ros-video] Done — {frame_idx} frames processed")


# ── Main loop (ROS mode) ───────────────────────────────────────────────────────

def main(model: str = "models/one_beacon.pt",
         display: bool = False,
         true_distance: float = 0.4826) -> None:

    _import_ros()   # pull in ROS2 / camera_interface / seabird_config

    DEBUG_DIR   = os.path.expanduser("~/seabird_dataset/beacon_debug")
    SAVE_EVERY_N = 30
    os.makedirs(DEBUG_DIR, exist_ok=True)

    rclpy.init()
    cam = BeaconCamera()

    if not cam.open():
        print("[beacon] Failed to open camera")
        rclpy.shutdown()
        return

    if not os.path.exists(model):
        print(f"[beacon] Model not found: {model}")
        cam.close()
        rclpy.shutdown()
        return

    print(f"[beacon] Loading model: {model}")
    if not cam.enable_detection(model):
        print("[beacon] Detection failed to start")
        cam.close()
        rclpy.shutdown()
        return

    print("[beacon] Detection ENABLED — class: beacon (color determined by CV)")
    print("[beacon] Publishing → /seabird/beacon_detections")

    frame_count       = 0
    intrinsics_printed = False

    try:
        while rclpy.ok():
            if not cam.grab():
                continue

            frame_count += 1
            rgb        = cam.get_rgb()
            depth      = cam.get_depth()
            intr       = cam._intrinsics
            drone_pos, drone_quat = cam.get_drone_pose()

            if intr and not intrinsics_printed:
                print(f"[beacon] Intrinsics ready: {intr.width}x{intr.height} "
                      f"fx={intr.fx:.1f} fy={intr.fy:.1f}")
                intrinsics_printed = True

            if rgb is None:
                continue

            dets = cam.get_detections()

            for d in dets:
                x1, y1, x2, y2 = d.bbox_2d

                # ── Color determination from the beacon's light area ──────────
                crop = rgb[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
                beacon_color, color_conf, light_mask = classify_beacon_color(crop)
                # ─────────────────────────────────────────────────────────────

                draw_color = _COLOR_BGR.get(beacon_color, (180, 180, 180))

                cv2.rectangle(rgb, (x1, y1), (x2, y2), draw_color, 2)

                # Overlay light mask back onto the frame (tinted)
                if light_mask is not None and light_mask.any():
                    lm_full = np.zeros(rgb.shape[:2], dtype=np.uint8)
                    lm_h = min(light_mask.shape[0], y2 - y1)
                    lm_w = min(light_mask.shape[1], x2 - x1)
                    lm_full[y1:y1+lm_h, x1:x1+lm_w] = light_mask[:lm_h, :lm_w]
                    tint = np.zeros_like(rgb)
                    tint[:] = draw_color
                    rgb[lm_full > 0] = cv2.addWeighted(
                        rgb, 0.5, tint, 0.5, 0
                    )[lm_full > 0]

                label_txt = (
                    f"beacon [{beacon_color}] "
                    f"conf={d.confidence:.2f} "
                    f"col_conf={color_conf:.2f}"
                )
                if d.tracking_id >= 0:
                    label_txt += f" #{d.tracking_id}"

                cv2.putText(rgb, label_txt, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1)

                # ── World-frame position (if pose available) ─────────────────
                world_pos  = None
                gps_coords = None
                if d.position_3d is not None and drone_pos is not None:
                    world_pos = camera_to_world(d.position_3d, drone_pos, drone_quat)
                    origin    = cam.get_gps_origin()
                    if origin is not None:
                        lat, lon, alt = local_enu_to_gps(world_pos, *origin)
                        gps_coords    = {"latitude": lat, "longitude": lon, "altitude": alt}

                    cv2.putText(
                        rgb,
                        f"W({world_pos[0]:.1f},{world_pos[1]:.1f},{world_pos[2]:.1f})",
                        (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, draw_color, 1,
                    )

                # ── Publish ──────────────────────────────────────────────────
                msg = String()
                msg.data = json.dumps({
                    "label":          "beacon",
                    "color":          beacon_color,
                    "color_confidence": color_conf,
                    "confidence":     float(d.confidence),
                    "bbox":           list(d.bbox_2d),
                    "position_3d":    d.position_3d.tolist() if d.position_3d is not None else None,
                    "world_position": world_pos.tolist()     if world_pos     is not None else None,
                    "gps_position":   gps_coords,
                    "drone_position": drone_pos.tolist()     if drone_pos     is not None else None,
                    "tracking_id":    d.tracking_id,
                    "timestamp":      time.time(),
                })
                cam.detection_pub.publish(msg)

                print(f"[beacon] {label_txt}"
                      + (f" pos3d=({d.position_3d[2]:.2f}m)" if d.position_3d is not None else ""))

            # ── Display ───────────────────────────────────────────────────────
            if display and rgb is not None:
                cv2.imshow("Beacon Detector", rgb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Periodic save ─────────────────────────────────────────────────
            if frame_count % SAVE_EVERY_N == 0 and rgb is not None:
                out_path = os.path.join(DEBUG_DIR, f"frame_{frame_count:06d}.png")
                cv2.imwrite(out_path, rgb)
                print(f"[beacon] Saved {out_path}")

    except KeyboardInterrupt:
        print("\n[beacon] Interrupted")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        print(f"[beacon] Done — {frame_count} frames processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="beacon_detector",
        description="Beacon detector with CV-based color classification",
    )
    parser.add_argument(
        "--model", "-m",
        default="models/one_beacon.pt",
        type=str,
        help="Path to YOLO beacon model (default: models/one_beacon.pt)",
    )
    parser.add_argument("--display", "-d", action="store_true", help="Show live cv2 window (ROS mode)")
    parser.add_argument(
        "--true_dist", "-td",
        type=float,
        default=0.4826,
        help="Known ground-truth distance to object in meters (ROS mode)",
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        type=str,
        metavar="VIDEO_PATH",
        help="Test on a local video file — no ROS, pure CV only",
    )
    parser.add_argument(
        "--ros-video", "-rv",
        default=None,
        type=str,
        metavar="VIDEO_PATH",
        help="Read frames from a video file AND publish detections to ROS topics",
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save annotated output video alongside the input (video modes only)",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold (video modes, default 0.5)",
    )
    args = parser.parse_args()

    if args.ros_video is not None:
        run_video_ros(args.ros_video, model_path=args.model, save_output=args.save, conf=args.conf)
    elif args.video is not None:
        run_video(args.video, model_path=args.model, save_output=args.save, conf=args.conf)
    else:
        main(args.model, args.display, args.true_dist)

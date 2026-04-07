#!/usr/bin/env python3
"""
SimCamera — ROS2 implementation of CameraInterface for Isaac Sim.

v4: Added /seabird/buoy_detections publisher (JSON over std_msgs/String).
    Each detection with a valid world-frame position is published for
    downstream consumers (sweep_and_detect.py, etc.).

Subscribes to Isaac's ROS2 camera topics, delivers frames through
the same interface that ZEDCamera will use on real hardware.

Runs as a standalone ROS2 node in a terminal (NOT inside Isaac Script Editor).

Usage:
    source /opt/ros/humble/setup.bash
    python3 ${ISAAC_ROS_WS}/isaac_ros_assets/scripts/seabirdCode/seabirdControlVision/scripts/sim_camera_zed.py

Load with different model:
    python3 ${ISAAC_ROS_WS}/isaac_ros_assets/scripts/seabirdCode/seabirdControlVision/scripts/sim_camera_zed.py -m "path/to/model"
Load with display:
    python3 ${ISAAC_ROS_WS}/isaac_ros_assets/scripts/seabirdCode/seabirdControlVision/scripts/sim_camera_zed.py -d
Load true distance of object detected
    python3 ${ISAAC_ROS_WS}/isaac_ros_assets/scripts/seabirdCode/seabirdControlVision/scripts/sim_camera_zed.py -td <DISTANCE IN METERS>

Prerequisites:
    pip3 install --user opencv-python numpy scipy
    (rclpy, sensor_msgs, geometry_msgs, message_filters come from ROS2 Humble)
"""

import sys
import argparse
# Do NOT insert Python 3.11 site-packages here — sim_camera runs under
# system Python 3.10 (required by ROS2 Humble's rclpy). The 3.11 numpy
# is compiled for cpython-311 and crashes under 3.10.
# ultralytics must be installed for 3.10: python3 -m pip install --user ultralytics
import os
sys.path.insert(0, os.path.expanduser("~/seabird/scripts"))

import threading
import numpy as np
from typing import List, Optional, Tuple
from std_msgs.msg import String
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import message_filters

from camera_interface import (
    CameraInterface, CameraConfig, Detection, Intrinsics
)
from seabird_config import (
    IMG_W, IMG_H, FX, FY, CX, CY,
    camera_to_world, nearest_buoy_error, BUOY_POSITIONS
)
from yolo_detector import YoloDetector
import os
import cv2

# Default topic prefix — matches init_scene.py's ROS2CameraGraph config
#DEFAULT_TOPIC_PREFIX = "/iris_0/front_cam"
DEFAULT_TOPIC_PREFIX = "/zed/zed_node"

# Drone pose topic — published by Pegasus ROS2Backend
#DRONE_POSE_TOPIC = "/drone00/state/pose"
DRONE_POSE_TOPIC = "/zed/zed_node/pose"

# Known ground-truth distance from camera to object (meters)
#TRUE_DISTANCE = 0.4826


class SimCamera(CameraInterface, Node):
    """
    ROS2-based camera for Isaac Sim.

    Subscribes to:
        /zed/zed_node/rgb/color/rect/image         — RGB image (rgb8 encoding)
        /zed/zed_node/depth/depth_registered       — depth map (32FC1, meters)
        /zed/zed_node/rgb/color/rect/camera_info — intrinsics (grabbed once)
        /drone00/state/pose           — drone pose (PoseStamped from ROS2Backend)

    Publishes:
        /seabird/buoy_detections      — JSON detection messages (std_msgs/String)
    """

    def __init__(self, node_name: str = "sim_camera",
                 topic_prefix: str = DEFAULT_TOPIC_PREFIX):
        Node.__init__(self, node_name)

        self._topic_prefix = topic_prefix

        # Frame storage — written by ROS2 callback thread, read by main thread.
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._intrinsics: Optional[Intrinsics] = None
        self._new_frame: bool = False
        self._frame_lock = threading.Lock()

        # Drone pose — written by pose callback, read when transforming detections.
        # Separate lock because pose updates faster than camera frames.
        self._drone_pos: Optional[np.ndarray] = None
        self._drone_quat_wxyz: Optional[np.ndarray] = None
        self._pose_lock = threading.Lock()

        # State
        self._is_open: bool = False
        self._config: CameraConfig = CameraConfig()

        # Detection (Level 2)
        self._detector: Optional[YoloDetector] = None

        # Publisher — created in open(), used by main loop
        self.detection_pub = None

        self.get_logger().info(f"SimCamera created — topics: {topic_prefix}/*")

    # ── Lifecycle ──────────────────────────────────────────────────

    def configure(self, config: CameraConfig) -> None:
        self._config = config
        self.get_logger().info(
            f"Configured: {config.width}x{config.height} @ {config.fps}fps"
        )

    def open(self) -> bool:
        if self._is_open:
            self.get_logger().warn("Already open")
            return True

        # QoS: BEST_EFFORT matches Isaac's publishers (Lesson #42)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # camera_info — grab once, then unsubscribe
        self._info_sub = self.create_subscription(
            CameraInfo,
            f"{self._topic_prefix}/rgb/color/rect/camera_info",
            self._on_camera_info,
            qos
        )

        # RGB + depth — synced via ApproximateTimeSynchronizer
        rgb_sub = message_filters.Subscriber(
            self, Image, f"{self._topic_prefix}/rgb/color/rect/image", qos_profile=qos
        )
        depth_sub = message_filters.Subscriber(
            self, Image, f"{self._topic_prefix}/depth/depth_registered", qos_profile=qos
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=5,
            slop=0.05
        )
        self._sync.registerCallback(self._on_synced_frame)

        # Detection publisher — JSON over String, consumed by sweep_and_detect.py
        self.detection_pub = self.create_publisher(
            String, '/seabird/buoy_detections', 10
        )

        # Drone pose — from Pegasus ROS2Backend
        # PoseStamped contains position (x,y,z) + orientation (x,y,z,w quaternion)
        # in Isaac world frame (ENU)
        self._pose_sub = self.create_subscription(
            PoseStamped,
            DRONE_POSE_TOPIC,
            self._on_drone_pose,
            qos
        )

        self._is_open = True
        self.get_logger().info("SimCamera open — waiting for frames + pose...")
        return True

    def close(self) -> None:
        self._is_open = False
        self.get_logger().info("SimCamera closed")

    def grab(self) -> bool:
        if not self._is_open:
            return False
        rclpy.spin_once(self, timeout_sec=0.05)
        with self._frame_lock:
            if self._new_frame:
                self._new_frame = False
                return True
        return False

    # ── Level 1: Data Source ───────────────────────────────────────

    def get_rgb(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._rgb.copy() if self._rgb is not None else None

    def get_depth(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._depth.copy() if self._depth is not None else None

    def get_intrinsics(self) -> Optional[Intrinsics]:
        return self._intrinsics

    def get_drone_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (position, quaternion_wxyz) or (None, None) if no pose yet.
        Position is in Isaac world frame (ENU, meters).
        Quaternion is [w, x, y, z] — matches Isaac's convention.
        """
        with self._pose_lock:
            if self._drone_pos is None:
                return None, None
            return self._drone_pos.copy(), self._drone_quat_wxyz.copy()

    # ── Level 2: Perception ───────────────────────────────────────

    def enable_detection(self, model_path: str, class_names: List[str],
                         enable_tracking: bool = True) -> bool:
        self._detector = YoloDetector(
            weights=model_path,
            class_names=class_names,
            imgsz=320,
            conf_thresh=0.5,
        )
        ok = self._detector.start(enable_tracking=enable_tracking)
        if not ok:
            self._detector = None
            self.get_logger().error("YoloDetector failed to start")
        return ok

    def get_detections(self) -> List[Detection]:
        if self._detector is None:
            return []

        with self._frame_lock:
            rgb = self._rgb.copy() if self._rgb is not None else None
            depth = self._depth.copy() if self._depth is not None else None

        if rgb is None:
            return []

        return self._detector.detect(rgb, depth, self._intrinsics)

    def is_open(self) -> bool:
        return self._is_open

    # ── ROS2 Callbacks ─────────────────────────────────────────────

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._intrinsics is not None:
            return
        K = msg.k
        self.get_logger().warn(
            f"ROS2 camera_info reports fx={K[0]:.1f} — WRONG (default lens). "
            f"Using seabird_config: fx={FX:.1f}"
        )
        self._intrinsics = Intrinsics(
            fx=FX, fy=FY, cx=CX, cy=CY,
            width=IMG_W, height=IMG_H
        )
        self.get_logger().info(
            f"Intrinsics (from config): fx={FX:.1f} fy={FY:.1f} "
            f"cx={CX:.1f} cy={CY:.1f} {IMG_W}x{IMG_H}"
        )
        self.destroy_subscription(self._info_sub)

    def _on_synced_frame(self, rgb_msg: Image, depth_msg: Image) -> None:
        channels = len(rgb_msg.data) // (rgb_msg.height * rgb_msg.width)
        rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, channels
        )
        bgr = rgb[:, :, :3][:, :, ::-1].copy()  # RGB(A) → drop alpha if present → BGR
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
            depth_msg.height, depth_msg.width
        ).copy()
        with self._frame_lock:
            self._rgb = bgr
            self._depth = depth
            self._new_frame = True

    def _on_drone_pose(self, msg: PoseStamped) -> None:
        """
        Fired when Pegasus ROS2Backend publishes drone state.

        ROS PoseStamped quaternion is (x, y, z, w) in the message fields.
        We store as (w, x, y, z) to match Isaac's convention and what
        camera_to_world() expects.
        """
        p = msg.pose.position
        q = msg.pose.orientation
        with self._pose_lock:
            self._drone_pos = np.array([p.x, p.y, p.z])
            self._drone_quat_wxyz = np.array([q.w, q.x, q.y, q.z])


# ─── Standalone Test ──────────────────────────────────────────────────

def main(model="yolov8s.pt", display=False, TRUE_DISTANCE = 0.4826):
    """
    Opens SimCamera with YOLO detection, world-frame validation,
    and publishes detections to /seabird/buoy_detections.

    For each detection:
      1. YOLO gives camera-frame 3D position (from depth back-projection)
      2. camera_to_world() transforms it to Isaac world coordinates
      3. nearest_buoy_error() compares against ground truth
      4. Publishes JSON detection msg for sweep_and_detect.py

    Run while Isaac + init_scene.py + PX4 are active and drone is flying.
    """

    """
    YOLO_WEIGHTS = os.path.expanduser(
        "models/best_alex.pt"
    )"""

    YOLO_DEFAULTS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    YOLO_WEIGHTS = model
    CLASS_NAMES = ["red_buoy", "green_buoy", "blue_buoy"]
    DRAW_COLORS = {
        "red_buoy":   (0, 0, 255),
        "green_buoy": (0, 255, 0),
        "blue_buoy":  (255, 0, 0),
    }

    KNOWN_DISTANCE = 1.0414  # meters

    DEBUG_DIR = os.path.expanduser("~/seabird_dataset/debug_live")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    SAVE_EVERY_N = 30

    rclpy.init()
    cam = SimCamera()
    cam.configure(CameraConfig(width=IMG_W, height=IMG_H))

    if not cam.open():
        print("[sim_camera] Failed to open")
        rclpy.shutdown()
        return

    # Enable detection if weights exist
    detection_on = False
    if os.path.exists(YOLO_WEIGHTS) or YOLO_WEIGHTS in YOLO_DEFAULTS:
        print(f"[sim_camera] Loading YOLO: {YOLO_WEIGHTS}")
        detection_on = cam.enable_detection(
            YOLO_WEIGHTS, CLASS_NAMES, enable_tracking=True
        )
        if detection_on:
            print("[sim_camera] Detection ENABLED")
        else:
            print("[sim_camera] Detection failed — running RGB-only")
    else:
        print(f"[sim_camera] No weights at {YOLO_WEIGHTS} — RGB-only mode")

    print("[sim_camera] Waiting for frames + drone pose...")
    print("[sim_camera] Publishing detections → /seabird/buoy_detections")
    print("[sim_camera] (debug frames → ~/seabird_dataset/debug_live/)")

    frame_count = 0
    intrinsics_printed = False
    pose_printed = False

    try:
        while rclpy.ok():
            if cam.grab():
                frame_count += 1
                rgb = cam.get_rgb()
                depth = cam.get_depth()
                intr = cam.get_intrinsics()
                drone_pos, drone_quat = cam.get_drone_pose()

                if intr and not intrinsics_printed:
                    print(f"[sim_camera] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                          f"cx={intr.cx:.1f} cy={intr.cy:.1f} "
                          f"{intr.width}x{intr.height}")
                    intrinsics_printed = True

                # Log first pose receipt
                if drone_pos is not None and not pose_printed:
                    print(f"[sim_camera] Drone pose received: "
                          f"pos=({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}) "
                          f"quat=({drone_quat[0]:.3f}, {drone_quat[1]:.3f}, "
                          f"{drone_quat[2]:.3f}, {drone_quat[3]:.3f})")
                    pose_printed = True

                # Run detection, transform to world, validate, publish
                if detection_on and rgb is not None:
                    dets = cam.get_detections()

                    for d in dets:
                        x1, y1, x2, y2 = d.bbox_2d
                        color = DRAW_COLORS.get(d.label, (255, 255, 255))
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)

                        # Build annotation text
                        txt = f"{d.label} {d.confidence:.2f}"
                        if d.tracking_id >= 0:
                            txt += f" #{d.tracking_id}"

                        # Depth error vs known ground-truth distance
                        if d.position_3d is not None and intr is not None:
                            cx_px = (d.bbox_2d[0] + d.bbox_2d[2]) // 2
                            cy_px = (d.bbox_2d[1] + d.bbox_2d[3]) // 2
                            dx = (cx_px - intr.cx) / intr.fx
                            dy = (cy_px - intr.cy) / intr.fy
                            expected_z = TRUE_DISTANCE / np.sqrt(1 + dx**2 + dy**2)
                            depth_err = d.position_3d[2] - expected_z
                            txt += f" dz={depth_err:+.3f}m"

                        # World-frame transform + validation + publish
                        world_pos = None
                        if d.position_3d is not None and drone_pos is not None:
                            world_pos = camera_to_world(
                                d.position_3d, drone_pos, drone_quat
                            )
                            gt_name, err_m, gt_pos = nearest_buoy_error(world_pos)

                            # Publish detection for sweep_and_detect.py
                            det_msg = String()
                            det_msg.data = json.dumps({
                                'color': d.label.replace('_buoy', ''),
                                'world_position': world_pos.tolist(),
                                'confidence': float(d.confidence),
                                'drone_position': drone_pos.tolist(),
                                'timestamp': time.time(),
                            })
                            cam.detection_pub.publish(det_msg)

                            # Draw world position on frame
                            cv2.putText(
                                rgb,
                                f"W({world_pos[0]:.1f},{world_pos[1]:.1f},{world_pos[2]:.1f})",
                                (x1, y2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
                            )
                        elif d.position_3d is not None:
                            txt += f" ({d.position_3d[2]:.1f}m)"

                        cv2.putText(rgb, txt, (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                    # Periodic console logging with world-frame validation
                    if frame_count % 30 == 0 and dets:
                        print(f"--- Frame {frame_count} | "
                              f"drone=({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}) ---"
                              if drone_pos is not None else
                              f"--- Frame {frame_count} | NO POSE ---")
                        for d in dets:
                            cam_str = ""
                            world_str = ""
                            depth_err_str = ""
                            if d.position_3d is not None:
                                cx, cy, cz = d.position_3d
                                cam_str = f" cam=({cx:.2f},{cy:.2f},{cz:.2f})"
                                if intr is not None:
                                    cx_px = (d.bbox_2d[0] + d.bbox_2d[2]) // 2
                                    cy_px = (d.bbox_2d[1] + d.bbox_2d[3]) // 2
                                    dx = (cx_px - intr.cx) / intr.fx
                                    dy = (cy_px - intr.cy) / intr.fy
                                    expected_z = TRUE_DISTANCE / np.sqrt(1 + dx**2 + dy**2)
                                    depth_err_str = f" depth_err={cz - expected_z:+.4f}m(measured={cz:.4f} expected={expected_z:.4f})"
                            if d.position_3d is not None and drone_pos is not None:
                                wp = camera_to_world(
                                    d.position_3d, drone_pos, drone_quat
                                )
                                gt_name, err_m, gt_pos = nearest_buoy_error(wp)
                                world_str = (
                                    f" world=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})"
                                    f" → {gt_name} gt={gt_pos}"
                                    f" err={err_m:.2f}m"
                                )
                            print(f"  {d.label} conf={d.confidence:.2f} "
                                  f"tid={d.tracking_id}{cam_str}{depth_err_str}{world_str}")

                # Live display — press 'q' to quit
                if display and rgb is not None:
                    cv2.imshow("SimCamera - detections", rgb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Save annotated frames to disk
                if frame_count % SAVE_EVERY_N == 0 and rgb is not None:
                    
                    out_path = os.path.join(DEBUG_DIR, f"frame_{frame_count:06d}.png")
                    cv2.imwrite(out_path, rgb)
                    print(f"Saved {frame_count:06d}.png to {DEBUG_DIR}")
                    if depth is not None:
                        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        depth_clipped = np.clip(depth_clean, 0, 30)
                        depth_norm = (depth_clipped / 30.0 * 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        depth_path = os.path.join(DEBUG_DIR, f"depth_{frame_count:06d}.png")
                        cv2.imwrite(depth_path, depth_color)

                if frame_count % 100 == 0:
                    if depth is not None:
                        valid = depth[np.isfinite(depth)]
                        if len(valid) > 0:
                            print(f"[sim_camera] Frame {frame_count} | "
                                  f"depth min={valid.min():.2f}m "
                                  f"max={valid.max():.2f}m "
                                  f"median={np.median(valid):.2f}m")

    except KeyboardInterrupt:
        print("\n[sim_camera] Interrupted")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        print(f"[sim_camera] Done — {frame_count} frames processed")
        print(f"[sim_camera] Debug frames saved to {DEBUG_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Seabird object detector", description = "Run seabird object detector with chosen model")
    parser.add_argument('--model', '-m', default="models/best_alex.pt", type=str, help="enter model paths")
    parser.add_argument('--display', '-d', action='store_true', help="show live cv2 window")
    parser.add_argument('--true_dist', '-td', type=float, default=0.4826, help="input distance of desired object")
    args = parser.parse_args()
    main(args.model, args.display, args.true_dist)

#!/usr/bin/env python3
"""
gt_label_viz.py — Visual verification of 3D-to-pixel projection math.

Reads ground-truth drone pose from gt_pose_writer.py (JSON file),
grabs camera frames from SimCamera, projects known buoy world positions
into pixel coordinates, and draws circles + bounding boxes on the image.

If the circles sit exactly on the buoys in the camera feed, the math is correct
and we can use it to generate YOLO training labels.

Run in terminal with Isaac + init_scene.py + gt_pose_writer.py + PX4 all running:
    source /opt/ros/humble/setup.bash
    cd ~/seabird/scripts
    python3 gt_label_viz.py

Press 'q' to quit. Press 'd' to toggle debug text overlay.
"""

import json
import os
import sys
import time
import cv2
import numpy as np
import rclpy

from camera_interface import CameraConfig, Intrinsics
from sim_camera import SimCamera

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — must match init_scene.py
# ═══════════════════════════════════════════════════════════════════

POSE_FILE = os.path.expanduser("~/seabird/logs/drone_pose.json")

# Camera mount in drone body frame (from init_scene.py)
CAMERA_MOUNT_TRANSLATION = np.array([0.30, 0.0, 0.05])  # 30cm forward, 5cm up
CAMERA_PITCH_DEG = 15.0
CAMERA_MOUNT_EULER_XYZ = [90.0 + CAMERA_PITCH_DEG, 0.0, -90.0]  # [105, 0, -90]

# Buoy physical diameter for bounding box estimation
BUOY_DIAMETER_M = 0.46  # ~18 inches

# Colors for drawing (BGR)
BUOY_COLORS = {
    "red_buoy":   (0, 0, 255),
    "green_buoy": (0, 255, 0),
    "blue_buoy":  (255, 0, 0),
}

# YOLO class IDs
BUOY_CLASS_IDS = {
    "red_buoy":   0,
    "green_buoy": 1,
    "blue_buoy":  2,
}


# ═══════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════

def quat_xyzw_to_rotmat(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    
    This is the standard formula. The quaternion represents a rotation:
    R @ v rotates vector v by the rotation described by q.
    
    Pegasus state.attitude uses [x,y,z,w] (scipy convention) and represents
    the body-to-world rotation: p_world = R @ p_body + pos_drone.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)]
    ])


def euler_xyz_to_rotmat(angles_deg):
    """Convert extrinsic XYZ euler angles (degrees) to rotation matrix.
    
    USD RotateXYZ applies: first rotate about X, then Y, then Z (fixed axes).
    Matrix form: R = Rz(c) @ Ry(b) @ Rx(a)
    
    This matrix transforms points FROM the prim's local frame TO the parent frame:
        p_parent = R @ p_local + translation
    """
    a, b, c = np.radians(angles_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    Ry = np.array([
        [ np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    Rz = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


# Precompute camera mount rotation (body frame → camera local frame)
# R_body_cam transforms FROM camera local TO body: p_body = R @ p_cam_local + t
# To go body → camera: p_cam_local = R^T @ (p_body - t)
R_BODY_CAM = euler_xyz_to_rotmat(CAMERA_MOUNT_EULER_XYZ)


def project_buoy_to_pixel(p_world_buoy, drone_pos, drone_quat_xyzw, intrinsics):
    """
    Full projection chain: world 3D → camera pixel (u, v).
    
    Returns (u, v, depth_m) or None if buoy is behind the camera.
    
    The chain:
        1. World → Body:   subtract drone position, rotate by inverse of drone orientation
        2. Body → Camera:  subtract mount offset, rotate by inverse of mount rotation
        3. USD → OpenCV:   flip Y and Z (USD: +Y up, -Z forward → OpenCV: +Y down, +Z forward)
        4. Pinhole:        u = fx * X/Z + cx,  v = fy * Y/Z + cy
    """
    p_world = np.array(p_world_buoy)
    pos = np.array(drone_pos)

    # Step 1: World → Body frame
    # Pegasus attitude quat is body-to-world: p_world = R_b2w @ p_body + pos
    # Inverse: p_body = R_b2w^T @ (p_world - pos)
    R_body_to_world = quat_xyzw_to_rotmat(drone_quat_xyzw)
    R_world_to_body = R_body_to_world.T
    p_body = R_world_to_body @ (p_world - pos)

    # Step 2: Body → Camera (USD local frame)
    # Camera prim transform: p_body = R_BODY_CAM @ p_cam_local + mount_t
    # Inverse: p_cam_local = R_BODY_CAM^T @ (p_body - mount_t)
    p_cam_usd = R_BODY_CAM.T @ (p_body - CAMERA_MOUNT_TRANSLATION)

    # Step 3: USD camera → OpenCV camera convention
    # USD:    +X right, +Y up,   -Z into scene (viewing direction)
    # OpenCV: +X right, +Y down, +Z into scene (depth)
    p_cam = np.array([p_cam_usd[0], -p_cam_usd[1], -p_cam_usd[2]])

    # Behind camera check
    if p_cam[2] <= 0.1:
        return None

    # Step 4: Pinhole projection
    u = intrinsics.fx * p_cam[0] / p_cam[2] + intrinsics.cx
    v = intrinsics.fy * p_cam[1] / p_cam[2] + intrinsics.cy

    return (int(round(u)), int(round(v)), float(p_cam[2]))


def estimate_bbox_pixels(depth_m, fx):
    """Estimate buoy bounding box size in pixels from depth.
    
    A buoy of known diameter at a given depth projects to:
        pixel_size = real_diameter * focal_length / depth
    """
    if depth_m <= 0:
        return 0
    return int(round(BUOY_DIAMETER_M * fx / depth_m))


# ═══════════════════════════════════════════════════════════════════
# POSE FILE READER
# ═══════════════════════════════════════════════════════════════════

def read_pose():
    """Read the latest drone pose + buoy positions from gt_pose_writer's JSON file.
    
    Returns (drone_pos, drone_quat_xyzw, buoy_dict) or (None, None, None) on failure.
    Handles partial writes gracefully — gt_pose_writer uses atomic rename.
    """
    try:
        with open(POSE_FILE, 'r') as f:
            data = json.load(f)
        return (
            data["position"],
            data["attitude_xyzw"],
            data["buoys"]
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None, None, None


# ═══════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    cam = SimCamera()
    cam.configure(CameraConfig(width=640, height=480))

    if not cam.open():
        print("[gt_viz] Failed to open SimCamera")
        rclpy.shutdown()
        return

    print("[gt_viz] Waiting for frames and pose data...")
    print("[gt_viz] Press 'q' to quit, 'd' to toggle debug overlay")

    show_debug = True
    frame_count = 0
    intrinsics = None

    try:
        while rclpy.ok():
            if not cam.grab():
                continue

            frame_count += 1
            rgb = cam.get_rgb()  # BGR uint8
            if rgb is None:
                continue

            # Get intrinsics (once)
            if intrinsics is None:
                intrinsics = cam.get_intrinsics()
                if intrinsics is None:
                    continue
                print(f"[gt_viz] Intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
                      f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}")

            # Read drone pose
            drone_pos, drone_quat, buoys = read_pose()
            if drone_pos is None or buoys is None:
                # No pose yet — just show raw feed
                cv2.imshow("GT Label Viz", rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Project each buoy and draw
            viz = rgb.copy()
            for label, buoy_world_pos in buoys.items():
                color = BUOY_COLORS.get(label, (255, 255, 255))

                result = project_buoy_to_pixel(
                    buoy_world_pos, drone_pos, drone_quat, intrinsics
                )

                if result is None:
                    continue  # behind camera

                u, v, depth = result

                # Skip if outside image with generous margin
                margin = 50
                if (u < -margin or u > intrinsics.width + margin or
                    v < -margin or v > intrinsics.height + margin):
                    continue

                # Bounding box size from projected diameter
                box_size = estimate_bbox_pixels(depth, intrinsics.fx)
                half = box_size // 2

                # Draw circle at projected center
                cv2.circle(viz, (u, v), max(half, 5), color, 2)

                # Draw bounding box
                cv2.rectangle(viz,
                    (u - half, v - half),
                    (u + half, v + half),
                    color, 1)

                # Label with class name and depth
                if show_debug:
                    text = f"{label.split('_')[0]} {depth:.1f}m"
                    cv2.putText(viz, text, (u - half, v - half - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Debug overlay: drone position
            if show_debug:
                pos_text = f"drone: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})"
                cv2.putText(viz, pos_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(viz, f"frame: {frame_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("GT Label Viz", viz)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug

    except KeyboardInterrupt:
        print("\n[gt_viz] Interrupted")
    finally:
        cam.close()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print(f"[gt_viz] Done — {frame_count} frames")


if __name__ == "__main__":
    main()

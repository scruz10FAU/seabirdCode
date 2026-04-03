#!/usr/bin/env python3
"""
seabird/sweep_and_detect.py  v3
================================
Flies an expanding rectangular spiral over the course area.
Stops early once all target buoy colors are detected.
Reads detections from /seabird/buoy_detections via rclpy in a background thread.
Publishes flight path on /seabird/flight_path (nav_msgs/Path) for RViz2.
Also publishes /seabird/path_markers (visualization_msgs/Marker) as a LINE_STRIP
  so you can see the trail even without a full TF tree.

Prerequisites (all should already be running before this script):
  1. Isaac Sim + spawn_drone.py completed ("[init] Done")
  2. PX4 SITL connected ("Ready for takeoff!")
  3. buoy_detector.py running in another terminal

PX4 params to set once (in pxh, then param save):
  param set MPC_XY_CRUISE 2.0
  param set MPC_XY_VEL_MAX 3.0
  param set MPC_Z_VEL_MAX_UP 1.5
  param set MPC_Z_VEL_MAX_DN 1.0
  param set SYS_HAS_MAG 0
  param set COM_ARM_MAG_STR 0
  param set EKF2_ABL_LIM 5.0
  param save
"""

import asyncio
import math
import json
import threading
import sys

import rclpy
from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw
    from mavsdk.action import ActionError
except ImportError:
    print("[ERROR] mavsdk not installed. Run: pip install mavsdk --break-system-packages")
    sys.exit(1)


# ── Mission configuration ─────────────────────────────────────────────────────
TAKEOFF_ALT_M     = 5.0
WAYPOINT_TOL_M    = 2.5     # slightly relaxed for slow cruise
WAYPOINT_TIMEOUT  = 60.0
HOVER_STABILIZE_S = 8.0

SPIRAL_STEP_M     = 8.0
SPIRAL_RINGS      = 5
SPIRAL_BIAS_EAST  = 1.4
SPIRAL_BIAS_PLUS  = 0.2
SPIRAL_BIAS_NORTH = 0.6

SWEEP_YAW_DEG     = 270.0
TARGET_COLORS     = {"red", "green", "blue"}
MAVSDK_ADDRESS    = "udp://:14540"

# ── Shared state ──────────────────────────────────────────────────────────────
_detected_buoys: set = set()
_lock = threading.Lock()

_path_pub = None
_path_msg = None
_marker_pub = None
_marker_msg = None
_rclpy_node = None


# ── ROS2 listener thread ──────────────────────────────────────────────────────

def _rclpy_thread_fn():
    global _path_pub, _path_msg, _marker_pub, _marker_msg, _rclpy_node
    rclpy.init()
    node = rclpy.create_node("sweep_listener")
    _rclpy_node = node

    # Path publisher (nav_msgs/Path)
    _path_pub = node.create_publisher(Path, "/seabird/flight_path", 10)
    _path_msg = Path()
    _path_msg.header.frame_id = "map"

    # Marker publisher (LINE_STRIP — works in RViz2 without full TF)
    _marker_pub = node.create_publisher(Marker, "/seabird/path_marker", 10)
    _marker_msg = Marker()
    _marker_msg.header.frame_id = "map"
    _marker_msg.ns = "flight_path"
    _marker_msg.id = 0
    _marker_msg.type = Marker.LINE_STRIP
    _marker_msg.action = Marker.ADD
    _marker_msg.scale.x = 0.3  # line width
    _marker_msg.color = ColorRGBA(r=0.0, g=1.0, b=0.5, a=1.0)  # bright green
    _marker_msg.pose.orientation.w = 1.0

    def _cb(msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        color = data.get("color", "")
        if not color:
            return
        with _lock:
            if color not in _detected_buoys:
                _detected_buoys.add(color)
                node.get_logger().info(
                    f"[sweep_listener] ★ NEW: {color}   "
                    f"total={len(_detected_buoys)}/{len(TARGET_COLORS)}"
                )

    node.create_subscription(String, "/seabird/buoy_detections", _cb, 10)
    node.get_logger().info("[sweep_listener] Subscribed to /seabird/buoy_detections")
    node.get_logger().info("[sweep_listener] Publishing: /seabird/flight_path, /seabird/path_marker")
    rclpy.spin(node)


def start_rclpy_listener() -> threading.Thread:
    t = threading.Thread(target=_rclpy_thread_fn, daemon=True, name="rclpy_listener")
    t.start()
    return t


# ── Spiral waypoint generator ─────────────────────────────────────────────────

def generate_spiral_waypoints(rings, step, alt, yaw):
    wps = []
    for ring in range(1, rings + 1):
        r = ring * step
        n_max =  r * SPIRAL_BIAS_NORTH
        n_min = -r * SPIRAL_BIAS_NORTH
        e_max =  r * SPIRAL_BIAS_PLUS
        e_min = -r * SPIRAL_BIAS_EAST
        d     = -alt

        wps += [
            (n_max, e_max, d, yaw),
            (n_max, e_min, d, yaw),
            (n_min, e_min, d, yaw),
            (n_min, e_max, d, yaw),
        ]
    return wps


# ── Drone state tracker ──────────────────────────────────────────────────────

class _DroneState:
    north_m: float = 0.0
    east_m:  float = 0.0
    down_m:  float = 0.0

_state = _DroneState()


async def track_position(drone: System):
    async for pv in drone.telemetry.position_velocity_ned():
        _state.north_m = pv.position.north_m
        _state.east_m  = pv.position.east_m
        _state.down_m  = pv.position.down_m


# ── Path publishing ──────────────────────────────────────────────────────────

def publish_path_point():
    """Append current drone position to Path + Marker and publish both."""
    if _path_pub is None or _rclpy_node is None:
        return

    now = _rclpy_node.get_clock().now().to_msg()

    # Nav path
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = now
    pose.pose.position.x = _state.north_m
    pose.pose.position.y = _state.east_m
    pose.pose.position.z = -_state.down_m
    _path_msg.header.stamp = now
    _path_msg.poses.append(pose)
    _path_pub.publish(_path_msg)

    # LINE_STRIP marker
    pt = Point(x=_state.north_m, y=_state.east_m, z=-_state.down_m)
    _marker_msg.header.stamp = now
    _marker_msg.points.append(pt)
    _marker_pub.publish(_marker_msg)


# ── Flight helpers ────────────────────────────────────────────────────────────

async def fly_to(drone, north, east, down, yaw,
                 tol=WAYPOINT_TOL_M, timeout=WAYPOINT_TIMEOUT):
    await drone.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw))
    t0 = asyncio.get_event_loop().time()
    last_pub = 0

    while True:
        await asyncio.sleep(0.1)

        dist = math.sqrt(
            (_state.north_m - north) ** 2
            + (_state.east_m - east) ** 2
            + (_state.down_m - down) ** 2
        )
        if dist < tol:
            publish_path_point()
            return True

        elapsed = asyncio.get_event_loop().time() - t0
        if elapsed > timeout:
            print(f"  [fly_to] ⚠ Timeout ({timeout:.0f}s) at N={north:.1f} E={east:.1f}")
            return False

        # Publish path breadcrumb every 0.5s
        if elapsed - last_pub >= 0.5:
            publish_path_point()
            last_pub = elapsed

        # Log progress every 5 seconds
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            print(
                f"  [fly_to] → N={_state.north_m:.1f}/{north:.1f}  "
                f"E={_state.east_m:.1f}/{east:.1f}  "
                f"dist={dist:.1f}m  t={elapsed:.0f}s"
            )


def all_found():
    with _lock:
        return _detected_buoys >= TARGET_COLORS

def found_count():
    with _lock:
        return len(_detected_buoys)

def found_set():
    with _lock:
        return set(_detected_buoys)


# ── Mission ───────────────────────────────────────────────────────────────────

async def run_mission():

    drone = System()
    print(f"[sweep] Connecting to PX4 at {MAVSDK_ADDRESS}...")
    await drone.connect(system_address=MAVSDK_ADDRESS)

    print("[sweep] Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[sweep] ✓ Connected to PX4")
            break

    print("[sweep] Waiting for GPS fix and home position...")
    async for health in drone.telemetry.health():
        gps_ok  = health.is_global_position_ok
        home_ok = health.is_home_position_ok
        if gps_ok and home_ok:
            print("[sweep] ✓ GPS OK, home set")
            break
        if not gps_ok:
            print("  [sweep] ... waiting for global position estimate")
        await asyncio.sleep(1.0)

    asyncio.ensure_future(track_position(drone))
    await asyncio.sleep(0.5)
    print(f"[sweep] Drone at N={_state.north_m:.2f} E={_state.east_m:.2f} "
          f"D={_state.down_m:.2f}")

    print("[sweep] Arming...")
    try:
        await drone.action.arm()
    except ActionError as e:
        print(f"[sweep] Arm failed: {e} — is PX4 ready?")
        return
    print("[sweep] ✓ Armed")

    print(f"[sweep] Taking off to {TAKEOFF_ALT_M}m...")
    await drone.action.set_takeoff_altitude(TAKEOFF_ALT_M)
    await drone.action.takeoff()
    print(f"[sweep] Waiting {HOVER_STABILIZE_S:.0f}s to stabilize at altitude...")
    await asyncio.sleep(HOVER_STABILIZE_S)
    print(f"[sweep] ✓ Airborne — N={_state.north_m:.1f} E={_state.east_m:.1f} "
          f"Alt≈{-_state.down_m:.1f}m")

    print("[sweep] Switching to offboard mode...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(_state.north_m, _state.east_m, -TAKEOFF_ALT_M, SWEEP_YAW_DEG)
    )
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[sweep] Offboard start failed: {e}")
        await drone.action.land()
        return
    print("[sweep] ✓ Offboard active")

    waypoints = generate_spiral_waypoints(
        rings=SPIRAL_RINGS, step=SPIRAL_STEP_M,
        alt=TAKEOFF_ALT_M, yaw=SWEEP_YAW_DEG,
    )
    total_wps = len(waypoints)
    print(f"\n[sweep] Spiral: {SPIRAL_RINGS} rings × 4 corners = {total_wps} waypoints")
    print(f"[sweep] Step={SPIRAL_STEP_M}m  Alt={TAKEOFF_ALT_M}m  Yaw={SWEEP_YAW_DEG}°")
    print(f"[sweep] Target: {TARGET_COLORS}")
    print("[sweep] ─────────────────────────────────────")

    for i, (n, e, d, yaw) in enumerate(waypoints):
        ring = i // 4 + 1
        corner = ["NE", "SE", "SW", "NW"][i % 4]

        if all_found():
            print(f"\n[sweep] ★★★ All {len(TARGET_COLORS)} buoys found! "
                  f"Exiting sweep at WP {i+1}/{total_wps} (ring {ring})")
            break

        print(
            f"[sweep] WP {i+1:2d}/{total_wps}  ring={ring} {corner:2s}  "
            f"N={n:+6.1f}  E={e:+6.1f}  "
            f"found={found_count()}/{len(TARGET_COLORS)} {found_set()}"
        )

        reached = await fly_to(drone, n, e, d, yaw)
        if not reached:
            print(f"  [sweep] Skipping to next waypoint")

        await asyncio.sleep(1.5)

    print("\n[sweep] Returning to launch...")
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    await drone.action.return_to_launch()

    print("[sweep] Waiting for landing...")
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("[sweep] ✓ Landed")
            break
        await asyncio.sleep(1.0)

    found   = found_set()
    missing = TARGET_COLORS - found
    print("\n" + "═" * 50)
    print("  SEABIRD SWEEP COMPLETE")
    print("═" * 50)
    print(f"  Buoys detected : {sorted(found) or 'none'}")
    if missing:
        print(f"  NOT found      : {sorted(missing)}")
    else:
        print("  ✓ All target buoys accounted for")
    print("═" * 50 + "\n")


def main():
    print("=" * 60)
    print("  SEABIRD — Sweep and Detect")
    print("=" * 60)
    print("  Make sure these are already running:")
    print("  1. Isaac Sim + spawn_drone.py ('[init] Done')")
    print("  2. PX4 SITL ('Ready for takeoff!')")
    print("  3. python3 buoy_detector.py")
    print("=" * 60 + "\n")

    start_rclpy_listener()
    print("[sweep] ROS2 detection listener started\n")

    asyncio.run(run_mission())


if __name__ == "__main__":
    main()
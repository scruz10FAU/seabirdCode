#!/usr/bin/env python3
"""
sweep_and_detect.py — Autonomous lawnmower sweep with live buoy detection.

Architecture:
  - Main thread: MAVSDK asyncio loop (arm → takeoff → offboard lawnmower)
  - Background thread: rclpy subscriber for /seabird/buoy_detections
  - Shared: DetectionLedger (thread-safe, proximity-deduped)

Runs as: terminal script (Python 3.10)
Prerequisites:
  - python3 -m pip install --user mavsdk
  - source /opt/ros/humble/setup.bash
  - sim_camera.py running and publishing to /seabird/buoy_detections
  - PX4 SITL running and ready

Usage:
  source /opt/ros/humble/setup.bash
  python3 ~/seabird/scripts/sweep_and_detect.py
"""

import sys
import os
import asyncio
import json
import time
import threading
import math
from dataclasses import dataclass, asdict
from typing import List

# Seabird config lives at ~/seabird/scripts/
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "seabird", "scripts"))
from seabird_config import BUOY_POSITIONS

# ROS2 (system Python 3.10 — do NOT insert 3.11 paths here, Lesson #52)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw


# ═══════════════════════════════════════════════════════
# SWEEP CONFIGURATION
# Stays here until validated, then moves to seabird_config
# ═══════════════════════════════════════════════════════

SWEEP_ALTITUDE = 8.0         # meters — within YOLO training range
SWEEP_SPACING = 8.0          # meters between east-west passes
CONFIDENCE_THRESHOLD = 0.60  # minimum YOLO confidence to log
DEDUP_RADIUS = 3.0           # meters — detections closer than this = same buoy
WAYPOINT_TOLERANCE = 1.5     # meters — "close enough" to advance
WAYPOINT_TIMEOUT = 60.0      # seconds — skip waypoint if not reached

# Survey area in Isaac ENU coordinates (what we want the camera to cover)
# Buoys are at roughly x=[-3.3, 3.1], y=[-15.3, -13.5] — add margin
SURVEY_X_MIN = -8.0    # East boundary (west end of sweep)
SURVEY_X_MAX = 8.0     # East boundary (east end of sweep)
SURVEY_Y_MIN = -18.0   # North boundary (deep into marina)
SURVEY_Y_MAX = -8.0    # North boundary (closer to spawn)

# PX4 home position in Isaac ENU — must match init_scene.py spawn position
# NED waypoints are computed relative to this
HOME_ENU = (0.0, -2.0, 0.0)

# Output
LEDGER_PATH = os.path.expanduser("~/seabird/sweep_ledger.json")


# ═══════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════

@dataclass
class BuoyDetection:
    """Single detection received from sim_camera via ROS2."""
    color: str
    world_position: List[float]   # Isaac ENU [x, y, z]
    confidence: float
    drone_position: List[float]   # Isaac ENU [x, y, z]
    timestamp: float


@dataclass
class LedgerEntry:
    """Confirmed buoy sighting in the detection ledger."""
    color: str
    position: List[float]   # best-estimate Isaac ENU [x, y, z]
    confidence: float       # highest confidence observed
    observations: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0


class DetectionLedger:
    """Thread-safe buoy ledger with proximity-based deduplication.

    Why thread-safe: the ROS2 subscriber callback runs in a background
    thread while the MAVSDK asyncio loop reads the ledger from the main
    thread. The lock prevents torn reads/writes on the entries list.
    """

    def __init__(self, dedup_radius: float, conf_threshold: float):
        self.entries: List[LedgerEntry] = []
        self.dedup_radius = dedup_radius
        self.conf_threshold = conf_threshold
        self._lock = threading.Lock()

    def try_add(self, det: BuoyDetection) -> bool:
        """Process a detection. Returns True only if a NEW buoy was added.

        Dedup logic: if a detection has the same color AND is within
        dedup_radius of an existing entry, it's the same buoy — we just
        update the observation count and keep the highest-confidence position.
        """
        if det.confidence < self.conf_threshold:
            return False

        with self._lock:
            for entry in self.entries:
                if entry.color != det.color:
                    continue
                if _dist3(det.world_position, entry.position) < self.dedup_radius:
                    # Same buoy — update stats
                    entry.observations += 1
                    entry.last_seen = det.timestamp
                    if det.confidence > entry.confidence:
                        entry.confidence = det.confidence
                        entry.position = list(det.world_position)
                    return False

            # No match — new buoy
            self.entries.append(LedgerEntry(
                color=det.color,
                position=list(det.world_position),
                confidence=det.confidence,
                observations=1,
                first_seen=det.timestamp,
                last_seen=det.timestamp,
            ))
            return True

    def get_entries(self) -> List[LedgerEntry]:
        with self._lock:
            return list(self.entries)

    def summary(self) -> str:
        with self._lock:
            if not self.entries:
                return "  Ledger: empty"
            lines = [f"  Ledger ({len(self.entries)} buoys):"]
            for i, e in enumerate(self.entries):
                lines.append(
                    f"    [{i+1}] {e.color:6s} "
                    f"pos=({e.position[0]:.2f}, {e.position[1]:.2f}, {e.position[2]:.2f}) "
                    f"conf={e.confidence:.2f} obs={e.observations}"
                )
            return "\n".join(lines)

    def save(self, path: str):
        """Dump ledger to JSON for post-analysis."""
        with self._lock:
            data = [asdict(e) for e in self.entries]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[LEDGER] Saved {len(data)} entries → {path}")


# ═══════════════════════════════════════════════════════
# ROS2 SUBSCRIBER (runs in background thread)
# ═══════════════════════════════════════════════════════

class DetectionSubscriber(Node):
    """Subscribes to /seabird/buoy_detections, feeds the ledger.

    This node does almost no work — just JSON parse + ledger insert.
    The heavy compute (YOLO, depth, transforms) is in sim_camera.py,
    a completely separate process. Fault isolation: if sim_camera
    crashes, this node just stops getting messages. The sweep continues.
    """

    def __init__(self, ledger: DetectionLedger):
        super().__init__("sweep_detection_listener")
        self.ledger = ledger

        # Default QoS (reliable, depth=10) — matches sim_camera's publisher
        self.sub = self.create_subscription(
            String,
            "/seabird/buoy_detections",
            self._on_detection,
            10,
        )
        self.get_logger().info("Listening on /seabird/buoy_detections")

    def _on_detection(self, msg: String):
        try:
            d = json.loads(msg.data)
            det = BuoyDetection(
                color=d["color"],
                world_position=d["world_position"],
                confidence=d["confidence"],
                drone_position=d["drone_position"],
                timestamp=d["timestamp"],
            )
            is_new = self.ledger.try_add(det)
            if is_new:
                self.get_logger().info(
                    f"*** NEW BUOY: {det.color} at "
                    f"({det.world_position[0]:.2f}, {det.world_position[1]:.2f}, "
                    f"{det.world_position[2]:.2f}) conf={det.confidence:.2f} ***"
                )
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Bad detection msg: {e}")


def _start_ros2_listener(ledger: DetectionLedger) -> threading.Thread:
    """Spin rclpy in a daemon thread so it doesn't block asyncio."""
    def _run():
        rclpy.init()
        node = DetectionSubscriber(ledger)
        try:
            rclpy.spin(node)
        except Exception:
            pass
        finally:
            node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ═══════════════════════════════════════════════════════
# LAWNMOWER WAYPOINT GENERATION
# ═══════════════════════════════════════════════════════

def _enu_to_ned_wp(x_enu: float, y_enu: float, alt: float, yaw: float) -> PositionNedYaw:
    """Convert an Isaac ENU survey point to a MAVSDK NED waypoint.

    ENU→NED mapping:
      NED_North = ENU_Y - home_Y   (Isaac Y = North)
      NED_East  = ENU_X - home_X   (Isaac X = East)
      NED_Down  = -altitude         (NED Down is positive, altitude is up)

    Yaw is in degrees: 0=North, 90=East, 270=West.
    """
    north = y_enu - HOME_ENU[1]
    east = x_enu - HOME_ENU[0]
    down = -alt
    return PositionNedYaw(north, east, down, yaw)


def generate_lawnmower() -> List[PositionNedYaw]:
    """Generate east-west lawnmower passes over the survey area.

    Passes run along X (east-west), stepping along Y (north-south).
    Alternates direction each pass for efficient coverage.
    Yaw rotates to face direction of travel so the front camera
    scans the ground ahead.
    """
    waypoints = []
    y = SURVEY_Y_MIN
    eastbound = True

    while y <= SURVEY_Y_MAX + 0.1:  # +0.1 for float tolerance
        if eastbound:
            waypoints.append(_enu_to_ned_wp(SURVEY_X_MIN, y, SWEEP_ALTITUDE, 90.0))
            waypoints.append(_enu_to_ned_wp(SURVEY_X_MAX, y, SWEEP_ALTITUDE, 90.0))
        else:
            waypoints.append(_enu_to_ned_wp(SURVEY_X_MAX, y, SWEEP_ALTITUDE, 270.0))
            waypoints.append(_enu_to_ned_wp(SURVEY_X_MIN, y, SWEEP_ALTITUDE, 270.0))
        y += SWEEP_SPACING
        eastbound = not eastbound

    return waypoints


# ═══════════════════════════════════════════════════════
# MAVSDK FLIGHT CONTROL
# ═══════════════════════════════════════════════════════

async def wait_for_waypoint(drone, wp: PositionNedYaw) -> bool:
    """Block until drone reaches waypoint (within tolerance) or timeout.

    Uses MAVSDK's position_velocity_ned async stream — yields NED
    position updates from PX4. We compute horizontal distance to
    target and return when it's within WAYPOINT_TOLERANCE.
    """
    t0 = time.time()
    async for pv in drone.telemetry.position_velocity_ned():
        dn = pv.position.north_m - wp.north_m
        de = pv.position.east_m - wp.east_m
        dist = math.sqrt(dn**2 + de**2)
        if dist < WAYPOINT_TOLERANCE:
            return True
        if time.time() - t0 > WAYPOINT_TIMEOUT:
            print(f"  TIMEOUT at dist={dist:.1f}m")
            return False
    return False


async def run_sweep():
    """Main mission: connect → arm → takeoff → lawnmower → detect → land."""

    # --- Detection ledger + ROS2 listener ---
    ledger = DetectionLedger(DEDUP_RADIUS, CONFIDENCE_THRESHOLD)
    _start_ros2_listener(ledger)
    print("[SWEEP] Detection listener started")
    await asyncio.sleep(1)  # let subscriber establish connection

    # --- MAVSDK connection ---
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("[SWEEP] Waiting for PX4 connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[SWEEP] Connected")
            break

    print("[SWEEP] Waiting for GPS lock + home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[SWEEP] GPS OK, home set")
            break

    # --- Arm + takeoff ---
    print("[SWEEP] Arming...")
    await drone.action.arm()
    await asyncio.sleep(1)

    print(f"[SWEEP] Takeoff → {SWEEP_ALTITUDE}m")
    await drone.action.set_takeoff_altitude(SWEEP_ALTITUDE)
    await drone.action.takeoff()

    # Wait for altitude — 10s is conservative but safe (Lesson #41)
    print("[SWEEP] Climbing... (10s stabilize)")
    await asyncio.sleep(10)

    # --- Generate waypoints ---
    wps = generate_lawnmower()
    print(f"\n[SWEEP] Lawnmower plan: {len(wps)} waypoints, "
          f"{SWEEP_SPACING}m spacing, {SWEEP_ALTITUDE}m alt")
    for i, wp in enumerate(wps):
        print(f"  WP{i}: N={wp.north_m:7.1f}  E={wp.east_m:7.1f}  "
              f"D={wp.down_m:6.1f}  Yaw={wp.yaw_deg:.0f}")

    # --- Start offboard mode ---
    # PX4 requires at least one setpoint BEFORE offboard engages
    await drone.offboard.set_position_ned(wps[0])
    try:
        await drone.offboard.start()
        print("\n[SWEEP] Offboard engaged — starting lawnmower")
    except OffboardError as e:
        print(f"[SWEEP] Offboard start FAILED: {e}")
        print("  Falling back to land.")
        await drone.action.land()
        return

    # --- Fly the pattern ---
    mission_start = time.time()

    for i, wp in enumerate(wps):
        print(f"\n[WP {i}/{len(wps)-1}] → N={wp.north_m:.1f} E={wp.east_m:.1f} "
              f"Yaw={wp.yaw_deg:.0f}")
        await drone.offboard.set_position_ned(wp)

        reached = await wait_for_waypoint(drone, wp)
        status = "REACHED" if reached else "SKIPPED (timeout)"
        print(f"  {status}")
        print(ledger.summary())

    elapsed = time.time() - mission_start

    # ═══════════════════════════════════════════════════
    # MISSION COMPLETE — RESULTS
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  SWEEP COMPLETE — {elapsed:.1f}s flight time")
    print(f"{'='*55}")
    print(ledger.summary())

    # --- Ground truth validation ---
    print("\n--- Ground Truth Validation ---")
    entries = ledger.get_entries()
    found_colors = set()
    if not entries:
        print("  NO BUOYS DETECTED — check sim_camera output")
    for entry in entries:
        found_colors.add(entry.color)
        best_dist = float("inf")
        best_name = "unknown"
        for name, gt in BUOY_POSITIONS.items():
            d = _dist3(entry.position, list(gt))
            if d < best_dist:
                best_dist = d
                best_name = name
        print(f"  {entry.color:6s} → {best_name:12s} err={best_dist:.2f}m  "
              f"obs={entry.observations}")

    # Check for missed buoys
    expected = {"red", "green", "blue"}
    missed = expected - found_colors
    if missed:
        print(f"\n  MISSED: {', '.join(sorted(missed))}")
    else:
        print(f"\n  ALL 3 BUOYS DETECTED — Phase 0 checkpoint PASSED")

    # --- Save ledger ---
    ledger.save(LEDGER_PATH)

    # --- Land ---
    print("\n[SWEEP] Stopping offboard, landing...")
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    await drone.action.land()
    await asyncio.sleep(8)
    print("[SWEEP] Done.")


# ═══════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════

def _dist3(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(run_sweep())

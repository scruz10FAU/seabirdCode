#!/usr/bin/env python3
"""
takeoff_test.py — Simple MAVSDK arm + takeoff + hover + land.

Run in a terminal AFTER Isaac + init_scene.py + PX4 are all running
and PX4 shows "Ready for takeoff!"

Usage:
    /usr/share/isaac-sim/python.sh ~/seabird/scripts/takeoff_test.py

Uses Isaac's Python (3.11) because mavsdk was pip-installed there.
"""

import asyncio
from mavsdk import System
from mavsdk.action import ActionError

HOVER_TIME = 10  # seconds to hover before landing


async def main():
    drone = System()

    # udp://:14540 — Lesson #20: udpin:// fails with "No network interface supplied"
    await drone.connect(system_address="udp://:14540")

    print("[takeoff] Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[takeoff] Connected to PX4")
            break

    print("[takeoff] Waiting for position estimate...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("[takeoff] Local position OK")
            break

    # Arm — no force flag needed, PX4 params are clean
    try:
        await drone.action.arm()
        print("[takeoff] Armed")
    except ActionError as e:
        print(f"[takeoff] Arm FAILED: {e}")
        return

    await asyncio.sleep(1)

    # Takeoff — uses MIS_TAKEOFF_ALT (default 2.5m)
    try:
        await drone.action.takeoff()
        print(f"[takeoff] Takeoff command sent — hovering for {HOVER_TIME}s")
    except ActionError as e:
        print(f"[takeoff] Takeoff FAILED: {e}")
        return

    await asyncio.sleep(HOVER_TIME)

    # Land
    try:
        await drone.action.land()
        print("[takeoff] Landing...")
    except ActionError as e:
        print(f"[takeoff] Land FAILED: {e}")
        return

    # Wait for touchdown
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("[takeoff] Landed — done")
            break

asyncio.run(main())

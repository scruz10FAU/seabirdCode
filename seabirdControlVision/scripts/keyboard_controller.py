#!/usr/bin/env python3
"""
keyboard_controller.py — Fly the drone with WASD + velocity offboard mode.

Run in terminal (NOT Script Editor):
    /usr/share/isaac-sim/python.sh ~/seabird/scripts/keyboard_controller.py

Controls:
    W / S     — forward / backward
    A / D     — left / right
    SPACE / C — up / down
    Q / E     — yaw left / yaw right
    1-5       — set speed (1=0.5 m/s, 2=1.0, 3=2.0, 4=3.0, 5=5.0)
    T         — takeoff (arm + takeoff to 2.5m)
    L         — land
    X         — land and exit

Requires PX4 SITL already running.
"""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.11/site-packages"))

import asyncio
import os
import tty
import termios
import select
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# --- Config ---
PX4_ADDR = "udp://:14540"
DEFAULT_SPEED = 1.0        # m/s
YAW_RATE = 30.0            # deg/s
TICK_HZ = 20               # offboard setpoint rate
SPEED_PRESETS = {
    '1': 0.5, '2': 1.0, '3': 2.0, '4': 3.0, '5': 5.0,
}

# --- Terminal raw-mode key reader ---
class RawKeyReader:
    """Non-blocking single-char reader using raw terminal mode."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        print("\033[2J\033[H", end="")  # clear screen on exit

    def read(self):
        """Return a character if available, else empty string."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return ""


def clear_and_print(lines):
    """Rewrite the HUD in place without full curses."""
    # Move cursor to top-left, clear screen
    out = "\033[H\033[2J"
    out += "\n".join(lines)
    sys.stdout.write(out)
    sys.stdout.flush()


async def run():
    with RawKeyReader() as keys:
        # --- Connect ---
        clear_and_print([f"Connecting to {PX4_ADDR}..."])
        drone = System()
        await drone.connect(system_address=PX4_ADDR)

        # Wait for connection
        async for state in drone.core.connection_state():
            if state.is_connected:
                break
        clear_and_print(["Connected! Waiting for GPS fix..."])

        # Wait for global position estimate
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break
        clear_and_print(["GPS OK. Ready — press T to take off."])

        speed = DEFAULT_SPEED
        offboard_active = False
        flying = False
        alt = 0.0
        status = "Press T to take off"

        # Background altitude poller
        async def poll_altitude():
            nonlocal alt
            async for pos in drone.telemetry.position():
                alt = pos.relative_altitude_m

        alt_task = asyncio.ensure_future(poll_altitude())

        try:
            while True:
                key = keys.read().lower()

                # --- Takeoff ---
                if key == 't' and not flying:
                    status = "Arming..."
                    await drone.action.arm()
                    status = "Taking off..."
                    await drone.action.takeoff()
                    await asyncio.sleep(5)
                    flying = True

                    # Send initial setpoint then start offboard
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0, 0, 0, 0))
                    try:
                        await drone.offboard.start()
                        offboard_active = True
                        status = "Offboard active — fly with WASD!"
                    except OffboardError as e:
                        status = f"Offboard failed: {e}"

                # --- Land ---
                elif key == 'l':
                    if offboard_active:
                        await drone.offboard.stop()
                        offboard_active = False
                    await drone.action.land()
                    flying = False
                    status = "Landing..."

                # --- Exit ---
                elif key == 'x':
                    if offboard_active:
                        await drone.offboard.stop()
                        offboard_active = False
                    if flying:
                        await drone.action.land()
                    status = "Exiting..."
                    clear_and_print([status])
                    await asyncio.sleep(2)
                    break

                # --- Speed presets ---
                elif key in SPEED_PRESETS:
                    speed = SPEED_PRESETS[key]
                    status = f"Speed → {speed:.1f} m/s"

                # --- Build velocity from key ---
                vf, vr, vd, yr = 0.0, 0.0, 0.0, 0.0
                if key == 'w':   vf = speed
                elif key == 's': vf = -speed
                if key == 'a':   vr = -speed
                elif key == 'd': vr = speed
                if key == ' ':   vd = -speed     # NED: negative = up
                elif key == 'c': vd = speed      # down
                if key == 'q':   yr = -YAW_RATE
                elif key == 'e': yr = YAW_RATE

                # --- Send velocity setpoint ---
                if offboard_active:
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(vf, vr, vd, yr))

                # --- HUD ---
                hud = [
                    "=== SEABIRD KEYBOARD CONTROLLER ===",
                    f"Speed: {speed:.1f} m/s  |  Alt: {alt:.1f}m  |  Offboard: {'ON' if offboard_active else 'OFF'}",
                    f"Vel: fwd={vf:+.1f}  right={vr:+.1f}  down={vd:+.1f}  yaw={yr:+.0f}°/s",
                    f"Status: {status}",
                    "",
                    "W/S = fwd/back    A/D = left/right",
                    "SPACE/C = up/down  Q/E = yaw",
                    "1-5 = speed        T = takeoff",
                    "L = land           X = exit",
                ]
                clear_and_print(hud)

                # Tick rate
                await asyncio.sleep(1.0 / TICK_HZ)

        finally:
            alt_task.cancel()
            if offboard_active:
                try:
                    await drone.offboard.stop()
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run())

# test_static_hover.py — Static hover perception test
# ──────────────────────────────────────────────────────────────────
# Spawns drone facing the buoy field. Hover at 5m via PX4 commander.
# All 3 buoys visible from this position. Run buoy_estimator.py in a
# separate terminal to validate the pixel-to-world back-projection.
#
# Run via Script Editor one-liner:
#   exec(open("/home/tgarcia/drone_sim/workspace/scripts/test_static_hover.py").read())

import asyncio
import sys
import math

# ── Make workspace importable inside Isaac's exec() environment ───
sys.path.insert(0, "/home/tgarcia/drone_sim/workspace/scripts")
from seabird_spawn import spawn_scene

# ── Test configuration ────────────────────────────────────────────

# Drone spawns at origin, facing -Y (toward the buoy field)
# Quaternion for -90° yaw about Z: [0, 0, sin(-45°), cos(-45°)]
YAW_NEG90 = [0.0, 0.0, -math.sin(math.pi / 4), math.cos(math.pi / 4)]

DRONE_POS        = [0.0, 0.0, 2.5]     # spawn height (PX4 takeoff overrides)
DRONE_ROT        = YAW_NEG90           # face -Y toward buoys
CAMERA_PITCH_DEG = 15.0                # ~15° down — puts buoys near frame center at 5m alt
CAMERA_RES       = (320, 240)

# ── Ground truth: buoy positions baked into marina_dock.usd ───────
# These are the KNOWN positions we validate our estimates against.
BUOY_GROUND_TRUTH = {
    "blue":  (0.0, -18.0, 0.0),
    "green": (-6.0, -14.0, 0.0),
    "red":   (6.0, -14.0, 0.0),
}


async def main():
    pi, stage = await spawn_scene(
        drone_pos=DRONE_POS,
        drone_rot=DRONE_ROT,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        camera_resolution=CAMERA_RES,
    )

    if pi is None:
        print("[test] FAILED — spawn_scene returned None")
        return

    # ── Print test setup ──────────────────────────────────────────
    print("")
    print("=" * 60)
    print("  STATIC HOVER TEST — Phase 1 Perception Validation")
    print("=" * 60)
    print(f"  Drone spawn:  {DRONE_POS}")
    print(f"  Drone yaw:    -90° (facing -Y toward buoy field)")
    print(f"  Camera pitch: {CAMERA_PITCH_DEG}° down")
    print(f"  Resolution:   {CAMERA_RES[0]}x{CAMERA_RES[1]}")
    print("")
    print("  Buoy ground truth (Isaac world frame):")
    for color, pos in BUOY_GROUND_TRUTH.items():
        print(f"    {color:6s}  →  x={pos[0]:6.1f}  y={pos[1]:6.1f}  z={pos[2]:6.1f}")
    print("")
    print("  STEPS:")
    print("  1. Start PX4:  cd ~/drone_sim/workspace/PX4-Autopilot")
    print("                 pkill -9 -f px4 && sleep 2 && make px4_sitl none_iris")
    print("  2. Wait for:   'Ready for takeoff!'")
    print("  3. Takeoff:    commander takeoff (default 2.5m) or 'commander takeoff -a 5'")
    print("  4. Run estimator: python3 ~/drone_sim/workspace/scripts/buoy_estimator.py")
    print("  5. Estimator will print position estimates + error vs ground truth")
    print("=" * 60)


asyncio.ensure_future(main())
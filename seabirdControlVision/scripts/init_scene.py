"""
init_scene.py — Master scene initialization for Seabird.

Loads marina, spawns drone with PX4 backend + ROS2Backend,
sets camera FOV to match ZED 2i wide lens.

All parameters come from seabird_config.py — single source of truth.

Usage (Script Editor):
    exec(open(os.path.expanduser("~/seabird/scripts/init_scene.py")).read())
"""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.11/site-packages"))

# ══════════════════════════════════════════════════════════════════
# rclpy fix — MUST run before ANY Pegasus import (Lessons #55, #57)
#
# Isaac ships a complete Python 3.11 rclpy at the path below, but
# the system ROS2 Humble rclpy (3.10) gets found first and cached
# as a failed import in sys.modules.  Subsequent sys.path.insert
# can't override a cached failure.  So we:
#   1. Flush any cached rclpy/rpyutils entries from sys.modules
#   2. Remove all python3.10 paths so the 3.10 rclpy can't be found
#   3. Insert Isaac's 3.11 rclpy path at the front
# This lets ROS2Backend import cleanly inside Isaac's Script Editor.
# ══════════════════════════════════════════════════════════════════
# Flush ALL ROS2-related cached modules — not just rclpy.
# The system python3.10 rcl_interfaces, rosidl_*, builtin_interfaces etc.
# get cached before we can redirect, and create_node() needs them all as 3.11.
# Also flush pegasus.*.ros2_backend — it caches a reference to whatever
# rclpy was available when it was first imported. On re-runs in Script
# Editor, that's the OLD (broken) rclpy. Flushing forces reimport with
# the correct Isaac 3.11 rclpy we're about to put on sys.path.
_ros2_prefixes = (
    "rclpy", "rpyutils", "rcl_interfaces", "rosidl_",
    "builtin_interfaces", "action_msgs", "actionlib_msgs",
    "geometry_msgs", "sensor_msgs", "std_msgs", "std_srvs",
    "nav_msgs", "tf2_", "visualization_msgs", "vision_msgs",
    "rosgraph_msgs", "lifecycle_msgs", "composition_interfaces",
    "statistics_msgs", "unique_identifier_msgs", "shape_msgs",
    "stereo_msgs", "trajectory_msgs", "diagnostic_msgs",
    "ament_", "launch", "rcutils", "rmw_",
    "pegasus.simulator.logic.backends.ros2_backend",
)
_to_remove = [k for k in sys.modules if k.startswith(_ros2_prefixes)]
for _k in _to_remove:
    del sys.modules[_k]
if _to_remove:
    print(f"Cleared {len(_to_remove)} cached ROS2 modules")
sys.path = [p for p in sys.path if "python3.10" not in p]
_isaac_rclpy = "/usr/share/isaac-sim/exts/isaacsim.ros2.bridge/humble/rclpy"
if _isaac_rclpy not in sys.path:
    sys.path.insert(0, _isaac_rclpy)

import asyncio
import omni.usd
from pxr import UsdGeom, Gf
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend

# ── Import shared config ──
sys.path.insert(0, os.path.expanduser("~/seabird/scripts"))
from seabird_config import (
    ASSETS_DIR,
    DRONE_PRIM_PATH, CAMERA_PRIM_PATH,
    DRONE_SPAWN_POS, DRONE_SPAWN_QUAT_WXYZ,
    IMG_W, IMG_H,
    CAM_OFFSET_BODY, CAM_PITCH_DEG,
    CAMERA_FOCAL_LENGTH_MM, CAMERA_H_APERTURE_MM, CAMERA_V_APERTURE_MM,
    CAMERA_CLIPPING_NEAR, CAMERA_CLIPPING_FAR,
    PX4_CONNECTION_TYPE,
    print_camera_config,
)

MARINA_USD   = f"{ASSETS_DIR}/marina_dock.usd"
CAMERA_NAME  = "front_cam"

# Pegasus takes init_orientation as [x, y, z, w] — convert from our wxyz config
_qw, _qx, _qy, _qz = DRONE_SPAWN_QUAT_WXYZ
DRONE_ROT_XYZW = [_qx, _qy, _qz, _qw]


def ensure_scene(world):
    """Force _scene to exist before any vehicle spawn."""
    if hasattr(world, '_scene') and world._scene is not None:
        return True
    try:
        from isaacsim.core.api.scenes.scene import Scene
        world._scene = Scene()
        print("[scene] Patched via isaacsim.core.api.scenes.scene")
        return True
    except Exception:
        pass
    try:
        from omni.isaac.core.scenes.scene import Scene
        world._scene = Scene()
        print("[scene] Patched via omni.isaac.core.scenes.scene")
        return True
    except Exception:
        pass
    print("[scene] FATAL: could not create Scene — restart Isaac Sim")
    return False


async def main():
    print("[init] Starting Seabird init_scene.py (ROS2Backend enabled)")
    print_camera_config()

    pi = PegasusInterface()
    if pi.world is None:
        pi.initialize_world()
    await pi.world.initialize_simulation_context_async()

    # Clear registries — NO world.stop() before spawn (Rule #8)
    if pi.vehicle_manager.vehicles:
        pi.vehicle_manager.remove_all_vehicles()
    if hasattr(pi.world, '_scene') and pi.world._scene is not None:
        try:
            pi.world.scene.remove_object(DRONE_PRIM_PATH, registry_only=True)
        except Exception:
            pass
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DRONE_PRIM_PATH).IsValid():
        stage.RemovePrim(DRONE_PRIM_PATH)

    # Load marina as sublayer
    root_layer = stage.GetRootLayer()
    if MARINA_USD not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(MARINA_USD)
    print(f"[init] Loaded marina: {MARINA_USD}")

    if not ensure_scene(pi.world):
        print("[init] FATAL — could not initialize scene, restart Isaac")
        return

    cam_graph = ROS2CameraGraph(
        camera_prim_path=f"body/{CAMERA_NAME}",
        config={
            "resolution":  [IMG_W, IMG_H],
            "types":       ["rgb", "depth", "camera_info"],
            "namespace":   "/iris_0",
            "topic":       f"/{CAMERA_NAME}",
            "tf_frame_id": CAMERA_NAME,
        }
    )

    px4_config = PX4MavlinkBackendConfig(config={
        "vehicle_id":          0,
        "px4_autolaunch":      False,
        "connection_type":     PX4_CONNECTION_TYPE,
        "connection_ip":       "localhost",
        "connection_baseport": 4560,
        "enable_lockstep":     False,
    })

    # ROS2Backend needs rclpy initialized before it creates its node.
    # Guard so re-runs in Script Editor don't crash on double-init.
    import rclpy
    try:
        rclpy.init()
        print("[ros2] rclpy.init() — OK")
    except RuntimeError:
        # Already initialized from a previous run in this session
        print("[ros2] rclpy already initialized — reusing")

    config = MultirotorConfig()
    config.backends = [PX4MavlinkBackend(px4_config), ROS2Backend(vehicle_id=0, num_rotors=4)]
    config.graphical_sensors = []
    config.graphs            = [cam_graph]

    drone = Multirotor(
        stage_prefix=DRONE_PRIM_PATH,
        usd_file=ROBOTS["Iris"],
        init_pos=DRONE_SPAWN_POS,
        init_orientation=DRONE_ROT_XYZW,
        config=config,
    )

    await pi.world.reset_async()

    # ── Fix camera transform ──
    # ROS2CameraGraph bakes world spawn Z into the prim — we override with
    # our known mount offset and pitch from config.
    cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if cam_prim.IsValid():
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        cam_prim.RemoveProperty("xformOp:translate")
        cam_prim.RemoveProperty("xformOp:orient")
        cam_prim.RemoveProperty("xformOp:rotateXYZ")
        cam_prim.RemoveProperty("xformOp:scale")

        xf.AddTranslateOp().Set(Gf.Vec3d(
            float(CAM_OFFSET_BODY[0]),
            float(CAM_OFFSET_BODY[1]),
            float(CAM_OFFSET_BODY[2])
        ))
        # Pitch sign: (90 - pitch) = nose-DOWN. Lesson #47.
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 - CAM_PITCH_DEG, 0.0, -90.0))
        xf.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))

        # ── Set camera lens to match ZED 2i ──
        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(CAMERA_FOCAL_LENGTH_MM)
        cam.GetHorizontalApertureAttr().Set(CAMERA_H_APERTURE_MM)
        cam.GetVerticalApertureAttr().Set(CAMERA_V_APERTURE_MM)
        cam.GetClippingRangeAttr().Set((CAMERA_CLIPPING_NEAR, CAMERA_CLIPPING_FAR))

        print(f"[camera] Ready: pitch={CAM_PITCH_DEG}°, ZED 2i FOV, "
              f"mount=({CAM_OFFSET_BODY[0]}, {CAM_OFFSET_BODY[1]}, {CAM_OFFSET_BODY[2]})")
    else:
        print(f"[camera] ERROR — prim not found at {CAMERA_PRIM_PATH}")

    print("[init] Done — ROS2Backend active, /drone_0/state should publish")
    print("[init]   cd ~/seabird/PX4-Autopilot && pkill -9 -f px4; sleep 2; make px4_sitl none_iris")
    print("[init]   Verify: ros2 topic list | grep drone_0")

asyncio.ensure_future(main())

# seabird_spawn.py — Reusable Isaac-side spawn logic for Seabird
# ──────────────────────────────────────────────────────────────────
# All battle-tested plumbing lives here. Test scripts import spawn_scene().
# This file is NEVER exec()'d directly — it's imported as a module.
#
# Usage from any test script:
#   import sys; sys.path.insert(0, "/home/tgarcia/drone_sim/workspace/scripts")
#   from seabird_spawn import spawn_scene
#   asyncio.ensure_future(my_test())

import asyncio
import omni.usd
from pxr import UsdGeom, Gf
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.params import ROBOTS

# ── Constants (never change per test) ─────────────────────────────
MARINA_USD  = "/home/tgarcia/drone_sim/workspace/marina_dock.usd"
DRONE_PATH  = "/World/Iris"
CAMERA_NAME = "front_cam"


def _ensure_scene(world):
    """Patch World._scene if missing (cold-start edge case, Lesson id=1/Error id=8)."""
    if hasattr(world, '_scene') and world._scene is not None:
        return True

    for path in [
        "isaacsim.core.api.scenes.scene",
        "omni.isaac.core.scenes.scene",
    ]:
        try:
            mod = __import__(path, fromlist=["Scene"])
            world._scene = mod.Scene()
            print(f"[scene] Patched via {path}")
            return True
        except Exception:
            continue

    print("[scene] FATAL: could not create Scene — restart Isaac Sim")
    return False


async def spawn_scene(
    drone_pos=(0.0, 0.0, 2.5),
    drone_rot=(0.0, 0.0, 0.0, 1.0),
    camera_pitch_deg=15.0,
    camera_resolution=(320, 240),
):
    """
    Load marina world, spawn Iris drone with PX4 backend + ROS2 camera.

    Args:
        drone_pos:         (x, y, z) spawn position in Isaac world frame
        drone_rot:         (qx, qy, qz, qw) quaternion — NOT euler (Lesson id=3)
        camera_pitch_deg:  positive = nose down. 0 = level horizon.
        camera_resolution: (width, height) for RGB/depth/camera_info

    Returns:
        (pi, stage) on success — PegasusInterface + USD stage handle
        (None, None) on failure

    After this returns, start PX4 SITL in a separate terminal.
    """

    # ── 1. World init (Lesson id=1: PegasusInterface must own the World) ──
    pi = PegasusInterface()
    if pi.world is None:
        pi.initialize_world()
    await pi.world.initialize_simulation_context_async()
    print("[spawn] sim context ready")

    # ── 2. Clear all three registries (Lesson id=2) ───────────────────────
    #   Registry 1: VehicleManager (physics bodies + ROS nodes)
    #   Registry 2: World.scene internal dict ("name not unique" error)
    #   Registry 3: USD stage prim
    #   Missing ANY one causes a different error on next spawn.
    if pi.vehicle_manager.vehicles:
        pi.vehicle_manager.remove_all_vehicles()

    if hasattr(pi.world, '_scene') and pi.world._scene is not None:
        try:
            pi.world.scene.remove_object(DRONE_PATH, registry_only=True)
        except Exception:
            pass

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DRONE_PATH).IsValid():
        stage.RemovePrim(DRONE_PATH)

    # ── 3. Load marina as sublayer (Lesson id=15: sublayer, NOT reference) ─
    root_layer = stage.GetRootLayer()
    if MARINA_USD not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(MARINA_USD)
        print("[spawn] marina sublayer loaded")

    # ── 4. Ensure _scene before spawning ──────────────────────────────────
    if not _ensure_scene(pi.world):
        return None, None

    # ── 5. Camera (Lesson id=4: ROS2CameraGraph only, no MonocularCamera) ─
    cam_graph = ROS2CameraGraph(
        camera_prim_path=f"body/{CAMERA_NAME}",   # relative — Lesson id=5
        config={
            "resolution":  list(camera_resolution),
            "types":       ["rgb", "depth", "camera_info"],
            "namespace":   "/iris_0",
            "topic":       f"/{CAMERA_NAME}",
            "tf_frame_id": CAMERA_NAME,
        },
    )

    # ── 6. PX4 backend (tcpin = Pegasus listens, PX4 connects) ────────────
    px4_config = PX4MavlinkBackendConfig(config={
        "vehicle_id":          0,
        "px4_autolaunch":      False,
        "connection_type":     "tcpin",
        "connection_ip":       "localhost",
        "connection_baseport": 4560,
        "enable_lockstep":     True,   # Must match PX4's lockstep_scheduler
    })

    config = MultirotorConfig()
    config.backends = [
        PX4MavlinkBackend(px4_config),            # flight control
        ROS2Backend(vehicle_id=0, num_rotors=4),   # sensor topic publishing (Lesson id=8)
    ]
    config.graphical_sensors = []   # MonocularCamera broken (Lesson id=4)
    config.graphs            = [cam_graph]

    # ── 7. Spawn drone (Lesson id=3: orientation is quaternion [x,y,z,w]) ─
    Multirotor(
        stage_prefix=DRONE_PATH,
        usd_file=ROBOTS["Iris"],
        init_pos=list(drone_pos),
        init_orientation=list(drone_rot),
        config=config,
    )
    print(f"[spawn] Iris at pos={list(drone_pos)}")

    # ── 8. Activate physics + ROS2 graphs ─────────────────────────────────
    await pi.world.reset_async()

    # ── 9. Fix camera transform (Error id=8: baked world offset) ──────────
    #   Formula: Gf.Vec3f(90 + pitch, 0, -90). Pitch in X ONLY (Error id=9).
    cam_prim = stage.GetPrimAtPath(f"{DRONE_PATH}/body/{CAMERA_NAME}")
    if cam_prim.IsValid():
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        cam_prim.RemoveProperty("xformOp:translate")
        cam_prim.RemoveProperty("xformOp:orient")
        cam_prim.RemoveProperty("xformOp:rotateXYZ")
        cam_prim.RemoveProperty("xformOp:scale")
        xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.05))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 + camera_pitch_deg, 0.0, -90.0))
        xf.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        print(f"[spawn] camera: pitch={camera_pitch_deg}° down")
    else:
        print("[spawn] ERROR — camera prim not found!")

    print("[spawn] ✓ Scene ready — start PX4 now")
    return pi, stage
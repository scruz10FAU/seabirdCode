import asyncio
import omni.usd
from pxr import UsdGeom, Gf
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.params import ROBOTS

MARINA_USD       = "/home/tgarcia/drone_sim/workspace/marina_dock.usd"
DRONE_POS        = [0.0, 0.0, 2.5]
DRONE_ROT        = [0.0, 0.0, 0.0, 1.0]   # quaternion [x,y,z,w] — NOT euler
CAMERA_NAME      = "front_cam"
DRONE_PATH       = "/World/Iris"
CAMERA_PITCH_DEG = -10.0                    


def ensure_scene(world):
    if hasattr(world, '_scene') and world._scene is not None:
        print("[scene] _scene already present — no patch needed")
        return True

    print("[scene] _scene missing — attempting to patch...")
    try:
        from isaacsim.core.api.scenes.scene import Scene
        world._scene = Scene()
        print("[scene] Patched via isaacsim.core.api.scenes.scene")
        return True
    except Exception as e:
        print(f"[scene] isaacsim path failed: {e}")

    try:
        from omni.isaac.core.scenes.scene import Scene
        world._scene = Scene()
        print("[scene] Patched via omni.isaac.core.scenes.scene")
        return True
    except Exception as e:
        print(f"[scene] omni.isaac path failed: {e}")

    print("[scene] FATAL: could not create Scene — restart Isaac Sim and try again")
    return False


async def main():

    # ── 1. World init ─────────────────────────────────────────────────────
    pi = PegasusInterface()
    if pi.world is None:
        pi.initialize_world()
    await pi.world.initialize_simulation_context_async()

    has_scene = hasattr(pi.world, '_scene') and pi.world._scene is not None
    print(f"[init] simulation context ready | _scene present: {has_scene}")

    # ── 2. Clear all three registries ────────────────────────────────────
    if pi.vehicle_manager.vehicles:
        pi.vehicle_manager.remove_all_vehicles()
        print("[cleanup] VehicleManager cleared")

    if hasattr(pi.world, '_scene') and pi.world._scene is not None:
        try:
            pi.world.scene.remove_object(DRONE_PATH, registry_only=True)
            print(f"[cleanup] scene registry cleared for {DRONE_PATH}")
        except Exception as e:
            print(f"[cleanup] scene.remove_object skipped: {e}")
    else:
        print("[cleanup] _scene not present — skipping scene registry cleanup")

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DRONE_PATH).IsValid():
        stage.RemovePrim(DRONE_PATH)
        print(f"[cleanup] USD prim {DRONE_PATH} removed")

    # ── 3. Load marina as sublayer ────────────────────────────────────────
    root_layer = stage.GetRootLayer()
    if MARINA_USD not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(MARINA_USD)
        print("[scene] marina sublayer loaded")
    else:
        print("[scene] marina sublayer already present")

    # ── 4. Ensure _scene exists before spawning ───────────────────────────
    if not ensure_scene(pi.world):
        print("[init] Aborting — could not initialize scene")
        return

    # ── 5. Camera ─────────────────────────────────────────────────────────
    cam_graph = ROS2CameraGraph(
        camera_prim_path=f"body/{CAMERA_NAME}",
        config={
            "resolution":  [320, 240],
            "types":       ["rgb", "depth", "camera_info"],
            "namespace":   "/iris_0",
            "topic":       f"/{CAMERA_NAME}",
            "tf_frame_id": CAMERA_NAME,
        }
    )

    # ── 6. PX4 backend ────────────────────────────────────────────────────
    px4_config = PX4MavlinkBackendConfig(config={
        "vehicle_id":          0,
        "px4_autolaunch":      False,
        "connection_type":     "tcpin",
        "connection_ip":       "localhost",
        "connection_baseport": 4560,
        "enable_lockstep":     False,
    })

    config = MultirotorConfig()
    config.backends = [
        PX4MavlinkBackend(px4_config),          # flight control
        ROS2Backend(vehicle_id=0, num_rotors=4), # sensor topic publishing
    ]
    config.graphical_sensors = []
    config.graphs            = [cam_graph]

    # ── 7. Spawn drone ────────────────────────────────────────────────────
    drone = Multirotor(
        stage_prefix=DRONE_PATH,
        usd_file=ROBOTS["Iris"],
        init_pos=DRONE_POS,
        init_orientation=DRONE_ROT,
        config=config,
    )
    print(f"[spawn] Multirotor created at {DRONE_PATH}")

    # ── 8. reset_async — activates physics, camera graphs, ROS2 ──────────
    await pi.world.reset_async()
    print("[world] reset_async complete")

    # ── 9. Lock camera transform ──────────────────────────────────────────
    cam_prim = stage.GetPrimAtPath("/World/Iris/body/front_cam")
    if cam_prim.IsValid():
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        cam_prim.RemoveProperty("xformOp:translate")
        cam_prim.RemoveProperty("xformOp:orient")
        cam_prim.RemoveProperty("xformOp:rotateXYZ")
        cam_prim.RemoveProperty("xformOp:scale")
        xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.05))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 + CAMERA_PITCH_DEG, 0.0, -90.0))
        xf.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        print(f"[camera] Locked: 30cm front, 5cm up, {CAMERA_PITCH_DEG}deg nose-down")
    else:
        print("[camera] ERROR — prim not found, searching stage:")
        for p in stage.Traverse():
            if "cam" in str(p.GetPath()).lower():
                print(f"  {p.GetPath()}")

    # ── Done ──────────────────────────────────────────────────────────────
    print("")
    print("[init] ✓ Isaac ready — NOW start PX4 in a new terminal:")
    print("         cd ~/drone_sim/workspace/PX4-Autopilot")
    print("         pkill -9 -f px4 && sleep 2")
    print("         make px4_sitl none_iris")
    print("")
    print("[init] Wait for: INFO  [simulator_mavlink] Simulator connected on TCP port 4560")
    print("[init] Then:     INFO  [commander] Ready for takeoff!")
    print("")
    print("[ros2] Sensors should now be live:")
    print("         ros2 topic list | grep drone_0")
    print("")
    print("[ros2] Camera preview:")
    print("         ros2 run rqt_image_view rqt_image_view /iris_0/front_cam/rgb")

asyncio.ensure_future(main())
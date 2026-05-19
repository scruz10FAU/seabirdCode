# Seabird Scripts

Autonomous drone buoy detection system built on Isaac Sim, PX4, ROS2, and YOLOv8.

The system has three phases that run in sequence:
1. **Simulation setup** — initialize the Isaac Sim world and drone
2. **Training data generation** — fly the drone and auto-label frames
3. **Autonomous mission** — fly a lawnmower sweep and detect buoys in real time

---

## Architecture

```
Isaac Sim (init_scene.py)
  └─► /iris_0/front_cam/rgb+depth  (ROS2 topics)
  └─► /drone00/state/pose          (ROS2 topics)
        │
        ▼
  sim_camera.py  (ROS2 node, system terminal)
    ├── yolo_detector.py  →  detections in camera frame
    ├── camera_to_world() →  detections in world frame
    └─► /seabird/buoy_detections   (JSON over std_msgs/String)
              │
              ▼
        sweep_and_detect.py
          ├── DetectionLedger  (deduplicated buoy map)
          └── MAVSDK  →  PX4 SITL  →  drone motion in Isaac
```

---

## Scripts

### `seabird_config.py`
**Single source of truth for all parameters.** Every other script imports from here — no magic numbers elsewhere.

Contains:
- File paths (`DATASET_DIR`, `LOGS_DIR`, etc.)
- Camera intrinsics for the ZED 2i wide lens (`CAMERA_FX`, `CAMERA_FY`, `CAMERA_CX`, `CAMERA_CY`)
- Render resolution (`IMG_W = 640`, `IMG_H = 480`)
- Buoy world positions (`BUOY_POSITIONS`), class names, and colors
- Drone spawn pose
- Precomputed body→camera rotation matrix `R_BODY_TO_CAM`
- Helper functions: `camera_to_world()`, `nearest_buoy_error()`, `print_camera_config()`

**Usage:** Import only — never run directly.
```python
from seabird_config import CAMERA_FX, BUOY_POSITIONS, camera_to_world
```

If a camera is swapped, buoy layout changes, or resolution changes — edit this file and everything stays consistent.

---

### `init_scene.py`
**Master scene initializer.** Run this first inside the Isaac Sim Script Editor to bring up the simulation world.

What it does:
1. Fixes the Python 3.10/3.11 rclpy conflict by flushing cached ROS2 modules
2. Loads the marina USD scene as a sublayer
3. Spawns the Iris drone with `PX4MavlinkBackend` (flight control) and `ROS2Backend` (state publishing)
4. Attaches a `ROS2CameraGraph` to publish RGB, depth, and camera_info topics under `/iris_0/front_cam/`
5. Sets the camera prim transform and lens attributes to match the ZED 2i config from `seabird_config.py`

**Usage (Isaac Script Editor):**
```python
exec(open("/home/<user>/seabird/scripts/init_scene.py").read())
```

After running, start PX4 SITL in a separate terminal:
```bash
cd ~/seabird/PX4-Autopilot
pkill -9 -f px4; sleep 2; make px4_sitl none_iris
```

---

### `camera_interface.py`
**Abstract base class for all camera backends.** Defines the universal contract so mission code never needs to know whether it is talking to the simulator or real ZED hardware.

Defines three dataclasses:
- `Detection` — a single detected object (label, confidence, 2D bbox, optional 3D position)
- `Intrinsics` — pinhole camera parameters (fx, fy, cx, cy, width, height)
- `CameraConfig` — settings applied before opening the camera (resolution, fps, depth mode)

Defines the `CameraInterface` abstract base class with two lifecycle levels:
- **Level 1 (data source):** `configure()` → `open()` → `grab()` → `get_rgb()` / `get_depth()` / `get_intrinsics()`
- **Level 2 (perception):** `enable_detection()` → `get_detections()`

Also provides a default `back_project()` implementation (pixel + depth → 3D camera frame).

**Usage:** Import the dataclasses and subclass `CameraInterface` — never instantiate directly.

---

### `sim_camera.py`
**ROS2 camera implementation for Isaac Sim.** Implements `CameraInterface` as a ROS2 node that subscribes to Isaac's camera topics.

Subscribes to:
- `/iris_0/front_cam/rgb` — RGB image
- `/iris_0/front_cam/depth` — depth map (float32, meters)
- `/iris_0/front_cam/camera_info` — used once to trigger intrinsics setup (then overridden by config)
- `/drone00/state/pose` — drone pose from Pegasus ROS2Backend

Publishes:
- `/seabird/buoy_detections` — JSON detection messages consumed by `sweep_and_detect.py`

Key behavior: camera intrinsics are intentionally taken from `seabird_config.py` rather than the ROS2 `camera_info` topic, because Isaac's default camera reports incorrect values until the lens is set by `init_scene.py`.

**Usage (system terminal, after Isaac + init_scene.py are running):**
```bash
source /opt/ros/humble/setup.bash
python3 ~/seabird/scripts/sim_camera.py
```

Requires YOLO weights at `~/seabird_dataset/runs/v1/weights/best.pt`. Without them it runs in RGB-only mode. Debug frames are saved to `~/seabird_dataset/debug_live/`.

---

### `sim_camera_zed.py`
**ZED camera variant of SimCamera** with added GPS origin support and depth error validation. Intended for transitioning from simulation to real ZED 2i hardware. Drop-in replacement for `sim_camera.py` — same published topics and `CameraInterface` contract.

Differences from `sim_camera.py`:
- Subscribes to ZED SDK topic paths (`/zed/zed_node/...`) instead of Isaac's `/iris_0/front_cam/...`
- Reads drone pose from `/mavros/local_position/pose` instead of Pegasus ROS2Backend
- Subscribes to `/mavros/global_position/gp_origin` to latch the EKF GPS origin and convert local ENU detections to lat/lon/alt
- Adds depth error validation: computes expected depth from a known ground-truth object distance and reports the per-frame error as `dz=` on each detection
- Detection messages include an additional `gps_position` field (`{latitude, longitude, altitude}`) alongside the existing `world_position`
- Handles RGBA images from the ZED SDK (drops the alpha channel before processing)

**Usage:**
```bash
source /opt/ros/humble/setup.bash
python3 ~/seabird/scripts/sim_camera_zed.py [OPTIONS]
```

**Arguments:**

| Flag | Short | Type | Default | Description | Options |
|------|-------|------|---------|-------------|---------|
| `--model` | `-m` | `str` | `models/best_alex.pt` | Path to YOLO weights file | Any `.pt` file path, or a standard YOLOv8 model name |
| `--display` | `-d` | flag | `False` | Show a live OpenCV window with annotated detections | Present = enabled, absent = disabled |
| `--true_dist` | `-td` | `float` | `0.4826` | Known ground-truth distance (meters) from camera to target object, used to compute depth error | Any positive float (meters) |

**Argument details:**

`--model` / `-m` — selects the YOLO weights to load. Accepts either a local `.pt` file path or any standard ultralytics model name (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`). If the path does not exist and is not a recognized default, detection is disabled and the node runs in RGB-only mode.

`--display` / `-d` — opens a `cv2.imshow` window showing the annotated camera feed in real time. Press `q` in the window to stop the node. Omit this flag when running headless (e.g. over SSH).

`--true_dist` / `-td` — the actual physical distance in meters from the camera to a target object placed at a known location. Used to compute `dz` (depth error): the difference between the measured depth and the expected depth accounting for the pixel's off-center angle. Useful for calibrating or validating depth accuracy. The default (`0.4826 m`) matches the original bench test setup.

**Examples:**
```bash
# Default model, no display
python3 sim_camera_zed.py

# Custom weights with live display
python3 sim_camera_zed.py -m path/to/best.pt -d

# Validate depth against a target 2 meters away
python3 sim_camera_zed.py -td 2.0

# Combine all flags
python3 sim_camera_zed.py -m models/best_rf.pt -d -td 1.5
```

---

### `yolo_detector.py`
**Pure perception module.** Takes an RGB frame and optionally a depth map, runs YOLOv8 inference, and returns a `List[Detection]`. Knows nothing about ROS2, MAVSDK, world frames, or which camera is active.

Key details:
- Lazy-loads ultralytics on first `start()` call to avoid import-time GPU initialization
- Uses ByteTrack (`track()`) for persistent IDs across frames when tracking is enabled
- Depth back-projection samples a 5×5 patch around the bbox center and takes the median (robust to missing/noisy depth pixels)
- Returns `position_3d` in **camera frame** — world-frame transform is the caller's responsibility

**Usage:**
```python
from yolo_detector import YoloDetector
det = YoloDetector(weights="path/to/best.pt", class_names=["red_buoy", "green_buoy", "blue_buoy"])
det.start(enable_tracking=True)
detections = det.detect(rgb_frame, depth_map, intrinsics)
```

---

### `ground_truth_labeler.py`
**Auto-generates YOLO training labels while the drone flies in Isaac Sim.** Run inside the Isaac Script Editor after `init_scene.py`.

How it works:
1. Hooks into Isaac's render event stream (`get_update_event_stream`)
2. Every Nth frame (configurable via `LABELER_SAVE_EVERY_N` in config), reads the drone's world pose via `XFormPrim.get_world_pose()`
3. Projects each known buoy world position through the full transform chain: **World → Body → Camera → Pixel** using `R_BODY_TO_CAM` from config
4. Estimates bbox size from known buoy physical radius and depth: `px_radius = fx * BUOY_RADIUS_M / depth`
5. Saves PNG images + YOLO-format `.txt` label files to `~/seabird_dataset/`, with automatic 80/20 train/val split
6. Writes `dataset.yaml` ready for `yolo train`
7. Saves debug images with drawn bboxes for the first 100 frames

**Usage (Isaac Script Editor):**
```python
exec(open("/home/<user>/seabird/scripts/ground_truth_labeler.py").read())
```

Fly the drone with `keyboard_controller.py` or another script to collect frames. Stop recording:
```python
_labeler_sub = None   # run this in Script Editor
```

Train after collecting data:
```bash
yolo detect train data=~/seabird_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=320
```

---

### `gt_label_viz.py`
**Visual verification tool for the ground truth projection math.** Pulls live camera frames from `SimCamera`, reads drone pose from a JSON file written by a pose writer script, projects known buoy positions into pixel space, and draws circles and bounding boxes on a live OpenCV window.

If the projected circles sit exactly on the buoys in the camera feed, the projection math in `ground_truth_labeler.py` is correct and will produce accurate labels.

**Usage (system terminal, with Isaac + init_scene.py + PX4 running):**
```bash
source /opt/ros/humble/setup.bash
cd ~/seabird/scripts
python3 gt_label_viz.py
```

Controls: `q` to quit, `d` to toggle debug text overlay.

---

### `sweep_and_detect.py`
**Full autonomous mission script.** Arms the drone, flies a lawnmower coverage pattern over the buoy field, and collects detections from `sim_camera.py` in real time.

Architecture — two concurrent threads:
- **Main thread (asyncio):** MAVSDK flight control — arm → takeoff → offboard lawnmower → land
- **Background thread (rclpy):** `DetectionSubscriber` listens on `/seabird/buoy_detections` and feeds the `DetectionLedger`

`DetectionLedger` is thread-safe and deduplicates detections by proximity (default 3m radius) — multiple sightings of the same buoy update observation count and best-confidence position rather than creating duplicate entries.

After the sweep completes:
- Validates detected buoy positions against `BUOY_POSITIONS` ground truth from config
- Reports which buoys were missed
- Saves the full ledger to `~/seabird/sweep_ledger.json`

**Usage (system terminal, with Isaac + init_scene.py + PX4 + sim_camera.py all running):**
```bash
source /opt/ros/humble/setup.bash
python3 ~/seabird/scripts/sweep_and_detect.py
```

---

### `keyboard_controller.py`
**Manual WASD flight controller** using MAVSDK velocity offboard mode. Useful for manually flying the drone to collect training data with `ground_truth_labeler.py`.

Controls:
| Key | Action |
|-----|--------|
| `W` / `S` | Forward / backward |
| `A` / `D` | Left / right |
| `SPACE` / `C` | Up / down |
| `Q` / `E` | Yaw left / right |
| `1`–`5` | Speed preset (0.5 – 5.0 m/s) |
| `T` | Arm and take off |
| `L` | Land |
| `X` | Land and exit |

**Usage (system terminal, with PX4 SITL running):**
```bash
python3 ~/seabird/scripts/keyboard_controller.py
```

Publishes `VelocityBodyYawspeed` setpoints at 20 Hz while offboard mode is active.

---

### `takeoff_test.py`
**Minimal connectivity test.** Connects to PX4 via MAVSDK, arms, takes off to a fixed altitude, hovers briefly, then lands. Used to verify that Isaac Sim + PX4 SITL + MAVSDK are all wired up correctly before running more complex scripts.

**Usage:**
```bash
python3 ~/seabird/scripts/takeoff_test.py
```

---

### `inspect_usd.py`
**USD asset inspector.** Opens the marina USD file and prints its sublayers, prim references, and payload metadata. Used to debug the asset structure when something is missing or broken in the scene.

**Usage (Isaac Script Editor):**
```python
exec(open("/home/<user>/seabird/scripts/inspect_usd.py").read())
```

---

### `kill_all.sh`
**Process cleanup script.** Kills all Seabird-related processes except Isaac Sim itself.

Kills:
- PX4 SITL
- MAVSDK-based flight scripts
- Seabird Python scripts
- ROS2 nodes (`sim_camera`, `sweep_and_detect`, etc.)
- micro-XRCE / DDS bridge agent

**Usage:**
```bash
bash ~/seabird/scripts/kill_all.sh
```

> The ground truth labeler runs inside Isaac Sim and cannot be killed from outside. Stop it from the Script Editor with `_labeler_sub = None`.

---

## Typical Workflow

### Phase 1 — Collect Training Data

1. Start Isaac Sim, open the Script Editor
2. Run `init_scene.py` in the Script Editor
3. Start PX4 SITL in a terminal
4. Run `ground_truth_labeler.py` in the Script Editor
5. Run `keyboard_controller.py` in a terminal and fly the drone around the buoys
6. Stop the labeler with `_labeler_sub = None` when enough frames are saved
7. Train YOLO: `yolo detect train data=~/seabird_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=320`

### Phase 2 — Verify Projection Math (Optional)

1. Run Isaac Sim + `init_scene.py` + PX4 + a pose writer script
2. Run `gt_label_viz.py` and confirm the projected circles align with buoys in the live feed

### Phase 3 — Autonomous Sweep

1. Isaac Sim + `init_scene.py` + PX4 already running
2. Terminal 1: `python3 sim_camera.py` — starts YOLO detection and detection publisher
3. Terminal 2: `python3 sweep_and_detect.py` — runs the autonomous lawnmower mission
4. After landing, check `~/seabird/sweep_ledger.json` for results

### Cleanup
```bash
bash ~/seabird/scripts/kill_all.sh
```

---

## Dependencies

| Package | Used by |
|---------|---------|
| `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `message_filters` | `sim_camera.py`, `sweep_and_detect.py`, `gt_label_viz.py` |
| `ultralytics` (YOLOv8) | `yolo_detector.py` |
| `mavsdk` | `sweep_and_detect.py`, `keyboard_controller.py`, `takeoff_test.py` |
| `opencv-python` | `sim_camera.py`, `gt_label_viz.py`, `ground_truth_labeler.py` |
| `scipy` | `seabird_config.py` (`camera_to_world`) |
| `numpy` | everywhere |
| `omni.*`, `pxr`, `pegasus.*` | `init_scene.py`, `ground_truth_labeler.py`, `inspect_usd.py` (Isaac Sim only) |

ROS2 nodes run under **system Python 3.10**. Isaac Script Editor scripts run under **Isaac Python 3.11**. Do not mix site-packages between them.

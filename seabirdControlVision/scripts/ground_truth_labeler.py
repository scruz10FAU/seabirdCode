"""
ground_truth_labeler.py — Isaac Script Editor script

Projects known buoy world positions through the camera to generate
YOLO-format bounding box labels automatically while the drone flies.

No HSV needed — we're in sim, so we use ground truth positions directly.
This teaches the full projection math: World → Body → Camera → Pixel.

All parameters come from seabird_config.py — single source of truth.

Usage (Script Editor, AFTER init_scene.py):
    exec(open(os.path.expanduser("~/seabird/scripts/ground_truth_labeler.py")).read())

Then fly the drone (commander takeoff, or MAVSDK script).
Run `_labeler_sub = None` in Script Editor to stop recording.
"""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.11/site-packages"))

import numpy as np
from pxr import UsdGeom, Usd
import omni.usd
import omni.kit.app

# ── Import shared config ──
sys.path.insert(0, os.path.expanduser("~/seabird/scripts"))
from seabird_config import (
    HOME, DATASET_DIR,
    IMG_W, IMG_H,
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    CAMERA_FOCAL_LENGTH_MM, CAMERA_H_APERTURE_MM, CAMERA_V_APERTURE_MM,
    CAM_OFFSET_BODY, CAM_PITCH_DEG,
    R_BODY_TO_CAM,
    BUOY_RADIUS_M, BUOY_POSITIONS,
    CLASS_MAP, CLASS_NAMES, CLASS_COLORS_BGR,
    DRONE_BODY_PATH, CAMERA_PRIM_PATH,
    LABELER_SAVE_EVERY_N, LABELER_MAX_FRAMES,
    LABELER_DEBUG_MODE, LABELER_MIN_BBOX_PX,
    print_camera_config,
)

# Print config so we can verify in console
print_camera_config()


# ═══════════════════════════════════════════════════════════════
# MATH — the projection chain
# ═══════════════════════════════════════════════════════════════

def quat_to_matrix(q):
    """
    Quaternion [w, x, y, z] → 3x3 rotation matrix.
    XFormPrim.get_world_pose() returns scalar-first [w,x,y,z].
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
    ])


def project_buoy(buoy_world, drone_pos, drone_quat_wxyz, fx, fy, cx, cy):
    """
    Project a 3D world point to 2D pixel coordinates.

    Transform chain:
      1. World → Body:   p_body = R_drone⁻¹ @ (p_world - p_drone)
      2. Body → Camera:  p_cam  = R_body_to_cam @ (p_body - cam_offset)
      3. Camera → Pixel: u = fx * x/z + cx,  v = fy * y/z + cy

    Returns (u, v, depth_m) or None if point is behind/too close to camera.
    """
    # Step 1: World → Body
    R_drone = quat_to_matrix(drone_quat_wxyz)
    p_body = R_drone.T @ (buoy_world - drone_pos)

    # Step 2: Body → Camera
    p_cam = R_BODY_TO_CAM @ (p_body - CAM_OFFSET_BODY)

    # Behind camera or too close
    if p_cam[2] < 0.5:
        return None

    # Step 3: Camera → Pixel (pinhole projection)
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy

    return (u, v, p_cam[2])


# USD-to-OpenCV flip: USD camera has Y=up, Z=-forward; OpenCV has Y=down, Z=forward
_F_USD_TO_CV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

def project_buoy_via_camera(buoy_world, cam_pos, cam_quat_wxyz, fx, fy, cx, cy):
    """
    DIAGNOSTIC: Project using the camera prim's own world pose.
    Bypasses the body→camera decomposition entirely.
    If this matches the render but project_buoy() doesn't, the bug is in our
    body-to-camera transform or camera offset.
    """
    R_cam_usd = quat_to_matrix(cam_quat_wxyz)  # camera local → world
    # World → camera (OpenCV convention)
    p_cam = _F_USD_TO_CV @ R_cam_usd.T @ (buoy_world - cam_pos)

    if p_cam[2] < 0.5:
        return None

    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy

    return (u, v, p_cam[2])


def make_yolo_label(u, v, depth, fx, class_id):
    """
    Generate a YOLO-format label from a projected buoy center.
    Bounding box estimated from known buoy radius and distance:
      pixel_radius = fx * real_radius / depth
    Returns "class_id cx cy w h" (normalized 0-1) or None if too small/offscreen.
    """
    px_r = fx * BUOY_RADIUS_M / depth

    x0 = max(0.0, u - px_r)
    y0 = max(0.0, v - px_r)
    x1 = min(float(IMG_W), u + px_r)
    y1 = min(float(IMG_H), v + px_r)

    if (x1 - x0) < LABELER_MIN_BBOX_PX or (y1 - y0) < LABELER_MIN_BBOX_PX:
        return None

    cx_n = ((x0 + x1) / 2) / IMG_W
    cy_n = ((y0 + y1) / 2) / IMG_H
    w_n  = (x1 - x0) / IMG_W
    h_n  = (y1 - y0) / IMG_H

    return f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


# ═══════════════════════════════════════════════════════════════
# SCENE SETUP
# ═══════════════════════════════════════════════════════════════

stage = omni.usd.get_context().get_stage()

# ── Use intrinsics from config (not from USD prim) ──
# The config is the source of truth. init_scene.py sets the USD prim
# to match, but we don't depend on reading it back correctly.
_fx, _fy, _cx, _cy = CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
print(f"[labeler] Using config intrinsics: fx={_fx:.1f}  fy={_fy:.1f}  "
      f"cx={_cx:.1f}  cy={_cy:.1f}")

# ── Sanity check: compare config vs USD prim ──
cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
if cam_prim.IsValid():
    cam = UsdGeom.Camera(cam_prim)
    usd_fl = cam.GetFocalLengthAttr().Get()
    usd_ha = cam.GetHorizontalApertureAttr().Get()
    if usd_fl and usd_ha:
        usd_fx = usd_fl * IMG_W / usd_ha
        if abs(usd_fx - _fx) > 1.0:
            print(f"[labeler] ⚠ USD prim intrinsics MISMATCH: usd_fx={usd_fx:.1f} vs config_fx={_fx:.1f}")
            print(f"[labeler]   USD prim: fl={usd_fl}, ha={usd_ha}")
            print(f"[labeler]   Config:   fl={CAMERA_FOCAL_LENGTH_MM}, ha={CAMERA_H_APERTURE_MM}")
            print(f"[labeler]   Using CONFIG values (source of truth). Fix init_scene.py if renderer looks wrong.")
        else:
            print(f"[labeler] USD prim intrinsics match config ✓")

# ── Build buoy list from config positions ──
buoys = []
for color, pos in BUOY_POSITIONS.items():
    buoys.append({
        "class_id": CLASS_MAP[color],
        "class_name": CLASS_NAMES[CLASS_MAP[color]],
        "pos": pos,
    })

print(f"[labeler] {len(buoys)} buoys from config:")
for b in buoys:
    print(f"  {b['class_name']:12s} at ({b['pos'][0]:7.2f}, {b['pos'][1]:7.2f}, {b['pos'][2]:7.2f})")

# ── Drone pose accessor ──
try:
    from isaacsim.core.api.prims import XFormPrim
except ImportError:
    from omni.isaac.core.prims import XFormPrim

drone_xform = XFormPrim(DRONE_BODY_PATH)
cam_xform = XFormPrim(CAMERA_PRIM_PATH)   # for diagnostic comparison

# Verify drone pose is readable
# Isaac's XFormPrim.get_world_pose() returns quaternion as [w, x, y, z] (scalar-first).
# NEVER use a heuristic to guess the format — q and -q are the same rotation,
# so |w| vs |z| comparisons break for ~90° yaw rotations. (Lesson #46)
try:
    _tp, _tq = drone_xform.get_world_pose()
    print(f"[labeler] Drone pose: pos=({_tp[0]:.2f}, {_tp[1]:.2f}, {_tp[2]:.2f})  "
          f"quat_wxyz=({_tq[0]:.3f}, {_tq[1]:.3f}, {_tq[2]:.3f}, {_tq[3]:.3f})")
except Exception as e:
    print(f"[labeler] ERROR reading drone pose: {e}")

# ── Image capture via Replicator annotator ──
USE_ANNOTATOR = False
try:
    import omni.replicator.core as rep
    rp = rep.create.render_product(CAMERA_PRIM_PATH, (IMG_W, IMG_H))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])
    USE_ANNOTATOR = True
    print("[labeler] Replicator RGB annotator attached ✓")
except Exception as e:
    print(f"[labeler] Replicator not available: {e}")

# ── Output directories ──
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

if LABELER_DEBUG_MODE:
    os.makedirs(os.path.join(DATASET_DIR, "debug"), exist_ok=True)

# ── dataset.yaml ──
yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {DATASET_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(CLASS_NAMES)}\n")
    f.write(f"names: {CLASS_NAMES}\n")
print(f"[labeler] dataset.yaml → {yaml_path}")


# ═══════════════════════════════════════════════════════════════
# FRAME CALLBACK
# ═══════════════════════════════════════════════════════════════

_frame_count = 0
_saved_count = 0
_labeler_sub = None


def _on_frame(event):
    global _frame_count, _saved_count, _labeler_sub

    if _saved_count >= LABELER_MAX_FRAMES:
        if _labeler_sub is not None:
            _labeler_sub = None
            print(f"\n[labeler] DONE — saved {_saved_count} frames to {DATASET_DIR}")
        return

    _frame_count += 1
    if _frame_count % LABELER_SAVE_EVERY_N != 0:
        return

    # ── Get drone pose ──
    try:
        d_pos, d_quat = drone_xform.get_world_pose()
    except Exception:
        return

    # get_world_pose() returns [w,x,y,z] — use directly
    q_wxyz = np.array(d_quat)

    # Debug: log drone pose on first saved frame
    if _saved_count == 0:
        print(f"\n[labeler] FIRST FRAME DEBUG:")
        print(f"  drone pos=({d_pos[0]:.2f}, {d_pos[1]:.2f}, {d_pos[2]:.2f})")
        print(f"  drone quat_wxyz=({q_wxyz[0]:.3f}, {q_wxyz[1]:.3f}, {q_wxyz[2]:.3f}, {q_wxyz[3]:.3f})")

        # Also read camera world pose for diagnostic comparison
        try:
            c_pos, c_quat = cam_xform.get_world_pose()
            print(f"  camera pos=({c_pos[0]:.3f}, {c_pos[1]:.3f}, {c_pos[2]:.3f})")
            print(f"  camera quat_wxyz=({c_quat[0]:.3f}, {c_quat[1]:.3f}, {c_quat[2]:.3f}, {c_quat[3]:.3f})")

            # Compare body-based vs camera-based projection
            print(f"\n  PROJECTION COMPARISON (body-chain vs camera-direct):")
            for buoy in buoys:
                r1 = project_buoy(buoy["pos"], d_pos, q_wxyz, _fx, _fy, _cx, _cy)
                r2 = project_buoy_via_camera(buoy["pos"], c_pos, c_quat, _fx, _fy, _cx, _cy)
                u1, v1 = (r1[0], r1[1]) if r1 else (None, None)
                u2, v2 = (r2[0], r2[1]) if r2 else (None, None)
                if u1 is not None and u2 is not None:
                    print(f"  {buoy['class_name']:12s}  body=({u1:6.1f}, {v1:6.1f})  "
                          f"cam=({u2:6.1f}, {v2:6.1f})  "
                          f"delta=({u2-u1:+.1f}, {v2-v1:+.1f})")
                else:
                    print(f"  {buoy['class_name']:12s}  body={'BEHIND' if r1 is None else 'ok'}  "
                          f"cam={'BEHIND' if r2 is None else 'ok'}")
        except Exception as e:
            print(f"  camera pose read failed: {e}")

    # ── Project each buoy ──
    labels = []
    projections = []

    for buoy in buoys:
        result = project_buoy(buoy["pos"], d_pos, q_wxyz, _fx, _fy, _cx, _cy)

        # Debug: log every buoy's projection on first saved frame
        if _saved_count == 0:
            if result is None:
                print(f"  {buoy['class_name']:12s} → BEHIND CAMERA")
            else:
                u, v, depth = result
                in_frame = 0 <= u < IMG_W and 0 <= v < IMG_H
                print(f"  {buoy['class_name']:12s} → pixel=({u:.0f}, {v:.0f})  "
                      f"depth={depth:.1f}m  {'IN FRAME' if in_frame else 'OUT OF FRAME'}")

        if result is None:
            continue
        u, v, depth = result
        label = make_yolo_label(u, v, depth, _fx, buoy["class_id"])
        if label:
            labels.append(label)
            projections.append((u, v, depth, buoy["class_id"]))

    if not labels:
        return

    # ── Get image ──
    rgb = None
    if USE_ANNOTATOR:
        try:
            rgba = rgb_annot.get_data()
            if rgba is not None and rgba.size > 0:
                rgb = np.array(rgba[:, :, :3])
        except Exception:
            pass

    # ── Save ──
    split = "val" if _saved_count % 5 == 0 else "train"
    fname = f"frame_{_saved_count:06d}"

    if rgb is not None:
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(rgb).save(
                os.path.join(DATASET_DIR, "images", split, f"{fname}.png"))
        except ImportError:
            np.save(os.path.join(DATASET_DIR, "images", split, f"{fname}.npy"), rgb)

    with open(os.path.join(DATASET_DIR, "labels", split, f"{fname}.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")

    # Debug images
    if LABELER_DEBUG_MODE and rgb is not None and _saved_count < 100:
        try:
            import cv2
            debug_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for u, v, depth, cid in projections:
                px_r = int(_fx * BUOY_RADIUS_M / depth)
                color = CLASS_COLORS_BGR[cid]
                cv2.rectangle(debug_img,
                              (int(u - px_r), int(v - px_r)),
                              (int(u + px_r), int(v + px_r)), color, 2)
                cv2.putText(debug_img, f"{CLASS_NAMES[cid]} {depth:.1f}m",
                            (int(u - px_r), int(v - px_r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(os.path.join(DATASET_DIR, "debug", f"{fname}.png"), debug_img)
        except ImportError:
            pass

    _saved_count += 1
    if _saved_count % 50 == 0:
        print(f"[labeler] {_saved_count}/{LABELER_MAX_FRAMES} saved  "
              f"({len(labels)} buoys visible, frame #{_frame_count})")


# ── Register ──
_labeler_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(_on_frame)

print(f"\n[labeler] ▸ RECORDING — every {LABELER_SAVE_EVERY_N} frames, max {LABELER_MAX_FRAMES}")
print(f"[labeler] ▸ Output: {DATASET_DIR}")
print(f"[labeler] ▸ Debug images: {'ON (first 100 frames)' if LABELER_DEBUG_MODE else 'OFF'}")
print(f"[labeler] ▸ Fly the drone to collect data!")
print(f"[labeler] ▸ Stop: run _labeler_sub = None in Script Editor\n")

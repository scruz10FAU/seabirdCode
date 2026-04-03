"""
breadcrumb_trail.py  v3
=======================
Run in Isaac Sim Script Editor AFTER spawn_drone.py + PX4 are connected.
Drops small glowing spheres along the drone's flight path.

Uses Isaac's dynamic control interface to read live physics position.

Usage:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/breadcrumb_trail.py").read())

Stop:   import builtins; builtins._breadcrumb_sub = None
Clear:  stage.RemovePrim("/World/Breadcrumbs")
"""

import builtins
import time
import omni.usd
import omni.kit.app
from pxr import UsdGeom, UsdShade, Gf, Sdf

# ── Try to get dynamic control interface (reads live physics transforms) ──────
_dc = None
_dc_mod = None
_body_handle = None

# Isaac Sim 4.5 pip install path
try:
    from isaacsim.core.utils.types import DynamicState
    import isaacsim.core.api  # ensure extension loaded
    from omni.isaac.dynamic_control import _dynamic_control
    _dc_mod = _dynamic_control
    _dc = _dynamic_control.acquire_dynamic_control_interface()
    print("[breadcrumb] Using dynamic_control interface (isaacsim path)")
except Exception as e1:
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        _dc_mod = _dynamic_control
        _dc = _dynamic_control.acquire_dynamic_control_interface()
        print("[breadcrumb] Using dynamic_control interface (omni path)")
    except Exception as e2:
        print(f"[breadcrumb] dynamic_control not available: {e1} / {e2}")
        print("[breadcrumb] Will fall back to XFormPrim")

# ── Fallback: try XFormPrim which reads physics state ─────────────────────────
_XFormPrim = None
if _dc is None:
    try:
        from isaacsim.core.prims import XFormPrim as _XFP
        _XFormPrim = _XFP
        print("[breadcrumb] Fallback: using isaacsim.core.prims.XFormPrim")
    except Exception:
        try:
            from omni.isaac.core.prims import XFormPrim as _XFP
            _XFormPrim = _XFP
            print("[breadcrumb] Fallback: using omni.isaac.core.prims.XFormPrim")
        except Exception:
            print("[breadcrumb] WARNING: No physics position API available")

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_PATH      = "/World/Iris"
DRONE_BODY_PATH = "/World/Iris/body"
CRUMB_ROOT      = "/World/Breadcrumbs"
DROP_INTERVAL   = 0.5
SPHERE_RADIUS   = 0.15
MIN_ALT         = 0.5

# ── State ─────────────────────────────────────────────────────────────────────
_crumb_idx = 0
_last_drop = 0.0
_last_pos = None
_xform_prim = None  # lazy-init XFormPrim instance
_dc_initialized = False

stage = omni.usd.get_context().get_stage()

# Kill previous subscription
if hasattr(builtins, '_breadcrumb_sub') and builtins._breadcrumb_sub is not None:
    builtins._breadcrumb_sub = None
    print("[breadcrumb] Stopped previous trail")

# Clean up old breadcrumbs
if stage.GetPrimAtPath(CRUMB_ROOT).IsValid():
    stage.RemovePrim(CRUMB_ROOT)
UsdGeom.Xform.Define(stage, CRUMB_ROOT)

# Create shared material
mat_path = f"{CRUMB_ROOT}/CrumbMat"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.SetSourceAsset("OmniPBR.mdl", "mdl")
shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
shader.CreateIdAttr("OmniPBR")
mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.4))
shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.3)
shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.4))
shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(2000.0)
print("[breadcrumb] Material created")


def _init_dc_handle():
    """Lazy-init the rigid body handle for the drone."""
    global _body_handle, _dc_initialized
    if _dc is None or _dc_initialized:
        return
    _dc_initialized = True

    # Try articulation root body first
    art = _dc.get_articulation(DRONE_PATH)
    if art != _dc_mod.INVALID_HANDLE:
        _body_handle = _dc.get_articulation_root_body(art)
        if _body_handle != _dc_mod.INVALID_HANDLE:
            print(f"[breadcrumb] DC: got articulation root body handle")
            return

    # Try rigid body on /body
    rb = _dc.get_rigid_body(DRONE_BODY_PATH)
    if rb != _dc_mod.INVALID_HANDLE:
        _body_handle = rb
        print(f"[breadcrumb] DC: got rigid body handle at {DRONE_BODY_PATH}")
        return

    # Try rigid body on root
    rb = _dc.get_rigid_body(DRONE_PATH)
    if rb != _dc_mod.INVALID_HANDLE:
        _body_handle = rb
        print(f"[breadcrumb] DC: got rigid body handle at {DRONE_PATH}")
        return

    print("[breadcrumb] DC: no valid rigid body handle found")


def _get_pos_dc():
    """Read position from dynamic control interface."""
    if _body_handle is None or _body_handle == _dc_mod.INVALID_HANDLE:
        return None
    try:
        pose = _dc.get_rigid_body_pose(_body_handle)
        return (pose.p.x, pose.p.y, pose.p.z)
    except Exception:
        return None


def _get_pos_xformprim():
    """Read position from XFormPrim.get_world_pose()."""
    global _xform_prim
    if _XFormPrim is None:
        return None
    try:
        if _xform_prim is None:
            _xform_prim = _XFormPrim(prim_path=DRONE_PATH)
        pos, _ = _xform_prim.get_world_pose()
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except Exception:
        return None


def _get_drone_pos():
    """Try all methods to get live drone position."""
    _init_dc_handle()

    # Method 1: dynamic control
    pos = _get_pos_dc()
    if pos is not None:
        return pos

    # Method 2: XFormPrim
    pos = _get_pos_xformprim()
    if pos is not None:
        return pos

    return None


def _drop_breadcrumb(px, py, pz):
    global _crumb_idx
    name = f"c_{_crumb_idx:05d}"
    path = f"{CRUMB_ROOT}/{name}"

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(SPHERE_RADIUS)

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))

    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path))
    )
    _crumb_idx += 1


_debug_counter = 0

def _on_update(event):
    global _last_drop, _last_pos, _debug_counter

    now = time.time()
    if now - _last_drop < DROP_INTERVAL:
        return

    pos = _get_drone_pos()

    # Debug: print every ~5 seconds
    _debug_counter += 1
    if _debug_counter % 10 == 1:
        if pos:
            print(f"[breadcrumb] Pos: X={pos[0]:.2f} Y={pos[1]:.2f} Z={pos[2]:.2f}  crumbs={_crumb_idx}")
        else:
            print(f"[breadcrumb] Pos: None (no valid read)  crumbs={_crumb_idx}")

    if pos is None:
        _last_drop = now
        return

    if pos[2] < MIN_ALT:
        _last_drop = now
        return

    # Skip if hasn't moved
    if _last_pos is not None:
        dx = pos[0] - _last_pos[0]
        dy = pos[1] - _last_pos[1]
        dz = pos[2] - _last_pos[2]
        if (dx*dx + dy*dy + dz*dz) < 0.01:  # < 0.1m
            _last_drop = now
            return

    _drop_breadcrumb(pos[0], pos[1], pos[2])
    _last_pos = pos
    _last_drop = now


_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    _on_update, name="breadcrumb_trail"
)
builtins._breadcrumb_sub = _sub

print(f"[breadcrumb] Trail active — dropping sphere every {DROP_INTERVAL}s")
print(f"[breadcrumb] Debug: printing position every ~5s")
print(f"[breadcrumb] Stop: import builtins; builtins._breadcrumb_sub = None")
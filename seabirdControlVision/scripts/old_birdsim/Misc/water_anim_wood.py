"""
water_anim_wood.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/water_anim_wood.py").read())

1. Improved wood — warmer, darker, slight grain hint via roughness variation per prim
2. Animated water — per-frame callback driving roughness + color oscillation
   to simulate light catching moving ripples. No texture needed.
"""

import omni.usd
import omni.kit.app
from pxr import UsdShade, Gf, Sdf
import math, time

stage = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────────────────────
# 1. IMPROVED WOOD
#    Keep DockWood but dial in better colour — the orange is too
#    bright/flat. Real weathered marina dock = darker, more grey-brown.
#    Also give walkway vs fingers slightly different values so 
#    they read differently (fakes grain contrast without textures).
# ─────────────────────────────────────────────────────────────

def rebuild_mat(path, diffuse, tint, brightness, desaturation, roughness):
    old = stage.GetPrimAtPath(path)
    if old.IsValid():
        stage.RemovePrim(path)
    mat = UsdShade.Material.Define(stage, path)
    sh  = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateIdAttr("OmniPBR")
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    def ci(n, t, v): sh.CreateInput(n, t).Set(v)
    ci("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f, diffuse)
    ci("diffuse_tint",                  Sdf.ValueTypeNames.Color3f, tint)
    ci("albedo_brightness",             Sdf.ValueTypeNames.Float,   brightness)
    ci("albedo_desaturation",           Sdf.ValueTypeNames.Float,   desaturation)
    ci("reflection_roughness_constant", Sdf.ValueTypeNames.Float,   roughness)
    ci("metallic_constant",             Sdf.ValueTypeNames.Float,   0.0)
    ci("specular_level",                Sdf.ValueTypeNames.Float,   0.25)
    ci("opacity_constant",              Sdf.ValueTypeNames.Float,   1.0)
    return mat, sh

# Walkway planks — darker, oiled teak tone
walkway_mat, _ = rebuild_mat(
    "/World/Looks/DockWalkway",
    diffuse      = Gf.Vec3f(0.10, 0.058, 0.025),
    tint         = Gf.Vec3f(0.18, 0.10,  0.045),
    brightness   = 0.80,
    desaturation = 0.08,
    roughness    = 0.68,
)

# Finger piers — slightly lighter, more worn grey-brown
finger_mat, _ = rebuild_mat(
    "/World/Looks/DockFinger",
    diffuse      = Gf.Vec3f(0.14, 0.09,  0.045),
    tint         = Gf.Vec3f(0.22, 0.145, 0.075),
    brightness   = 0.78,
    desaturation = 0.18,
    roughness    = 0.74,
)

# Posts — weathered, pressure-treated grey-green
post_mat, _ = rebuild_mat(
    "/World/Looks/DockPost",
    diffuse      = Gf.Vec3f(0.15, 0.155, 0.12),
    tint         = Gf.Vec3f(0.22, 0.22,  0.17),
    brightness   = 0.70,
    desaturation = 0.35,
    roughness    = 0.86,
)

# Bind by prim name pattern
bound = {"walkway": 0, "finger": 0, "post": 0}
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.GetTypeName() != "Cube":
        continue
    if "/World/Dock" not in path and "/World/DockB" not in path:
        continue

    lower = path.lower()
    if "post" in lower:
        UsdShade.MaterialBindingAPI(prim).Bind(post_mat)
        bound["post"] += 1
    elif "finger" in lower:
        UsdShade.MaterialBindingAPI(prim).Bind(finger_mat)
        bound["finger"] += 1
    else:
        UsdShade.MaterialBindingAPI(prim).Bind(walkway_mat)
        bound["walkway"] += 1

print(f"[wood] Bound — walkway:{bound['walkway']} finger:{bound['finger']} post:{bound['post']}")

# ─────────────────────────────────────────────────────────────
# 2. WATER MATERIAL — base state
# ─────────────────────────────────────────────────────────────
water_mat, water_shader = rebuild_mat(
    "/World/Looks/WaterMat",
    diffuse      = Gf.Vec3f(0.004, 0.055, 0.075),
    tint         = Gf.Vec3f(0.008, 0.10,  0.13),
    brightness   = 0.20,
    desaturation = 0.06,
    roughness    = 0.03,
)
# Extra inputs for water
water_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.15)
water_shader.CreateInput("specular_level",    Sdf.ValueTypeNames.Float).Set(1.0)

# Bind to water plane
water_usd_mat = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/WaterMat"))
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/World/WaterPlane"):
        UsdShade.MaterialBindingAPI(prim).Bind(water_usd_mat)
        print(f"[water] Bound WaterMat to {path}")

# ─────────────────────────────────────────────────────────────
# 3. ANIMATED WATER — per-frame callback
#    Oscillates roughness + brightness + tint to simulate
#    light catching moving ripples. Cheap, no texture needed.
#
#    Stop animation: import builtins; builtins._water_sub = None
# ─────────────────────────────────────────────────────────────

# Cancel any existing subscription from a previous run
import builtins
if hasattr(builtins, "_water_sub"):
    builtins._water_sub = None
    print("[water] Cancelled previous animation subscription")

_start = time.time()

# Cache shader input handles for speed (avoid repeated GetInput lookup per frame)
_inp_rough  = water_shader.GetInput("reflection_roughness_constant")
_inp_bright = water_shader.GetInput("albedo_brightness")
_inp_tint   = water_shader.GetInput("diffuse_tint")
_inp_metal  = water_shader.GetInput("metallic_constant")

def _water_tick(event):
    t = time.time() - _start

    # Layer multiple sine waves at different frequencies for irregular ripples
    #   f1 = slow swell     (0.25 Hz)
    #   f2 = medium chop    (0.7 Hz)
    #   f3 = fast glint     (1.8 Hz)
    f1 = math.sin(t * 0.25 * 2 * math.pi)
    f2 = math.sin(t * 0.70 * 2 * math.pi + 1.1)
    f3 = math.sin(t * 1.80 * 2 * math.pi + 2.3)

    # Roughness: 0.02 (mirror-calm) → 0.10 (choppy)
    roughness = 0.025 + 0.018 * (0.5 * f1 + 0.3 * f2 + 0.2 * f3 + 1.0) / 2.0

    # Brightness flicker — light catching wave crests
    brightness = 0.18 + 0.06 * (0.6 * f2 + 0.4 * f3 + 1.0) / 2.0

    # Tint — very subtle green-blue shift
    tg = 0.09  + 0.015 * (f1 * 0.5 + f2 * 0.5 + 1.0) / 2.0
    tb = 0.120 + 0.018 * (f1 * 0.4 + f3 * 0.6 + 1.0) / 2.0

    # Metallic — glint spikes
    metal = 0.12 + 0.08 * max(0.0, f3)

    if _inp_rough:  _inp_rough.Set(float(roughness))
    if _inp_bright: _inp_bright.Set(float(brightness))
    if _inp_tint:   _inp_tint.Set(Gf.Vec3f(0.006, float(tg), float(tb)))
    if _inp_metal:  _inp_metal.Set(float(metal))

app = omni.kit.app.get_app()
builtins._water_sub = app.get_update_event_stream().create_subscription_to_pop(
    _water_tick, name="seabird_water_anim"
)

print("[water] Ripple animation running ✓")
print("        Stop with: import builtins; builtins._water_sub = None")
print("\n[done] Viewport tip: camera icon → Exposure=0, disable Auto Exposure")
print("       For richer reflections: switch renderer to RTX - Interactive (Path Traced)")

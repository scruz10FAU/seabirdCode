"""
add_sky.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/add_sky.py").read())

Tries in order:
  1. omni.kit.environment.core dynamic sky (procedural sun + clouds)
  2. RtxDynamicSky USD prim (Omniverse native)
  3. DomeLight with bundled HDR texture search
  4. Fallback: DomeLight with sky-blue tint + DistantLight tuning
"""

import omni.usd
import omni.kit.app
from pxr import UsdGeom, UsdLux, UsdShade, Gf, Sdf
import os, glob

stage = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def remove_if_exists(path):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)
        print(f"[sky] Removed existing {path}")

def find_hdr_files():
    """Search common Isaac Sim / system paths for HDR sky files."""
    search_roots = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/isaacsim"),
        os.path.expanduser("~/.local/share/ov"),
        "/home/tgarcia/drone_sim",
        "/usr/local/lib",
    ]
    hdrs = []
    keywords = ["sky", "outdoor", "cloud", "sun", "hdri", "environment", "clear"]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for ext in ("*.hdr", "*.exr", "*.HDR", "*.EXR"):
            for f in glob.glob(os.path.join(root, "**", ext), recursive=True):
                lower = f.lower()
                if any(k in lower for k in keywords):
                    hdrs.append(f)
    return hdrs

# ─────────────────────────────────────────────────────────────
# ATTEMPT 1 — omni.kit.environment.core dynamic sky
# ─────────────────────────────────────────────────────────────
sky_done = False

try:
    import omni.kit.environment.core as env_core
    # Enable the extension if not already running
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("omni.kit.environment.core"):
        manager.set_extension_enabled_immediate("omni.kit.environment.core", True)
        print("[sky] Enabled omni.kit.environment.core")

    # The extension provides a sky creation API
    env = env_core.get_environment_interface()
    if env is not None:
        env.set_dynamic_sky(True)
        print("[sky] Dynamic sky enabled via omni.kit.environment.core ✓")
        sky_done = True
    else:
        print("[sky] env interface returned None, trying next method...")
except Exception as e:
    print(f"[sky] omni.kit.environment.core not available: {e}")

# ─────────────────────────────────────────────────────────────
# ATTEMPT 2 — RtxDynamicSky USD prim (Omniverse native prim type)
# ─────────────────────────────────────────────────────────────
if not sky_done:
    try:
        remove_if_exists("/World/Sky")
        sky_prim = stage.DefinePrim("/World/Sky", "RtxDynamicSky")
        if sky_prim.IsValid():
            # Set sky parameters
            def set_attr(prim, name, value):
                attr = prim.GetAttribute(name)
                if attr:
                    attr.Set(value)
                else:
                    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)

            # Sun position — afternoon Florida sun
            sky_prim.CreateAttribute("inputs:azimuth",              Sdf.ValueTypeNames.Float).Set(210.0)   # SW direction
            sky_prim.CreateAttribute("inputs:elevation",            Sdf.ValueTypeNames.Float).Set(42.0)    # mid-afternoon
            sky_prim.CreateAttribute("inputs:sky_brightness",       Sdf.ValueTypeNames.Float).Set(1.0)
            sky_prim.CreateAttribute("inputs:sun_disk_intensity",   Sdf.ValueTypeNames.Float).Set(1.0)
            sky_prim.CreateAttribute("inputs:ground_color",         Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.07, 0.05))
            sky_prim.CreateAttribute("inputs:horizon_height",       Sdf.ValueTypeNames.Float).Set(0.01)
            sky_prim.CreateAttribute("inputs:horizon_fuzziness",    Sdf.ValueTypeNames.Float).Set(0.1)
            sky_prim.CreateAttribute("inputs:cloud_coverage",       Sdf.ValueTypeNames.Float).Set(0.35)   # ~1/3 cloud cover
            sky_prim.CreateAttribute("inputs:cloud_scale",          Sdf.ValueTypeNames.Float).Set(2.0)
            sky_prim.CreateAttribute("inputs:saturation",           Sdf.ValueTypeNames.Float).Set(1.1)
            print("[sky] RtxDynamicSky prim created at /World/Sky ✓")
            sky_done = True
        else:
            print("[sky] RtxDynamicSky prim type not recognized in this Isaac version")
    except Exception as e:
        print(f"[sky] RtxDynamicSky failed: {e}")

# ─────────────────────────────────────────────────────────────
# ATTEMPT 3 — DomeLight with a bundled HDR texture
# ─────────────────────────────────────────────────────────────
if not sky_done:
    hdrs = find_hdr_files()
    if hdrs:
        print(f"[sky] Found {len(hdrs)} HDR file(s):")
        for h in hdrs[:5]:
            print(f"  {h}")

        # Pick the most sky-like one
        chosen = hdrs[0]
        for h in hdrs:
            if "sky" in h.lower() or "outdoor" in h.lower():
                chosen = h
                break

        remove_if_exists("/World/DomeLight")
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(1200.0)
        dome.GetExposureAttr().Set(0.0)
        dome.CreateTextureFileAttr().Set(chosen)
        dome.CreateTextureFormatAttr().Set("latlong")

        xf = UsdGeom.Xformable(dome.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 270.0))  # orient HDR north

        print(f"[sky] DomeLight with HDR: {chosen} ✓")
        sky_done = True
    else:
        print("[sky] No bundled HDR sky files found")

# ─────────────────────────────────────────────────────────────
# ATTEMPT 4 — Fallback: sky-blue DomeLight + tuned DistantLight
#   Not as good but always works and is stable
# ─────────────────────────────────────────────────────────────
if not sky_done:
    print("[sky] Using fallback: sky-blue DomeLight + DistantLight")
    remove_if_exists("/World/DomeLight")
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.GetIntensityAttr().Set(900.0)
    dome.GetExposureAttr().Set(0.0)

    # Sky blue color — Florida afternoon
    dome_prim = dome.GetPrim()
    dome_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.53, 0.74, 0.95)
    )
    print("[sky] Fallback sky-blue DomeLight created")

# ─────────────────────────────────────────────────────────────
# REGARDLESS OF METHOD — tune DistantLight for afternoon sun
# ─────────────────────────────────────────────────────────────
remove_if_exists("/World/DistantLight")
distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant.GetIntensityAttr().Set(7500.0)
distant.GetAngleAttr().Set(0.8)   # slight softness to shadow edge

# Warm afternoon sun color
distant.GetPrim().CreateAttribute(
    "inputs:color", Sdf.ValueTypeNames.Color3f
).Set(Gf.Vec3f(1.0, 0.93, 0.78))   # warm golden-white

# Afternoon sun: coming from SW, about 42deg elevation
sun_xf = UsdGeom.Xformable(distant.GetPrim())
sun_xf.ClearXformOpOrder()
sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 0.0, 210.0))

print("[sun] DistantLight: 7500 intensity, warm white, SW afternoon angle ✓")

# ─────────────────────────────────────────────────────────────
# TRY to add OmniSky via Create menu path (extension-based)
# ─────────────────────────────────────────────────────────────
try:
    import omni.kit.commands
    # This command is registered by omni.kit.environment.core when loaded
    omni.kit.commands.execute("CreateDynamicSkyCommand",
        sky_prim_path="/World/OmniSky"
    )
    print("[sky] CreateDynamicSkyCommand executed ✓")
except Exception as e:
    print(f"[sky] CreateDynamicSkyCommand not available: {e}")

print("\n[done] Sky setup complete.")
print("  If sky is still plain grey, try manually:")
print("  Create menu → Environment → Dynamic Sky (or Sky)")
print("  Then paste this into Script Editor to tune it:")
print("""
# Tune sky prim after adding via menu — find its path first:
for prim in stage.Traverse():
    if 'Sky' in str(prim.GetPath()) or 'sky' in prim.GetTypeName().lower():
        print(prim.GetPath(), prim.GetTypeName())
        for attr in prim.GetAttributes():
            print(' ', attr.GetName(), '=', attr.Get())
""")

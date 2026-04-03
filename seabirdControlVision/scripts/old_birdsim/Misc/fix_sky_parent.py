"""
fix_sky_parent.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_sky_parent.py").read())

Isaac 4.5 path tracing is running (1/512 spp confirmed) but sky is grey.
Root cause: RtxDynamicSky must be a CHILD of the DomeLight prim, not a sibling.
Correct structure: /World/DomeLight/Sky  (not /World/Sky)
"""

import omni.usd
import carb.settings
from pxr import UsdLux, UsdGeom, Gf, Sdf, Usd

stage    = omni.usd.get_context().get_stage()
settings = carb.settings.get_settings()

def remove_if_exists(path):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)
        print(f"  Removed {path}")

# ─────────────────────────────────────────────────────────────
# 1. REMOVE OLD PRIMS
# ─────────────────────────────────────────────────────────────
print("[cleanup] Removing old sky/dome prims...")
remove_if_exists("/World/Sky")
remove_if_exists("/World/DomeLight")

# ─────────────────────────────────────────────────────────────
# 2. CREATE DOMELIGHT
# ─────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_prim = dome.GetPrim()

# No texture — sky child prim drives the appearance
dome.GetIntensityAttr().Set(1.0)
dome.GetExposureAttr().Set(0.0)

# Rotate so sky north aligns with scene
dome_xf = UsdGeom.Xformable(dome_prim)
dome_xf.ClearXformOpOrder()
dome_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 270.0))

print("[dome] DomeLight created at /World/DomeLight")

# ─────────────────────────────────────────────────────────────
# 3. CREATE RtxDynamicSky AS CHILD OF DOMELIGHT
#    This is the correct Isaac 4.5 hierarchy
# ─────────────────────────────────────────────────────────────
sky_path = "/World/DomeLight/Sky"
sky_prim = stage.DefinePrim(sky_path, "RtxDynamicSky")

if sky_prim.IsValid():
    print(f"[sky] RtxDynamicSky created at {sky_path} (child of DomeLight)")

    def sky_set(name, type_name, value):
        attr = sky_prim.GetAttribute(name)
        if attr and attr.IsValid():
            attr.Set(value)
        else:
            sky_prim.CreateAttribute(name, type_name).Set(value)

    # Florida afternoon — scattered cumulus
    sky_set("inputs:cloud_coverage",     Sdf.ValueTypeNames.Float,   0.55)
    sky_set("inputs:cloud_scale",        Sdf.ValueTypeNames.Float,   1.2)
    sky_set("inputs:sky_brightness",     Sdf.ValueTypeNames.Float,   1.0)
    sky_set("inputs:saturation",         Sdf.ValueTypeNames.Float,   1.1)
    sky_set("inputs:azimuth",            Sdf.ValueTypeNames.Float,   200.0)
    sky_set("inputs:elevation",          Sdf.ValueTypeNames.Float,   38.0)
    sky_set("inputs:sun_disk_intensity", Sdf.ValueTypeNames.Float,   1.5)
    sky_set("inputs:horizon_height",     Sdf.ValueTypeNames.Float,   0.0)
    sky_set("inputs:horizon_fuzziness",  Sdf.ValueTypeNames.Float,   0.06)
    sky_set("inputs:ground_color",       Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.04, 0.04, 0.035))

    print("[sky] Attributes set: 55% cloud cover, Florida afternoon sun")
else:
    print("[sky] ERROR — RtxDynamicSky prim type not valid, trying fallback...")

    # Fallback: HDR-less dome with strong sky blue
    dome.GetIntensityAttr().Set(800.0)
    dome_prim.CreateAttribute(
        "inputs:color", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(0.42, 0.62, 0.88))
    print("[sky] Fallback: sky-blue DomeLight at 800 intensity")

# ─────────────────────────────────────────────────────────────
# 4. TUNE DISTANTLIGHT — compensate for sky-driven ambient
# ─────────────────────────────────────────────────────────────
sun_prim = stage.GetPrimAtPath("/World/DistantLight")
if sun_prim.IsValid():
    distant = UsdLux.DistantLight(sun_prim)
    distant.GetIntensityAttr().Set(5500.0)
    print("[sun] DistantLight → 5500")

# ─────────────────────────────────────────────────────────────
# 5. CONFIRM PATH TRACING IS STILL ON
# ─────────────────────────────────────────────────────────────
try:
    mode = settings.get("/rtx/rendermode")
    print(f"[renderer] Current mode: {mode}")
    if mode != "PathTracing":
        settings.set("/rtx/rendermode", "PathTracing")
        print("[renderer] Re-enabled PathTracing")
    else:
        print("[renderer] PathTracing already active ✓")
except Exception as e:
    print(f"[renderer] {e}")

# ─────────────────────────────────────────────────────────────
# 6. DUMP final structure so we can verify
# ─────────────────────────────────────────────────────────────
print("\n=== DomeLight subtree ===")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/World/DomeLight"):
        depth = path.count("/") - 2
        print(f"{'  '*depth}[{prim.GetTypeName()}] {path}")
        for attr in prim.GetAttributes():
            v = attr.Get()
            if v is not None:
                print(f"{'  '*depth}  {attr.GetName()} = {v}")
print("=== End ===")

print("""
[done] If sky still grey after this:
  - Try: Create menu → Lights → Dome Light  (then delete /World/DomeLight above,
    use the one Isaac creates — it may set internal flags ours can't)
  - OR: accept RTX Real-Time mode with blue DomeLight — switch renderer back:
      import carb.settings
      carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
""")

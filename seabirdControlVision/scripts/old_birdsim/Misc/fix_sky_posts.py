"""
fix_sky_posts.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_sky_posts.py").read())
"""

import omni.usd
import omni.kit.app
from pxr import UsdGeom, UsdLux, UsdShade, Gf, Sdf
import omni.kit.commands

stage = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────────────────────
# 1. DUMP /World/Sky so we can see all real attribute names
# ─────────────────────────────────────────────────────────────
sky_prim = stage.GetPrimAtPath("/World/Sky")
if sky_prim.IsValid():
    print(f"\n=== /World/Sky  type={sky_prim.GetTypeName()} ===")
    for attr in sky_prim.GetAttributes():
        print(f"  {attr.GetName():50s} = {attr.Get()}")
    print("=== End Sky attrs ===\n")
else:
    print("[sky] /World/Sky not found")

# ─────────────────────────────────────────────────────────────
# 2. FIX SKY — RtxDynamicSky needs a DomeLight to drive it,
#    AND the DomeLight must reference the sky prim via
#    light:shaderId or the sky prim needs to be under /World/Render
#    Try multiple activation paths.
# ─────────────────────────────────────────────────────────────

def remove_if_exists(path):
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)

# Approach A: DomeLight with texture format set to "automatic" 
# and color that ISN'T white — the sky prim drives the background
# but lighting still comes from the DomeLight
remove_if_exists("/World/DomeLight")
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(1000.0)
dome.GetExposureAttr().Set(0.0)

dome_prim = dome.GetPrim()

# Color: sky blue — if RtxDynamicSky doesn't render, at least we get blue
dome_prim.CreateAttribute("inputs:color",         Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.65, 0.90))
dome_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
dome_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)

# Some Isaac versions need this to activate RtxDynamicSky
try:
    dome_prim.CreateAttribute("rtx:shaderId", Sdf.ValueTypeNames.Token).Set("RtxDynamicSky")
    print("[sky] Set rtx:shaderId on DomeLight")
except:
    pass

# Orient dome
dome_xf = UsdGeom.Xformable(dome_prim)
dome_xf.ClearXformOpOrder()
dome_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 270.0))
print("[sky] DomeLight rebuilt with sky-blue color")

# Approach B: Try activating via omni.kit.commands
for cmd in ["CreatePhysicsSkyCommand", "AddSkyCommand", "CreateSkyDomeCommand"]:
    try:
        omni.kit.commands.execute(cmd)
        print(f"[sky] {cmd} succeeded")
        break
    except:
        pass

# Approach C: Set RtxDynamicSky attributes using the CORRECT names
# (the ones that actually exist on the prim, seen from the dump above)
if sky_prim.IsValid():
    def set_sky(name, type_name, value):
        attr = sky_prim.GetAttribute(name)
        if attr:
            attr.Set(value)
            return True
        else:
            sky_prim.CreateAttribute(name, type_name).Set(value)
            return True

    # Try every known naming convention for RtxDynamicSky
    attrs_to_try = [
        # (attr_name, type, value)
        ("inputs:cloudCoverage",         Sdf.ValueTypeNames.Float,   0.35),
        ("inputs:cloud_coverage",        Sdf.ValueTypeNames.Float,   0.35),
        ("cloudCoverage",                Sdf.ValueTypeNames.Float,   0.35),
        ("inputs:sunAzimuthAngle",       Sdf.ValueTypeNames.Float,   210.0),
        ("inputs:sun_azimuth",           Sdf.ValueTypeNames.Float,   210.0),
        ("inputs:azimuth",               Sdf.ValueTypeNames.Float,   210.0),
        ("inputs:sunElevationAngle",     Sdf.ValueTypeNames.Float,   42.0),
        ("inputs:elevation",             Sdf.ValueTypeNames.Float,   42.0),
        ("inputs:skyBrightness",         Sdf.ValueTypeNames.Float,   1.0),
        ("inputs:sky_brightness",        Sdf.ValueTypeNames.Float,   1.0),
        ("inputs:groundColor",           Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.06, 0.05, 0.04)),
        ("inputs:ground_color",          Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.06, 0.05, 0.04)),
    ]
    for name, t, v in attrs_to_try:
        set_sky(name, t, v)

    print("[sky] Set all RtxDynamicSky candidate attributes")

# ─────────────────────────────────────────────────────────────
# 3. DISTANTLIGHT — warm afternoon sun
# ─────────────────────────────────────────────────────────────
remove_if_exists("/World/DistantLight")
distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant.GetIntensityAttr().Set(6000.0)
distant.GetAngleAttr().Set(0.8)
distant.GetPrim().CreateAttribute(
    "inputs:color", Sdf.ValueTypeNames.Color3f
).Set(Gf.Vec3f(1.0, 0.94, 0.80))
sun_xf = UsdGeom.Xformable(distant.GetPrim())
sun_xf.ClearXformOpOrder()
sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 0.0, 210.0))
print("[sun] DistantLight set: 6000 intensity, warm white, SW afternoon")

# ─────────────────────────────────────────────────────────────
# 4. FIX POSTS — were too olive/green, fix to proper 
#    pressure-treated brown-grey (no green tint)
# ─────────────────────────────────────────────────────────────
def rebuild_mat(path, diffuse, tint, brightness, desaturation, roughness, specular=0.2):
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
    ci("specular_level",                Sdf.ValueTypeNames.Float,   specular)
    ci("opacity_constant",              Sdf.ValueTypeNames.Float,   1.0)
    return UsdShade.Material(stage.GetPrimAtPath(path))

# Posts: warm brown-grey, NO green — like old creosote/CCA timber
post_mat = rebuild_mat(
    "/World/Looks/DockPost",
    diffuse      = Gf.Vec3f(0.18, 0.13, 0.09),   # warm brown base
    tint         = Gf.Vec3f(0.24, 0.18, 0.13),   # slightly lighter warm tint
    brightness   = 0.72,
    desaturation = 0.20,                           # less desaturation = less grey-green
    roughness    = 0.88,
    specular     = 0.12,
)
print("[posts] Rebuilt DockPost: warm brown-grey (no green)")

# Walkway: richer dark brown — the name filter missed some last time
walkway_mat = rebuild_mat(
    "/World/Looks/DockWalkway",
    diffuse      = Gf.Vec3f(0.12, 0.068, 0.030),
    tint         = Gf.Vec3f(0.20, 0.115, 0.052),
    brightness   = 0.82,
    desaturation = 0.06,
    roughness    = 0.66,
    specular     = 0.28,
)

finger_mat = rebuild_mat(
    "/World/Looks/DockFinger",
    diffuse      = Gf.Vec3f(0.15, 0.090, 0.042),
    tint         = Gf.Vec3f(0.23, 0.145, 0.068),
    brightness   = 0.80,
    desaturation = 0.10,
    roughness    = 0.72,
    specular     = 0.22,
)

# ─────────────────────────────────────────────────────────────
# 5. REBIND — fixed logic: walkway = everything that's NOT post/finger
# ─────────────────────────────────────────────────────────────
bound = {"walkway": 0, "finger": 0, "post": 0}
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.GetTypeName() != "Cube":
        continue
    if "/World/Dock" not in path and "/World/DockB" not in path:
        continue

    # Check parent prim name for classification
    parent_name = path.split("/")[-2].lower()   # e.g. "Post_L_0_0", "Finger_L_0", "Walkway"

    if "post" in parent_name:
        UsdShade.MaterialBindingAPI(prim).Bind(post_mat)
        bound["post"] += 1
    elif "finger" in parent_name:
        UsdShade.MaterialBindingAPI(prim).Bind(finger_mat)
        bound["finger"] += 1
    else:
        # Walkway, Connector, anything else
        UsdShade.MaterialBindingAPI(prim).Bind(walkway_mat)
        bound["walkway"] += 1

print(f"[bind] walkway:{bound['walkway']} finger:{bound['finger']} post:{bound['post']}")

print("\n[done] Sky dump is above — paste it here so we can see real attr names.")
print("       If sky still white: Create menu → Environment → Sky (manual fallback)")

"""
fix_wood_water.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_wood_water.py").read())

Strategy:
  1. Try injecting texture_scale into the remote MDL shader (sometimes works)
  2. Build a LOCAL OmniPBR wood material we fully control — no S3 dependency
  3. Bind it to all dock Cubes
  4. Improve water
"""

import omni.usd
from pxr import UsdShade, UsdGeom, Gf, Sdf

stage = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────────────────────
# STEP 1 — Try injecting inputs into the remote MDL
# ─────────────────────────────────────────────────────────────
walnut_shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Wood_Tiles_Walnut/Shader"))
if walnut_shader:
    try:
        walnut_shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(6.0, 6.0))
        print("[mdl] Injected texture_scale=(6,6) into Walnut MDL")
    except Exception as e:
        print(f"[mdl] texture_scale inject failed: {e}")
    try:
        # Some vMaterials use this name instead
        walnut_shader.CreateInput("uv_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(6.0, 6.0))
        print("[mdl] Injected uv_scale=(6,6)")
    except Exception as e:
        print(f"[mdl] uv_scale inject failed: {e}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Build a LOCAL OmniPBR wood we fully control
#   No S3 URL, no MDL mystery — just OmniPBR with good values.
#   Looks like oiled teak/mahogany: warm dark brown, low gloss.
# ─────────────────────────────────────────────────────────────
WOOD_MAT_PATH = "/World/Looks/DockWood"

# Remove old if re-running
old = stage.GetPrimAtPath(WOOD_MAT_PATH)
if old.IsValid():
    stage.RemovePrim(WOOD_MAT_PATH)

mat = UsdShade.Material.Define(stage, WOOD_MAT_PATH)
shader = UsdShade.Shader.Define(stage, WOOD_MAT_PATH + "/Shader")
shader.SetSourceAsset("OmniPBR.mdl", "mdl")
shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
shader.CreateIdAttr("OmniPBR")
mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

def ci(name, type_name, value):
    shader.CreateInput(name, type_name).Set(value)

# Warm oiled wood — dark walnut/teak tones
ci("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.18, 0.10, 0.045))
ci("diffuse_tint",                  Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.28, 0.16, 0.07))
ci("albedo_brightness",             Sdf.ValueTypeNames.Float,   0.85)
ci("albedo_desaturation",           Sdf.ValueTypeNames.Float,   0.05)

# Slightly reflective — oiled dock wood, not matte raw lumber
ci("reflection_roughness_constant", Sdf.ValueTypeNames.Float,   0.62)
ci("metallic_constant",             Sdf.ValueTypeNames.Float,   0.0)

# Specular — subtle sheen from weatherproofing
ci("specular_level",                Sdf.ValueTypeNames.Float,   0.3)

# No transparency
ci("opacity_constant",              Sdf.ValueTypeNames.Float,   1.0)

print(f"[wood] Created local OmniPBR DockWood at {WOOD_MAT_PATH}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Bind DockWood to every Dock/DockB Cube
# ─────────────────────────────────────────────────────────────
dock_wood_mat = UsdShade.Material(stage.GetPrimAtPath(WOOD_MAT_PATH))
bound = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.GetTypeName() == "Cube" and (
        "/World/Dock/" in path or "/World/DockB/" in path
    ):
        UsdShade.MaterialBindingAPI(prim).Bind(dock_wood_mat)
        bound += 1

print(f"[wood] DockWood bound to {bound} Cube prims")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Make posts slightly lighter/greyer than planks
#   Real dock posts are often bare pressure-treated wood, 
#   slightly greyer than oiled decking.
# ─────────────────────────────────────────────────────────────
POST_MAT_PATH = "/World/Looks/DockPost"
old = stage.GetPrimAtPath(POST_MAT_PATH)
if old.IsValid():
    stage.RemovePrim(POST_MAT_PATH)

pmat = UsdShade.Material.Define(stage, POST_MAT_PATH)
pshader = UsdShade.Shader.Define(stage, POST_MAT_PATH + "/Shader")
pshader.SetSourceAsset("OmniPBR.mdl", "mdl")
pshader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
pshader.CreateIdAttr("OmniPBR")
pmat.CreateSurfaceOutput("mdl").ConnectToSource(pshader.ConnectableAPI(), "out")

def pci(name, type_name, value):
    pshader.CreateInput(name, type_name).Set(value)

# Weathered grey-brown pressure-treated post
pci("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.22, 0.17, 0.12))
pci("diffuse_tint",                  Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.30, 0.24, 0.18))
pci("albedo_brightness",             Sdf.ValueTypeNames.Float,   0.75)
pci("albedo_desaturation",           Sdf.ValueTypeNames.Float,   0.25)  # more grey = weathered
pci("reflection_roughness_constant", Sdf.ValueTypeNames.Float,   0.82)  # rough bare wood
pci("metallic_constant",             Sdf.ValueTypeNames.Float,   0.0)
pci("opacity_constant",              Sdf.ValueTypeNames.Float,   1.0)

post_mat = UsdShade.Material(stage.GetPrimAtPath(POST_MAT_PATH))
post_bound = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if prim.GetTypeName() == "Cube" and (
        "/Post_" in path or "/WalkPost_" in path
    ):
        UsdShade.MaterialBindingAPI(prim).Bind(post_mat)
        post_bound += 1

print(f"[posts] DockPost (weathered grey-brown) bound to {post_bound} Cube prims")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Water: deeper, more reflective, slight normal-map tint
# ─────────────────────────────────────────────────────────────
WATER_PATH = "/World/Looks/WaterMat"
old = stage.GetPrimAtPath(WATER_PATH)
if old.IsValid():
    stage.RemovePrim(WATER_PATH)

wmat = UsdShade.Material.Define(stage, WATER_PATH)
wshader = UsdShade.Shader.Define(stage, WATER_PATH + "/Shader")
wshader.SetSourceAsset("OmniPBR.mdl", "mdl")
wshader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
wshader.CreateIdAttr("OmniPBR")
wmat.CreateSurfaceOutput("mdl").ConnectToSource(wshader.ConnectableAPI(), "out")

def wci(name, type_name, value):
    wshader.CreateInput(name, type_name).Set(value)

# Intracoastal water — murky teal-green, highly reflective surface
wci("diffuse_color_constant",        Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.004, 0.072, 0.095))
wci("diffuse_tint",                  Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.008, 0.13,  0.16))
wci("albedo_brightness",             Sdf.ValueTypeNames.Float,   0.22)
wci("albedo_desaturation",           Sdf.ValueTypeNames.Float,   0.08)
wci("reflection_roughness_constant", Sdf.ValueTypeNames.Float,   0.03)   # near-mirror
wci("metallic_constant",             Sdf.ValueTypeNames.Float,   0.12)   # wet surface sheen
wci("specular_level",                Sdf.ValueTypeNames.Float,   1.0)    # full specular
wci("opacity_constant",              Sdf.ValueTypeNames.Float,   1.0)
wci("enable_emission",               Sdf.ValueTypeNames.Bool,    False)

# Bind to WaterPlane mesh
water_plane = stage.GetPrimAtPath("/World/WaterPlane")
if water_plane.IsValid():
    # Bind to the mesh prim (may be WaterPlane itself or a child)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/WaterPlane") and prim.GetTypeName() in ("Mesh", "Xform"):
            UsdShade.MaterialBindingAPI(prim).Bind(
                UsdShade.Material(stage.GetPrimAtPath(WATER_PATH))
            )
            print(f"[water] WaterMat bound to {path}")
            break
    # Also bind to the root WaterPlane xform for good measure
    UsdShade.MaterialBindingAPI(water_plane).Bind(
        UsdShade.Material(stage.GetPrimAtPath(WATER_PATH))
    )
    print("[water] WaterMat bound to /World/WaterPlane")
else:
    print("[water] WARNING — /World/WaterPlane not found")

print("\n[done] Materials applied:")
print("  DockWood  — warm oiled walnut-teak for all planks/walkways/fingers")
print("  DockPost  — weathered grey-brown for all posts")
print("  WaterMat  — dark teal, near-mirror reflective surface")
print("\nIf it still looks flat: Window → Rendering → RTX Settings → Path Tracing")
print("Or check viewport: camera icon → Exposure=0, disable Auto Exposure")

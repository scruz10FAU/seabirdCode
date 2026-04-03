
import omni.usd
from pxr import UsdShade, UsdGeom, Gf, Sdf

stage = omni.usd.get_context().get_stage()

# ─────────────────────────────────────────────────────────────
# 1. INSPECT WALNUT SHADER — dump all inputs so we can find
#    the texture_scale / texture_translate controls
# ─────────────────────────────────────────────────────────────
WALNUT_PATH   = "/World/Looks/Wood_Tiles_Walnut"
WALNUT_SHADER = WALNUT_PATH + "/Shader"

walnut_prim = stage.GetPrimAtPath(WALNUT_PATH)
if not walnut_prim.IsValid():
    print("[wood] ERROR — Walnut material not found at", WALNUT_PATH)
    print("[wood] Open the Materials browser and drag Wood_Tiles_Walnut into the stage first.")
else:
    shader = UsdShade.Shader(stage.GetPrimAtPath(WALNUT_SHADER))
    if shader:
        print("\n=== Walnut Shader Inputs ===")
        for inp in shader.GetInputs():
            print(f"  {inp.GetFullName():50s} = {inp.Get()}")
        print("=== End Walnut Inputs ===\n")
    else:
        # Some VMATERIALS use a nested shader path — try to find it
        print("[wood] Shader not found at expected path, traversing...")
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(walnut_prim.GetPath()):
                if prim.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(prim)
                    print(f"[wood] Found shader at: {prim.GetPath()}")
                    for inp in shader.GetInputs():
                        print(f"  {inp.GetFullName():50s} = {inp.Get()}")
                    break

# ─────────────────────────────────────────────────────────────
# 2. BIND WALNUT TO ALL DOCK CUBE PRIMS
#    The issue: material bound to Xform parent doesn't propagate
#    to Cube children — must bind each Cube directly.
# ─────────────────────────────────────────────────────────────
if walnut_prim.IsValid():
    walnut_mat = UsdShade.Material(walnut_prim)
    bound_count = 0
    skipped = []

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        # Target: Cube prims under Dock or DockB (not buoys, not water, not drone)
        if prim.GetTypeName() == "Cube" and (
            "/World/Dock/" in path_str or "/World/DockB/" in path_str
        ):
            UsdShade.MaterialBindingAPI(prim).Bind(walnut_mat)
            bound_count += 1

    print(f"[wood] Walnut bound to {bound_count} Cube prims")

# ─────────────────────────────────────────────────────────────
# 3. SET WALNUT TEXTURE TILING
#    Dock planks are large (21m long, 2.5m wide) — default UV 1:1
#    makes the texture tile too large and look flat.
#    We try the most common VMATERIALS texture_scale inputs.
# ─────────────────────────────────────────────────────────────
if walnut_prim.IsValid() and shader:
    # VMATERIALS typically uses one of these for UV scale:
    scale_input_candidates = [
        "texture_scale",
        "project_uvw_scale",
        "uv_scale",
        "texture_translate",  # just to see it
        "world_or_object",
        "texture_rotate",
    ]
    print("\n[wood] Checking for texture scale inputs...")
    for name in scale_input_candidates:
        inp = shader.GetInput(name)
        if inp:
            print(f"  FOUND: {name} = {inp.Get()}")

    # Try setting texture_scale — most common in VMATERIALS wood shaders
    # A value of (4, 4) means the texture tiles 4x across each meter → good for planks
    tex_scale = shader.GetInput("texture_scale")
    if tex_scale:
        current = tex_scale.Get()
        tex_scale.Set(Gf.Vec2f(8.0, 8.0))   # tighter tiling
        print(f"[wood] texture_scale: {current} → (8.0, 8.0)")
    else:
        print("[wood] texture_scale input not found — check dump above for correct name")

# ─────────────────────────────────────────────────────────────
# 4. IMPROVE WATER MATERIAL
#    Make it more reflective, darker, add slight chop via
#    OmniPBR roughness and metallic boost
# ─────────────────────────────────────────────────────────────
WATER_SHADER_PATH = "/World/WaterPlane/OmniPBR/Shader"
water_shader_prim = stage.GetPrimAtPath(WATER_SHADER_PATH)

if not water_shader_prim.IsValid():
    # Try alternate path if material was inlined differently
    print("[water] Shader not at expected path, searching...")
    for prim in stage.Traverse():
        if "Water" in str(prim.GetPath()) or "water" in str(prim.GetPath()):
            if prim.GetTypeName() == "Shader":
                WATER_SHADER_PATH = str(prim.GetPath())
                water_shader_prim = prim
                print(f"[water] Found at: {WATER_SHADER_PATH}")
                break

if water_shader_prim.IsValid():
    ws = UsdShade.Shader(water_shader_prim)

    def set_or_create(shader, name, type_name, value):
        inp = shader.GetInput(name)
        if inp:
            inp.Set(value)
        else:
            shader.CreateInput(name, type_name).Set(value)

    # Deeper, darker intracoastal — more blue-green, less grey
    set_or_create(ws, "diffuse_color_constant",         Sdf.ValueTypeNames.Color3f,  Gf.Vec3f(0.005, 0.085, 0.11))
    set_or_create(ws, "diffuse_tint",                   Sdf.ValueTypeNames.Color3f,  Gf.Vec3f(0.01,  0.15,  0.18))
    set_or_create(ws, "albedo_brightness",              Sdf.ValueTypeNames.Float,    0.28)
    set_or_create(ws, "albedo_desaturation",            Sdf.ValueTypeNames.Float,    0.05)
    set_or_create(ws, "albedo_add",                     Sdf.ValueTypeNames.Float,    0.0)
    # Low roughness = more mirror-like reflections (water)
    set_or_create(ws, "reflection_roughness_constant",  Sdf.ValueTypeNames.Float,    0.04)
    # Metallic gives it that wet sheen without making it look like metal
    set_or_create(ws, "metallic_constant",              Sdf.ValueTypeNames.Float,    0.08)
    # Opacity — full for water surface
    set_or_create(ws, "opacity_constant",               Sdf.ValueTypeNames.Float,    1.0)

    print("[water] Material updated — darker, more reflective intracoastal look")
    print("[water] NOTE: For animated water, we'll add a scrolling normal map next.")
else:
    print("[water] ERROR — could not find water shader prim")

print("\n[done] Run this, check the shader input dump for 'texture_scale' name,")
print("       then adjust TILING value if needed.")
print("       If Walnut is still flat, paste the shader input list here.")








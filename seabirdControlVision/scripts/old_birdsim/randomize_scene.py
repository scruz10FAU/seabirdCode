"""
seabird/randomize_scene.py
==========================
Isaac Script Editor script — randomizes buoy count, positions, lighting,
and water appearance for training data collection.

Run in Script Editor:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/randomize_scene.py").read())

What it does:
  1. Removes any previously spawned extra buoys (/World/BuoyField/*)
  2. Moves the 3 original buoys to random water positions
  3. Spawns N additional buoys (cloned from originals) at random positions
  4. Randomizes DistantLight angle + DomeLight intensity/color
  5. Jitters water material tint and roughness
  6. Prints summary of new layout

Re-run between sweep flights to get diverse training data.
"""

import random
import omni.usd
from pxr import UsdGeom, UsdShade, UsdLux, Gf, Sdf

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Water zone where buoys can spawn (Isaac coords: X=east, Y=north, Z=up)
# Tighter zone = more buoys in camera FOV during sweeps
SPAWN_X_RANGE = (-15.0, 15.0)
SPAWN_Y_RANGE = (-25.0, -6.0)     # south of dock
SPAWN_Z       = 0.25               # slightly above water surface so buoys float visibly

# How many EXTRA buoys to spawn (on top of the 3 originals)
EXTRA_BUOYS_MIN = 15
EXTRA_BUOYS_MAX = 30

# Minimum distance between any two buoys (meters)
MIN_BUOY_SPACING = 2.5

# Buoy models were imported in centimeters with xformOp:scale:unitsResolve = 0.01.
# We bake that into the user scale so we don't depend on unitsResolve staying in opOrder.
# Effective IRL size = BAKED_SCALE * raw_mesh_cm ≈ 0.00525 * ~87cm ≈ 0.46m (18in)
CM_TO_M = 0.01
USER_SCALE_MIN = 0.45
USER_SCALE_MAX = 0.61
# Baked = user_scale * cm_to_m
# e.g. 0.525 * 0.01 = 0.00525

# Original buoy prims
ORIGINALS = {
    "red":   "/World/bouy_red",
    "green": "/World/bouy_green",
    "blue":  "/World/bouy_blue",
}

# Container for spawned extras
FIELD_ROOT = "/World/BuoyField"

# ── Lighting randomization ───────────────────────────────────────────────────
SUN_ELEVATION_RANGE = (-55.0, -25.0)
SUN_AZIMUTH_RANGE   = (170.0, 250.0)
SUN_INTENSITY_RANGE = (5000, 10000)
DOME_INTENSITY_RANGE = (800, 1600)
DOME_TEMP_RANGE     = (5500, 9000)

# ── Water randomization ──────────────────────────────────────────────────────
WATER_TINT_BASE     = [0.008, 0.10, 0.13]
WATER_TINT_JITTER   = 0.04
WATER_ROUGH_RANGE   = (0.02, 0.12)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

stage = omni.usd.get_context().get_stage()


def random_pos(existing_positions: list) -> tuple:
    """Pick a random (x, y) respecting MIN_BUOY_SPACING from all others."""
    for _attempt in range(200):
        x = random.uniform(*SPAWN_X_RANGE)
        y = random.uniform(*SPAWN_Y_RANGE)
        too_close = False
        for ex, ey in existing_positions:
            if ((x - ex) ** 2 + (y - ey) ** 2) < MIN_BUOY_SPACING ** 2:
                too_close = True
                break
        if not too_close:
            return (x, y)
    return (random.uniform(*SPAWN_X_RANGE), random.uniform(*SPAWN_Y_RANGE))


def nuke_all_xform_ops(prim):
    """Remove every xformOp property from a prim so we start clean."""
    to_remove = []
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") or name == "xformOpOrder":
            to_remove.append(name)
    for name in to_remove:
        prim.RemoveProperty(name)


def set_buoy_transform(prim_path: str, tx: float, ty: float, tz: float, user_scale: float):
    """Set translate + baked scale (user_scale * 0.01) on a buoy prim.
    Clears ALL existing xform ops first to avoid conflicts from CopySpec."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [WARN] prim not found: {prim_path}")
        return

    # Nuclear cleanup — remove every xformOp property
    nuke_all_xform_ops(prim)

    # Fresh xformable
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    # Bake cm→m into scale so we don't need unitsResolve in the op order
    baked = user_scale * CM_TO_M
    xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    xf.AddScaleOp().Set(Gf.Vec3d(baked, baked, baked))


def _find_material_for_color(color: str):
    """Walk the original buoy prim tree — return the first material found on any mesh."""
    src_prim = stage.GetPrimAtPath(ORIGINALS[color])
    if not src_prim.IsValid():
        return None
    for desc in _iter_all(src_prim):
        if desc.GetTypeName() != "Mesh":
            continue
        binding = UsdShade.MaterialBindingAPI(desc)
        mat = binding.GetDirectBinding().GetMaterial()
        if mat:
            return mat
        # Also check computed binding (inherited from parent)
        mat_bound, _ = binding.ComputeBoundMaterial()
        if mat_bound:
            return mat_bound
    return None


def _bind_all_meshes(root_path: str, material):
    """Bind material to every Mesh prim under root_path."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not material:
        return 0
    count = 0
    for desc in _iter_all(root):
        if desc.GetTypeName() == "Mesh":
            UsdShade.MaterialBindingAPI(desc).Bind(material)
            count += 1
    return count


def clone_buoy(src_path: str, dst_path: str, material) -> bool:
    """Deep-copy a buoy prim subtree then force-bind the correct material."""
    layer = stage.GetRootLayer()
    if stage.GetPrimAtPath(dst_path).IsValid():
        stage.RemovePrim(dst_path)
    success = Sdf.CopySpec(layer, src_path, layer, dst_path)
    if not success:
        print(f"  [ERROR] CopySpec failed: {src_path} → {dst_path}")
        return False
    _bind_all_meshes(dst_path, material)
    return True


def _iter_all(prim):
    """Recursively yield all descendant prims."""
    for child in prim.GetChildren():
        yield child
        yield from _iter_all(child)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("[randomize] Starting scene randomization...")
print("=" * 60)

# ── 1. Clean up previous extra buoys ─────────────────────────────────────────
if stage.GetPrimAtPath(FIELD_ROOT).IsValid():
    stage.RemovePrim(FIELD_ROOT)
    print(f"[randomize] Cleared previous {FIELD_ROOT}")
UsdGeom.Xform.Define(stage, FIELD_ROOT)

# ── 1b. Discover materials on original buoys (before we move them) ────────────
buoy_materials = {}
for color in ORIGINALS:
    mat = _find_material_for_color(color)
    if mat:
        buoy_materials[color] = mat
        print(f"  {color:6s} material: {mat.GetPath()}")
    else:
        print(f"  {color:6s} WARNING — no material found on original!")
print(f"[randomize] Found materials for {len(buoy_materials)}/{len(ORIGINALS)} colors")

# ── 2. Randomize original buoy positions ─────────────────────────────────────
placed_positions = []
print("[randomize] Moving original buoys:")
for color, path in ORIGINALS.items():
    x, y = random_pos(placed_positions)
    s = random.uniform(USER_SCALE_MIN, USER_SCALE_MAX)
    set_buoy_transform(path, x, y, SPAWN_Z, s)
    placed_positions.append((x, y))
    print(f"  {color:6s} → ({x:+6.1f}, {y:+6.1f})  scale={s:.3f}")

# ── 3. Spawn extra buoys (cloned from originals) ────────────────────────────
n_extra = random.randint(EXTRA_BUOYS_MIN, EXTRA_BUOYS_MAX)

# Weighted toward red/green (nav markers)
weighted_colors = ["red"] * 3 + ["green"] * 3 + ["blue"] * 2

# Some buoys spawn in clusters of 2-4 (like real channel markers / mooring fields)
# This forces the detector to handle nearby/overlapping buoys
CLUSTER_CHANCE = 0.35      # probability a buoy starts a cluster
CLUSTER_SIZE   = (2, 4)    # extra buoys in the cluster
CLUSTER_RADIUS = 4.0       # meters — how tight the cluster is

print(f"[randomize] Spawning {n_extra} extra buoys (with clusters):")
i = 0
spawned = 0
while spawned < n_extra:
    color = random.choice(weighted_colors)
    src_path = ORIGINALS[color]
    dst_path = f"{FIELD_ROOT}/buoy_{i:02d}_{color}"

    if not clone_buoy(src_path, dst_path):
        i += 1
        continue

    x, y = random_pos(placed_positions)
    s = random.uniform(USER_SCALE_MIN, USER_SCALE_MAX)
    set_buoy_transform(dst_path, x, y, SPAWN_Z, s)
    placed_positions.append((x, y))
    baked = s * CM_TO_M
    print(f"  buoy_{i:02d}_{color:6s} → ({x:+6.1f}, {y:+6.1f})  scale={s:.3f}")
    i += 1
    spawned += 1

    # Possibly spawn a cluster around this position
    if spawned < n_extra and random.random() < CLUSTER_CHANCE:
        cluster_n = min(random.randint(*CLUSTER_SIZE), n_extra - spawned)
        for c_j in range(cluster_n):
            c_color = random.choice(weighted_colors)
            c_src = ORIGINALS[c_color]
            c_dst = f"{FIELD_ROOT}/buoy_{i:02d}_{c_color}"
            if not clone_buoy(c_src, c_dst):
                i += 1
                continue
            # Offset from cluster center
            cx = x + random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
            cy = y + random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
            # Clamp to spawn zone
            cx = max(SPAWN_X_RANGE[0], min(SPAWN_X_RANGE[1], cx))
            cy = max(SPAWN_Y_RANGE[0], min(SPAWN_Y_RANGE[1], cy))
            c_s = random.uniform(USER_SCALE_MIN, USER_SCALE_MAX)
            set_buoy_transform(c_dst, cx, cy, SPAWN_Z, c_s)
            placed_positions.append((cx, cy))
            print(f"  buoy_{i:02d}_{c_color:6s} → ({cx:+6.1f}, {cy:+6.1f})  scale={c_s:.3f}  [cluster]")
            i += 1
            spawned += 1

# ── 4. Randomize lighting ────────────────────────────────────────────────────
print("[randomize] Lighting:")

# Distant light (sun)
dl_path = "/World/DistantLight"
dl_prim = stage.GetPrimAtPath(dl_path)
if dl_prim.IsValid():
    sun_el = random.uniform(*SUN_ELEVATION_RANGE)
    sun_az = random.uniform(*SUN_AZIMUTH_RANGE)
    sun_int = random.uniform(*SUN_INTENSITY_RANGE)

    nuke_all_xform_ops(dl_prim)
    xf = UsdGeom.Xformable(dl_prim)
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(sun_el, 0.0, sun_az))

    dl = UsdLux.DistantLight(dl_prim)
    dl.GetIntensityAttr().Set(sun_int)
    r = random.uniform(0.92, 1.0)
    g = random.uniform(0.85, 0.96)
    b = random.uniform(0.70, 0.88)
    dl.GetColorAttr().Set(Gf.Vec3f(r, g, b))
    dl.GetAngleAttr().Set(random.uniform(0.5, 1.5))
    print(f"  Sun: elev={sun_el:.0f}° az={sun_az:.0f}° int={sun_int:.0f}")

# Dome light (sky)
dome_path = "/World/DomeLight"
dome_prim = stage.GetPrimAtPath(dome_path)
if dome_prim.IsValid():
    dome = UsdLux.DomeLight(dome_prim)
    dome_int = random.uniform(*DOME_INTENSITY_RANGE)
    dome.GetIntensityAttr().Set(dome_int)
    dome_temp = random.uniform(*DOME_TEMP_RANGE)
    dome.GetColorTemperatureAttr().Set(dome_temp)
    print(f"  Dome: int={dome_int:.0f} temp={dome_temp:.0f}K")

# ── 5. Randomize water material ──────────────────────────────────────────────
print("[randomize] Water:")
water_shader_path = "/World/Looks/WaterMat/Shader"
water_shader_prim = stage.GetPrimAtPath(water_shader_path)
if water_shader_prim.IsValid():
    shader = UsdShade.Shader(water_shader_prim)
    tint = Gf.Vec3f(
        max(0, WATER_TINT_BASE[0] + random.uniform(-WATER_TINT_JITTER, WATER_TINT_JITTER)),
        max(0, WATER_TINT_BASE[1] + random.uniform(-WATER_TINT_JITTER, WATER_TINT_JITTER)),
        max(0, WATER_TINT_BASE[2] + random.uniform(-WATER_TINT_JITTER, WATER_TINT_JITTER)),
    )
    rough = random.uniform(*WATER_ROUGH_RANGE)
    shader.GetInput("diffuse_tint").Set(tint)
    shader.GetInput("reflection_roughness_constant").Set(rough)
    print(f"  Tint=({tint[0]:.3f}, {tint[1]:.3f}, {tint[2]:.3f})  rough={rough:.3f}")
else:
    print("  [WARN] WaterMat shader not found")

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(ORIGINALS) + spawned
print(f"\n[randomize] Done — {total} buoys in scene ({len(ORIGINALS)} original + {spawned} cloned)")

# ── Diagnostic: check material bindings ───────────────────────────────────────
bound_count = 0
unbound_prims = []
for prim in stage.Traverse():
    path_str = str(prim.GetPath())
    if ("bouy_" in path_str or "BuoyField" in path_str) and prim.GetTypeName() == "Mesh":
        binding = UsdShade.MaterialBindingAPI(prim)
        mat = binding.GetDirectBinding().GetMaterial()
        if mat:
            bound_count += 1
        else:
            unbound_prims.append(path_str)
print(f"[randomize] Material check: {bound_count} mesh prims with materials bound")
if unbound_prims:
    print(f"[randomize] WARNING — {len(unbound_prims)} mesh prims have NO material (will render grey):")
    for p in unbound_prims[:5]:
        print(f"  {p}")
    if len(unbound_prims) > 5:
        print(f"  ... and {len(unbound_prims) - 5} more")
print("[randomize] Run sweep_and_detect.py + record_training_data.py to collect data")
print("=" * 60 + "\n")
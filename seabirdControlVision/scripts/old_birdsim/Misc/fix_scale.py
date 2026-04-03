"""
fix_scale.py
Measures actual buoy mesh bounds, corrects scale to IRL 18"x18" base,
fixes post heights to extend above dock, and repositions buoys to channel entrance.

Run via Script Editor:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_scale.py").read())
"""

import omni.usd
from pxr import UsdGeom, Gf, Sdf

stage = omni.usd.get_context().get_stage()

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DOCK_Z      = 0.3     # dock surface Z
PLANK_H     = 0.2     # dock plank thickness
WALK_W      = 2.5     # walkway width
WALK_LEN    = 21.0    # each dock section length
BAY_WIDTH   = 7.0     # center-to-center bay spacing

# Post IRL target:
#   Real pilings: 8-10in diameter (0.25m) ✓ current is 0.3 close enough
#   Extend 0.75m (2.5ft) above dock surface
#   Underwater anchor: keep bottom at same depth as before (~2.7m below surface)
POST_ABOVE_DOCK = 0.75          # how far above dock surface posts extend (m)
POST_BOTTOM_Z   = -2.5          # absolute Z of post bottom (underwater anchor)
POST_DIAM       = 0.25          # post diameter (m) — slight correction from 0.3

# IRL buoy base: 18in x 18in = 0.4572m
BUOY_BASE_IRL = 0.4572          # meters

# ─── STEP 1: MEASURE BUOY MESH BOUNDS ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Buoy mesh bounds (raw, before any scale)")
print("="*60)

buoy_names   = ["bouy_blue", "bouy_green", "bouy_red"]
buoy_info    = {}

for name in buoy_names:
    mesh_path = f"/World/{name}/geometry_0/geometry_0"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        print(f"  [{name}] MISSING at {mesh_path}")
        continue

    mesh      = UsdGeom.Mesh(mesh_prim)
    pts_attr  = mesh.GetPointsAttr()
    pts       = pts_attr.Get()

    if pts is None or len(pts) == 0:
        print(f"  [{name}] No points found in mesh")
        continue

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    raw_w = max(xs) - min(xs)   # X extent in source units (cm)
    raw_d = max(ys) - min(ys)   # Y extent
    raw_h = max(zs) - min(zs)   # Z extent (height)

    # Effective size in meters = raw_size * unitsResolve(0.01) * user_scale(2)
    eff_w = raw_w * 0.01 * 2
    eff_d = raw_d * 0.01 * 2
    eff_h = raw_h * 0.01 * 2

    buoy_info[name] = {
        "raw_w": raw_w, "raw_d": raw_d, "raw_h": raw_h,
        "eff_w": eff_w, "eff_d": eff_d, "eff_h": eff_h,
    }

    print(f"\n  [{name}]")
    print(f"    Mesh raw size (source cm): {raw_w:.1f} x {raw_d:.1f} x {raw_h:.1f} cm")
    print(f"    Current effective size:    {eff_w:.3f} x {eff_d:.3f} x {eff_h:.3f} m")
    print(f"    Current base (X):          {eff_w:.3f}m ({eff_w/0.0254:.1f} in)")

# ─── STEP 2: CALCULATE CORRECT BUOY SCALE ────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Scale correction to hit 18in (0.4572m) base")
print("="*60)

# The xform stack is:  scale:unitsResolve(0.01) then user scale(S)
# effective_size = raw_cm * 0.01 * S
# We want effective_size = 0.4572
# → S = 0.4572 / (raw_cm * 0.01) = 45.72 / raw_cm

buoy_scales = {}
for name, info in buoy_info.items():
    raw_base = info["raw_w"]   # use X as the base dimension
    S = BUOY_BASE_IRL / (raw_base * 0.01)
    # Height scales proportionally
    new_h = info["raw_h"] * 0.01 * S
    buoy_scales[name] = S
    print(f"\n  [{name}]")
    print(f"    Raw base: {raw_base:.1f} cm")
    print(f"    New user scale: {S:.4f}  (was 2.0)")
    print(f"    New effective base: {raw_base * 0.01 * S:.4f}m ({raw_base * 0.01 * S / 0.0254:.1f} in) ✓")
    print(f"    New height:         {new_h:.3f}m ({new_h/0.0254:.1f} in)")

# ─── STEP 3: FIX POST HEIGHTS ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Fix post heights (extend above dock surface)")
print("="*60)

# Target:
#   Top    = DOCK_Z + POST_ABOVE_DOCK = 0.3 + 0.75 = 1.05m
#   Bottom = POST_BOTTOM_Z = -2.5m
#   Height = 1.05 - (-2.5) = 3.55m
#   Center = (1.05 + (-2.5)) / 2 = -0.725m

post_top    = DOCK_Z + POST_ABOVE_DOCK
post_height = post_top - POST_BOTTOM_Z
post_center = (post_top + POST_BOTTOM_Z) / 2.0

print(f"  Post top:    {post_top:.3f}m  ({post_top/0.0254:.1f} in above water)")
print(f"  Post bottom: {POST_BOTTOM_Z:.3f}m  (underwater anchor)")
print(f"  Post height: {post_height:.3f}m  ({post_height/0.0254:.1f} in total)")
print(f"  Post center: {post_center:.3f}m")
print(f"  Post diam:   {POST_DIAM:.3f}m  ({POST_DIAM/0.0254:.1f} in)")
print(f"\n  Was: translate_z=-1.2, scale_z=3.0  (top flush with dock — no protrusion)")
print(f"  Now: translate_z={post_center:.3f}, scale_z={post_height:.3f}  (top {POST_ABOVE_DOCK}m above dock)")

print("\n  Applying to all posts...")

post_count = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    name = prim.GetName()

    # Target: Post_L/R and WalkPost_L/R prims (the Xform wrappers, not the Cubes)
    is_post = (
        prim.GetTypeName() == "Xform" and
        ("Post_" in name or "WalkPost_" in name) and
        ("/World/Dock" in path or "/World/DockB" in path)
    )
    if not is_post:
        continue

    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}

    if "xformOp:translate" not in ops:
        continue

    # Keep existing X and Y, only change Z
    current_t = ops["xformOp:translate"].Get()
    new_t = Gf.Vec3d(current_t[0], current_t[1], post_center)
    ops["xformOp:translate"].Set(new_t)

    # Fix scale
    if "xformOp:scale" in ops:
        current_s = ops["xformOp:scale"].Get()
        new_s = Gf.Vec3d(POST_DIAM, POST_DIAM, post_height)
        ops["xformOp:scale"].Set(new_s)

    post_count += 1

print(f"  Updated {post_count} posts ✓")

# ─── STEP 4: APPLY BUOY SCALE ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Apply corrected buoy scale")
print("="*60)

for name, S in buoy_scales.items():
    prim = stage.GetPrimAtPath(f"/World/{name}")
    if not prim.IsValid():
        print(f"  [{name}] prim not found — skipping")
        continue

    xf   = UsdGeom.Xformable(prim)
    ops  = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}

    if "xformOp:scale" in ops:
        ops["xformOp:scale"].Set(Gf.Vec3d(S, S, S))
        print(f"  [{name}] scale set to {S:.4f} ✓")
    else:
        print(f"  [{name}] WARNING — no scale op found, skipping")

# ─── STEP 5: REPOSITION BUOYS TO CHANNEL ENTRANCE ────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Reposition buoys to channel entrance")
print("="*60)

# Dock layout:
#   Walkway X center: 0, extends from Y=-10.5 to Y=31.5
#   Dock total X span: -8.25 to +8.25 (finger tips)
#   Channel entrance: Y side (the open end is at Y ~ -10.5)
#
# IALA buoyage convention (US):
#   Red "right returning" → starboard (right) side of channel when entering
#   Green → port (left) side of channel when entering
#   Blue → special mark / informational
#
# We place them just outside the dock entrance in a triangle formation:
#   Assuming boats approach from -Y direction (south), entering toward +Y
#   Red   → starboard = +X side of channel entrance
#   Green → port      = -X side of channel entrance
#   Blue  → behind (further out) center informational marker
#
# Channel entrance is at Y ~ -12m (a couple meters past dock end at -10.5)
# Flanking the walkway at X ≈ ±6m (well clear of finger tips at ±8.25)

ENTRANCE_Y  = -14.0    # just past the dock entrance
FLANK_X     = 6.0      # distance from center to each side marker
BACK_Y      = -18.0    # blue informational marker further out

buoy_positions = {
    "bouy_red":   (  FLANK_X,  ENTRANCE_Y, 0.0),   # starboard (right entering)
    "bouy_green": ( -FLANK_X,  ENTRANCE_Y, 0.0),   # port (left entering)
    "bouy_blue":  (  0.0,      BACK_Y,     0.0),   # center informational
}

for name, (tx, ty, tz) in buoy_positions.items():
    prim = stage.GetPrimAtPath(f"/World/{name}")
    if not prim.IsValid():
        print(f"  [{name}] prim not found — skipping")
        continue

    xf  = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}

    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(tx, ty, tz))
        print(f"  [{name}] moved to ({tx}, {ty}, {tz}) ✓")
    else:
        print(f"  [{name}] WARNING — no translate op found")

# ─── DONE ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE — Summary of changes")
print("="*60)
print(f"  Posts:  now extend {POST_ABOVE_DOCK}m ({POST_ABOVE_DOCK/0.0254:.1f}in) above dock surface")
print(f"  Buoys:  scaled to {BUOY_BASE_IRL}m ({BUOY_BASE_IRL/0.0254:.0f}in) base")
print(f"  Buoys:  repositioned to channel entrance (Y={ENTRANCE_Y})")
print(f"  Save:   File → Save (Ctrl+S) to persist to marina_dock.usd")
print("="*60 + "\n")

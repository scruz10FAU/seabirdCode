"""
fix_clouds.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_clouds.py").read())

The RtxDynamicSky prim exists and has correct attrs, but clouds aren't
showing because the DomeLight isn't wired to it. This script connects them.
"""

import omni.usd
from pxr import UsdGeom, UsdLux, Gf, Sdf

stage = omni.usd.get_context().get_stage()

DOME_PATH = "/World/DomeLight"
SKY_PATH  = "/World/Sky"

dome_prim = stage.GetPrimAtPath(DOME_PATH)
sky_prim  = stage.GetPrimAtPath(SKY_PATH)

if not sky_prim.IsValid():
    print("[sky] ERROR — /World/Sky not found, run fix_sky_posts.py first")
else:
    # ─────────────────────────────────────────────────────────
    # The key: DomeLight must point at the RtxDynamicSky prim
    # via its texture file attr using the special omniverse:// 
    # dynamic sky token, OR via a direct prim relationship.
    # Isaac Sim 4.x uses a "inputs:sky" relationship on the dome.
    # ─────────────────────────────────────────────────────────

    # Method A — relationship (Isaac 4.x preferred)
    try:
        rel = dome_prim.CreateRelationship("inputs:sky", custom=False)
        rel.SetTargets([Sdf.Path(SKY_PATH)])
        print("[sky] Set inputs:sky relationship on DomeLight → /World/Sky")
    except Exception as e:
        print(f"[sky] relationship method failed: {e}")

    # Method B — texture file token (older Omniverse path)
    try:
        dome = UsdLux.DomeLight(dome_prim)
        dome.CreateTextureFileAttr().Set("dynamic://RtxDynamicSky")
        print("[sky] Set texture file to dynamic://RtxDynamicSky")
    except Exception as e:
        print(f"[sky] texture token method failed: {e}")

    # Method C — set the sky prim as a light portal target
    try:
        dome_prim.CreateAttribute(
            "inputs:texture:format", Sdf.ValueTypeNames.Token
        ).Set("automatic")
        print("[sky] Set texture format to automatic")
    except Exception as e:
        print(f"[sky] texture format: {e}")

    # ─────────────────────────────────────────────────────────
    # Tune sky for good cloud appearance
    # ─────────────────────────────────────────────────────────
    def sky_set(name, value):
        attr = sky_prim.GetAttribute(name)
        if attr:
            attr.Set(value)
            print(f"[sky] {name} = {value}")
        else:
            print(f"[sky] attr not found: {name}")

    sky_set("inputs:cloud_coverage",   0.55)   # 55% cover — scattered cumulus
    sky_set("inputs:cloud_scale",      1.4)    # tighter, more defined clouds
    sky_set("inputs:sky_brightness",   1.1)    # slightly punchier blue
    sky_set("inputs:saturation",       1.15)   # richer colour
    sky_set("inputs:azimuth",          200.0)  # sun slightly more south
    sky_set("inputs:elevation",        38.0)   # slightly lower = longer shadows
    sky_set("inputs:sun_disk_intensity", 1.2)  # visible sun disk
    sky_set("inputs:horizon_height",   0.0)    # horizon at water level
    sky_set("inputs:horizon_fuzziness", 0.08)  # crisp horizon

    # Ground color — dark water/dock, not sand
    attr = sky_prim.GetAttribute("inputs:ground_color")
    if attr:
        attr.Set(Gf.Vec3f(0.04, 0.04, 0.035))
    print("[sky] Ground color darkened")

    # ─────────────────────────────────────────────────────────
    # DomeLight intensity — reduce slightly so clouds don't wash out
    # ─────────────────────────────────────────────────────────
    dome = UsdLux.DomeLight(dome_prim)
    dome.GetIntensityAttr().Set(850.0)
    print("[dome] Intensity → 850")

    # ─────────────────────────────────────────────────────────
    # Dump DomeLight attrs so we can see if relationship took
    # ─────────────────────────────────────────────────────────
    print("\n=== DomeLight attrs after edit ===")
    for attr in dome_prim.GetAttributes():
        v = attr.Get()
        if v is not None:
            print(f"  {attr.GetName():45s} = {v}")
    for rel in dome_prim.GetRelationships():
        print(f"  [rel] {rel.GetName():41s} → {rel.GetTargets()}")
    print("=== End DomeLight attrs ===\n")

print("\n[done] If clouds still not visible:")
print("  → In viewport top-left, switch from 'RTX - Real-Time' to")
print("    'RTX - Interactive (Path Traced)' — clouds render in path tracing")
print("  → Or: Stage panel → select /World/Sky → Property panel →")
print("    check inputs:cloud_coverage is set to 0.55")

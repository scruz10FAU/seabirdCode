"""
activate_clouds.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/activate_clouds.py").read())

The "Failed to upload DomeLight texture dynamic://RtxDynamicSky" error means
the texture token approach is wrong for Isaac 4.5.

Correct approach for Isaac 4.5:
  - DomeLight has NO texture set
  - RtxDynamicSky prim just exists in the stage
  - Renderer must be in RTX Interactive (Path Traced) mode
  - RTX renderer picks up the Sky prim automatically via scene scan
"""

import omni.usd
import omni.kit.app
import carb.settings
from pxr import UsdLux, UsdGeom, Gf, Sdf

stage  = omni.usd.get_context().get_stage()
settings = carb.settings.get_settings()

# ─────────────────────────────────────────────────────────────
# 1. CLEAN UP DOMELIGHT — remove bad texture, clear color override
#    A pure, textureless DomeLight lets RtxDynamicSky take over
# ─────────────────────────────────────────────────────────────
dome_prim = stage.GetPrimAtPath("/World/DomeLight")
if dome_prim.IsValid():
    dome = UsdLux.DomeLight(dome_prim)

    # Remove the bad texture path that caused the upload error
    tex_attr = dome_prim.GetAttribute("inputs:texture:file")
    if tex_attr:
        tex_attr.Clear()
        dome_prim.RemoveProperty("inputs:texture:file")
        print("[dome] Cleared bad texture:file attribute")

    # Remove the color override — let sky prim control appearance
    col_attr = dome_prim.GetAttribute("inputs:color")
    if col_attr:
        col_attr.Clear()
        print("[dome] Cleared color override — sky prim now controls appearance")

    # Set intensity for sky-driven lighting
    dome.GetIntensityAttr().Set(1.0)   # very low — sky prim drives the light
    dome.GetExposureAttr().Set(0.0)

    # rtx:shaderId relationship — correct way to connect in 4.5
    dome_prim.RemoveProperty("rtx:shaderId")  # clear the attr version
    try:
        rel = dome_prim.GetRelationship("inputs:sky")
        if not rel:
            rel = dome_prim.CreateRelationship("inputs:sky", custom=True)
        rel.SetTargets([Sdf.Path("/World/Sky")])
        print("[dome] inputs:sky relationship → /World/Sky")
    except Exception as e:
        print(f"[dome] relationship: {e}")

    print("[dome] DomeLight cleaned — intensity=1, no texture, sky prim linked")
else:
    print("[dome] ERROR — DomeLight not found")

# ─────────────────────────────────────────────────────────────
# 2. VERIFY SKY PRIM SETTINGS
# ─────────────────────────────────────────────────────────────
sky_prim = stage.GetPrimAtPath("/World/Sky")
if sky_prim.IsValid():
    print(f"\n[sky] Prim active: {sky_prim.IsActive()}, type: {sky_prim.GetTypeName()}")
    sky_prim.SetActive(True)

    def sky_set(name, value):
        attr = sky_prim.GetAttribute(name)
        if attr: attr.Set(value)

    sky_set("inputs:cloud_coverage",    0.60)   # good scattered cumulus
    sky_set("inputs:cloud_scale",       1.2)    # realistic scale
    sky_set("inputs:sky_brightness",    1.0)
    sky_set("inputs:saturation",        1.1)
    sky_set("inputs:azimuth",           200.0)
    sky_set("inputs:elevation",         38.0)
    sky_set("inputs:sun_disk_intensity", 1.5)
    sky_set("inputs:horizon_height",    0.0)
    sky_set("inputs:horizon_fuzziness", 0.06)
    print("[sky] Cloud coverage set to 60%")

# ─────────────────────────────────────────────────────────────
# 3. SWITCH RENDERER TO PATH TRACED via carb settings
#    RtxDynamicSky clouds ONLY render in path tracing mode.
#    RTX Real-Time shows sky color but skips volumetric clouds.
# ─────────────────────────────────────────────────────────────
try:
    # Get current renderer
    current = settings.get("/rtx/rendermode")
    print(f"\n[renderer] Current mode: {current}")

    # Switch to path traced
    settings.set("/rtx/rendermode", "PathTracing")
    print("[renderer] Switched to PathTracing ✓")
    print("[renderer] Note: first few frames may be noisy — it converges quickly")
except Exception as e:
    print(f"[renderer] Could not switch via carb.settings: {e}")

    # Fallback: try omni.kit.viewport approach
    try:
        import omni.kit.viewport.utility as vp_util
        vp = vp_util.get_active_viewport()
        if vp:
            vp.set_hd_engine("rtx")
            print("[renderer] Set via viewport utility")
    except Exception as e2:
        print(f"[renderer] viewport utility also failed: {e2}")
        print("[renderer] MANUAL: click 'RTX - Real-Time' dropdown → RTX - Interactive (Path Traced)")

# ─────────────────────────────────────────────────────────────
# 4. TUNE DISTANT LIGHT to complement sky
# ─────────────────────────────────────────────────────────────
sun_prim = stage.GetPrimAtPath("/World/DistantLight")
if sun_prim.IsValid():
    distant = UsdLux.DistantLight(sun_prim)
    distant.GetIntensityAttr().Set(4500.0)  # lower — sky prim adds ambient
    print("[sun] DistantLight intensity reduced to 4500 (sky prim adds ambient)")

print("""
[done] Three things to check in the viewport:
  1. Renderer shows 'RTX - Interactive (Path Traced)' in top-left dropdown
     (if still 'RTX - Real-Time' → switch it manually, clouds won't show otherwise)
  2. Stage panel → /World/Sky → Property panel → inputs:cloud_coverage = 0.6
  3. Stage panel → /World/DomeLight → no texture:file attribute
     
If clouds visible but scene too dark:
  → Increase DistantLight intensity back to 6000
  → Or bump DomeLight intensity to 500
""")

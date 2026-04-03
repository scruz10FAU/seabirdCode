"""
restore_realtime.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/restore_realtime.py").read())

Path tracing = 3 FPS = unusable for RL training.
Revert to RTX Real-Time, clean blue sky, restore 80+ FPS.
Clouds via RtxDynamicSky simply don't work in Real-Time mode — accept that.
"""

import omni.usd
import carb.settings
from pxr import UsdLux, UsdGeom, Gf, Sdf

stage    = omni.usd.get_context().get_stage()
settings = carb.settings.get_settings()

# ─────────────────────────────────────────────────────────────
# 1. SWITCH BACK TO RTX REAL-TIME
# ─────────────────────────────────────────────────────────────
settings.set("/rtx/rendermode", "RaytracedLighting")
settings.set("/rtx/pathtracing/spp", 1)
settings.set("/rtx/pathtracing/totalSpp", 1)
print("[renderer] Reverted to RTX Real-Time (RaytracedLighting) ✓")

# ─────────────────────────────────────────────────────────────
# 2. REMOVE RtxDynamicSky — useless in Real-Time mode
# ─────────────────────────────────────────────────────────────
for path in ["/World/DomeLight/Sky", "/World/Sky", "/World/DomeLight"]:
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        stage.RemovePrim(path)
        print(f"[sky] Removed {path}")

# ─────────────────────────────────────────────────────────────
# 3. SIMPLE DOMELIGHT — sky blue, works great in Real-Time
#    This is all we need. Reflections on water look good.
#    No clouds but 80+ FPS for training.
# ─────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_prim = dome.GetPrim()
dome.GetIntensityAttr().Set(1200.0)
dome.GetExposureAttr().Set(0.0)

# Florida afternoon sky — slightly hazy blue
dome_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(0.55, 0.72, 0.95)
)
dome_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(7500.0)
dome_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)

# Orient sky
xf = UsdGeom.Xformable(dome_prim)
xf.ClearXformOpOrder()
xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 270.0))
print("[dome] Sky-blue DomeLight → 1200 intensity ✓")

# ─────────────────────────────────────────────────────────────
# 4. RESTORE DISTANT LIGHT
# ─────────────────────────────────────────────────────────────
sun_prim = stage.GetPrimAtPath("/World/DistantLight")
if sun_prim.IsValid():
    distant = UsdLux.DistantLight(sun_prim)
    distant.GetIntensityAttr().Set(8000.0)
    distant.GetAngleAttr().Set(0.8)
    distant.GetPrim().CreateAttribute(
        "inputs:color", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(1.0, 0.94, 0.82))
    sun_xf = UsdGeom.Xformable(sun_prim)
    sun_xf.ClearXformOpOrder()
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 0.0, 210.0))
    print("[sun] DistantLight → 8000, warm afternoon ✓")

# ─────────────────────────────────────────────────────────────
# 5. RESET TONEMAP to defaults
# ─────────────────────────────────────────────────────────────
try:
    settings.set("/rtx/post/tonemap/op", 6)
    settings.set("/rtx/post/tonemap/filmIso", 100.0)
    settings.set("/rtx/post/tonemap/fStop", 5.6)
    settings.set("/rtx/post/tonemap/exposureValue", 0.0)
    print("[tonemap] Reset to defaults")
except:
    pass

print("""
[done] RTX Real-Time restored. FPS should return to 80+.
  Sky = solid blue — no clouds, but great performance.
  Water animation + reflections still work in Real-Time.

  What we have now:
    ✓ Blue sky (DomeLight 1200, color temp 7500K)  
    ✓ Warm afternoon sun (DistantLight 8000)
    ✓ Animated water ripples
    ✓ Dark walnut dock planks
    ✓ Grey-brown posts
    ✓ 80+ FPS for RL training
    
  Clouds: not worth the FPS cost for RL training.
  Can revisit for final renders/videos later with path tracing.
""")

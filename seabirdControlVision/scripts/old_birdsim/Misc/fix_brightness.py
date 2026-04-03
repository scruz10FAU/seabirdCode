"""
fix_brightness.py
Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/fix_brightness.py").read())

Sky structure is correct. Two problems:
  1. DomeLight intensity=1 → scene black
  2. PathTracing renderer keeps reverting → switch it + boost intensity for both modes
"""

import omni.usd
import carb.settings
from pxr import UsdLux, Gf

stage    = omni.usd.get_context().get_stage()
settings = carb.settings.get_settings()

# ─────────────────────────────────────────────────────────────
# 1. BOOST DOMELIGHT — works in BOTH Real-Time and Path Traced
# ─────────────────────────────────────────────────────────────
dome_prim = stage.GetPrimAtPath("/World/DomeLight")
if dome_prim.IsValid():
    dome = UsdLux.DomeLight(dome_prim)
    dome.GetIntensityAttr().Set(2500.0)
    dome.GetExposureAttr().Set(0.0)
    # Remove the white color override so RtxDynamicSky controls color
    col = dome_prim.GetAttribute("inputs:color")
    if col and col.Get() == Gf.Vec3f(1, 1, 1):
        col.Set(Gf.Vec3f(1.0, 1.0, 1.0))  # keep white — sky child overrides it
    print("[dome] Intensity → 2500")
else:
    print("[dome] ERROR — /World/DomeLight not found")

# ─────────────────────────────────────────────────────────────
# 2. DISTANTLIGHT — needs to be strong enough to light the scene
# ─────────────────────────────────────────────────────────────
sun_prim = stage.GetPrimAtPath("/World/DistantLight")
if sun_prim.IsValid():
    UsdLux.DistantLight(sun_prim).GetIntensityAttr().Set(8000.0)
    print("[sun] DistantLight → 8000")

# ─────────────────────────────────────────────────────────────
# 3. SWITCH TO PATH TRACING — and set convergence limit higher
#    so it doesn't revert on viewport interaction
# ─────────────────────────────────────────────────────────────
try:
    settings.set("/rtx/rendermode", "PathTracing")
    # Max samples per pixel — stops reverting at 512
    settings.set("/rtx/pathtracing/spp", 64)          # fast convergence
    settings.set("/rtx/pathtracing/totalSpp", 4096)   # high quality ceiling
    settings.set("/rtx/pathtracing/clampSpp", 0)      # don't clamp
    # Firefly suppression (helps with noisy first frames)
    settings.set("/rtx/pathtracing/fireflyFilter/enabled", True)
    settings.set("/rtx/pathtracing/fireflyFilter/maxIntensityPerSample", 10000.0)
    mode = settings.get("/rtx/rendermode")
    print(f"[renderer] Mode: {mode}")
except Exception as e:
    print(f"[renderer] {e}")

# ─────────────────────────────────────────────────────────────
# 4. VIEWPORT EXPOSURE — make sure it's not crushing the image
# ─────────────────────────────────────────────────────────────
try:
    settings.set("/rtx/post/tonemap/enabled", True)
    settings.set("/rtx/post/tonemap/op", 6)        # ACES filmic
    settings.set("/rtx/post/tonemap/filmIso", 100.0)
    settings.set("/rtx/post/tonemap/fStop", 8.0)   # slightly stopped down
    settings.set("/rtx/post/tonemap/exposureValue", 0.0)
    print("[tonemap] ACES filmic tonemapper set")
except Exception as e:
    print(f"[tonemap] {e}")

print("""
[done] Scene should be bright now.
  If renderer reverted to Real-Time again:
    → Click dropdown top-left → 'RTX - Interactive (Path Traced)'
    → Clouds should appear within 2-3 seconds of switching
  
  Tweak brightness:
    DomeLight intensity: 1000-4000  (sky ambient)  
    DistantLight: 5000-10000        (sun direct)
""")

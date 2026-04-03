"""
inspect_stage.py
Run from Isaac Script Editor to dump everything currently in the stage.
Paste into Script Editor and hit play, or load via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/inspect_stage.py").read())
"""

import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

print("\n" + "="*70)
print("STAGE INSPECTION — all prims")
print("="*70)

counts = {}
all_prims = []

for prim in stage.Traverse():
    path     = str(prim.GetPath())
    typ      = prim.GetTypeName() or "Xform/Group"
    depth    = path.count("/") - 1

    counts[typ] = counts.get(typ, 0) + 1
    all_prims.append((depth, path, typ, prim))

    indent = "  " * depth
    print(f"{indent}{path.split('/')[-1]}  [{typ}]  → {path}")

print("\n" + "="*70)
print("SUMMARY — prim type counts")
print("="*70)
for typ, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {typ:<30} {count}")

print(f"\n  TOTAL PRIMS: {len(all_prims)}")

print("\n" + "="*70)
print("SCALE / TRANSFORM SNAPSHOT — top-level Xforms and Meshes only")
print("="*70)

for depth, path, typ, prim in all_prims:
    if depth > 2:
        continue  # only show top 2 levels for transforms
    if typ not in ("Xform", "Mesh", "Cube", "Cylinder", "Cone", "Sphere", "Capsule"):
        continue

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    op_summary = []
    for op in ops:
        name = op.GetOpName()
        val  = op.Get()
        if val is not None:
            op_summary.append(f"{name}={val}")

    if op_summary:
        print(f"  {path}")
        for s in op_summary:
            print(f"      {s}")
    else:
        print(f"  {path}  (no xform ops)")

print("\n" + "="*70)
print("SUBLAYERS")
print("="*70)
root = stage.GetRootLayer()
if root.subLayerPaths:
    for sl in root.subLayerPaths:
        print(f"  {sl}")
else:
    print("  (none)")

print("\n" + "="*70)
print("INSPECTION COMPLETE")
print("="*70 + "\n")

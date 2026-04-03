"""
inspect_walnut.py
Dumps the FULL prim tree under Wood_Tiles_Walnut to find where
the actual shader inputs live in the VMATERIALS MDL structure.

Run via:
  exec(open("/home/tgarcia/drone_sim/workspace/scripts/inspect_walnut.py").read())
"""

import omni.usd
from pxr import UsdShade, Sdf

stage = omni.usd.get_context().get_stage()

WALNUT_ROOT = "/World/Looks/Wood_Tiles_Walnut"

walnut_prim = stage.GetPrimAtPath(WALNUT_ROOT)
if not walnut_prim.IsValid():
    print("ERROR: Walnut prim not found. Run stage.Traverse() to find its actual path.")
    for prim in stage.Traverse():
        if "Walnut" in str(prim.GetPath()) or "walnut" in str(prim.GetPath()):
            print("  Found:", prim.GetPath(), "type:", prim.GetTypeName())
else:
    print(f"\n=== Full prim tree under {WALNUT_ROOT} ===")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(WALNUT_ROOT):
            indent = "  " * (path.count("/") - WALNUT_ROOT.count("/"))
            print(f"{indent}[{prim.GetTypeName()}] {path}")

            # Print all attributes on this prim
            for attr in prim.GetAttributes():
                val = attr.Get()
                print(f"{indent}  attr: {attr.GetName()} = {val}")

            # If it's a shader, try GetInputs()
            if prim.GetTypeName() == "Shader":
                shader = UsdShade.Shader(prim)
                inputs = shader.GetInputs()
                if inputs:
                    print(f"{indent}  --- Shader.GetInputs() ---")
                    for inp in inputs:
                        print(f"{indent}    {inp.GetFullName()} = {inp.Get()}")
                else:
                    print(f"{indent}  (no shader inputs exposed)")

            # Check for references / payloads
            refs = prim.GetPrimStack()
            if refs:
                for spec in refs:
                    if spec.referenceList.prependedItems or spec.payloadList.prependedItems:
                        print(f"{indent}  refs: {spec.referenceList.prependedItems}")
                        print(f"{indent}  payloads: {spec.payloadList.prependedItems}")

    print("=== End tree ===\n")

# Also: check what material is actually bound to a dock cube
print("=== Material binding check (first 3 Dock Cubes) ===")
count = 0
for prim in stage.Traverse():
    if prim.GetTypeName() == "Cube" and "/World/Dock/" in str(prim.GetPath()):
        binding = UsdShade.MaterialBindingAPI(prim)
        mat = binding.GetDirectBinding().GetMaterial()
        print(f"  {prim.GetPath()} → {mat.GetPath() if mat else 'NOTHING'}")
        count += 1
        if count >= 3:
            break

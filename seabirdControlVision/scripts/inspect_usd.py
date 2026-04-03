from pxr import Usd, Sdf
import os
stage = Usd.Stage.Open(os.path.expanduser("~/seabird/assets/marina_dock.usd"))
root = stage.GetRootLayer()
print("Sublayers:", root.subLayerPaths)
for prim in stage.Traverse():
    refs = prim.GetMetadata("references")
    if refs:
        print(f"REF {prim.GetPath()}: {refs}")
    payload = prim.GetMetadata("payload")
    if payload:
        print(f"PAYLOAD {prim.GetPath()}: {payload}")
print("Done inspecting")

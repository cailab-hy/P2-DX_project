"""Stage 24 (add_obj_v1): collider overlap check of the furnished scene (no Isaac needed).

Axis-aligned bounds of every collider under /Furniture are compared with the building colliders (floor excluded) and with
the colliders of the other furniture pieces; overlaps thicker than 3 mm are listed. Parts of one articulation that touch
by design (drawer front in its frame, sliding door in its track) show up as '<piece>/<body> x <piece>/<body>' pairs of the
same piece and are expected. Exit code 1 if any furniture piece overlaps the building or another piece.
"""
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

SCENE = "/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1_scene.usda"
st = Usd.Stage.Open(SCENE)
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"])
B, Fu = [], {}
for p in st.Traverse():
    if not p.HasAPI(UsdPhysics.CollisionAPI):
        continue
    s = str(p.GetPath())
    r = bb.ComputeWorldBound(p).ComputeAlignedRange()
    if "/Furniture/" in s:
        Fu.setdefault("/".join(s.split("/")[4:7]), []).append((s, r))      # Room/Piece/Body
    elif not p.GetName().startswith("floor"):
        B.append((s, r))


def ov(r1, r2, tol=0.003):
    i = Gf.Range3d.GetIntersection(r1, r2)
    return (not i.IsEmpty()) and min(*(i.GetMax() - i.GetMin())) > tol


bad = 0
for n, parts in Fu.items():
    hits = sorted({(bs.split("/")[-1], s.split("/")[-1]) for s, r in parts for bs, br in B if ov(r, br)})
    if hits:
        bad += 1
        print(n, "x building:", hits[:6])
names = list(Fu)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        h = [(s1.split("/")[-1], s2.split("/")[-1]) for s1, r1 in Fu[a] for s2, r2 in Fu[b] if ov(r1, r2)]
        if h:
            same = a.split("/")[:2] == b.split("/")[:2]
            bad += not same
            print(a, "x", b, h[:4], "(same articulated piece: expected)" if same else "")
print(f"checked {len(Fu)} bodies, {bad} unexpected overlaps")
sys.exit(1 if bad else 0)

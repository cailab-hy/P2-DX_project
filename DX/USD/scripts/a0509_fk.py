"""Forward kinematics for the Doosan A0509 USD (links are flat siblings under /Root/a0509; joints carry the frames).

world_child = F1^-1 * Rz(q) * F0 * world_parent   (Gf row-vector convention; F0 / F1 = joint localPos/Rot in body0 / body1)
Verified: q = 0 reproduces the authored link transforms within 0.5 mm.
"""
import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _T(pos, rot):
    m = Gf.Matrix4d()
    m.SetTransform(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(pos))
    return m


def posed_links(usd_path, q_deg):
    """{link name: world Matrix4d (robot root frame)} for joint angles q_deg = [j1..j6] in degrees."""
    st = Usd.Stage.Open(str(usd_path))
    xc = UsdGeom.XformCache()
    root = st.GetPrimAtPath("/Root/a0509")
    W = {p.GetName(): xc.GetLocalToWorldTransform(p) for p in root.GetChildren() if p.IsA(UsdGeom.Xformable)}
    joints = {}
    for p in st.Traverse():
        if p.GetTypeName() == "PhysicsRevoluteJoint":
            j = UsdPhysics.Joint(p)
            joints[p.GetName()] = (j.GetBody0Rel().GetTargets()[0].name, j.GetBody1Rel().GetTargets()[0].name,
                                   _T(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                                   _T(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))
    out = {"base_link": W["base_link"]}
    for k in range(1, 7):
        b0, b1, F0, F1 = joints[f"joint_{k}"]
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), q_deg[k - 1]))
        out[b1] = F1.GetInverse() * R * F0 * out[b0]
    return out

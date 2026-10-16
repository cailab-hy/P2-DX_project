"""Shared I/O helpers for the DX Hub scan -> USD pipeline."""
import struct
from pathlib import Path

import numpy as np

DATA = Path("/home/cai/P2&DX_project/P2&DX/DATA/DX/DX_Hub_20260622165437/data")
LAS = DATA / "pointcloud/260622_dx_scan01_2026-06-22-14-23-30_otter/260622_dx_scan01_2026-06-22-14-23-30_otter.las"
TRAJ = DATA / "pointcloud/260622_dx_scan01_2026-06-22-14-23-30_otter/260622_dx_scan01_2026-06-22-14-23-30_otter.trajectory.las"
GSPLAT = DATA / "model/260622_dx_scan01_Gaussian_2.ply"
OUT = Path("/home/cai/P2&DX_project/P2&DX/USD/DX_Hub")
WORK = Path("/home/cai/P2&DX_project/P2&DX/USD/work")

# LAS 1.2, point format 3 (XYZ, intensity, GPS time, RGB).
LAS_PF3 = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("I", "<u2"), ("flags", "u1"), ("cls", "u1"),
                    ("ang", "i1"), ("ud", "u1"), ("psid", "<u2"), ("t", "<f8"), ("R", "<u2"), ("G", "<u2"), ("B", "<u2")])


def read_las(path):
    """Return (xyz float64 [N,3], structured records) for a LAS 1.2 PF3 file."""
    with open(path, "rb") as f:
        h = f.read(227)
    assert h[:4] == b"LASF" and h[104] == 3
    offset = struct.unpack("<I", h[96:100])[0]
    n = struct.unpack("<I", h[107:111])[0]
    scale = np.array(struct.unpack("<3d", h[131:155]))
    origin = np.array(struct.unpack("<3d", h[155:179]))
    rec = np.memmap(path, dtype=LAS_PF3, mode="r", offset=offset, shape=(n,))
    xyz = np.empty((n, 3), np.float64)
    for i, k in enumerate("XYZ"):
        xyz[:, i] = rec[k] * scale[i] + origin[i]
    return xyz, rec


def read_gs_ply(path):
    """Return (header bytes, property names, float32 array [N, P]) for a 3DGS PLY."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        props = [l.split()[-1].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
        n = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype=np.float32).reshape(n, len(props))
    return hdr, props, data


def write_gs_ply(path, props, data):
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    hdr += [f"property float {p}" for p in props] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def floor_plane(P, cell=1.0, min_count=200, pct=2.0, band=0.3):
    """Robust floor plane z = a x + b y + c from a low percentile per XY cell."""
    g = np.floor(P[:, :2] / cell).astype(np.int64)
    key = g[:, 0] * 1_000_003 + g[:, 1]
    o = np.argsort(key, kind="stable")
    _, idx, cnt = np.unique(key[o], return_index=True, return_counts=True)
    fl = np.array([[*P[o[i], :2], np.percentile(P[o[i:i + c], 2], pct)] for i, c in zip(idx, cnt) if c >= min_count])
    fl = fl[fl[:, 2] < np.percentile(fl[:, 2], 30) + band]
    A = np.c_[fl[:, :2], np.ones(len(fl))]
    z = fl[:, 2]
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ coef
        keep = np.abs(r) < 3 * np.median(np.abs(r)) + 0.01
        A, z = A[keep], z[keep]
    return coef, float(r[keep].std()), int(keep.sum())


def level_rotation(coef):
    """Rotation that maps the plane normal (-a, -b, 1) to +Z."""
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), n @ z
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot_z(deg):
    t = np.radians(deg)
    R = np.eye(4)
    R[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return R

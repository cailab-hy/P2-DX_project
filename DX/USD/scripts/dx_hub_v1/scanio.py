"""FJD LAS reader shared by register_scan.py and build_mesh.py."""
import struct

import numpy as np


def read_las(path):
    head = open(path, 'rb').read(400)
    off = struct.unpack('<I', head[96:100])[0]
    reclen = struct.unpack('<H', head[105:107])[0]
    n = struct.unpack('<I', head[107:111])[0]
    scale = np.array(struct.unpack('<3d', head[131:155]))
    offset = np.array(struct.unpack('<3d', head[155:179]))
    dt = np.dtype([('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('I', '<u2'), ('f', 'u1'), ('c', 'u1'), ('a', 'i1'),
                   ('u', 'u1'), ('s', '<u2'), ('t', '<f8'), ('R', '<u2'), ('G', '<u2'), ('B', '<u2'),
                   ('pad', f'V{reclen - 34}')])
    a = np.fromfile(path, dtype=dt, count=n, offset=off)
    xyz = np.stack([a['X'], a['Y'], a['Z']], 1) * scale + offset
    rgb = np.stack([a['R'], a['G'], a['B']], 1).astype(np.float64)
    rgb /= 255.0 if rgb.max() <= 255 else 65535.0
    return xyz, rgb

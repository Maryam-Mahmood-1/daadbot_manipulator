#!/usr/bin/env python3
"""Tiny dependency-light STL renderer (numpy + PIL only).

Renders an STL with a z-buffer, flat shading, coloured per connected shell, so
you can identify which shell is which. Usage:
    stl_render.py STL OUT.png [--elev 25 --azim -60 --w 900 --h 900]
"""
import argparse, sys
import numpy as np
from PIL import Image
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from stl_shells import load_stl, shells


def rot(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl'); ap.add_argument('out')
    ap.add_argument('--elev', type=float, default=25)
    ap.add_argument('--azim', type=float, default=-60)
    ap.add_argument('--w', type=int, default=900); ap.add_argument('--h', type=int, default=900)
    ap.add_argument('--highlight', default='', help='comma shell idxs drawn red')
    args = ap.parse_args()

    normals, tris = load_stl(args.stl)
    comps = shells(tris)
    hl = {int(x) for x in args.highlight.split(',')} if args.highlight else set()

    # per-triangle shell id
    shell_of = np.zeros(len(tris), dtype=int)
    for i, c in enumerate(comps):
        shell_of[c] = i

    R = rot(args.elev, args.azim)
    V = tris.reshape(-1, 3) @ R.T
    V = V.reshape(-1, 3, 3)

    # orthographic: x->screen x, z->screen y(up), y-> depth
    sx, sy, depth = V[:, :, 0], V[:, :, 2], V[:, :, 1]
    W, H = args.w, args.h
    mnx, mxx = sx.min(), sx.max(); mny, mxy = sy.min(), sy.max()
    pad = 0.06
    span = max(mxx - mnx, mxy - mny) * (1 + pad)
    cx, cy = (mnx + mxx) / 2, (mny + mxy) / 2
    px = (sx - cx) / span * W + W / 2
    py = H / 2 - (sy - cy) / span * H

    # light from camera
    nrm = normals @ R.T
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    shade = np.clip(np.abs(nrm[:, 1]), 0.25, 1.0)

    palette = (np.array([
        [70, 130, 180], [60, 179, 113], [238, 130, 60], [186, 85, 211],
        [205, 92, 92], [218, 165, 32], [95, 158, 160], [199, 21, 133],
        [100, 149, 237], [154, 205, 50], [255, 140, 0], [147, 112, 219],
        [220, 20, 60], [0, 139, 139], [233, 150, 122], [107, 142, 35],
        [72, 61, 139], [46, 139, 87], [210, 105, 30]]) )

    img = np.full((H, W, 3), 30, dtype=np.uint8)
    zbuf = np.full((H, W), 1e9)

    order = np.argsort(-depth.mean(axis=1))  # far first (painter assist; zbuf is real)
    for ti in order:
        x = px[ti]; y = py[ti]; z = depth[ti]
        col = np.array([220, 60, 60]) if shell_of[ti] in hl else palette[shell_of[ti] % len(palette)]
        col = (col * shade[ti]).astype(np.uint8)
        minx, maxx = int(max(0, np.floor(x.min()))), int(min(W - 1, np.ceil(x.max())))
        miny, maxy = int(max(0, np.floor(y.min()))), int(min(H - 1, np.ceil(y.max())))
        if minx > maxx or miny > maxy:
            continue
        xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        x0, y0 = x[0], y[0]; x1, y1 = x[1], y[1]; x2, y2 = x[2], y[2]
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-9:
            continue
        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / d
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / d
        c = 1 - a - b
        m = (a >= 0) & (b >= 0) & (c >= 0)
        if not m.any():
            continue
        zz = a * z[0] + b * z[1] + c * z[2]
        yy, xx = ys[m], xs[m]; zz = zz[m]
        cur = zbuf[yy, xx]
        upd = zz < cur
        yy, xx = yy[upd], xx[upd]
        zbuf[yy, xx] = zz[upd]
        img[yy, xx] = col

    Image.fromarray(img).save(args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

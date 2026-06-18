#!/usr/bin/env python3
"""Inspect / split / edit a binary STL by connected shell.

A "shell" = a set of triangles connected through shared (welded) vertices.
Many CAD-exported STLs bundle several disconnected solids into one file; this
lets you see them and drop the ones you don't want -- without a CAD tool.

Subcommands:
  list   STL                     # print every shell with size/center
  split  STL OUTDIR              # write each shell as OUTDIR/shell_<i>.stl
  remove STL -k I[,J..] -o OUT   # write OUT with shells I,J.. deleted
                                 # (original is backed up to *.bak first if OUT==STL)
"""
import argparse, os, struct, shutil
import numpy as np
from collections import defaultdict


def load_stl(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    floats = data[:, :48].copy().view('<f4').reshape(n, 12)
    normals = floats[:, 0:3]
    tris = floats[:, 3:12].reshape(n, 3, 3)
    return normals, tris


def write_stl(path, normals, tris):
    n = len(tris)
    out = np.zeros((n, 50), dtype=np.uint8)
    buf = np.zeros((n, 12), dtype='<f4')
    buf[:, 0:3] = normals
    buf[:, 3:12] = tris.reshape(n, 9)
    out[:, :48] = buf.view(np.uint8).reshape(n, 48)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', n))
        f.write(out.tobytes())


def shells(tris, tol=1e-5):
    n = len(tris)
    keys = np.round(tris.reshape(-1, 3) / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(n, 3)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    v2t = defaultdict(list)
    for ti in range(n):
        for vi in inv[ti]:
            v2t[vi].append(ti)
    for vts in v2t.values():
        r0 = find(vts[0])
        for k in vts[1:]:
            parent[find(k)] = r0
    comp = defaultdict(list)
    for ti in range(n):
        comp[find(ti)].append(ti)
    # stable order: biggest first
    return sorted(comp.values(), key=len, reverse=True)


def describe(tris, comps):
    for i, c in enumerate(comps):
        v = tris[c].reshape(-1, 3)
        mn, mx = v.min(0), v.max(0)
        print(f"  shell {i:2d}: {len(c):6d} tris  "
              f"size={np.round(mx - mn, 4)}  center={np.round((mn + mx) / 2, 4)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list'); p.add_argument('stl')
    p = sub.add_parser('split'); p.add_argument('stl'); p.add_argument('outdir')
    p = sub.add_parser('remove'); p.add_argument('stl')
    p.add_argument('-k', '--keep-out', required=True,
                   help='comma-separated shell indices to REMOVE')
    p.add_argument('-o', '--output', required=True)
    args = ap.parse_args()

    normals, tris = load_stl(args.stl)
    comps = shells(tris)

    if args.cmd == 'list':
        print(f"{args.stl}: {len(tris)} tris, {len(comps)} shells")
        describe(tris, comps)
        return

    if args.cmd == 'split':
        os.makedirs(args.outdir, exist_ok=True)
        for i, c in enumerate(comps):
            out = os.path.join(args.outdir, f"shell_{i}.stl")
            write_stl(out, normals[c], tris[c])
            print(f"  wrote {out}  ({len(c)} tris)")
        return

    if args.cmd == 'remove':
        drop = {int(x) for x in args.keep_out.split(',')}
        bad = [d for d in drop if d < 0 or d >= len(comps)]
        if bad:
            raise SystemExit(f"shell index out of range: {bad} (have 0..{len(comps)-1})")
        keep_idx = [i for i in range(len(comps)) if i not in drop]
        keep_tris = np.concatenate([comps[i] for i in keep_idx])
        if os.path.abspath(args.output) == os.path.abspath(args.stl):
            bak = args.stl + '.bak'
            if not os.path.exists(bak):
                shutil.copy2(args.stl, bak)
                print(f"  backed up original -> {bak}")
        write_stl(args.output, normals[keep_tris], tris[keep_tris])
        print(f"  removed shells {sorted(drop)}; kept {len(keep_tris)}/{len(tris)} tris "
              f"-> {args.output}")


if __name__ == '__main__':
    main()

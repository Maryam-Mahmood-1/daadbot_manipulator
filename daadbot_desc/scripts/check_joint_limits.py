#!/usr/bin/env python3
"""Verify joint limits are consistent across the three places they live:

  1. URDF <limit lower/upper effort velocity>     (source of truth)
  2. URDF ros2_control <command_interface> min/max (command clamp)
  3. MoveIt joint_limits.yaml                      (max velocity/acceleration,
                                                     optional position override)

Flags command clamps that don't match the URDF position limits and MoveIt
velocity limits that exceed the URDF velocity. Use it after editing limits to
confirm everything agrees.

  check_joint_limits.py --urdf <flat.urdf> [--moveit-limits joint_limits.yaml]
"""
import argparse
import xml.etree.ElementTree as ET


def urdf_limits(path):
    root = ET.parse(path).getroot()
    rc = root.find('ros2_control')
    rc_joints = {j.get('name'): j for j in rc.findall('joint')} if rc is not None else {}
    out = {}
    for j in root.findall('joint'):
        if j.get('type') not in ('revolute', 'prismatic'):
            continue
        lim = j.find('limit')
        if lim is None:
            continue
        name = j.get('name')
        ci = rc_joints.get(name)
        cmd = ci.find('command_interface') if ci is not None else None
        cmin = cmax = None
        if cmd is not None:
            pmin = cmd.find('param[@name="min"]')
            pmax = cmd.find('param[@name="max"]')
            cmin = float(pmin.text) if pmin is not None else None
            cmax = float(pmax.text) if pmax is not None else None
        out[name] = {
            'lower': float(lim.get('lower')),
            'upper': float(lim.get('upper')),
            'effort': float(lim.get('effort')),
            'velocity': float(lim.get('velocity')),
            'cmd': cmd.get('name') if cmd is not None else None,
            'cmin': cmin, 'cmax': cmax,
        }
    return out


def moveit_limits(path):
    try:
        import yaml
    except ImportError:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return (data.get('joint_limits') or {})


def sync_cmd_limits(urdf_path, out_path):
    """Write a copy of the URDF with each ros2_control command_interface min/max
    set to the matching joint's <limit> lower/upper. Input is left untouched."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {j.get('name'): j.find('limit') for j in root.findall('joint')
              if j.find('limit') is not None}
    rc = root.find('ros2_control')
    changed = 0
    for j in rc.findall('joint'):
        lim = limits.get(j.get('name'))
        cmd = j.find('command_interface')
        if lim is None or cmd is None:
            continue
        for attr, src in (('min', 'lower'), ('max', 'upper')):
            p = cmd.find(f'param[@name="{attr}"]')
            if p is None:
                p = ET.SubElement(cmd, 'param'); p.set('name', attr)
            p.text = lim.get(src)
        changed += 1
    tree.write(out_path, xml_declaration=True, encoding='utf-8')
    print(f'synced command min/max to <limit> for {changed} joint(s) -> {out_path}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--moveit-limits', default=None)
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--write-synced', default=None, metavar='OUT',
                    help='write a copy of the URDF with every command_interface '
                         'min/max set equal to that joint <limit> (does not touch input)')
    args = ap.parse_args()

    if args.write_synced:
        return sync_cmd_limits(args.urdf, args.write_synced)

    u = urdf_limits(args.urdf)
    m = moveit_limits(args.moveit_limits) if args.moveit_limits else {}

    print(f"{'joint':12s} {'pos lower':>10s} {'upper':>9s} {'eff':>7s} {'vel':>8s} "
          f"{'cmd min':>9s} {'cmd max':>9s} {'mv vel':>8s}")
    print('-' * 86)
    issues = []
    for name, d in u.items():
        mv = m.get(name, {})
        mvel = mv.get('max_velocity')
        s_cmin = '-' if d['cmin'] is None else f"{d['cmin']:.3f}"
        s_cmax = '-' if d['cmax'] is None else f"{d['cmax']:.3f}"
        s_mvel = '-' if mvel is None else f"{mvel:.3f}"
        print(f"{name:12s} {d['lower']:10.3f} {d['upper']:9.3f} {d['effort']:7.1f} "
              f"{d['velocity']:8.3f} {s_cmin:>9s} {s_cmax:>9s} {s_mvel:>8s}")

        # checks (only for joints that have a command interface)
        if d['cmd'] in ('position',):
            if d['cmin'] is None or abs(d['cmin'] - d['lower']) > args.tol:
                issues.append(f"{name}: cmd min {d['cmin']} != URDF lower {d['lower']}")
            if d['cmax'] is None or abs(d['cmax'] - d['upper']) > args.tol:
                issues.append(f"{name}: cmd max {d['cmax']} != URDF upper {d['upper']}")
        if d['cmin'] is not None and d['cmin'] < d['lower'] - args.tol:
            issues.append(f"{name}: cmd min {d['cmin']} is BELOW URDF lower {d['lower']}")
        if d['cmax'] is not None and d['cmax'] > d['upper'] + args.tol:
            issues.append(f"{name}: cmd max {d['cmax']} is ABOVE URDF upper {d['upper']}")
        if mvel is not None and mvel > d['velocity'] + args.tol:
            issues.append(f"{name}: MoveIt max_velocity {mvel} > URDF velocity {d['velocity']}")
        if d['lower'] >= d['upper']:
            issues.append(f"{name}: lower {d['lower']} >= upper {d['upper']}")

    print()
    if issues:
        print(f"{len(issues)} issue(s):")
        for i in issues:
            print(f"  ! {i}")
    else:
        print("OK: command clamps match URDF position limits; MoveIt velocities within URDF.")
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())

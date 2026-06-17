#!/usr/bin/env python3
"""Re-bake a joint's initial angle into a URDF.

Produce a new URDF in which a chosen joint's ZERO position corresponds to a
given angle of the original model. The whole sub-tree below the joint is
rotated accordingly (handled implicitly by baking the rotation into the joint
<origin>), and the joint <limit>/safety soft-limits are shifted so the physical
reachable envelope is unchanged -- only renumbered.

Math (revolute joint):
    Child frame:  T(q)  = Origin . Rot(axis, q)
    We want q'=0  to equal old q=theta0:
        Origin' = Origin . Rot(axis, theta0)
        q'      = q_old - theta0
    => origin rotation is right-multiplied by Rot(axis, theta0)
       (xyz translation is unchanged, the axis passes through the origin)
    => limits/soft-limits shift by -theta0
Prismatic joints are handled analogously (theta0 is a translation along axis,
which shifts origin xyz and the limits).

Usage:
    rebake_joint_angle.py INPUT.urdf JOINT_NAME ANGLE_DEG -o OUTPUT.urdf
e.g.
    rebake_joint_angle.py base.urdf joint_6 -90 -o new_base.urdf
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np


# ---------- rotation helpers (URDF fixed-axis XYZ convention) ----------
def rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) . Ry(pitch) . Rx(roll)  (URDF/ROS convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_rpy(R):
    """Inverse of rpy_to_matrix. Returns (roll, pitch, yaw)."""
    # pitch = asin(-R[2,0]); handle gimbal lock.
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(abs(sp) - 1.0) < 1e-9:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def axis_angle_matrix(axis, theta):
    """Rodrigues rotation about a (not-necessarily-unit) axis by theta rad."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("joint axis has zero length")
    a = a / n
    x, y, z = a
    c, s, C = math.cos(theta), math.sin(theta), 1 - math.cos(theta)
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def fmt(v):
    """Tidy float formatting for URDF attributes."""
    if abs(v) < 1e-12:
        v = 0.0
    return repr(round(v, 12))


# ---------- core ----------
def parse_xyz(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return list(default)
    return [float(x) for x in s.split()]


def rebake(tree, joint_name, theta0):
    """Mutate `tree` in place. theta0 in radians (rad for revolute, m for prismatic)."""
    root = tree.getroot()
    joint = next((j for j in root.findall('joint')
                  if j.get('name') == joint_name), None)
    if joint is None:
        raise SystemExit(f"joint '{joint_name}' not found in URDF")

    jtype = joint.get('type')
    axis_el = joint.find('axis')
    axis = parse_xyz(axis_el.get('xyz') if axis_el is not None else None,
                     default=(1.0, 0.0, 0.0))

    origin = joint.find('origin')
    if origin is None:
        origin = ET.SubElement(joint, 'origin')
    rpy = parse_xyz(origin.get('rpy'))
    xyz = parse_xyz(origin.get('xyz'))

    report = {'type': jtype, 'axis': axis,
              'old_rpy': list(rpy), 'old_xyz': list(xyz)}

    if jtype in ('revolute', 'continuous'):
        # Bake rotation: R_new = R_old . Rot(axis, theta0); xyz unchanged.
        R_new = rpy_to_matrix(*rpy) @ axis_angle_matrix(axis, theta0)
        rpy = list(matrix_to_rpy(R_new))
        origin.set('rpy', ' '.join(fmt(v) for v in rpy))
        # xyz unchanged
        shift = -theta0  # limits renumber: q' = q_old - theta0
    elif jtype == 'prismatic':
        # Bake translation: xyz_new = xyz_old + R_old . (axis_unit * theta0)
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        xyz = list(np.asarray(xyz) + rpy_to_matrix(*rpy) @ (a * theta0))
        origin.set('xyz', ' '.join(fmt(v) for v in xyz))
        shift = -theta0
    else:
        raise SystemExit(f"joint type '{jtype}' is not offsettable (fixed?)")

    report['new_rpy'] = list(rpy)
    report['new_xyz'] = list(xyz)

    # Shift hard limits (revolute/prismatic have <limit lower/upper>).
    limit = joint.find('limit')
    if limit is not None and limit.get('lower') is not None:
        lo = float(limit.get('lower')) + shift
        hi = float(limit.get('upper')) + shift
        limit.set('lower', fmt(lo))
        limit.set('upper', fmt(hi))
        report['old_limit'] = (float(limit.get('lower')) - shift,
                               float(limit.get('upper')) - shift)
        report['new_limit'] = (lo, hi)

    # Shift safety_controller soft limits if present.
    safety = joint.find('safety_controller')
    if safety is not None:
        for attr in ('soft_lower_limit', 'soft_upper_limit'):
            if safety.get(attr) is not None:
                safety.set(attr, fmt(float(safety.get(attr)) + shift))
        report['soft_limits_shifted'] = True

    report['shift'] = shift
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('joint')
    ap.add_argument('angle_deg', type=float,
                    help='new zero = this angle of the original model (deg; '
                         'meters for prismatic)')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--radians', action='store_true',
                    help='treat angle as radians/meters, not degrees')
    args = ap.parse_args()

    theta0 = args.angle_deg if args.radians else math.radians(args.angle_deg)

    tree = ET.parse(args.input)
    rep = rebake(tree, args.joint, theta0)
    tree.write(args.output, xml_declaration=True, encoding='utf-8')

    unit = 'rad' if args.radians else 'deg'
    print(f"Re-baked '{args.joint}' ({rep['type']}) so q'=0 == old "
          f"{args.angle_deg}{unit}.  Written: {args.output}")
    print(f"  axis     : {rep['axis']}")
    print(f"  origin rpy: {rep['old_rpy']}  ->  {rep['new_rpy']}")
    print(f"  origin xyz: {rep['old_xyz']}  ->  {rep['new_xyz']}")
    if 'new_limit' in rep:
        print(f"  limits   : {rep['new_limit'][0]-rep['shift']:.6f}..{rep['new_limit'][1]-rep['shift']:.6f}"
              f"  ->  {rep['new_limit'][0]:.6f}..{rep['new_limit'][1]:.6f}")


if __name__ == '__main__':
    main()

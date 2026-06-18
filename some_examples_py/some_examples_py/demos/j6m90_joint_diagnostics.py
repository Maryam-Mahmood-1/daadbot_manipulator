#!/usr/bin/env python3
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.append("/opt/ros/humble/lib/python3.10/site-packages")
sys.path.append("/opt/ros/humble/local/lib/python3.10/dist-packages")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_URDF = (
    "/home/maryammahmood/daadbot_manipulator_ws/src/daadbot_manipulator/"
    "daadbot_desc/urdf/urdf_table_j6m90_effort/daadbot.urdf"
)
JOINTS = [f"joint_{i}" for i in range(1, 8)]


def load_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in JOINTS:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = {
            "lower": float(limit.attrib.get("lower", "-inf")),
            "upper": float(limit.attrib.get("upper", "inf")),
            "effort": float(limit.attrib.get("effort", "inf")),
            "velocity": float(limit.attrib.get("velocity", "inf")),
        }
    return limits


class JointDiagnostics(Node):
    def __init__(self, args):
        super().__init__("j6m90_joint_diagnostics")
        self.args = args
        self.limits = load_limits(args.urdf)
        self.samples = []
        self.create_subscription(JointState, args.topic, self.callback, 100)
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.maybe_finish)
        self.get_logger().info(
            f"Collecting {args.duration:.2f}s of {args.topic} for {', '.join(JOINTS)}"
        )

    def callback(self, msg):
        row = {"stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9}
        for name in JOINTS:
            if name not in msg.name:
                continue
            i = msg.name.index(name)
            row[name] = {
                "position": msg.position[i] if i < len(msg.position) else math.nan,
                "velocity": msg.velocity[i] if i < len(msg.velocity) else math.nan,
                "effort": msg.effort[i] if i < len(msg.effort) else math.nan,
            }
        self.samples.append(row)

    def maybe_finish(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.args.duration:
            return
        self.report()
        rclpy.shutdown()

    def report(self):
        if not self.samples:
            print("No joint states received.")
            return

        print(f"samples: {len(self.samples)}")
        print(
            "joint   pos_min   pos_max   lower_margin  upper_margin  "
            "vel_rms  vel_max  vel_limit_hits  effort_max  effort_sat"
        )
        for name in JOINTS:
            vals = [s[name] for s in self.samples if name in s]
            if not vals:
                continue
            q = np.array([v["position"] for v in vals], dtype=float)
            dq = np.array([v["velocity"] for v in vals], dtype=float)
            tau = np.array([v["effort"] for v in vals], dtype=float)
            lim = self.limits.get(name, {})
            lower = lim.get("lower", -np.inf)
            upper = lim.get("upper", np.inf)
            vlim = lim.get("velocity", np.inf)
            elim = lim.get("effort", np.inf)
            lower_margin = np.nanmin(q - lower)
            upper_margin = np.nanmin(upper - q)
            vel_hits = np.mean(np.abs(dq) >= 0.98 * vlim) if np.isfinite(vlim) else 0.0
            effort_sat = np.mean(np.abs(tau) >= 0.98 * elim) if np.isfinite(elim) else 0.0
            print(
                f"{name:7s} "
                f"{np.nanmin(q):8.4f} {np.nanmax(q):8.4f} "
                f"{lower_margin:12.4f} {upper_margin:12.4f} "
                f"{np.sqrt(np.nanmean(dq * dq)):8.3f} {np.nanmax(np.abs(dq)):8.3f} "
                f"{vel_hits:14.1%} {np.nanmax(np.abs(tau)):11.3f} {effort_sat:10.1%}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args(argv)

    if not Path(args.urdf).exists():
        raise FileNotFoundError(args.urdf)

    rclpy.init()
    node = JointDiagnostics(args)
    rclpy.spin(node)


if __name__ == "__main__":
    main()

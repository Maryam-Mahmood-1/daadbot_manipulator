#!/usr/bin/env python3
"""Angle-input slider GUI for a forward joint-group controller.

Reads the actuated joints and their limits straight from a URDF and publishes
std_msgs/Float64MultiArray to /<controller>/commands, in the controller's joint
order. Works with the generated lower-DOF sim launches (position/effort/velocity
JointGroup controllers).

Run it AFTER the sim is up, e.g.:
  ros2 launch daadbot_bringup sim_robot_table_pos_j6m90_2dof_trunc.launch.py
  python3 joint_command_gui.py \
      --controller position_arm_controller \
      --urdf .../urdf/urdf_table_pos_j6m90_2dof_trunc/daadbot.urdf

--joints overrides the auto-detected order; --topic overrides the destination.
"""
import argparse
import threading
import xml.etree.ElementTree as ET


def joints_and_limits(urdf_path, override=None):
    root = ET.parse(urdf_path).getroot()
    rc = root.find('ros2_control')
    # command joints = ros2_control joints that have a command_interface, in order
    cmd_joints = override or [
        j.get('name') for j in (rc.findall('joint') if rc is not None else [])
        if j.find('command_interface') is not None
    ]
    limits = {}
    for j in root.findall('joint'):
        lim = j.find('limit')
        if lim is not None and lim.get('lower') is not None:
            limits[j.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
    return cmd_joints, limits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller', default='position_arm_controller')
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--joints', nargs='+', default=None,
                    help='override joint order (must match the controller yaml)')
    ap.add_argument('--topic', default=None,
                    help='override topic (default /<controller>/commands)')
    args = ap.parse_args()

    joints, limits = joints_and_limits(args.urdf, args.joints)
    if not joints:
        raise SystemExit('No command joints found; pass --joints explicitly.')
    topic = args.topic or f'/{args.controller}/commands'

    import rclpy
    from std_msgs.msg import Float64MultiArray
    import tkinter as tk

    rclpy.init()
    node = rclpy.create_node('joint_command_gui')
    pub = node.create_publisher(Float64MultiArray, topic, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    sliders = []

    def publish(*_):
        msg = Float64MultiArray()
        msg.data = [float(v.get()) for v in sliders]
        pub.publish(msg)

    def zero_all():
        for v in sliders:
            v.set(0.0)
        publish()

    win = tk.Tk()
    win.title(f'angle input -> {topic}')
    for jn in joints:
        lo, hi = limits.get(jn, (-3.14159, 3.14159))
        row = tk.Frame(win)
        row.pack(fill='x', padx=8, pady=3)
        tk.Label(row, text=jn, width=14, anchor='w').pack(side='left')
        var = tk.DoubleVar(value=0.0)
        sliders.append(var)
        tk.Scale(row, from_=round(lo, 3), to=round(hi, 3), resolution=0.01,
                 orient='horizontal', length=320, variable=var,
                 command=publish).pack(side='left', fill='x', expand=True)
    tk.Button(win, text='zero all', command=zero_all).pack(pady=6)

    publish()  # send the initial all-zeros command
    try:
        win.mainloop()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

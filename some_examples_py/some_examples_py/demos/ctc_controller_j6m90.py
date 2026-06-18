#!/usr/bin/env python3
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

sys.path.append("/opt/ros/humble/lib/python3.10/site-packages")
sys.path.append("/opt/ros/humble/local/lib/python3.10/dist-packages")

import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


URDF_PATH = os.environ.get(
    "DAADBOT_CTC_URDF",
    "/home/maryammahmood/daadbot_manipulator_ws/src/daadbot_manipulator/"
    "daadbot_desc/urdf/urdf_table_j6m90_effort/daadbot.urdf",
)
COMMAND_TOPIC = os.environ.get(
    "DAADBOT_CTC_COMMAND_TOPIC",
    "/effort_arm_controller/commands",
)
COMMAND_LIMIT = float(os.environ.get("DAADBOT_CTC_COMMAND_LIMIT", "60.0"))
SLIDER_LIMIT_DEG = float(os.environ.get("DAADBOT_CTC_SLIDER_LIMIT_DEG", "45.0"))
SINE_AMP_RAD = float(os.environ.get("DAADBOT_CTC_SINE_AMP_RAD", "0.12"))

CONTROLLER_JOINT_ORDER = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

KP = np.array([80.0, 80.0, 240.0, 160.0, 320.0, 500.0, 1000.0])
KD = np.array([18.0, 18.0, 40.0, 30.0, 58.0, 90.0, 140.0])
ARMATURE = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05])


class PinocchioRosNode(Node):
    def __init__(self):
        super().__init__("j6m90_ctc_controller")

        full_model = pin.buildModelFromUrdf(URDF_PATH)
        lock_joint_ids = [
            joint_id
            for joint_id, name in enumerate(full_model.names)
            if joint_id > 0 and name not in CONTROLLER_JOINT_ORDER
        ]
        self.model = pin.buildReducedModel(
            full_model,
            lock_joint_ids,
            pin.neutral(full_model),
        )
        if self.model.nv != len(CONTROLLER_JOINT_ORDER):
            raise RuntimeError(
                f"Expected {len(CONTROLLER_JOINT_ORDER)} DoFs, got {self.model.nv}"
            )
        self.model.armature[:] = ARMATURE
        self.data = self.model.createData()

        self.q_current = np.zeros(self.model.nq)
        self.v_current = np.zeros(self.model.nv)
        self.q_gui_target = np.zeros(self.model.nq)
        self.state_received = False
        self.trajectory_active = False
        self.traj_start_time = 0.0
        self.state_index = {name: i for i, name in enumerate(CONTROLLER_JOINT_ORDER)}

        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        self.create_timer(0.001, self.control_loop)

        self.get_logger().info(f"J6M90 CTC ready. URDF={URDF_PATH}")
        self.get_logger().info(f"Publishing raw effort commands on {COMMAND_TOPIC}")

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            idx = self.state_index.get(name)
            if idx is None:
                continue
            self.q_current[idx] = msg.position[i]
            if i < len(msg.velocity):
                self.v_current[idx] = msg.velocity[i]

        if not self.state_received:
            self.q_gui_target = self.q_current.copy()
            self.state_received = True
            self.get_logger().info("Connected to Gazebo. Holding current pose.")

    def control_loop(self):
        if not self.state_received:
            return

        q_des = self.q_gui_target.copy()
        v_des = np.zeros(self.model.nv)
        a_des = np.zeros(self.model.nv)

        if self.trajectory_active:
            t = time.time() - self.traj_start_time
            omega = 1.0
            for i in range(self.model.nv):
                phase = 0.5 * i
                q_des[i] += SINE_AMP_RAD * math.sin(omega * t + phase)
                v_des[i] = SINE_AMP_RAD * omega * math.cos(omega * t + phase)
                a_des[i] = -SINE_AMP_RAD * omega * omega * math.sin(omega * t + phase)

        error_q = pin.difference(self.model, self.q_current, q_des)
        error_v = v_des - self.v_current
        desired_accel = a_des + KP * error_q + KD * error_v
        tau = pin.rnea(self.model, self.data, self.q_current, self.v_current, desired_accel)

        msg = Float64MultiArray()
        msg.data = np.clip(tau, -COMMAND_LIMIT, COMMAND_LIMIT).tolist()
        self.torque_pub.publish(msg)


def run_gui(node):
    root = tk.Tk()
    root.title("J6M90 Nominal CTC")
    root.geometry("400x700")

    ttk.Label(root, text="J6M90 NOMINAL CTC", font=("Arial", 12, "bold")).pack(pady=10)
    ttk.Label(root, text=COMMAND_TOPIC, justify=tk.CENTER).pack(pady=2)

    sliders = []
    suppress_slider_callback = {"value": False}

    def on_slider_update(_val=None):
        if suppress_slider_callback["value"]:
            return
        node.q_gui_target = np.array([np.deg2rad(slider.get()) for slider in sliders])

    for joint_name in CONTROLLER_JOINT_ORDER:
        ttk.Label(root, text=f"{joint_name} (deg)").pack(pady=2)
        slider = tk.Scale(
            root,
            from_=-SLIDER_LIMIT_DEG,
            to=SLIDER_LIMIT_DEG,
            orient=tk.HORIZONTAL,
            length=300,
            command=on_slider_update,
        )
        slider.set(0)
        slider.pack()
        sliders.append(slider)

    sliders_synced = {"done": False}

    def sync_sliders_to_robot_once():
        if not sliders_synced["done"] and node.state_received:
            suppress_slider_callback["value"] = True
            for i, slider in enumerate(sliders):
                slider.set(float(np.rad2deg(node.q_current[i])))
            suppress_slider_callback["value"] = False
            sliders_synced["done"] = True
        if not sliders_synced["done"]:
            root.after(100, sync_sliders_to_robot_once)

    def toggle_trajectory():
        if not node.trajectory_active:
            node.traj_start_time = time.time()
            node.trajectory_active = True
            btn.config(text="STOP SINE WAVE")
        else:
            node.trajectory_active = False
            btn.config(text="START SINE WAVE")

    root.after(100, sync_sliders_to_robot_once)
    ttk.Separator(root, orient="horizontal").pack(fill="x", pady=20)
    btn = ttk.Button(root, text="START SINE WAVE", command=toggle_trajectory)
    btn.pack(pady=10, ipady=10)
    root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = PinocchioRosNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        run_gui(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

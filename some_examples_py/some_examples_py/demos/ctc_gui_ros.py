#!/usr/bin/env python3
import sys
import os
import threading
import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk

# --- 1. CONDA + ROS FIX ---
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import pinocchio as pin

# --- 2. CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"
COMMAND_TOPIC = '/effort_arm_controller/commands' 
DYNAMICS_TOPIC = '/robot_joint_dynamics' 

CONTROLLER_JOINT_ORDER = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class PinocchioRosNode(Node):
    def __init__(self):
        super().__init__('pinocchio_gui_controller')
        
        # A. Setup Pinocchio
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # B. Internal Robot State
        self.q_current = np.zeros(self.nq)
        self.v_current = np.zeros(self.nv)
        self.state_received = False

        # C. Control Targets
        self.q_target = np.zeros(self.nq)      # The final goal from GUI
        self.q_smooth_ref = np.zeros(self.nq)  # The intermediate "moving" target

        # D. Joint Mapping
        self.output_map = []
        for name in CONTROLLER_JOINT_ORDER:
            if self.model.existJointName(name):
                idx = self.model.getJointId(name) - 1
                self.output_map.append(idx)
            else:
                self.get_logger().error(f"Joint '{name}' not found in URDF!")

        # E. ROS Interface
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        self.dynamics_pub = self.create_publisher(JointTrajectory, DYNAMICS_TOPIC, 10)
        
        # Gains
        self.kp = 120.0 
        self.kd = 2 * np.sqrt(self.kp)

        # F. Control Loop (100 Hz = 0.01s)
        self.dt = 0.01
        self.create_timer(self.dt, self.control_loop)
        
        # Max Velocity for Smoothing (rad/s)
        # 1.0 rad/s is approx 60 degrees/sec. Adjust as needed.
        self.max_velocity = 1.0 
        
        self.get_logger().info("GUI Controller Ready. Waiting for Gazebo...")

    def joint_state_callback(self, msg):
        if not self.state_received:
            self.get_logger().info("Connected to Gazebo!")
            # Initialize smooth ref to current position to prevent startup jump
            # We need to map ROS msg indices to Pinocchio indices first
            temp_q = np.zeros(self.nq)
            for i, name in enumerate(msg.name):
                if self.model.existJointName(name):
                    idx = self.model.getJointId(name) - 1
                    if 0 <= idx < self.nq:
                        temp_q[idx] = msg.position[i]
            self.q_smooth_ref = temp_q
            self.state_received = True

        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                idx = j_id - 1
                if 0 <= idx < self.nq:
                    self.q_current[idx] = msg.position[i]
                    self.v_current[idx] = msg.velocity[i]

    def control_loop(self):
        if not self.state_received:
            return

        # --- 0. TRAJECTORY SMOOTHING (The Fix) ---
        # Instead of jumping instantly to q_target, we move q_smooth_ref towards it
        # by a small step every loop.
        
        step_limit = self.max_velocity * self.dt
        
        diff = self.q_target - self.q_smooth_ref
        
        # If distance is larger than one step, move by one step
        # If smaller, just snap to target
        dist = np.linalg.norm(diff)
        
        if dist > 0.0001:
            # Move towards target at constant speed
            direction = diff / dist
            step = min(dist, step_limit)
            self.q_smooth_ref += direction * step
        else:
            self.q_smooth_ref = self.q_target.copy()

        # --- 1. COMPUTED TORQUE CONTROL ---
        # USE q_smooth_ref instead of q_target
        q_des = self.q_smooth_ref
        v_des = np.zeros(self.nv) # Target velocity is 0 (holding the reference)
        a_des = np.zeros(self.nv)

        error_q = pin.difference(self.model, self.q_current, q_des)
        error_v = v_des - self.v_current
        
        u_raw = a_des + self.kp * error_q + self.kd * error_v
        
        # Clipping is still good for safety, but it won't be hit constantly now
        acc_max = 10.0 
        u_clipped = np.clip(u_raw, -acc_max, acc_max)
        
        # Calculate ideal torque
        tau_ideal = pin.rnea(self.model, self.data, self.q_current, self.v_current, u_clipped)
        
        # --- 2. TORQUE SATURATION ---
        tau_final = np.clip(tau_ideal, -self.model.effortLimit, self.model.effortLimit)
        
        # --- 3. FORWARD DYNAMICS (For Data Collection) ---
        pin.aba(self.model, self.data, self.q_current, self.v_current, tau_final)
        a_realized = self.data.ddq 

        # --- 4. PUBLISH ---
        ordered_tau = [tau_final[i] for i in self.output_map]
        
        cmd_msg = Float64MultiArray()
        cmd_msg.data = ordered_tau
        self.torque_pub.publish(cmd_msg)

        # Publish Dynamics Data
        ordered_q = [self.q_current[i] for i in self.output_map]
        ordered_v = [self.v_current[i] for i in self.output_map]
        ordered_a = [a_realized[i] for i in self.output_map] 
        
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = CONTROLLER_JOINT_ORDER
        
        point = JointTrajectoryPoint()
        point.positions = ordered_q
        point.velocities = ordered_v
        point.accelerations = ordered_a  
        point.effort = ordered_tau
        traj_msg.points = [point]
        self.dynamics_pub.publish(traj_msg)

# --- GUI CODE REMAINS SAME ---
def run_gui(node):
    root = tk.Tk()
    root.title(f"ROS 2 Control: {node.model.name}")
    root.geometry("400x600")

    ttk.Label(root, text="GAZEBO TORQUE CONTROL", font=("Arial", 12, "bold")).pack(pady=10)

    sliders = []
    
    def on_submit():
        new_q = np.zeros(node.nq)
        for i in range(node.nq):
            deg = sliders[i].get()
            new_q[i] = np.deg2rad(deg)
        node.q_target = new_q
        # We DO NOT update q_smooth_ref here. The loop will handle the interpolation.
        print(f"Target Updated: {np.round(new_q, 2)}")

    for i in range(node.nq):
        joint_name = node.model.names[i+1] if i+1 < len(node.model.names) else f"Joint {i}"
        lbl = ttk.Label(root, text=f"{joint_name} (deg)")
        lbl.pack(pady=5)
        
        scale = tk.Scale(root, from_=-180, to=180, orient=tk.HORIZONTAL, length=300)
        scale.set(0)
        scale.pack()
        sliders.append(scale)

    btn = ttk.Button(root, text="MOVE ROBOT", command=on_submit)
    btn.pack(pady=20)

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

if __name__ == '__main__':
    main()
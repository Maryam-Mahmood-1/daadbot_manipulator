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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin

# --- 2. CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_2.urdf"
COMMAND_TOPIC = '/effort_arm_controller/commands' 

# IMPORTANT: Must match your controllers.yaml
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
        
        # B. Internal Robot State (from Gazebo)
        self.q_current = np.zeros(self.nq)
        self.v_current = np.zeros(self.nv)
        self.state_received = False

        # C. Control Targets (Set by GUI)
        # Initialize target to "Neutral" or "Zero"
        self.q_gui_target = np.zeros(self.nq) 
        self.trajectory_active = False
        self.traj_start_time = 0.0

        # D. Joint Mapping (Controller -> Pinocchio)
        self.output_map = []
        for name in CONTROLLER_JOINT_ORDER:
            if self.model.existJointName(name):
                # Pinocchio ID - 1 = q index
                idx = self.model.getJointId(name) - 1
                self.output_map.append(idx)
            else:
                self.get_logger().error(f"Joint '{name}' not found in URDF!")

        # E. ROS Interface
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
        # Gains
        self.kp = 100.0 
        self.kd = 2 * np.sqrt(self.kp)

        # F. Control Loop (1 kHz)
        self.dt = 0.001
        self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("GUI Controller Ready. Waiting for Gazebo...")

    def joint_state_callback(self, msg):
        """Read real state from Gazebo"""
        if not self.state_received:
            self.get_logger().info("Connected to Gazebo!")
            self.state_received = True

        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                j_id = self.model.getJointId(name)
                idx = j_id - 1
                if 0 <= idx < self.nq:
                    self.q_current[idx] = msg.position[i]
                    self.v_current[idx] = msg.velocity[i]

    def control_loop(self):
        """Physics Loop: Calculates Torque based on GUI targets"""
        if not self.state_received:
            return

        # 1. Determine Desired State (q_des, v_des, a_des)
        q_des = self.q_gui_target.copy()
        v_des = np.zeros(self.nv)
        a_des = np.zeros(self.nv)

        if self.trajectory_active:
            # Sine Wave Logic
            t = time.time() - self.traj_start_time
            omega = 1.0  # rad/s
            amp = 0.5    # rad
            
            for i in range(self.nq):
                phase = i * 0.5
                # Center oscillation around the slider position
                q_des[i] += amp * math.sin(omega * t + phase)
                v_des[i] = amp * omega * math.cos(omega * t + phase)
                a_des[i] = -amp * (omega**2) * math.sin(omega * t + phase)

        # 2. Computed Torque Control Law
        error_q = pin.difference(self.model, self.q_current, q_des)
        error_v = v_des - self.v_current
        
        u = a_des + self.kp * error_q + self.kd * error_v
        
        # 3. Inverse Dynamics (RNEA)
        tau = pin.rnea(self.model, self.data, self.q_current, self.v_current, u)
        
        # 4. Map & Publish
        ordered_tau = [tau[idx] for idx in self.output_map]
        
        msg = Float64MultiArray()
        msg.data = ordered_tau
        self.torque_pub.publish(msg)


# --- 3. GUI SETUP ---
def run_gui(node):
    root = tk.Tk()
    root.title(f"ROS 2 Control: {node.model.name}")
    root.geometry("400x700")

    ttk.Label(root, text="GAZEBO TORQUE CONTROL", font=("Arial", 12, "bold")).pack(pady=10)

    # Sliders
    sliders = []
    
    def on_slider_update(val=None):
        # Update the ROS Node's target variable in real-time
        new_q = np.zeros(node.nq)
        for i in range(node.nq):
            deg = sliders[i].get()
            new_q[i] = np.deg2rad(deg)
        node.q_gui_target = new_q

    # Create a slider for each joint
    for i in range(node.nq):
        # Clean name logic
        joint_name = node.model.names[i+1] if i+1 < len(node.model.names) else f"Joint {i}"
        
        lbl = ttk.Label(root, text=f"{joint_name} (deg)")
        lbl.pack(pady=2)
        
        scale = tk.Scale(root, from_=-180, to=180, orient=tk.HORIZONTAL, length=300, command=on_slider_update)
        scale.set(0)
        scale.pack()
        sliders.append(scale)

    # Button Logic
    def toggle_trajectory():
        if not node.trajectory_active:
            node.traj_start_time = time.time()
            node.trajectory_active = True
            btn.config(text="STOP SINE WAVE (Return to Sliders)")
        else:
            node.trajectory_active = False
            btn.config(text="START SINE WAVE (Dance!)")

    ttk.Separator(root, orient='horizontal').pack(fill='x', pady=20)
    btn = ttk.Button(root, text="START SINE WAVE (Dance!)", command=toggle_trajectory)
    btn.pack(pady=10, ipady=10)

    ttk.Label(root, text="Note: Sine wave centers around\ncurrent slider positions.", justify=tk.CENTER).pack()

    # Start GUI Loop
    root.mainloop()


# --- 4. MAIN EXECUTION ---
def main(args=None):
    rclpy.init(args=args)
    
    # Create Node
    node = PinocchioRosNode()
    
    # Run ROS in a separate thread so GUI doesn't freeze
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    try:
        # Run GUI in main thread
        run_gui(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
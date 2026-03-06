#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# --- SINDY PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq = dq[0]**2, dq[1]**2
        dq_cross = dq[0] * dq[1]
        
        H_x = np.array([
            1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
            dq0_sq, dq1_sq, dq_cross,
            dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
            dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
            np.sign(dq[0]), np.sign(dq[1])
        ])
        
        G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
        a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
        b_hat = np.zeros((2, 2))
        b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
        b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
        return a_hat, b_hat

ALL_JOINTS = ["baseHinge", "interArm"]

class TaskStabilizationNode(Node):
    def __init__(self):
        super().__init__('task_stabilization_node')
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        
        try:
            self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
        except FileNotFoundError:
            self.get_logger().error("SINDy weight files not found at " + ws_path)
            sys.exit(1)
            
        self.q_quantile = self.sindy.q_quantile
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        self.start_time = None
        self.q = np.array([0.0, 0.1])
        self.dq = np.zeros(2)
        self.tau_limits = np.array([500.0, 300.0]) 
        self.target_pos = np.array([0.8, 0.2])
        
        self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[]}
        self.V0 = None
        self.lock = threading.Lock()

    def cb_joints(self, msg):
        q_buf, dq_buf = [None]*2, [None]*2
        for i, name in enumerate(ALL_JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                q_buf[i] = msg.position[idx]
                dq_buf[i] = msg.velocity[idx]
        
        if all(v is not None for v in q_buf):
            with self.lock:
                self.q = np.arctan2(np.sin(q_buf), np.cos(q_buf))
                self.dq = np.array(dq_buf)

    def compute_kinematics(self, q, dq):
        L1, L2 = 0.75, 1.0
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        x, y = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12
        J = np.array([[-L1*s1 - L2*s12, -L2*s12], [ L1*c1 + L2*c12,  L2*c12]])
        dJ = np.array([[-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])], 
                       [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]])
        return np.array([x, y]), J @ dq, J, dJ

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time
        
        with self.lock:
            q, dq = self.q.copy(), self.dq.copy()

        x_task, dx_task, J, dJ = self.compute_kinematics(q, dq)
        ref_pos, ref_vel, ref_acc = self.target_pos, np.zeros(2), np.zeros(2)

        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, ref_pos, ref_vel)
        
        LfV, LgV, V, gamma, robust_clf_term = self.clf_ctrl.get_lyapunov_constraints(
            x_task, dx_task, ref_pos, ref_vel, u_nom, q_quantile=self.q_quantile, J=J
        )

        if self.V0 is None: self.V0 = V
        v_bound = self.V0 * np.exp(-gamma * t_clock)

        # Safety Constraints
        A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), q_quantile=self.q_quantile)
        cbf_A, cbf_b = A_3d[:, :2], b_3d

        mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=robust_clf_term, cbf_A=cbf_A, cbf_b=cbf_b)

        if feasible:
            b_inv = np.linalg.pinv(b_hat)
            manipulability = np.sqrt(max(0, np.linalg.det(J @ J.T)))
            epsilon = 0.05 * (1 - (manipulability / 0.1)**2) if manipulability < 0.1 else 1e-6
            J_pinv = J.T @ np.linalg.inv(J @ J.T + epsilon * np.eye(2))
            tau_cmd = b_inv @ (J_pinv @ (u_nom + mu - dJ @ dq) - a_hat)
        else:
            tau_cmd = -20.0 * dq 

        self.pub.publish(Float64MultiArray(data=np.clip(tau_cmd, -self.tau_limits, self.tau_limits).tolist()))
        
        with self.lock:
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['V'].append(V)
            self.log['V_bound'].append(v_bound)
            self.log['mu'].append(np.linalg.norm(mu))
            self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]*2))

def main():
    rclpy.init()
    node = TaskStabilizationNode()
    t_ros = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    t_ros.start()
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target')
    
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbf.radii[0], node.cbf.radii[1], node.cbf.power_n
    x_cage = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
    y_cage = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
    ax_traj.plot(x_cage, y_cage, 'k--', alpha=0.3, label='Safety Cage')
    
    ax_traj.set_xlim(-1.5, 1.5); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True); ax_traj.legend()
    
    ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$')
    ln_vb, = ax_v.plot([], [], 'r--', alpha=0.8, label='Theoretical Bound')
    ax_v.set_title("Lyapunov Stability"); ax_v.grid(True); ax_v.legend()
    
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction Magnitude $||\mu||$"); ax_mu.grid(True)

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu
            t, x, y, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'V', 'V_bound', 'mu']]
        
        ln_a.set_data(x, y)
        ln_v.set_data(t, v); ln_vb.set_data(t, vb)
        ln_mu.set_data(t, mu)
        
        ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
        ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2)
        ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
        return ln_a, ln_v, ln_vb, ln_mu

    try:
        ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt: pass
    finally:
        node.stop_robot()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
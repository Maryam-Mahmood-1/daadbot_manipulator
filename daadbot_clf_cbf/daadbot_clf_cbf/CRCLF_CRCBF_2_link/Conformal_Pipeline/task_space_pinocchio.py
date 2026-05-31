#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# --- SINDY PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1 = np.sin(q[0]), np.cos(q[0]); s2, c2 = np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
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

class TaskStabilizationPinocchioNode(Node):
    def __init__(self):
        super().__init__('task_stabilization_pinocchio_node')
        
        # 1. SETUP PINOCCHIO MODELS
        urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.ee_id = self.model.getFrameId("endEffector")
        
        self.model_phys = pin.buildModelFromUrdf(urdf_path)
        self.data_phys = self.model_phys.createData()

        # 2. CONTROLLER SETUP
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
        # 3. HIGH-POWER STATE INITIALIZATION
        self.q_sim = np.array([0.0, 0.1])
        self.v_sim = np.zeros(self.model_phys.nv)
        self.tau_command = np.zeros(2)
        self.tau_limits = np.array([20.0, 10.0]) # High torque limits
        self.target_pos = np.array([0.8, 0.2])
        self.dt_phys = 0.001  # True 1kHz Physics Rate
        
        # 4. TUNING FOR HIGH TORQUE
        self.error_int = np.zeros(2)
        self.ki = 0.5  # Reduced integrator gain
        self.kd_extra = 3.0  # Significantly increased damping to handle 500Nm inertia
        self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[]}
        self.V0 = None
        self.V_bound_running = None
        
        self.lock = threading.Lock()
        self.running = True
        
        # 5. START THREADS
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()
        self.control_timer = self.create_timer(0.01, self.control_loop) 
        self.start_time = None

    def physics_loop(self):
        """ Internal 1kHz Physics Simulator with simulated joint damping """
        next_tick = time.time()
        while self.running:
            with self.lock:
                tau = self.tau_command.copy()
            
            # Simulated joint damping (mimics motor friction missing in Pinocchio)
            physical_damping = -2.5 * self.v_sim
            tau_total = tau + physical_damping

            ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_total)
            self.v_sim += ddq * self.dt_phys
            self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            
            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def compute_kinematics(self, q, v):
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        x_task = self.data.oMf[self.ee_id].translation[:2]
        J_full = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:2, :]
        dj_dq = pin.getFrameAcceleration(self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
        return x_task, J, dj_dq

    def control_loop(self):
        """ Control Logic with Anti-Windup """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            q, dq = self.q_sim.copy(), self.v_sim.copy()

        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        x_task, J, dj_dq = self.compute_kinematics(q, dq)
        dx_task = J @ dq

        # 1. NOMINAL CONTROL WITH ANTI-WINDUP
        error = x_task - self.target_pos
        self.error_int += error * 0.01  
        self.error_int = np.clip(self.error_int, -0.5, 0.5) # Anti-windup clamping
        
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, self.target_pos, np.zeros(2))
        u_nom -= (self.ki * self.error_int + self.kd_extra * dx_task)

        # 2. CONSTRAINTS (RESCLF)
        LfV, LgV, V, gamma_clf, robust = self.clf_ctrl.get_lyapunov_constraints(
            x_task, dx_task, self.target_pos, np.zeros(2), u_nom, self.sindy.q_quantile, J
        )

        # 3. CUMULATIVE BOUND
        if self.V0 is None: self.V0 = V; self.V_bound_running = V
        self.V_bound_running *= np.exp(-gamma_clf * 0.01)

        # 4. OPTIMIZATION
        A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), self.sindy.q_quantile)
        mu, feasible = solve_optimization(LfV, LgV, V, gamma_clf, robust, A_3d[:, :2], b_3d)

        if feasible:
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
            ddq_des = J_pinv @ (u_nom + mu - dj_dq)
            tau_cmd = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
        else:
            tau_cmd = -40.0 * dq # Aggressive braking fallback

        with self.lock:
            self.tau_command = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
            
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock); self.log['V'].append(V); self.log['V_bound'].append(self.V_bound_running)
            self.log['mu'].append(np.linalg.norm(mu)); self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])

def main():
    rclpy.init()
    node = TaskStabilizationPinocchioNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual (Sim)')
    ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target')
    ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True)
    
    ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$'); ln_vb, = ax_v.plot([], [], 'r--', label='Theoretical Bound')
    ax_v.set_title("Lyapunov Stability Comparison"); ax_v.grid(True); ax_v.legend()
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title(r"Robust Correction $||\mu||$")

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu
            t, x, y, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'V', 'V_bound', 'mu']]
        ln_a.set_data(x, y); ln_v.set_data(t, v); ln_vb.set_data(t, vb); ln_mu.set_data(t, mu)
        ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
        ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2); ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
        return ln_a, ln_v, ln_vb, ln_mu

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    node.running = False
    node.phys_thread.join()
    rclpy.shutdown()

if __name__ == '__main__': main()

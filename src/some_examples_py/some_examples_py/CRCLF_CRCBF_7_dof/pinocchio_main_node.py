#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 
import os

# --- MODULAR IMPORTS ---
# Ensure these packages are in your PYTHONPATH
from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

import os
from ament_index_python.packages import get_package_share_directory

# NOISY PHYSICS: Points to the noisy URDF version
URDF_NOISY = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf",
    "2link_robot_noisy_3.urdf"
)

# TRUE PHYSICS: Points to the original URDF (The ground truth)
URDF_TRUE = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf",
    "2link_robot.urdf"
)
EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class PinocchioConformalRobustNode(Node):
    def __init__(self):
        super().__init__('pinocchio_cr_node')
        
        # --- 1. CONFORMAL PARAMETERS ---
        # Increasing this value makes the robot more "cautious" about its noisy model
        self.q_quantile = 30000.0  
        
        # --- 2. CONTROLLER SETUP (Uses Noisy Model) ---
        self.robot_ctrl = RobotDynamics(URDF_NOISY, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # Safety Barrier (Super-Ellipsoid)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], 
            lengths=[1.1, 1.1, 3.0], 
            power_n=4, k_pos=21.0, k_vel=12.0
        )
        self.cbf_active = False 

        # --- 3. PINOCCHIO PHYSICS ENGINE SETUP (Uses True Model) ---
        self.model_phys = pin.buildModelFromUrdf(URDF_TRUE)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

        # --- 4. STATE INITIALIZATION ---
        self.q_sim = pin.neutral(self.model_phys)
        self.v_sim = np.zeros(self.model_phys.nv)
        
        # Start with a slight joint offset to avoid singularity
        self.q_sim[self.model_phys.joints[self.phys_joint_ids[1]].idx_q] = 0.1

        self.tau_command = np.zeros(2)
        self.tau_limits = np.array([40.0, 30.0]) 
        self.lock = threading.Lock()

        # --- 5. THREADING & ROS ---
        self.running = True
        self.dt_phys = 0.001 # 1kHz Physics
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        self.control_rate = 100.0 # 100Hz Control
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        
        # --- 6. LOGGING ---
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def physics_loop(self):
        """ The High-Frequency Simulator (True Physics) """
        print("--- Pinocchio True Physics Thread Started ---")
        next_tick = time.time()
        
        while self.running:
            with self.lock:
                current_tau = self.tau_command.copy()

            tau_full = np.zeros(self.model_phys.nv)
            damping = 0.15 * self.v_sim 
            
            for i, jid in enumerate(self.phys_joint_ids):
                idx_v = self.model_phys.joints[jid].idx_v
                tau_full[idx_v] = current_tau[i] - damping[idx_v]

            try:
                # Forward Dynamics (ABA) and Numerical Integration
                ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                self.v_sim += ddq * self.dt_phys
                self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            except Exception as e:
                print(f"Physics Integration Error: {e}")

            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def control_loop(self):
        """ The Robust Controller (Using Noisy Model) """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # Extract state for the controller
        with self.lock:
            q_curr = np.array([self.q_sim[self.model_phys.joints[jid].idx_q] for jid in self.phys_joint_ids])
            dq_curr = np.array([self.v_sim[self.model_phys.joints[jid].idx_v] for jid in self.phys_joint_ids])

        # A. DYNAMICS (Using NOISY Internal Model)
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(q_curr, dq_curr)
        J = J[0:2, :]; dJ = dJ[0:2, :]; x_2d = x[0:2]; dx_2d = dx[0:2]

        # B. TRAJECTORY
        xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
        xd, vd, ad = xd_f[:2], vd_f[:2], ad_f[:2]

        # C. CR-CLF (Robust Lyapunov)
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
        u_ref = ad + u_nom
        LfV, LgV, V, gamma, robust_clf_term = self.clf_ctrl.get_lyapunov_constraints(
            x_2d, dx_2d, xd, vd, q_quantile=self.q_quantile
        )

        # D. CR-CBF (Robust Safety)
        cbf_A, cbf_b = None, None
        x_3d = np.array([x_2d[0], x_2d[1], 0.0])
        h_val = self.cbf.get_h_value(x_3d)

        if self.cbf_active:
            dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
            u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
            # Pass quantile to robustify the barrier boundary
            A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.q_quantile/10.0)
            cbf_A = A_temp[:, :2] 
            cbf_b = b_temp

        # E. QP SETUP
        J_pinv = np.linalg.pinv(J)
        drift_acc = u_ref - (dJ @ dq_curr)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        # F. SOLVE OPTIMIZATION
        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            robust_clf_term=robust_clf_term, 
            torque_A=A_tau, torque_b=b_tau, 
            cbf_A=cbf_A, cbf_b=cbf_b
        )

        # G. CALCULATE FINAL TORQUE
        if feasible:
            acc_cmd = u_ref + mu 
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_curr))) + nle
        else:
            tau_out = -8.0 * dq_curr + nle # Emergency damping

        # Apply to Physics Engine
        with self.lock:
            self.tau_command = np.clip(tau_out, -self.tau_limits, self.tau_limits)
            
            # H. LOGGING
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            self.log['h'].append(h_val)
            self.log['mu'].append(np.linalg.norm(mu))

def main(args=None):
    rclpy.init(args=args)
    node = PinocchioConformalRobustNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])
    ax_mu = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual (True Physics)') 
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target (Noisy Model Ref)')
    
    # Plot Super-Ellipsoid Boundary
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    n = node.cbf.power_n
    x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(x_b, y_b, 'g-', label='Safe Region')
    
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
    ax_traj.set_aspect('equal'); ax_traj.grid(True); ax_traj.legend()

    ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--')
    ax_h.set_title("Safety Margin h(x)"); ax_h.grid(True)
    
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Robust Correction ||μ||")
    ax_mu.grid(True)

    # UI Widgets
    ax_check = plt.axes([0.05, 0.05, 0.2, 0.06]) 
    check = CheckButtons(ax_check, ['Activate CR-CBF'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if not node.log['t']: return ln_a, ln_t, ln_h, ln_mu
            t_d = list(node.log['t'])
            x_d, y_d = list(node.log['x']), list(node.log['y'])
            xd_d, yd_d = list(node.log['xd']), list(node.log['yd'])
            h_d, mu_d = list(node.log['h']), list(node.log['mu'])

        ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
        ax_h.set_xlim(t_d[0], t_d[-1]); ax_mu.set_xlim(t_d[0], t_d[-1])
        ax_h.set_ylim(-1.0, 1.2); ax_mu.set_ylim(-2.0, 30.0)
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
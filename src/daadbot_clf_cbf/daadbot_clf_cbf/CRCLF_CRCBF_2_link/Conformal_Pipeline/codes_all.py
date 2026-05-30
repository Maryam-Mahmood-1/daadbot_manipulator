#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        """Reconstructs dynamics using SINDy learned coefficients."""
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
        
        # 1. MODELS
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
        
        # 3. CONSTRAINTS & TARGET
        self.q_sim = np.array([0.0, 0.1])
        self.v_sim = np.zeros(self.model_phys.nv)
        self.tau_command = np.zeros(2)
        self.tau_limits = np.array([20.0, 10.0]) 
        self.target_pos = np.array([0.8, 0.2])
        self.dt_phys = 0.001 
        
        # Log includes grad_norm for diagnostic plotting
        self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[], 'grad_norm':[]}
        self.V0 = None
        self.V_bound_running = None
        
        self.lock = threading.Lock()
        self.running = True
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()
        self.control_timer = self.create_timer(0.01, self.control_loop) 
        self.start_time = None

    def physics_loop(self):
        next_tick = time.time()
        while self.running:
            with self.lock:
                tau = self.tau_command.copy()
            physical_damping = -2.5 * self.v_sim 
            tau_total = tau + physical_damping
            ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_total)
            self.v_sim += ddq * self.dt_phys
            self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            next_tick += self.dt_phys
            time.sleep(max(0, next_tick - time.time()))

    def compute_kinematics(self, q, v):
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        x_task = self.data.oMf[self.ee_id].translation[:2]
        J_full = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:2, :]
        dj_dq = pin.getFrameAcceleration(self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
        return x_task, J, dj_dq

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            q, dq = self.q_sim.copy(), self.v_sim.copy()

        # Reconstruct Dynamics
        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        x_task, J, dj_dq = self.compute_kinematics(q, dq)
        dx_task = J @ dq
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))

        # 1. NOMINAL ACCELERATION (RES-CLF)
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, self.target_pos)

        # 2. CONFORMAL ROBUST CONSTRAINTS
        LfV, LgV, V, gamma_clf, robust = self.clf_ctrl.get_lyapunov_constraints(
            x_task, dx_task, self.target_pos, u_nom, self.sindy.q_quantile, J
        )

        # Calculate Gradient Magnitude for Diagnostics
        eta = np.hstack((x_task - self.target_pos, dx_task)).reshape(-1, 1)
        grad_V = 2 * self.clf_ctrl.P @ eta
        grad_norm = np.linalg.norm(grad_V)

        # 3. SAFETY CONSTRAINTS
        A_cbf, b_cbf = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), self.sindy.q_quantile)

        # 4. SOLVER
        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma_clf, a_hat, b_hat, u_nom, dj_dq, J_pinv, 
            self.tau_limits, robust, A_cbf[:, :2], b_cbf
        )

        if feasible:
            ddq_des = J_pinv @ (u_nom + mu - dj_dq)
            tau_cmd = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
        else:
            tau_cmd = -25.0 * dq 

        # 5. EXPONENTIAL BOUND TRACKING
        if self.V0 is None: self.V0 = V; self.V_bound_running = V
        self.V_bound_running *= np.exp(-gamma_clf * 0.01)

        with self.lock:
            self.tau_command = tau_cmd 
            self.log['t'].append(t_clock); self.log['V'].append(V); self.log['V_bound'].append(self.V_bound_running)
            self.log['mu'].append(np.linalg.norm(mu)); self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])
            self.log['grad_norm'].append(grad_norm)

def main():
    rclpy.init()
    node = TaskStabilizationPinocchioNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # 3x2 GridSpec for aligned diagnostic plots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_v    = fig.add_subplot(gs[0, 1])
    ax_mu   = fig.add_subplot(gs[1, 1])
    ax_grad = fig.add_subplot(gs[2, 1])   
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual Path')
    ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target')
    ax_traj.set_xlim(-1.5, 1.5); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True); ax_traj.legend()
    
    ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$')
    ln_vb, = ax_v.plot([], [], 'r--', label='Robust Bound')
    ax_v.set_title("CR-CLF Stability [cite: 150]"); ax_v.grid(True); ax_v.legend()
    
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title(r"Robust Correction $\|\mu\|$ [cite: 587]")
    ax_mu.grid(True)
    
    ln_grad, = ax_grad.plot([], [], 'b-'); ax_grad.set_title(r"Lyapunov Gradient Norm $\|\nabla V\|$")
    ax_grad.set_xlabel("Time (s)"); ax_grad.grid(True)

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu, ln_grad
            t = np.array(node.log['t'])
            x, y = np.array(node.log['x']), np.array(node.log['y'])
            v, vb = np.array(node.log['V']), np.array(node.log['V_bound'])
            mu = np.array(node.log['mu'])
            gn = np.array(node.log['grad_norm'])
            
        ln_a.set_data(x, y); ln_v.set_data(t, v); ln_vb.set_data(t, vb); ln_mu.set_data(t, mu)
        ln_grad.set_data(t, gn)
        
        for ax in [ax_v, ax_mu, ax_grad]: ax.set_xlim(t[0], t[-1])
        ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2)
        ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.1)
        ax_grad.set_ylim(0, np.nanmax(gn)*1.1 + 0.1)
        return ln_a, ln_v, ln_vb, ln_mu, ln_grad

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    node.running = False
    node.phys_thread.join()
    rclpy.shutdown()

if __name__ == '__main__': main()
















import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Controller:
    def __init__(self, dim_task=2):
        self.dim = dim_task
        # Linear error dynamics in normal form [cite: 473, 475]
        zero, eye = np.zeros((dim_task, dim_task)), np.eye(dim_task)
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # Stabilization gains
        self.kp, self.kd = 15.0, 15.0 # Increased for stiff stabilization
        
        # Lyapunov matrix P via ARE [cite: 478, 479]
        q_pos, q_vel = 300.0, 210.0 
        self.Q_mat = np.diag([q_pos]*dim_task + [q_vel]*dim_task)
        self.R_mat = np.eye(dim_task) * 0.05 
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, self.R_mat)
        
        eig_Q = np.min(np.linalg.eigvals(self.Q_mat).real)
        eig_P = np.max(np.linalg.eigvals(self.P).real)
        
        # Stability decay rate [cite: 486]
        self.gamma = (eig_Q / eig_P) 

    def get_nominal_acceleration(self, x, dx, x_target):
        """Pure stabilization: target velocity is zero."""
        error_pos = x_target - x
        error_vel = -dx # Target dx is 0
        return (self.kp * error_pos) + (self.kd * error_vel)

    def get_lyapunov_constraints(self, x, dx, x_target, u_nom, q_quantile=0.0, J=None):
        # Error state eta = [pos_err; vel_err] [cite: 472]
        eta = np.hstack((x - x_target, dx)).reshape(-1, 1)
        V = (eta.T @ self.P @ eta)[0, 0]
        
        # Lie Derivatives [cite: 482]
        LfV_open = (eta.T @ (self.P @ self.F + self.F.T @ self.P) @ eta)[0, 0]
        LgV = 2 * eta.T @ self.P @ self.G
        
        # Robustness term from Conformal Prediction 
        # Multiplied by J to map task uncertainty to joint leverage
        grad_V_actuated = (2 * self.P @ eta)[self.dim:, 0]
        mult_mat = J if J is not None else np.eye(self.dim)
        
        # The term ||dV/dx|| * q 
        robustness_cost = np.linalg.norm(grad_V_actuated @ mult_mat) * q_quantile
        
        # Closed-loop LfV including nominal PID-like u_nom
        LfV_closed = LfV_open + (LgV @ u_nom.reshape(-1, 1))[0, 0]
            
        return LfV_closed, LgV, V, self.gamma, robustness_cost
    














import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, a_hat, b_hat, u_nom, dj_dq, J_pinv, 
                       tau_lims, robust_clf_term=0.0, cbf_A=None, cbf_b=None):
    # 1. Numerical Sanity Check
    if any(np.isnan([LfV, V, robust_clf_term])): return np.zeros(2), False

    u_dim = 2 
    num_vars = u_dim + 1 # [mu_x, mu_y, delta]
    
    # 2. Balanced Cost Function
    # We use a smaller penalty initially to avoid 'domain error' from matrix ill-conditioning
    slack_penalty = 15 
    P = cvxopt.matrix(np.diag([1.0, 1.0, slack_penalty]))
    q = cvxopt.matrix(np.zeros(num_vars))

    G_list, h_list = [], []

    # 3. RES-CLF Constraint (Murtaza Eq. 16 + Hsu Robustness Eq. 4)
    # LgV*mu - delta <= -gamma*V - LfV - robust
    clf_row = np.zeros((1, num_vars))
    clf_row[0, :u_dim] = LgV.flatten()
    clf_row[0, -1] = -1.0 # The relaxation delta
    G_list.append(clf_row)
    h_list.append(np.array([[-gamma * V - LfV - robust_clf_term]]))

    # 4. Torque Constraints (Mapping mu -> tau)
    # b_hat is the learned control matrix from SINDy
    try:
        b_inv = np.linalg.pinv(b_hat)
        mapping_mu_to_tau = b_inv @ J_pinv
        const_tau_terms = b_inv @ (J_pinv @ (u_nom - dj_dq) - a_hat)

        # Torque Upper Bound
        G_tau_upper = np.zeros((2, num_vars))
        G_tau_upper[:, :u_dim] = mapping_mu_to_tau
        G_list.append(G_tau_upper)
        h_list.append((tau_lims - const_tau_terms).reshape(-1, 1))

        # Torque Lower Bound
        G_tau_lower = np.zeros((2, num_vars))
        G_tau_lower[:, :u_dim] = -mapping_mu_to_tau
        G_list.append(G_tau_lower)
        h_list.append((tau_lims + const_tau_terms).reshape(-1, 1))
    except np.linalg.LinAlgError:
        return np.zeros(2), False

    # 5. Solve
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-7
    try:
        sol = cvxopt.solvers.qp(P, q, cvxopt.matrix(np.vstack(G_list)), cvxopt.matrix(np.vstack(h_list)))
        if sol['status'] == 'optimal':
            res = np.array(sol['x']).flatten()
            return res[:u_dim], True
    except:
        pass
        
    return np.zeros(u_dim), False



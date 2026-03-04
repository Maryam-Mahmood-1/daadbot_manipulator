#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RadioButtons
import os
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
import sys

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

# --- SINDY PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1, s2, c2 = np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])
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

class SINDyConformalNode(Node):
    def __init__(self):
        super().__init__('sindy_conformal_node')
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        
        # --- 1. MODEL SETUP ---
        try:
            self.sindy = SINDyPredictor(
                os.path.join(ws_path, "sindy_Xi_state_space2.npy"), 
                os.path.join(ws_path, "q_quantile_state_space2.txt")
            )
        except FileNotFoundError:
            self.get_logger().error("SINDy files missing!")
            sys.exit(1)
        
        urdf_true = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
        self.model_phys = pin.buildModelFromUrdf(urdf_true)
        self.data_phys = self.model_phys.createData()
        
        urdf_noisy = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_noisy_3.urdf")
        self.model_ctrl = pin.buildModelFromUrdf(urdf_noisy)
        self.data_ctrl = self.model_ctrl.createData()
        self.ee_id = self.model_ctrl.getFrameId("endEffector")

        # --- 2. STATE ---
        self.q_sim = np.array([0.5, 0.5]) 
        self.v_sim = np.zeros(2)
        self.tau_cmd = np.zeros(2)
        self.tau_limits = np.array([500.0, 300.0])
        self.dt = 0.01 
        self.lock = threading.Lock()
        
        self.traj_gen = TrajectoryGenerator()
        self.clf_ctrl = RESCLF_Controller(dim_task=2)
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=20.0, k_vel=10.0)
        self.cbf_active = False
        self.active_mode = 'Task Track'
        
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[], 'V_bound':[]}
        self.V0 = None
        self.stab_start_t = 0.0

        # --- 3. LOOPS ---
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()
        self.timer = self.create_timer(self.dt, self.control_loop)

    def physics_loop(self):
        dt_p = 0.001
        while rclpy.ok():
            with self.lock:
                tau = self.tau_cmd.copy()
                q, v = self.q_sim.copy(), self.v_sim.copy()
            
            # Forward Dynamics (Internal Simulation)
            ddq = pin.aba(self.model_phys, self.data_phys, q, v, tau)
            v += ddq * dt_p
            q = pin.integrate(self.model_phys, q, v * dt_p)
            
            with self.lock:
                self.q_sim, self.v_sim = q, v
            time.sleep(dt_p)

    def generate_reference(self, t_clock, current_x_task):
        if self.active_mode == 'Joint Stab':
            return np.array([np.pi/4, -np.pi/4]), np.zeros(2), np.zeros(2)
        elif self.active_mode == 'Joint Track':
            return np.array([np.sin(t_clock), np.cos(t_clock)]), np.array([np.cos(t_clock), -np.sin(t_clock)]), np.array([-np.sin(t_clock), -np.cos(t_clock)])
        elif self.active_mode == 'Task Stab':
            return np.array([0.8, 0.2]), np.zeros(2), np.zeros(2)
        elif self.active_mode == 'Task Track':
            xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(current_x_task, (0,1)))
            return xd_f[:2], vd_f[:2], ad_f[:2]

    def control_loop(self):
        t_now = time.time()
        if not hasattr(self, 'init_t'): self.init_t = t_now
        t_clock = t_now - self.init_t

        with self.lock:
            q, v = self.q_sim.copy(), self.v_sim.copy()

        # Kinematics
        pin.forwardKinematics(self.model_ctrl, self.data_ctrl, q, v)
        pin.updateFramePlacements(self.model_ctrl, self.data_ctrl)
        x_task = self.data_ctrl.oMf[self.ee_id].translation[:2]
        J_full = pin.computeFrameJacobian(self.model_ctrl, self.data_ctrl, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:2, :]
        dx_task = J @ v
        acc_frame = pin.getFrameAcceleration(self.model_ctrl, self.data_ctrl, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dj_dq = acc_frame.linear[:2]

        # Reference
        ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
        is_task_space = 'Task' in self.active_mode
        curr_p, curr_v, J_clf = (x_task, dx_task, J) if is_task_space else (q, v, None)

        # Dynamics Prediction
        a_hat, b_hat = self.sindy.get_dynamics(q, v)
        u_nom = self.clf_ctrl.get_nominal_acceleration(curr_p, curr_v, ref_pos, ref_vel)

        # 1. CLF Constraints
        LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            curr_p, curr_v, ref_pos, ref_vel, u_nom, q_quantile=self.sindy.q_quantile, J=J_clf
        )

        # Exponential Bound Logic
        v_bound = np.nan
        if 'Stab' in self.active_mode:
            if self.V0 is None:
                self.V0 = V
                self.stab_start_t = t_clock
            v_bound = self.V0 * np.exp(-gamma * (t_clock - self.stab_start_t))
        else: self.V0 = None

        # 2. CBF Constraints
        cbf_A, cbf_b = None, None
        if self.cbf_active:
            u_ref = ref_acc + u_nom
            A_temp, b_temp = self.cbf.get_constraints(np.append(x_task,0), np.append(dx_task,0), np.append(u_ref,0), q_quantile=self.sindy.q_quantile)
            cbf_A, cbf_b = A_temp[:, :2], b_temp

        # 3. QP Solver
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=robust_term, cbf_A=cbf_A, cbf_b=cbf_b)

        if feasible:
            if is_task_space:
                J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
                ddq_des = J_pinv @ (ref_acc + u_nom + mu - dj_dq)
            else:
                ddq_des = ref_acc + u_nom + mu
            tau_out = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
        else:
            tau_out = -15.0 * v 

        with self.lock:
            self.tau_cmd = np.clip(tau_out, -self.tau_limits, self.tau_limits)
            
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])
            self.log['V'].append(V); self.log['V_bound'].append(v_bound)
            self.log['mu'].append(np.linalg.norm(mu))
            if is_task_space:
                self.log['xd'].append(ref_pos[0]); self.log['yd'].append(ref_pos[1])
            else:
                L1, L2 = 0.75, 1.0 # Approximate for visualization
                self.log['xd'].append(L1*np.cos(ref_pos[0]) + L2*np.cos(ref_pos[0]+ref_pos[1]))
                self.log['yd'].append(L1*np.sin(ref_pos[0]) + L2*np.sin(ref_pos[0]+ref_pos[1]))

def main():
    rclpy.init()
    node = SINDyConformalNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(bottom=0.25)
    
    ln_a, = ax_traj.plot([], [], 'r-', label='Actual EE')
    ln_t, = ax_traj.plot([], [], 'b--', label='Target Path')
    
    # Safety Boundary
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbf.radii[0], node.cbf.radii[1], node.cbf.power_n
    xc = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
    yc = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
    ax_traj.plot(xc, yc, 'k--', alpha=0.3, label='Cage')
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0); ax_traj.grid(True); ax_traj.legend()
    
    ln_v, = ax_v.plot([], [], 'g-', label='V(t)')
    ln_vb, = ax_v.plot([], [], 'r--', label='Theoretical')
    ax_v.set_title("Lyapunov Energy"); ax_v.legend(); ax_v.grid(True)
    
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction Force ||mu||"); ax_mu.grid(True)

    # UI Widgets
    radio = RadioButtons(plt.axes([0.05, 0.05, 0.15, 0.12]), ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
    check = CheckButtons(plt.axes([0.22, 0.05, 0.15, 0.05]), ['Safety Active'], [False])

    def switch_m(l):
        with node.lock:
            node.active_mode = l
            node.V0 = None
            for k in node.log: node.log[k] = []
    radio.on_clicked(switch_m)
    def toggle(l): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_t, ln_v, ln_vb, ln_mu
            t, x, y, xd, yd, v, vb, mu = [np.array(node.log[k]) for k in ['t','x','y','xd','yd','V','V_bound','mu']]
        
        ln_a.set_data(x, y); ln_t.set_data(xd, yd)
        ln_v.set_data(t, v); ln_vb.set_data(t, vb)
        ln_mu.set_data(t, mu)
        
        ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
        ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb) if not np.all(np.isnan(vb)) else 0) * 1.2 + 0.1)
        ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
        return ln_a, ln_t, ln_v, ln_vb, ln_mu

    try:
        ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt: pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
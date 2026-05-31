"""
Comparison: Standard (q=0) vs Conformally Robust (q>0) Controllers
Includes plots for Lyapunov Function V(x) and Gradient ||dV/dx||
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 
from ament_index_python.packages import get_package_share_directory

import os

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CRCLF_CRCBF.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CRCLF_CRCBF.utils.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CRCLF_CRCBF.crclf_formulation import RESCLF_Controller
from daadbot_clf_cbf.CRCLF_CRCBF.cr_qp_solver import solve_optimization
from daadbot_clf_cbf.CRCLF_CRCBF.crcbf_formulation import CBF_SuperEllipsoid 

# --- CONFIGURATIONS ---
URDF_PHYSICS = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)
URDF_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot_noisy_.urdf"
)

CALIBRATED_QUANTILE = 1000.0  # calibrated value from data 

EE_NAMES = ["gear1_claw", "gear2_claw"]
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
USE_JOINT_1 = False

class SimInstance:
    """ Holds state for ONE simulation instance """
    def __init__(self, name, urdf_phys, urdf_ctrl, quantile_val):
        self.name = name
        self.q_quantile = quantile_val 
        
        # 1. Controller (Noisy Model)
        self.robot_ctrl = RobotDynamics(urdf_ctrl, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4, k_pos=87.0, k_vel=60.0
        )
        
        # 2. Physics (Clean Model)
        self.model_phys = pin.buildModelFromUrdf(urdf_phys)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(n) for n in ALL_JOINTS]
        
        # 3. State
        q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.q_sim = pin.neutral(self.model_phys)
        self.v_sim = np.zeros(self.model_phys.nv)
        for i, jid in enumerate(self.phys_joint_ids):
             self.q_sim[self.model_phys.joints[jid].idx_q] = q_init[i]

        self.tau_cmd = np.zeros(7)
        # Added 'V' and 'gradV' to logs
        self.log = {'x':[], 'y':[], 'h':[], 'mu':[], 'V':[], 'gradV':[]} 

class RobustnessTestNode(Node):
    def __init__(self):
        super().__init__('robustness_test_node')
        
        self.sims = [
            SimInstance("Standard (q=0)", URDF_PHYSICS, URDF_CTRL, quantile_val=0.0),
            SimInstance(f"Robust (q={CALIBRATED_QUANTILE})", URDF_PHYSICS, URDF_CTRL, quantile_val=CALIBRATED_QUANTILE)
        ]
        
        self.traj_gen = TrajectoryGenerator()
        self.cbf_active = False 

        self.lock = threading.Lock()
        self.running = True
        self.dt_phys = 0.001
        self.log_t = [] 
        self.log_target = {'x':[], 'y':[]}

        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        self.control_rate = 200.0
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

    def physics_loop(self):
        next_tick = time.time()
        while self.running:
            with self.lock:
                for sim in self.sims:
                    self.step_single_physics(sim)
            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def step_single_physics(self, sim):
        tau_full = np.zeros(sim.model_phys.nv)
        damping = 0.3 * sim.v_sim 
        for i, jid in enumerate(sim.phys_joint_ids):
            idx_v = sim.model_phys.joints[jid].idx_v
            tau_full[idx_v] = sim.tau_cmd[i] - damping[idx_v]

        try:
            ddq = pin.aba(sim.model_phys, sim.data_phys, sim.q_sim, sim.v_sim, tau_full)
            sim.v_sim += ddq * self.dt_phys
            sim.q_sim = pin.integrate(sim.model_phys, sim.q_sim, sim.v_sim * self.dt_phys)
        except:
            pass

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            # Atomic Logging Cleanup
            if len(self.log_t) > 1500:
                self.log_t.pop(0)
                self.log_target['x'].pop(0); self.log_target['y'].pop(0)
                for sim in self.sims:
                    for key in sim.log: sim.log[key].pop(0)
            
            self.log_t.append(t_clock)

            for sim_idx, sim in enumerate(self.sims):
                # 1. State
                q_7dof = np.zeros(7); dq_7dof = np.zeros(7)
                for i, jid in enumerate(sim.phys_joint_ids):
                    q_7dof[i] = sim.q_sim[sim.model_phys.joints[jid].idx_q]
                    dq_7dof[i] = sim.v_sim[sim.model_phys.joints[jid].idx_v]

                # 2. Dynamics
                M, nle, J, dJ, x, dx = sim.robot_ctrl.compute_dynamics(q_7dof, dq_7dof, use_joint1=USE_JOINT_1)
                
                # 3. Ref
                xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)
                if sim_idx == 0:
                    self.log_target['x'].append(xd[0]); self.log_target['y'].append(xd[1])

                # 4. CLF & Gradient Calculation
                u_nom = sim.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
                u_ref = ad + u_nom
                
                LfV, LgV, V, gamma, robust_term = sim.clf_ctrl.get_lyapunov_constraints(
                    x, dx, xd, vd, q_quantile=sim.q_quantile
                )

                # --- EXPLICIT GRADIENT CALCULATION FOR PLOTTING ---
                # Recalculate eta to get the exact gradient used in the cost
                e = x - xd
                de = dx - vd
                eta = np.hstack((e, de)).reshape(-1, 1)
                # Gradient of V w.r.t eta: 2 * P * eta
                grad_V_full = 2 * (sim.clf_ctrl.P @ eta)
                # Actuated component (velocity part)
                grad_V_act = grad_V_full[sim.clf_ctrl.dim:, 0]
                grad_V_norm = np.linalg.norm(grad_V_act)

                # 5. CBF
                cbf_A, cbf_b = None, None
                h_val = sim.cbf.get_h_value(x)
                if self.cbf_active:
                    cbf_A, cbf_b = sim.cbf.get_constraints(x, dx, u_ref, q_quantile=sim.q_quantile/200.0)

                # 6. QP
                J_pinv = sim.robot_ctrl.get_pseudo_inverse(J)
                drift = u_ref - (dJ @ dq_7dof)
                b_tau_bias = (M @ J_pinv @ drift) + nle
                
                A_tau = np.vstack([M @ J_pinv, -M @ J_pinv])
                b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

                mu, feasible = solve_optimization(
                    LfV, LgV, V, gamma, 
                    robust_clf_term=robust_term, 
                    torque_A=A_tau, torque_b=b_tau, 
                    cbf_A=cbf_A, cbf_b=cbf_b
                )

                if feasible:
                    acc_cmd = u_ref + mu 
                    tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_7dof))) + nle
                else:
                    tau_out = -10.0 * dq_7dof + nle 

                if not USE_JOINT_1:
                    tau_out[0] = np.clip((-self.kp_lock * q_7dof[0]) - (self.kd_lock * dq_7dof[0]), -80, 80)
                
                sim.tau_cmd = np.clip(tau_out, -self.tau_limits, self.tau_limits)
                
                # Append Logs
                sim.log['x'].append(x[0]); sim.log['y'].append(x[1])
                sim.log['h'].append(h_val); sim.log['mu'].append(np.linalg.norm(mu))
                sim.log['V'].append(V); sim.log['gradV'].append(grad_V_norm)

def main(args=None):
    rclpy.init(args=args)
    node = RobustnessTestNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING LAYOUT ---
    # 2 Rows: Standard vs Robust
    # 5 Cols: Traj | V(x) | ||Grad V|| | h(x) | ||mu||
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 5)

    # --- Standard Row (Top) ---
    ax_s_traj  = fig.add_subplot(gs[0, 0])
    ax_s_V     = fig.add_subplot(gs[0, 1])
    ax_s_grad  = fig.add_subplot(gs[0, 2])
    ax_s_h     = fig.add_subplot(gs[0, 3])
    ax_s_mu    = fig.add_subplot(gs[0, 4])

    # --- Robust Row (Bottom) ---
    ax_r_traj  = fig.add_subplot(gs[1, 0])
    ax_r_V     = fig.add_subplot(gs[1, 1])
    ax_r_grad  = fig.add_subplot(gs[1, 2])
    ax_r_h     = fig.add_subplot(gs[1, 3])
    ax_r_mu    = fig.add_subplot(gs[1, 4])
    
    plt.subplots_adjust(bottom=0.1, wspace=0.35, hspace=0.4, left=0.05, right=0.98)

    # --- Initialize Lines ---
    # Trajectory
    ln_s_target, = ax_s_traj.plot([], [], 'b--', label='Target')
    ln_r_target, = ax_r_traj.plot([], [], 'b--', label='Target')
    ln_s_act, = ax_s_traj.plot([], [], 'r-', linewidth=2, label='Standard')
    ln_r_act, = ax_r_traj.plot([], [], 'g-', linewidth=2, label='Robust')

    # V(x)
    ln_s_V, = ax_s_V.plot([], [], 'r-', linewidth=1.5)
    ln_r_V, = ax_r_V.plot([], [], 'g-', linewidth=1.5)

    # Grad V
    ln_s_grad, = ax_s_grad.plot([], [], 'r-', linewidth=1.5)
    ln_r_grad, = ax_r_grad.plot([], [], 'g-', linewidth=1.5)

    # h(x)
    ln_s_h, = ax_s_h.plot([], [], 'r-')
    ln_r_h, = ax_r_h.plot([], [], 'g-')

    # mu
    ln_s_mu, = ax_s_mu.plot([], [], 'k-')
    ln_r_mu, = ax_r_mu.plot([], [], 'k-')

    # --- Decoration ---
    # Traj
    cbf = node.sims[0].cbf
    theta = np.linspace(0, 2*np.pi, 200)
    st = np.sin(theta); ct = np.cos(theta)
    n = cbf.power_n
    x_b = cbf.center[0] + cbf.radii[0] * np.sign(ct) * (np.abs(ct) ** (2/n))
    y_b = cbf.center[1] + cbf.radii[1] * np.sign(st) * (np.abs(st) ** (2/n))

    for ax in [ax_s_traj, ax_r_traj]:
        ax.plot(y_b, x_b, 'k-', linewidth=1)
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal'); ax.invert_xaxis()
        ax.grid(True)

    ax_s_traj.set_title("Standard Trajectory")
    ax_r_traj.set_title(f"Robust Trajectory (q={CALIBRATED_QUANTILE})")

    # V(x)
    for ax in [ax_s_V, ax_r_V]:
        ax.set_ylim(0, 50.0) # Adjust based on Q matrix
        ax.grid(True)
    ax_s_V.set_title("Standard V(x)"); ax_r_V.set_title("Robust V(x)")

    # Grad V
    for ax in [ax_s_grad, ax_r_grad]:
        ax.set_ylim(0, 1.0) 
        ax.grid(True)
    ax_s_grad.set_title("Standard ||∇_act V||"); ax_r_grad.set_title("Robust ||∇_act V||")

    # h(x)
    for ax in [ax_s_h, ax_r_h]:
        ax.axhline(0, color='r', linestyle='--')
        ax.set_ylim(-0.2, 1.2); ax.grid(True)
    ax_s_h.set_title("Standard h(x)"); ax_r_h.set_title("Robust h(x)")

    # mu
    for ax in [ax_s_mu, ax_r_mu]:
        ax.set_ylim(0, 20.0); ax.grid(True)
    ax_s_mu.set_title("Standard ||μ||"); ax_r_mu.set_title("Robust ||μ||")

    # Widget
    ax_check = plt.axes([0.05, 0.02, 0.1, 0.05])
    check = CheckButtons(ax_check, ['Activate Safety'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if len(node.log_t) == 0: return ln_s_act,
            
            t = list(node.log_t)
            tx = list(node.log_target['x']); ty = list(node.log_target['y'])
            
            sim_s = node.sims[0]; sim_r = node.sims[1]
            
            # Unpack Standard
            sx = list(sim_s.log['x']); sy = list(sim_s.log['y'])
            sh = list(sim_s.log['h']); smu = list(sim_s.log['mu'])
            sV = list(sim_s.log['V']); sG = list(sim_s.log['gradV'])

            # Unpack Robust
            rx = list(sim_r.log['x']); ry = list(sim_r.log['y'])
            rh = list(sim_r.log['h']); rmu = list(sim_r.log['mu'])
            rV = list(sim_r.log['V']); rG = list(sim_r.log['gradV'])

        min_len = min(len(t), len(sx), len(rx))
        t = t[:min_len]

        # --- Update Standard ---
        ln_s_target.set_data(ty[:min_len], tx[:min_len])
        ln_s_act.set_data(sy[:min_len], sx[:min_len])
        ln_s_V.set_data(t, sV[:min_len])
        ln_s_grad.set_data(t, sG[:min_len])
        ln_s_h.set_data(t, sh[:min_len])
        ln_s_mu.set_data(t, smu[:min_len])

        # --- Update Robust ---
        ln_r_target.set_data(ty[:min_len], tx[:min_len])
        ln_r_act.set_data(ry[:min_len], rx[:min_len])
        ln_r_V.set_data(t, rV[:min_len])
        ln_r_grad.set_data(t, rG[:min_len])
        ln_r_h.set_data(t, rh[:min_len])
        ln_r_mu.set_data(t, rmu[:min_len])

        if len(t) > 0:
            w_start = max(0, t[-1] - 8); w_end = t[-1] + 1
            for ax in [ax_s_V, ax_s_grad, ax_s_h, ax_s_mu, 
                       ax_r_V, ax_r_grad, ax_r_h, ax_r_mu]:
                ax.set_xlim(w_start, w_end)

        return ln_s_act, ln_r_act

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
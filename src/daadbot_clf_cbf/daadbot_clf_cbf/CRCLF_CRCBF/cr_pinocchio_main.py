"""
Main Node for Conformally Robust CLF-CBF Control (Internal Physics Version).
Simulates 'Reality' using Pinocchio (Clean URDF) while the Controller uses a 'Noisy' URDF.
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

EE_NAMES = ["gear1_claw", "gear2_claw"]
USE_JOINT_1 = False  
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# [IMPORTANT] Set this to the value output by your calibrate_quantile.py script!
CALIBRATED_QUANTILE = 29.0

class ResclfNode(Node):
    def __init__(self):
        super().__init__('resclf_modular_node')
        
        # --- 1. CONTROLLER SETUP (Uses NOISY URDF) ---
        self.robot = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
        # Initialize CR-CBF (Safety)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4,      
            k_pos=87.0, k_vel=60.0      
        )
        self.cbf_active = False
        self.q_quantile = CALIBRATED_QUANTILE
        
        # --- 2. PHYSICS ENGINE SETUP (Uses CLEAN URDF) ---
        self.model_phys = pin.buildModelFromUrdf(URDF_PHYSICS)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

        # --- 3. STATE INITIALIZATION ---
        q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        self.q_sim = pin.neutral(self.model_phys) 
        self.v_sim = np.zeros(self.model_phys.nv)
        
        # Inject initial 7DOF pose
        for i, jid in enumerate(self.phys_joint_ids):
             idx_q = self.model_phys.joints[jid].idx_q
             self.q_sim[idx_q] = q_init[i]

        # Shared Memory & Locks
        self.lock = threading.Lock()
        self.q_read = q_init.copy()
        self.dq_read = np.zeros(7)
        self.tau_command = np.zeros(7)
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

        # --- 4. THREADING & ROS ---
        self.running = True
        self.dt_phys = 0.001 
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        self.start_time = None

        # --- LOGGING ---
        self.log_t_clock = []
        self.log_target = {'x':[], 'y':[]} 
        self.log_actual = {'x':[], 'y':[]} 
        self.log_h = []       
        self.log_mu = []      

    def physics_loop(self):
        """ The Physics Engine Thread (Reality) """
        print("--- Physics Engine Started (Using Clean URDF) ---")
        next_tick = time.time()
        
        while self.running:
            # 1. READ COMMAND
            with self.lock:
                current_tau = self.tau_command.copy()

            # 2. MAP TO PINOCCHIO
            tau_full = np.zeros(self.model_phys.nv)
            damping = 0.3 * self.v_sim # Joint friction
            
            for i, jid in enumerate(self.phys_joint_ids):
                idx_v = self.model_phys.joints[jid].idx_v
                tau_full[idx_v] = current_tau[i] - damping[idx_v]

            # 3. DYNAMICS
            try:
                ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                self.v_sim += ddq * self.dt_phys
                self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            except Exception as e:
                print(f"Physics Error: {e}")

            # 4. UPDATE SHARED STATE
            q_7dof = np.zeros(7); dq_7dof = np.zeros(7)
            for i, jid in enumerate(self.phys_joint_ids):
                q_7dof[i] = self.q_sim[self.model_phys.joints[jid].idx_q]
                dq_7dof[i] = self.v_sim[self.model_phys.joints[jid].idx_v]

            with self.lock:
                self.q_read = q_7dof
                self.dq_read = dq_7dof

            # 5. SYNC CLOCK
            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # Read State from Physics Thread
        with self.lock:
            q_curr = self.q_read.copy()
            dq_curr = self.dq_read.copy()

        # A. Dynamics (Uses NOISY Internal Model)
        M, nle, J, dJ, x, dx = self.robot.compute_dynamics(q_curr, dq_curr, use_joint1=USE_JOINT_1)
        
        # B. Trajectory
        xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

        # C. Nominal Control
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
        u_ref = ad + u_nominal 
        
        # D. CR-CLF Constraints (Robust)
        LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            x, dx, xd, vd, q_quantile=self.q_quantile
        )

        # E. CR-CBF Constraints (Robust)
        cbf_A, cbf_b = None, None
        h_val = self.cbf.get_h_value(x)
        
        if self.cbf_active:
            cbf_A, cbf_b = self.cbf.get_constraints(
                x, dx, u_ref, q_quantile=5.0
            )

        # F. QP Solver
        J_pinv = self.robot.get_pseudo_inverse(J)
        drift_acc = u_ref - (dJ @ dq_curr)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
        b_tau = b_tau.reshape(-1, 1)

        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            robust_clf_term=robust_term, 
            torque_A=A_tau, torque_b=b_tau,
            cbf_A=cbf_A, cbf_b=cbf_b 
        )

        # G. Final Control
        if feasible:
            acc_cmd = u_ref + mu 
            # Torque calc uses NOISY model matrices (Model Mismatch)
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_curr))) + nle
        else:
            tau_out = -10.0 * dq_curr + nle # Emergency Braking

        if not USE_JOINT_1:
            tau_lock = (-self.kp_lock * q_curr[0]) - (self.kd_lock * dq_curr[0])
            tau_out[0] = np.clip(tau_lock, -80.0, 80.0)

        tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)

        # Write Command to Physics Thread
        with self.lock:
            self.tau_command = tau_out

        # --- LOGGING ---
        if len(self.log_actual['x']) > 2000: 
            self.log_actual['x'].pop(0); self.log_actual['y'].pop(0)
            self.log_target['x'].pop(0); self.log_target['y'].pop(0)
            self.log_h.pop(0); self.log_mu.pop(0); self.log_t_clock.pop(0)
            
        self.log_actual['x'].append(x[0]); self.log_actual['y'].append(x[1])
        self.log_target['x'].append(xd[0]); self.log_target['y'].append(xd[1])
        self.log_t_clock.append(t_clock)
        self.log_h.append(h_val)
        self.log_mu.append(np.linalg.norm(mu))

    def stop_robot(self):
        with self.lock:
            self.tau_command = np.zeros(7)

def main(args=None):
    rclpy.init(args=args)
    node = ResclfNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING SETUP ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])    
    ax_mu = fig.add_subplot(gs[1, 1])   
    
    plt.subplots_adjust(bottom=0.15) 
    
    # --- 1. Trajectory Plot ---
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=2, label='Target')
    
    # --- Visual Safe Set ---
    rx = node.cbf.radii[0]; ry = node.cbf.radii[1] 
    cx = node.cbf.center[0]; cy = node.cbf.center[1]
    n  = node.cbf.power_n   

    theta = np.linspace(0, 2*np.pi, 200)
    st = np.sin(theta); ct = np.cos(theta)
    x_boundary = cx + rx * np.sign(ct) * (np.abs(ct) ** (2 / n))
    y_boundary = cy + ry * np.sign(st) * (np.abs(st) ** (2 / n))

    ax_traj.plot(y_boundary, x_boundary, color='g', linewidth=2, linestyle='-', label=f'Safe Set (n={n})')
    ax_traj.set_xlim(-0.4, 0.4); ax_traj.set_ylim(-0.4, 0.4)
    ax_traj.set_title(f'Unified Controller (Tail: 330°)')
    ax_traj.invert_xaxis(); ax_traj.legend(loc='upper right'); ax_traj.grid(True)
    ax_traj.set_aspect('equal', 'box')

    # --- 2. Safety Metric h(x) ---
    ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
    ax_h.axhline(0, color='r', linestyle='--', label='Limit (h=0)')
    ax_h.set_title('Safety Barrier h(x)')
    ax_h.set_ylim(-0.1, 1.1)
    ax_h.grid(True)

    # --- 3. Control Effort ||mu|| ---
    ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
    ax_mu.set_title('QP Correction ||μ||')
    ax_mu.set_xlabel('Time [s]')
    ax_mu.set_ylim(0, 20.0) 
    ax_mu.grid(True)

    # --- WIDGET ---
    ax_check = plt.axes([0.05, 0.02, 0.15, 0.08]) 
    check = CheckButtons(ax_check, ['Activate Safety'], [False])
    def toggle_cbf(label):
        node.cbf_active = not node.cbf_active
        print(f"CBF Active: {node.cbf_active}")
    check.on_clicked(toggle_cbf)

    def update_plot(frame):
        # --- Thread-Safe Data Retrieval ---
        with node.lock:
            len_m = len(node.log_t_clock)
            if len_m == 0: return ln_t, ln_a, ln_h, ln_mu
            
            # Copy data for plotting
            tx_data = list(node.log_target['x'])
            ty_data = list(node.log_target['y'])
            ax_data = list(node.log_actual['x'])
            ay_data = list(node.log_actual['y'])
            time_data = list(node.log_t_clock)
            h_data = list(node.log_h)
            mu_data = list(node.log_mu)

        # Slice to sync lengths
        min_len = min(len(tx_data), len(ax_data), len(time_data))
        
        t_current = time_data[min_len-1]
        tail_duration = 11.0 
        t_start = t_current - tail_duration
        
        # Find start index for tail
        # (Simple linear search for simplicity in visualization)
        start_idx = 0
        if time_data[0] < t_start:
             for i, t_val in enumerate(time_data):
                 if t_val >= t_start:
                     start_idx = i
                     break

        ln_t.set_data(ty_data[start_idx:min_len], tx_data[start_idx:min_len])
        ln_a.set_data(ay_data[start_idx:min_len], ax_data[start_idx:min_len])
        ln_h.set_data(time_data[start_idx:min_len], h_data[start_idx:min_len])
        ln_mu.set_data(time_data[start_idx:min_len], mu_data[start_idx:min_len])
        
        t_window = 15.0
        t_min = max(0, t_current - t_window)
        ax_h.set_xlim(t_min, t_current + 0.5)
        ax_mu.set_xlim(t_min, t_current + 0.5)

        return ln_t, ln_a, ln_h, ln_mu

    ani = FuncAnimation(fig, update_plot, interval=100, blit=False)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
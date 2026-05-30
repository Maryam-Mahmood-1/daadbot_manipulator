"""
Main Node for Conformally Robust CLF-CBF Control.
Uses the CR-CLF and CR-CBF modules to guarantee stability and safety 
under model uncertainty defined by q_quantile.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 
# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os


# --- MODULAR IMPORTS (Assumes these files are updated with CR logic) ---
from daadbot_clf_cbf.CRCLF_CRCBF.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CRCLF_CRCBF.utils.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CRCLF_CRCBF.crclf_formulation import RESCLF_Controller
from daadbot_clf_cbf.CRCLF_CRCBF.cr_qp_solver import solve_optimization
from daadbot_clf_cbf.CRCLF_CRCBF.crcbf_formulation import CBF_SuperEllipsoid 

# --- CONFIGURATIONS ---
URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot_noisy_.urdf"
)


EE_NAMES = ["gear1_claw", "gear2_claw"]
USE_JOINT_1 = False  
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# [IMPORTANT] Set this to the value output by your calibrate_quantile.py script!
# Example: 0.196 was the quantile for 90% confidence in the paper example.
CALIBRATED_QUANTILE = 28.6 

class ResclfNode(Node):
    def __init__(self):
        super().__init__('resclf_modular_node')
        
        # 1. Initialize Robot Dynamics (Clean Model)
        self.robot = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        
        # 2. Initialize Trajectory Generator
        self.traj_gen = TrajectoryGenerator() 
        
        # 3. Initialize CR-CLF Controller
        # (Parameters tuned for high precision as discussed previously)
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
        # [cite_start]4. Initialize CR-CBF (Safety) [cite: 33, 34]
        # (Parameters tuned for anti-bounce: low stiffness, high damping)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4,      
            k_pos=87.0,     # Soft spring to prevent bounce
            k_vel=60.0      # Hard damper to absorb momentum
        )
        self.cbf_active = False
        
        # [NEW] Conformal Robustness Parameter
        self.q_quantile = CALIBRATED_QUANTILE
        
        # ROS Setup
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        self.start_time = None
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

        # --- LOGGING ---
        self.log_t_clock = []
        self.log_target = {'x':[], 'y':[]} 
        self.log_actual = {'x':[], 'y':[]} 
        self.log_h = []       
        self.log_mu = []      

    def cb_joints(self, msg):
        try:
            q_buf = [0.0] * 7
            dq_buf = [0.0] * 7
            for i, name in enumerate(ALL_JOINTS):
                if name in msg.name:
                    idx = msg.name.index(name)
                    q_buf[i] = msg.position[idx]
                    dq_buf[i] = msg.velocity[idx]
            self.q = np.array(q_buf)
            self.dq = np.array(dq_buf)
        except ValueError:
            pass

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # A. Dynamics
        M, nle, J, dJ, x, dx = self.robot.compute_dynamics(self.q, self.dq, use_joint1=USE_JOINT_1)
        
        # B. Trajectory
        xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

        # C. Nominal Control
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
        u_ref = ad + u_nominal 
        
        # D. CR-CLF Constraints
        # [UPDATED] Now returns 5 values, including robust_term
        LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            x, dx, xd, vd, q_quantile=self.q_quantile
        )

        # E. CR-CBF Constraints
        cbf_A, cbf_b = None, None
        h_val = self.cbf.get_h_value(x)
        
        if self.cbf_active:
            # [UPDATED] Passes q_quantile to calculate robustness penalty inside cbf class
            cbf_A, cbf_b = self.cbf.get_constraints(
                x, dx, u_ref, q_quantile=self.q_quantile
            )

        # F. QP Solver
        J_pinv = self.robot.get_pseudo_inverse(J)
        drift_acc = u_ref - (dJ @ self.dq)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
        b_tau = b_tau.reshape(-1, 1)

        # [UPDATED] Pass robust_clf_term to solver
        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            robust_clf_term=robust_term,  # <--- NEW ARGUMENT
            torque_A=A_tau, torque_b=b_tau,
            cbf_A=cbf_A, cbf_b=cbf_b 
        )

        # G. Final Control
        if feasible:
            acc_cmd = u_ref + mu 
            tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
        else:
            brake_dir = -np.sign(self.dq)
            brake_dir = np.nan_to_num(brake_dir) 
            tau_cmd = (brake_dir * (0.8 * self.tau_limits)) + nle

        if not USE_JOINT_1:
            tau_lock = (-self.kp_lock * self.q[0]) - (self.kd_lock * self.dq[0])
            tau_cmd[0] = np.clip(tau_lock, -80.0, 80.0)

        tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
        msg = Float64MultiArray(); msg.data = tau_cmd.tolist(); self.pub.publish(msg)

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
        msg = Float64MultiArray(data=[0.0]*7); self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ResclfNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING SETUP ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
    ax_traj = fig.add_subplot(gs[:, 0]) # Left Column
    ax_h = fig.add_subplot(gs[0, 1])    # Top Right
    ax_mu = fig.add_subplot(gs[1, 1])   # Bottom Right
    
    plt.subplots_adjust(bottom=0.15) 
    
    # --- 1. Trajectory Plot ---
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=2, label='Target')
    
    # --- Visual Safe Set (Superellipsoid) ---
    rx = node.cbf.radii[0] 
    ry = node.cbf.radii[1] 
    cx = node.cbf.center[0]
    cy = node.cbf.center[1]
    n  = node.cbf.power_n   

    # Parametric equations for superellipse
    theta = np.linspace(0, 2*np.pi, 200)
    st = np.sin(theta); ct = np.cos(theta)
    
    x_boundary = cx + rx * np.sign(ct) * (np.abs(ct) ** (2 / n))
    y_boundary = cy + ry * np.sign(st) * (np.abs(st) ** (2 / n))

    # Note: Plotted as (y, x) because graph axes are Robot Y vs Robot X
    ax_traj.plot(y_boundary, x_boundary, color='g', linewidth=2, linestyle='-', label=f'Safe Set (n={n})')
    
    ax_traj.set_xlim(-0.4, 0.4); ax_traj.set_ylim(-0.4, 0.4)
    ax_traj.set_xlabel('Robot Y [m]'); ax_traj.set_ylabel('Robot X [m]')
    ax_traj.set_title(f'Unified Controller (Tail: 330°)')
    ax_traj.invert_xaxis(); ax_traj.legend(loc='upper right'); ax_traj.grid(True)
    ax_traj.set_aspect('equal', 'box')

    # --- 2. Safety Metric h(x) ---
    ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
    ax_h.axhline(0, color='r', linestyle='--', label='Limit (h=0)')
    ax_h.set_title('Safety Barrier h(x)')
    ax_h.set_ylabel('h(x)')
    ax_h.set_ylim(-0.1, 1.1)
    ax_h.grid(True)

    # --- 3. Control Effort ||mu|| ---
    ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
    ax_mu.set_title('QP Correction ||μ||')
    ax_mu.set_xlabel('Time [s]')
    ax_mu.set_ylabel('m/s²')
    ax_mu.set_ylim(0, 10.0) 
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
        len_tx = len(node.log_target['x'])
        len_ty = len(node.log_target['y'])
        len_ax = len(node.log_actual['x'])
        len_ay = len(node.log_actual['y'])
        len_t = len(node.log_t_clock)
        len_h = len(node.log_h)
        len_mu = len(node.log_mu)

        min_t_len = min(len_tx, len_ty)
        min_a_len = min(len_ax, len_ay)
        min_m_len = min(len_t, len_h, len_mu)
        
        if min_m_len == 0: return ln_t, ln_a, ln_h, ln_mu

        tx_data = node.log_target['x'][:min_t_len]
        ty_data = node.log_target['y'][:min_t_len]
        ax_data = node.log_actual['x'][:min_a_len]
        ay_data = node.log_actual['y'][:min_a_len]
        time_data = np.array(node.log_t_clock[:min_m_len])
        h_data = node.log_h[:min_m_len]
        mu_data = node.log_mu[:min_m_len]

        t_current = time_data[-1]
        tail_duration = 11.0 
        t_start = t_current - tail_duration
        start_idx = np.searchsorted(time_data, t_start)
        tail_len = len(time_data) - start_idx
        
        if tail_len > 0:
            ln_t.set_data(ty_data[-tail_len:], tx_data[-tail_len:])
            ln_a.set_data(ay_data[-tail_len:], ax_data[-tail_len:])
            ln_h.set_data(time_data[start_idx:], h_data[start_idx:])
            ln_mu.set_data(time_data[start_idx:], mu_data[start_idx:])
        
        t_window = 15.0
        t_min = max(0, t_current - t_window)
        ax_h.set_xlim(t_min, t_current + 0.5)
        ax_mu.set_xlim(t_min, t_current + 0.5)

        return ln_t, ln_a, ln_h, ln_mu

    ani = FuncAnimation(fig, update_plot, interval=100, blit=False)
    plt.show()
    node.stop_robot(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
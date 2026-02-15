"""
Main Node for 7-DOF Robot: Conformal Robustness (CR) Implementation.
Supports Joint 1 locking and dual end-effector reference logic.
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
import os
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
# Ensure these point to your 7-DOF specific modules
from some_examples_py.CRCLF_CRCBF_7_dof.robot_dynamics import RobotDynamics
from some_examples_py.CRCLF_CRCBF_7_dof.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_7_dof.resclf_controller import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_7_dof.qp_solver import solve_optimization
from some_examples_py.CRCLF_CRCBF_7_dof.cbf_formulation import CBF_SuperEllipsoid 

# --- CONFIGURATIONS ---
URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot_noisy_.urdf"
)

EE_NAMES = ["gear1_claw", "gear2_claw"]
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
USE_JOINT_1 = False  # Locking Joint 1 as requested

class Gazebo7DOFConformalNode(Node):
    def __init__(self):
        super().__init__('gazebo_cr_7dof_node')
        
        # --- 1. CONFORMAL PARAMETERS ---
        self.q_quantile = 28.6  # From your 7-DOF calibration
        
        # --- 2. CONTROLLER SETUP ---
        self.robot = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=3) 
        
        # Safety Barrier
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4,      
            k_pos=87.0,     
            k_vel=60.0      
        )
        self.cbf_active = False 

        # --- 3. ROS INTERFACE ---
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        # --- 4. STATE & LIMITS ---
        self.start_time = None
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        
        # Lock Joint 1 PD Parameters
        self.kp_lock = 150.0
        self.kd_lock = 15.0

        # --- 5. LOGGING ---
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

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
        except (ValueError, IndexError):
            pass

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # A. DYNAMICS (7-DOF logic)
        M, nle, J, dJ, x, dx = self.robot.compute_dynamics(self.q, self.dq, use_joint1=USE_JOINT_1)

        # B. TRAJECTORY 
        xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

        # C. CR-CLF Constraints
        # Returns 5 values: LfV, LgV, V, gamma, and robust_term (||dV/dx|| * q_quantile)
        LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            x, dx, xd, vd, q_quantile=self.q_quantile
        )

        # D. CR-CBF Constraints
        cbf_A, cbf_b = None, None
        h_val = self.cbf.get_h_value(x)

        if self.cbf_active:
            u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
            u_ref_nominal = ad + u_nominal
            # Robust CBF constraint calculation
            cbf_A, cbf_b = self.cbf.get_constraints(
                x, dx, u_ref_nominal, q_quantile=self.q_quantile
            )

        # E. QP SETUP
        J_pinv = self.robot.get_pseudo_inverse(J)
        u_ref = ad + self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
        
        drift_acc = u_ref - (dJ @ self.dq)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        # F. SOLVE CR-QP (Integrating the robust_clf_term)
        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            robust_clf_term=robust_term, 
            torque_A=A_tau, torque_b=b_tau, 
            cbf_A=cbf_A, cbf_b=cbf_b
        )

        # G. FINAL TORQUE CALCULATION
        if feasible:
            acc_cmd = u_ref + mu 
            tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
        else:
            # Fallback: Safe braking
            tau_cmd = -2.0 * self.dq + nle 

        # H. JOINT 1 LOCKING LOGIC
        if not USE_JOINT_1:
            tau_lock = (-self.kp_lock * self.q[0]) - (self.kd_lock * self.dq[0])
            tau_cmd[0] = np.clip(tau_lock, -80.0, 80.0)

        tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
        msg = Float64MultiArray(data=tau_cmd.tolist()); self.pub.publish(msg)

        # I. LOGGING
        if len(self.log['t']) > 500:
            for k in self.log: self.log[k].pop(0)
        self.log['t'].append(t_clock)
        self.log['x'].append(x[0]); self.log['y'].append(x[1])
        self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
        self.log['h'].append(h_val)
        self.log['mu'].append(np.linalg.norm(mu))

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]*7))

def main(args=None):
    rclpy.init(args=args)
    node = Gazebo7DOFConformalNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING SETUP ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])    
    ax_mu = fig.add_subplot(gs[1, 1])   
    plt.subplots_adjust(bottom=0.15)
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    # Visual Safe Set Boundary
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    cx, cy = node.cbf.center[0], node.cbf.center[1]
    n  = node.cbf.power_n
    x_b = cx + rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = cy + ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(y_b, x_b, 'g-', label='Safe Set')
    
    ax_traj.set_xlim(-0.5, 0.5); ax_traj.set_ylim(-0.5, 0.5)
    ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True); ax_traj.legend()
    ax_traj.invert_xaxis() # Align with robot frame

    ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)")
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("QP Correction ||μ||")

    ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Activate CR-CBF'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
        t_d = list(node.log['t'])
        x_d, y_d = list(node.log['x']), list(node.log['y'])
        xd_d, yd_d = list(node.log['xd']), list(node.log['yd'])
        h_d, mu_d = list(node.log['h']), list(node.log['mu'])

        ln_a.set_data(y_d, x_d); ln_t.set_data(yd_d, xd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
        if len(t_d) > 0:
            ax_h.set_xlim(max(0, t_d[-1]-10), t_d[-1]+1)
            ax_mu.set_xlim(max(0, t_d[-1]-10), t_d[-1]+1)
            ax_h.set_ylim(-0.1, 1.1); ax_mu.set_ylim(0, 10.0)
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    
    node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

if __name__ == '__main__':
    main()
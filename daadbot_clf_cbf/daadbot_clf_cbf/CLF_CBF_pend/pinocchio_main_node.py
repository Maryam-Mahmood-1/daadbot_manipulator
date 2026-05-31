"""
Main Node for 2-DOF Robot using Real-Time Pinocchio Physics Simulation
Updated with Sliding Window Plotting & Higher Torque Limits
"""
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CLF_CBF_2_link.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CLF_CBF_2_link.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CLF_CBF_2_link.resclf_controller import RESCLF_Controller
from daadbot_clf_cbf.CLF_CBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CLF_CBF_2_link.qp_solver import solve_optimization

# Using the same URDF for both Physics and Control
URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf", 
    "2link_robot.urdf.xacro" 
)

EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class RealTimePhysicsNode(Node):
    def __init__(self):
        super().__init__('rt_physics_sim_node')
        
        # --- 1. CONTROLLER SETUP ---
        self.robot_ctrl = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # Safety Barrier (Planar Ellipse)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], 
            lengths=[1.2, 1.2, 3.0], 
            power_n=4, k_pos=21.0, k_vel=12.0
        )
        self.cbf_active = False # Default OFF 

        # --- 2. PINOCCHIO PHYSICS ENGINE SETUP ---
        self.model_phys = pin.buildModelFromUrdf(URDF_PATH)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

        # --- 3. STATE INITIALIZATION ---
        self.q_sim = pin.neutral(self.model_phys) 
        self.v_sim = np.zeros(self.model_phys.nv)
        
        # Initial offset to prevent singularity
        q_init = np.array([0.0, 0.1]) 
        for i, jid in enumerate(self.phys_joint_ids):
             idx_q = self.model_phys.joints[jid].idx_q
             self.q_sim[idx_q] = q_init[i]

        self.lock = threading.Lock()
        self.q_read = q_init.copy()
        self.dq_read = np.zeros(2)
        self.tau_command = np.zeros(2)
        
        # INCREASED LIMITS to prevent QP failure
        self.tau_limits = np.array([50.0, 30.0]) 

        # --- 4. THREADING & ROS ---
        self.running = True
        self.dt_phys = 0.001 # 1000Hz Physics Step
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        self.control_rate = 100.0 # 100Hz Control Rate
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        
        # --- 5. LOGGING ---
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def physics_loop(self):
        """ The Physics Engine Thread (Runs independently of the control loop) """
        print("--- Pinocchio Physics Engine Started ---")
        next_tick = time.time()
        
        while self.running:
            with self.lock:
                current_tau = self.tau_command.copy()

            tau_full = np.zeros(self.model_phys.nv)
            # Minimal joint damping for realistic simulation
            damping = 0.1 * self.v_sim 
            
            for i, jid in enumerate(self.phys_joint_ids):
                idx_v = self.model_phys.joints[jid].idx_v
                tau_full[idx_v] = current_tau[i] - damping[idx_v]

            try:
                # Forward Dynamics (ABA) and Integration
                ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                self.v_sim += ddq * self.dt_phys
                self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            except Exception as e:
                print(f"Physics Error: {e}")

            q_sys = np.zeros(2)
            dq_sys = np.zeros(2)
            for i, jid in enumerate(self.phys_joint_ids):
                q_sys[i] = self.q_sim[self.model_phys.joints[jid].idx_q]
                dq_sys[i] = self.v_sim[self.model_phys.joints[jid].idx_v]

            with self.lock:
                self.q_read = q_sys
                self.dq_read = dq_sys

            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def control_loop(self):
        """ The Control Brain """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            q_c = self.q_read.copy()
            dq_c = self.dq_read.copy()

        # A. DYNAMICS
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(q_c, dq_c)
        J = J[0:2, :]; dJ = dJ[0:2, :]; x_2d = x[0:2]; dx_2d = dx[0:2]

        # B. TRAJECTORY
        xd_full, vd_full, ad_full = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
        xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

        # C. NOMINAL CONTROL
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
        u_ref = ad + u_nominal 
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x_2d, dx_2d, xd, vd)

        # D. CBF
        cbf_A, cbf_b = None, None
        x_3d = np.array([x_2d[0], x_2d[1], 0.0])
        h_val = self.cbf.get_h_value(x_3d)

        if self.cbf_active:
            dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
            u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
            A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d)
            cbf_A = A_temp[:, :2] 
            cbf_b = b_temp

        # E. QP SETUP
        J_pinv = np.linalg.pinv(J)
        drift_acc = u_ref - (dJ @ dq_c)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        # F. SOLVE
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

        if feasible:
            acc_cmd = u_ref + mu 
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_c))) + nle
        else:
            tau_out = -5.0 * dq_c + nle 

        tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)
        mu_norm = np.linalg.norm(mu)

        # Push to Physics Engine
        with self.lock:
            self.tau_command = tau_out
            
            # G. LOGGING (Optimized Sliding Window)
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            self.log['h'].append(h_val)
            self.log['mu'].append(mu_norm)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePhysicsNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])
    ax_mu = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual') 
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    n = node.cbf.power_n
    x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    
    ax_traj.plot(x_b, y_b, 'g-', label='Safe Set')
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True)
    ax_traj.legend()

    ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)"); ax_h.grid(True)
    ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5); ax_mu.set_title("Correction ||μ||"); ax_mu.grid(True)

    ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Safety On'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
            t_d = list(node.log['t'])
            x_d = list(node.log['x']); y_d = list(node.log['y'])
            xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
            h_d = list(node.log['h']); mu_d = list(node.log['mu'])

        ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
        if len(t_d) > 0:
            ax_h.set_xlim(t_d[0], t_d[-1])
            ax_mu.set_xlim(t_d[0], t_d[-1])
            ax_h.set_ylim(-1.0, 1.0)
            ax_mu.set_ylim(-10.0, 20.0)
        
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    t_ros.join()

if __name__ == '__main__':
    main()

"""Noisy Vs Clean URDF based controllers comparison pinocchio physics simulation."""
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

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization
from daadbot_clf_cbf.CLF_CBF.cbf_formulation import CBF_SuperEllipsoid 

# --- CONFIGURATIONS ---

URDF_PHYSICS = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)
URDF_CLEAN_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)
URDF_NOISY_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot_noisy_.urdf"
)

EE_NAMES = ["gear1_claw", "gear2_claw"]
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
USE_JOINT_1 = False

class SimInstance:
    """ Holds the state for ONE simulation (Physics + Controller) """
    def __init__(self, name, urdf_phys, urdf_ctrl):
        self.name = name
        
        # 1. The "Mind" (Controller)
        self.robot_ctrl = RobotDynamics(urdf_ctrl, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4, k_pos=87.0, k_vel=60.0
        )
        
        # 2. The "Body" (Physics) - ALWAYS CLEAN URDF
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
        self.log = {'x':[], 'y':[], 'h':[], 'mu':[]} 

class DualSimNode(Node):
    def __init__(self):
        super().__init__('dual_sim_node')
        
        # --- 1. SETUP TWO SIMULATIONS ---
        self.sims = [
            SimInstance("Clean (Ideal)", URDF_PHYSICS, URDF_CLEAN_CTRL),
            SimInstance("Noisy (Robustness Test)", URDF_PHYSICS, URDF_NOISY_CTRL)
        ]
        
        self.traj_gen = TrajectoryGenerator()
        self.cbf_active = False 

        # --- 2. THREADING & SHARED MEMORY ---
        self.lock = threading.Lock()
        self.running = True
        self.dt_phys = 0.001
        self.log_t = [] 
        self.log_target = {'x':[], 'y':[]}

        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        # --- 3. ROS CONTROL ---
        self.control_rate = 200.0
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

    def physics_loop(self):
        """ Steps BOTH physics engines in parallel @ 1000Hz """
        print("--- Dual Physics Engine Started ---")
        next_tick = time.time()
        
        while self.running:
            with self.lock:
                for sim in self.sims:
                    self.step_single_physics(sim)
            
            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def step_single_physics(self, sim):
        # 1. Get Torque & Apply Friction
        tau_full = np.zeros(sim.model_phys.nv)
        damping = 0.3 * sim.v_sim
        
        for i, jid in enumerate(sim.phys_joint_ids):
            idx_v = sim.model_phys.joints[jid].idx_v
            tau_full[idx_v] = sim.tau_cmd[i] - damping[idx_v]

        # 2. Dynamics (ABA)
        try:
            ddq = pin.aba(sim.model_phys, sim.data_phys, sim.q_sim, sim.v_sim, tau_full)
            if np.isnan(ddq).any() or np.max(np.abs(ddq)) > 1e5:
                sim.v_sim = np.zeros(sim.model_phys.nv)
                ddq = np.zeros(sim.model_phys.nv)
            sim.v_sim += ddq * self.dt_phys
            sim.q_sim = pin.integrate(sim.model_phys, sim.q_sim, sim.v_sim * self.dt_phys)
        except:
            pass

    def control_loop(self):
        """ Computes Control for BOTH simulations @ 100Hz """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            # --- ATOMIC LOGGING ---
            if len(self.log_t) > 1560:
                self.log_t.pop(0)
                self.log_target['x'].pop(0); self.log_target['y'].pop(0)
                for sim in self.sims:
                    sim.log['x'].pop(0); sim.log['y'].pop(0)
                    sim.log['h'].pop(0); sim.log['mu'].pop(0)
            
            self.log_t.append(t_clock)

            for sim_idx, sim in enumerate(self.sims):
                # 1. Get State
                q_7dof = np.zeros(7); dq_7dof = np.zeros(7)
                for i, jid in enumerate(sim.phys_joint_ids):
                    q_7dof[i] = sim.q_sim[sim.model_phys.joints[jid].idx_q]
                    dq_7dof[i] = sim.v_sim[sim.model_phys.joints[jid].idx_v]

                # 2. Dynamics
                M, nle, J, dJ, x, dx = sim.robot_ctrl.compute_dynamics(q_7dof, dq_7dof, use_joint1=USE_JOINT_1)
                
                # 3. Trajectory
                xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)
                if sim_idx == 0:
                    self.log_target['x'].append(xd[0]); self.log_target['y'].append(xd[1])

                # 4. Control
                u_nom = sim.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
                u_ref = ad + u_nom
                LfV, LgV, V, gamma = sim.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)
                
                cbf_A, cbf_b = None, None
                h_val = sim.cbf.get_h_value(x)
                if self.cbf_active:
                    cbf_A, cbf_b = sim.cbf.get_constraints(x, dx, u_ref)

                J_pinv = sim.robot_ctrl.get_pseudo_inverse(J)
                drift = u_ref - (dJ @ dq_7dof)
                b_tau_bias = (M @ J_pinv @ drift) + nle
                
                A_tau = np.vstack([M @ J_pinv, -M @ J_pinv])
                b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

                mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

                if feasible:
                    acc_cmd = u_ref + mu 
                    tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_7dof))) + nle
                else:
                    tau_out = -10.0 * dq_7dof + nle 

                if not USE_JOINT_1:
                    tau_out[0] = np.clip((-self.kp_lock * q_7dof[0]) - (self.kd_lock * dq_7dof[0]), -80, 80)
                
                sim.tau_cmd = np.clip(tau_out, -self.tau_limits, self.tau_limits)
                
                sim.log['x'].append(x[0]); sim.log['y'].append(x[1])
                sim.log['h'].append(h_val); sim.log['mu'].append(np.linalg.norm(mu))

def main(args=None):
    rclpy.init(args=args)
    node = DualSimNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- DUAL ROW PLOTTING ---
    fig = plt.figure(figsize=(14, 10))
    # Create 2 Rows x 3 Cols
    # Row 1: Clean (Traj, Safety, Mu)
    # Row 2: Noisy (Traj, Safety, Mu)
    gs = fig.add_gridspec(2, 3)

    # --- ROW 1: CLEAN CONTROLLER ---
    ax_c_traj = fig.add_subplot(gs[0, 0])
    ax_c_h    = fig.add_subplot(gs[0, 1])
    ax_c_mu   = fig.add_subplot(gs[0, 2])

    # --- ROW 2: NOISY CONTROLLER ---
    ax_n_traj = fig.add_subplot(gs[1, 0])
    ax_n_h    = fig.add_subplot(gs[1, 1])
    ax_n_mu   = fig.add_subplot(gs[1, 2])
    
    plt.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)

    # --- INIT PLOT LINES ---
    # Targets
    ln_c_target, = ax_c_traj.plot([], [], 'b--', label='Target')
    ln_n_target, = ax_n_traj.plot([], [], 'b--', label='Target')

    # Data Lines
    ln_c_act, = ax_c_traj.plot([], [], 'g-', linewidth=2, label='Actual')
    ln_n_act, = ax_n_traj.plot([], [], 'r-', linewidth=2, label='Actual')

    ln_c_h_line, = ax_c_h.plot([], [], 'g-')
    ln_n_h_line, = ax_n_h.plot([], [], 'r-')

    ln_c_mu_line, = ax_c_mu.plot([], [], 'k-')
    ln_n_mu_line, = ax_n_mu.plot([], [], 'k-')

    # --- DECORATIONS ---
    # Safe Set Visualization
    theta = np.linspace(0, 2*np.pi, 200)
    cbf = node.sims[0].cbf
    x_b = cbf.center[0] + cbf.radii[0] * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** 0.5)
    y_b = cbf.center[1] + cbf.radii[1] * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** 0.5)

    for ax in [ax_c_traj, ax_n_traj]:
        ax.plot(y_b, x_b, 'k-', linewidth=1, label='Safe Set')
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal'); ax.invert_xaxis()
        ax.grid(True); ax.legend(loc='upper right')

    ax_c_traj.set_title("CLEAN: Trajectory")
    ax_n_traj.set_title("NOISY: Trajectory")

    # Safety Axes
    for ax in [ax_c_h, ax_n_h]:
        ax.axhline(0, color='r', linestyle='--')
        ax.set_ylim(-0.5, 1.5); ax.grid(True)
    
    ax_c_h.set_title("CLEAN: Safety h(x)")
    ax_n_h.set_title("NOISY: Safety h(x)")

    # Effort Axes
    for ax in [ax_c_mu, ax_n_mu]:
        ax.set_ylim(0, 25.0); ax.grid(True)

    ax_c_mu.set_title("CLEAN: Effort ||μ||")
    ax_n_mu.set_title("NOISY: Effort ||μ||")

    # Widget
    ax_check = plt.axes([0.05, 0.02, 0.1, 0.05])
    check = CheckButtons(ax_check, ['Safety On'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        with node.lock:
            if len(node.log_t) == 0: return ln_c_act,
            
            # Safe Copy
            t = list(node.log_t)
            tx = list(node.log_target['x']); ty = list(node.log_target['y'])
            
            sim_c = node.sims[0]; sim_n = node.sims[1]
            
            cx = list(sim_c.log['x']); cy = list(sim_c.log['y'])
            ch = list(sim_c.log['h']); cmu = list(sim_c.log['mu'])

            nx = list(sim_n.log['x']); ny = list(sim_n.log['y'])
            nh = list(sim_n.log['h']); nmu = list(sim_n.log['mu'])

        # Length check
        min_len = min(len(t), len(cx), len(nx))
        t = t[:min_len]

        # Update Clean Plots
        ln_c_target.set_data(ty[:min_len], tx[:min_len])
        ln_c_act.set_data(cy[:min_len], cx[:min_len])
        ln_c_h_line.set_data(t, ch[:min_len])
        ln_c_mu_line.set_data(t, cmu[:min_len])

        # Update Noisy Plots
        ln_n_target.set_data(ty[:min_len], tx[:min_len])
        ln_n_act.set_data(ny[:min_len], nx[:min_len])
        ln_n_h_line.set_data(t, nh[:min_len])
        ln_n_mu_line.set_data(t, nmu[:min_len])

        # Scroll X-axis
        if len(t) > 0:
            w_start = max(0, t[-1] - 10); w_end = t[-1] + 1
            for ax in [ax_c_h, ax_c_mu, ax_n_h, ax_n_mu]:
                ax.set_xlim(w_start, w_end)

        return ln_c_act, ln_n_act

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
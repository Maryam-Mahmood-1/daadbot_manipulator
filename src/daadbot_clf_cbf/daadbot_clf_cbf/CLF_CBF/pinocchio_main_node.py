"""The code for main_node using internal pinocchio physics simulation with Model Mismatch."""
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

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization
from daadbot_clf_cbf.CLF_CBF.cbf_formulation import CBF_SuperEllipsoid 

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os

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
    "daadbot_noisy.urdf"
)


EE_NAMES = ["gear1_claw", "gear2_claw"]
USE_JOINT_1 = False  
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

class RealTimePhysicsNode(Node):
    def __init__(self):
        super().__init__('rt_physics_sim_node')
        
        # --- 1. CONTROLLER SETUP (Uses NOISY URDF) ---
        # The controller thinks the robot has the parameters defined in URDF_CTRL
        self.robot_ctrl = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
        # Safety Barrier
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4, k_pos=87.0, k_vel=60.0
        )
        self.cbf_active = False 

        # --- 2. PHYSICS ENGINE SETUP (Uses CLEAN URDF) ---
        # The actual simulation follows the parameters defined in URDF_PHYSICS
        self.model_phys = pin.buildModelFromUrdf(URDF_PHYSICS)
        self.data_phys = self.model_phys.createData()
        self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

        # --- 3. STATE INITIALIZATION ---
        # Start at a pose where the robot is likely INSIDE the safe set.
        q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

        self.q_sim = pin.neutral(self.model_phys) 
        self.v_sim = np.zeros(self.model_phys.nv)
        
        # Inject initial 7DOF pose
        for i, jid in enumerate(self.phys_joint_ids):
             idx_q = self.model_phys.joints[jid].idx_q
             self.q_sim[idx_q] = q_init[i]

        # Shared Memory
        self.lock = threading.Lock()
        self.q_read = q_init.copy()
        self.dq_read = np.zeros(7)
        self.tau_command = np.zeros(7)
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

        # --- 4. THREADING ---
        self.running = True
        self.dt_phys = 0.001 
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()

        # --- 5. ROS CONTROL ---
        self.control_rate = 100.0
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)

        # Logging (Added 'mu')
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def physics_loop(self):
        """ The Physics Engine Thread (Simulating Reality) """
        print("--- Physics Engine Started (Using Clean URDF) ---")
        next_tick = time.time()
        
        while self.running:
            # 1. READ COMMAND
            with self.lock:
                current_tau = self.tau_command.copy()

            # 2. MAP TO PINOCCHIO
            tau_full = np.zeros(self.model_phys.nv)
            
            # --- USER SETTING: Kept Friction at 0.3 ---
            damping = 0.3 * self.v_sim 
            
            for i, jid in enumerate(self.phys_joint_ids):
                idx_v = self.model_phys.joints[jid].idx_v
                tau_full[idx_v] = current_tau[i] - damping[idx_v]

            # 3. DYNAMICS & SAFETY CHECK
            try:
                # This computes the "True" acceleration based on the Clean URDF
                ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                
                # CRITICAL: Catch Explosion before it crashes the app
                if np.isnan(ddq).any() or np.max(np.abs(ddq)) > 1e5:
                    print("!!! PHYSICS EXPLOSION DETECTED: Resetting Velocities !!!")
                    self.v_sim = np.zeros(self.model_phys.nv)
                    ddq = np.zeros(self.model_phys.nv)

                self.v_sim += ddq * self.dt_phys
                self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
            except Exception as e:
                print(f"Physics Error: {e}")

            # 4. UPDATE SHARED STATE
            q_7dof = np.zeros(7)
            dq_7dof = np.zeros(7)
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
        """ The Control Brain (Using Noisy Internal Model) """
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        with self.lock:
            q_c = self.q_read.copy()
            dq_c = self.dq_read.copy()

        # Dynamics (Model Mismatch happens here: This uses NOISY parameters)
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(q_c, dq_c, use_joint1=USE_JOINT_1)
        
        # Trajectory
        xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)
        
        # Control
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
        u_ref = ad + u_nominal 
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)
        
        cbf_A, cbf_b = None, None
        h_val = self.cbf.get_h_value(x)
        
        if self.cbf_active:
            cbf_A, cbf_b = self.cbf.get_constraints(x, dx, u_ref)

        J_pinv = self.robot_ctrl.get_pseudo_inverse(J)
        drift_acc = u_ref - (dJ @ dq_c)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

        if feasible:
            acc_cmd = u_ref + mu 
            # Torque calculation uses M and nle from the NOISY model
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_c))) + nle
        else:
            tau_out = -10.0 * dq_c + nle # Soft Braking

        if not USE_JOINT_1:
            tau_lock = (-self.kp_lock * q_c[0]) - (self.kd_lock * dq_c[0])
            tau_out[0] = np.clip(tau_lock, -80.0, 80.0)

        tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)
        
        # Calculate Norm of Mu for logging
        mu_norm = np.linalg.norm(mu)

        with self.lock:
            self.tau_command = tau_out
            
            # --- SAFE LOGGING (Inside Lock) ---
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['x'].append(x[0]); self.log['y'].append(x[1])
            self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
            self.log['h'].append(h_val)
            self.log['mu'].append(mu_norm)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePhysicsNode()
    
    # Run ROS in background
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1]) # 2 Rows, 2 Columns
    
    # 1. Trajectory (Left, spans both rows)
    ax_traj = fig.add_subplot(gs[:, 0]) 
    
    # 2. Safety h(x) (Top Right)
    ax_h = fig.add_subplot(gs[0, 1])
    
    # 3. Control Effort mu (Bottom Right)
    ax_mu = fig.add_subplot(gs[1, 1])
    
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    
    # Trajectory Plot
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual') 
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    # Safe Set Visualization
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    cx, cy = node.cbf.center[0], node.cbf.center[1]
    n = node.cbf.power_n
    x_bound = cx + rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_bound = cy + ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    
    ax_traj.plot(y_bound, x_bound, 'g-', label='Safe Set')
    ax_traj.set_xlim(-0.5, 0.5); ax_traj.set_ylim(-0.5, 0.5)
    ax_traj.set_aspect('equal')
    ax_traj.invert_xaxis()
    ax_traj.legend(loc='upper right')
    ax_traj.set_title("Trajectory")
    ax_traj.grid(True)

    # Safety Plot
    ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
    ax_h.set_title("Safety h(x)")
    ax_h.set_ylim(-0.2, 1.5)
    ax_h.axhline(0, color='r', linestyle='--')
    ax_h.grid(True)

    # Mu Plot (New)
    ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
    ax_mu.set_title("Correction ||μ||")
    ax_mu.set_xlabel("Time [s]")
    ax_mu.set_ylim(0, 20.0) # Adjust scale as needed
    ax_mu.grid(True)

    # Toggle Button
    ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Safety On'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        # Thread-safe read
        with node.lock:
            if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
            
            t_d = list(node.log['t'])
            x_d = list(node.log['x']); y_d = list(node.log['y'])
            xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
            h_d = list(node.log['h'])
            mu_d = list(node.log['mu'])

        # Update Plots
        ln_a.set_data(y_d, x_d) 
        ln_t.set_data(yd_d, xd_d)
        ln_h.set_data(t_d, h_d)
        ln_mu.set_data(t_d, mu_d)
        
        if len(t_d) > 0:
            window_start = max(0, t_d[-1] - 10)
            window_end = t_d[-1] + 1
            ax_h.set_xlim(window_start, window_end)
            ax_mu.set_xlim(window_start, window_end)
        
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    print("Shutting down...")
    node.running = False
    node.phys_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    t_ros.join()
    print("Done.")

if __name__ == '__main__':
    main()

    


# """The code for main_node using internal pinocchio physics simulation."""
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons 

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
# from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
# from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization
# from daadbot_clf_cbf.CLF_CBF.cbf_formulation import CBF_SuperEllipsoid 

# # --- CONFIGURATIONS ---
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy_.urdf"
# EE_NAMES = ["gear1_claw", "gear2_claw"]
# USE_JOINT_1 = False  
# ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# class RealTimePhysicsNode(Node):
#     def __init__(self):
#         super().__init__('rt_physics_sim_node')
        
#         # --- 1. CONTROLLER SETUP ---
#         self.robot_ctrl = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS, noise_level=0.0)
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
#         # Safety Barrier
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.72], 
#             lengths=[0.3, 0.24, 0.4], 
#             power_n=4, k_pos=87.0, k_vel=60.0
#         )
#         self.cbf_active = False 

#         # --- 2. PHYSICS ENGINE SETUP ---
#         self.model_phys = pin.buildModelFromUrdf(URDF_PATH)
#         self.data_phys = self.model_phys.createData()
#         self.phys_joint_ids = [self.model_phys.getJointId(name) for name in ALL_JOINTS]

#         # --- 3. STATE INITIALIZATION ---
#         # Start at a pose where the robot is likely INSIDE the safe set.
#         q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

#         self.q_sim = pin.neutral(self.model_phys) 
#         self.v_sim = np.zeros(self.model_phys.nv)
        
#         # Inject initial 7DOF pose
#         for i, jid in enumerate(self.phys_joint_ids):
#              idx_q = self.model_phys.joints[jid].idx_q
#              self.q_sim[idx_q] = q_init[i]

#         # Shared Memory
#         self.lock = threading.Lock()
#         self.q_read = q_init.copy()
#         self.dq_read = np.zeros(7)
#         self.tau_command = np.zeros(7)
#         self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
#         self.kp_lock = 150.0; self.kd_lock = 15.0

#         # --- 4. THREADING ---
#         self.running = True
#         self.dt_phys = 0.001 
#         self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
#         self.phys_thread.start()

#         # --- 5. ROS CONTROL ---
#         self.control_rate = 100.0
#         self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
#         self.start_time = None
#         self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)

#         # Logging (Added 'mu')
#         self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

#     def physics_loop(self):
#         """ The Physics Engine Thread """
#         print("--- Physics Engine Started ---")
#         next_tick = time.time()
        
#         while self.running:
#             # 1. READ COMMAND
#             with self.lock:
#                 current_tau = self.tau_command.copy()

#             # 2. MAP TO PINOCCHIO
#             tau_full = np.zeros(self.model_phys.nv)
            
#             # --- USER SETTING: Kept Friction at 0.3 ---
#             damping = 0.3 * self.v_sim 
            
#             for i, jid in enumerate(self.phys_joint_ids):
#                 idx_v = self.model_phys.joints[jid].idx_v
#                 tau_full[idx_v] = current_tau[i] - damping[idx_v]

#             # 3. DYNAMICS & SAFETY CHECK
#             try:
#                 ddq = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
                
#                 # CRITICAL: Catch Explosion before it crashes the app
#                 if np.isnan(ddq).any() or np.max(np.abs(ddq)) > 1e5:
#                     print("!!! PHYSICS EXPLOSION DETECTED: Resetting Velocities !!!")
#                     self.v_sim = np.zeros(self.model_phys.nv)
#                     ddq = np.zeros(self.model_phys.nv)

#                 self.v_sim += ddq * self.dt_phys
#                 self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * self.dt_phys)
#             except Exception as e:
#                 print(f"Physics Error: {e}")

#             # 4. UPDATE SHARED STATE
#             q_7dof = np.zeros(7)
#             dq_7dof = np.zeros(7)
#             for i, jid in enumerate(self.phys_joint_ids):
#                 q_7dof[i] = self.q_sim[self.model_phys.joints[jid].idx_q]
#                 dq_7dof[i] = self.v_sim[self.model_phys.joints[jid].idx_v]

#             with self.lock:
#                 self.q_read = q_7dof
#                 self.dq_read = dq_7dof

#             # 5. SYNC CLOCK
#             next_tick += self.dt_phys
#             sleep_time = next_tick - time.time()
#             if sleep_time > 0: time.sleep(sleep_time)

#     def control_loop(self):
#         """ The Control Brain """
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         with self.lock:
#             q_c = self.q_read.copy()
#             dq_c = self.dq_read.copy()

#         # Dynamics
#         M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(q_c, dq_c, use_joint1=USE_JOINT_1)
        
#         # Trajectory
#         xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)
        
#         # Control
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
#         u_ref = ad + u_nominal 
#         LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)
        
#         cbf_A, cbf_b = None, None
#         h_val = self.cbf.get_h_value(x)
        
#         if self.cbf_active:
#             cbf_A, cbf_b = self.cbf.get_constraints(x, dx, u_ref)

#         J_pinv = self.robot_ctrl.get_pseudo_inverse(J)
#         drift_acc = u_ref - (dJ @ dq_c)
#         b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
#         A_tau_base = M @ J_pinv
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

#         mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

#         if feasible:
#             acc_cmd = u_ref + mu 
#             tau_out = (M @ J_pinv @ (acc_cmd - (dJ @ dq_c))) + nle
#         else:
#             tau_out = -10.0 * dq_c + nle # Soft Braking

#         if not USE_JOINT_1:
#             tau_lock = (-self.kp_lock * q_c[0]) - (self.kd_lock * dq_c[0])
#             tau_out[0] = np.clip(tau_lock, -80.0, 80.0)

#         tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)
        
#         # Calculate Norm of Mu for logging
#         mu_norm = np.linalg.norm(mu)

#         with self.lock:
#             self.tau_command = tau_out
            
#             # --- SAFE LOGGING (Inside Lock) ---
#             if len(self.log['t']) > 500:
#                 for k in self.log: self.log[k].pop(0)
#             self.log['t'].append(t_clock)
#             self.log['x'].append(x[0]); self.log['y'].append(x[1])
#             self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
#             self.log['h'].append(h_val)
#             self.log['mu'].append(mu_norm)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RealTimePhysicsNode()
    
#     # Run ROS in background
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     # --- PLOTTING ---
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1]) # 2 Rows, 2 Columns
    
#     # 1. Trajectory (Left, spans both rows)
#     ax_traj = fig.add_subplot(gs[:, 0]) 
    
#     # 2. Safety h(x) (Top Right)
#     ax_h = fig.add_subplot(gs[0, 1])
    
#     # 3. Control Effort mu (Bottom Right)
#     ax_mu = fig.add_subplot(gs[1, 1])
    
#     plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    
#     # Trajectory Plot
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual') 
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
#     # Safe Set Visualization
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry = node.cbf.radii[0], node.cbf.radii[1]
#     cx, cy = node.cbf.center[0], node.cbf.center[1]
#     n = node.cbf.power_n
#     x_bound = cx + rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     y_bound = cy + ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    
#     ax_traj.plot(y_bound, x_bound, 'g-', label='Safe Set')
#     ax_traj.set_xlim(-0.5, 0.5); ax_traj.set_ylim(-0.5, 0.5)
#     ax_traj.set_aspect('equal')
#     ax_traj.invert_xaxis()
#     ax_traj.legend(loc='upper right')
#     ax_traj.set_title("Trajectory")
#     ax_traj.grid(True)

#     # Safety Plot
#     ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
#     ax_h.set_title("Safety h(x)")
#     ax_h.set_ylim(-0.2, 1.5)
#     ax_h.axhline(0, color='r', linestyle='--')
#     ax_h.grid(True)

#     # Mu Plot (New)
#     ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
#     ax_mu.set_title("Correction ||μ||")
#     ax_mu.set_xlabel("Time [s]")
#     ax_mu.set_ylim(0, 20.0) # Adjust scale as needed
#     ax_mu.grid(True)

#     # Toggle Button
#     ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
#     check = CheckButtons(ax_check, ['Safety On'], [False])
#     def toggle(label): node.cbf_active = not node.cbf_active
#     check.on_clicked(toggle)

#     def update(frame):
#         # Thread-safe read
#         with node.lock:
#             if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
            
#             t_d = list(node.log['t'])
#             x_d = list(node.log['x']); y_d = list(node.log['y'])
#             xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
#             h_d = list(node.log['h'])
#             mu_d = list(node.log['mu'])

#         # Update Plots
#         ln_a.set_data(y_d, x_d) 
#         ln_t.set_data(yd_d, xd_d)
#         ln_h.set_data(t_d, h_d)
#         ln_mu.set_data(t_d, mu_d)
        
#         if len(t_d) > 0:
#             window_start = max(0, t_d[-1] - 10)
#             window_end = t_d[-1] + 1
#             ax_h.set_xlim(window_start, window_end)
#             ax_mu.set_xlim(window_start, window_end)
        
#         return ln_a, ln_t, ln_h, ln_mu

#     ani = FuncAnimation(fig, update, interval=50)
#     plt.show()
    
#     print("Shutting down...")
#     node.running = False
#     node.phys_thread.join()
#     node.destroy_node()
#     rclpy.shutdown()
#     t_ros.join()
#     print("Done.")

# if __name__ == '__main__':
#     main()

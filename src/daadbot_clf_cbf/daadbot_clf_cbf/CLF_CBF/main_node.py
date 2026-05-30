
"""The code for main_node usign gazebo sim"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle 
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



URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

EE_NAMES = ["gear1_claw", "gear2_claw"]
USE_JOINT_1 = False  
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

class ResclfNode(Node):
    def __init__(self):
        super().__init__('resclf_modular_node')
        
        NOISE_LEVEL = 0.0 
        
        # Pass it to the dynamics module
        self.robot = RobotDynamics(
            URDF_PATH, 
            EE_NAMES, 
            ALL_JOINTS, 
            noise_level=NOISE_LEVEL
        )
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
        # [cite_start]Initialize CBF (Safety) [cite: 33, 34]
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.24, 0.4], 
            power_n=4,      
            k_pos=87.0,     
            k_vel=60.0      
        )
        self.cbf_active = False
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        self.start_time = None
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.tau_limits = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 
        self.kp_lock = 150.0; self.kd_lock = 15.0

        # --- LOGGING ---
        self.log_t_clock = [] # Time
        self.log_target = {'x':[], 'y':[]} 
        self.log_actual = {'x':[], 'y':[]} 
        # New Metrics
        self.log_h = []       # Safety Barrier Value
        self.log_mu = []      # Norm of Correction Input

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
        u_ref = ad + u_nominal # Full intent [cite: 137]
        
        # D. CLF Constraints
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)

        # E. CBF Constraints
        cbf_A, cbf_b = None, None
        
        # Calculate h(x) for logging regardless of activation
        h_val = self.cbf.get_h_value(x)
        
        if self.cbf_active:
            cbf_A, cbf_b = self.cbf.get_constraints(x, dx, u_ref)

        # F. QP Solver
        J_pinv = self.robot.get_pseudo_inverse(J)
        drift_acc = u_ref - (dJ @ self.dq)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
        b_tau = b_tau.reshape(-1, 1)

        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            torque_A=A_tau, torque_b=b_tau,
            cbf_A=cbf_A, cbf_b=cbf_b 
        )

        # G. Final Control
        if feasible:
            acc_cmd = u_ref + mu 
            tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
        else:
            # Fallback
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
    
    # --- UPDATED: Visual Safe Set (Superellipsoid) ---
    # Retrieve parameters from the active CBF
    rx = node.cbf.radii[0]  # Robot X semi-axis
    ry = node.cbf.radii[1]  # Robot Y semi-axis
    cx = node.cbf.center[0]
    cy = node.cbf.center[1]
    n  = node.cbf.power_n   # The power (e.g., 4)

    # Generate parametric points for the superellipse
    # Formula: x = a * sgn(cos t) * |cos t|^(2/n)
    theta = np.linspace(0, 2*np.pi, 200)
    st = np.sin(theta)
    ct = np.cos(theta)

    # Calculate coordinates in Robot Frame
    x_boundary = cx + rx * np.sign(ct) * (np.abs(ct) ** (2 / n))
    y_boundary = cy + ry * np.sign(st) * (np.abs(st) ** (2 / n))

    # Plot (Mapping: Plot X-axis = Robot Y, Plot Y-axis = Robot X)
    # linestyle='-' ensures it is just the outline (not filled)
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
        # --- 1. Thread-Safe Data Retrieval ---
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











# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Rectangle 
# from matplotlib.widgets import CheckButtons 

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
# from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
# from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization
# from daadbot_clf_cbf.CLF_CBF.cbf_formulation import CBF_SuperEllipsoid 

# # --- CONFIGURATIONS ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# EE_NAMES = ["gear1_claw", "gear2_claw"]
# USE_JOINT_1 = False  
# ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# class ResclfNode(Node):
#     def __init__(self):
#         super().__init__('resclf_modular_node')
        
#         self.robot = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS)
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
#         # [cite_start]Initialize CBF (Safety) [cite: 33, 34]
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.72], 
#             lengths=[0.3, 0.24, 0.4], 
#             power_n=4,      
#             k_pos=120.0,     
#             k_vel=22.0      
#         )
#         self.cbf_active = False
        
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         self.start_time = None
#         self.q = np.zeros(7)
#         self.dq = np.zeros(7)
#         self.tau_limits = np.array([80.0, 80.0, 60.0, 60.0, 55.0, 15.0, 15.0]) 
#         self.kp_lock = 150.0; self.kd_lock = 15.0

#         # --- LOGGING ---
#         self.log_t_clock = [] # Time
#         self.log_target = {'x':[], 'y':[]} 
#         self.log_actual = {'x':[], 'y':[]} 
#         # New Metrics
#         self.log_h = []       # Safety Barrier Value
#         self.log_mu = []      # Norm of Correction Input

#     def cb_joints(self, msg):
#         try:
#             q_buf = [0.0] * 7
#             dq_buf = [0.0] * 7
#             for i, name in enumerate(ALL_JOINTS):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     q_buf[i] = msg.position[idx]
#                     dq_buf[i] = msg.velocity[idx]
#             self.q = np.array(q_buf)
#             self.dq = np.array(dq_buf)
#         except ValueError:
#             pass

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # A. Dynamics
#         M, nle, J, dJ, x, dx = self.robot.compute_dynamics(self.q, self.dq, use_joint1=USE_JOINT_1)
        
#         # B. Trajectory
#         xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

#         # C. Nominal Control
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
#         u_ref = ad + u_nominal # Full intent [cite: 137]
        
#         # D. CLF Constraints
#         LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)

#         # E. CBF Constraints
#         cbf_A, cbf_b = None, None
        
#         # Calculate h(x) for logging regardless of activation
#         h_val = self.cbf.get_h_value(x)
        
#         if self.cbf_active:
#             cbf_A, cbf_b = self.cbf.get_constraints(x, dx, u_ref)

#         # F. QP Solver
#         J_pinv = self.robot.get_pseudo_inverse(J)
#         drift_acc = u_ref - (dJ @ self.dq)
#         b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
#         A_tau_base = M @ J_pinv
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
#         b_tau = b_tau.reshape(-1, 1)

#         mu, feasible = solve_optimization(
#             LfV, LgV, V, gamma, 
#             torque_A=A_tau, torque_b=b_tau,
#             cbf_A=cbf_A, cbf_b=cbf_b 
#         )

#         # G. Final Control
#         if feasible:
#             acc_cmd = u_ref + mu 
#             tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
#         else:
#             # Fallback
#             brake_dir = -np.sign(self.dq)
#             brake_dir = np.nan_to_num(brake_dir) 
#             tau_cmd = (brake_dir * (0.8 * self.tau_limits)) + nle

#         if not USE_JOINT_1:
#             tau_lock = (-self.kp_lock * self.q[0]) - (self.kd_lock * self.dq[0])
#             tau_cmd[0] = np.clip(tau_lock, -80.0, 80.0)

#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
#         msg = Float64MultiArray(); msg.data = tau_cmd.tolist(); self.pub.publish(msg)

#         # --- LOGGING ---
#         if len(self.log_actual['x']) > 2000: 
#             self.log_actual['x'].pop(0); self.log_actual['y'].pop(0)
#             self.log_target['x'].pop(0); self.log_target['y'].pop(0)
#             self.log_h.pop(0); self.log_mu.pop(0); self.log_t_clock.pop(0)
            
#         self.log_actual['x'].append(x[0]); self.log_actual['y'].append(x[1])
#         self.log_target['x'].append(xd[0]); self.log_target['y'].append(xd[1])
        
#         self.log_t_clock.append(t_clock)
#         self.log_h.append(h_val)
#         self.log_mu.append(np.linalg.norm(mu))

#     def stop_robot(self):
#         msg = Float64MultiArray(data=[0.0]*7); self.pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ResclfNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     # --- PLOTTING SETUP ---
#     # Layout: [Trajectory] [ h(x) ]
#     #         [          ] [ ||mu|| ]
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
#     ax_traj = fig.add_subplot(gs[:, 0]) # Left Column (Tall)
#     ax_h = fig.add_subplot(gs[0, 1])    # Top Right
#     ax_mu = fig.add_subplot(gs[1, 1])   # Bottom Right
    
#     plt.subplots_adjust(bottom=0.15) 
    
#     # --- 1. Trajectory Plot ---
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=2, label='Target')
    
#     rx = node.cbf.radii[0]; ry = node.cbf.radii[1]
#     # Keep color green regardless of state
#     rect = Rectangle((-ry, -rx), 2*ry, 2*rx, linewidth=2, edgecolor='g', facecolor='none', label='Safe Set')
#     ax_traj.add_patch(rect)
    
#     ax_traj.set_xlim(-0.4, 0.4); ax_traj.set_ylim(-0.4, 0.4)
#     ax_traj.set_xlabel('Robot Y [m]'); ax_traj.set_ylabel('Robot X [m]')
#     ax_traj.set_title('Unified Controller Trajectory')
#     ax_traj.invert_xaxis(); ax_traj.legend(loc='upper right'); ax_traj.grid(True)
#     ax_traj.set_aspect('equal', 'box')

#     # --- 2. Safety Metric h(x) Plot ---
#     ln_h, = ax_h.plot([], [], 'g-', linewidth=1.5)
#     ax_h.axhline(0, color='r', linestyle='--', label='Limit (h=0)')
#     ax_h.set_title('Safety Barrier h(x)')
#     ax_h.set_ylabel('h(x)')
#     ax_h.set_ylim(-0.1, 1.1) # h goes from 1 (center) to 0 (boundary)
#     ax_h.grid(True)

#     # --- 3. Control Effort ||mu|| Plot ---
#     ln_mu, = ax_mu.plot([], [], 'k-', linewidth=1.5)
#     ax_mu.set_title('QP Correction ||μ||')
#     ax_mu.set_xlabel('Time [s]')
#     ax_mu.set_ylabel('m/s²')
#     ax_mu.set_ylim(0, 10.0) # Adjust based on typical mu values
#     ax_mu.grid(True)

#     # --- WIDGET ---
#     ax_check = plt.axes([0.05, 0.02, 0.15, 0.08]) 
#     check = CheckButtons(ax_check, ['Activate Safety'], [False])
    
#     def toggle_cbf(label):
#         node.cbf_active = not node.cbf_active
#         # No visual color change for the box
#         print(f"CBF Active: {node.cbf_active}")
            
#     check.on_clicked(toggle_cbf)

#     def update_plot(frame):
#         # Update Data
#         ln_t.set_data(node.log_target['y'], node.log_target['x'])
#         ln_a.set_data(node.log_actual['y'], node.log_actual['x'])
        
#         # Update Metrics (Sliding window for time plots)
#         time_data = node.log_t_clock
#         h_data = node.log_h
#         mu_data = node.log_mu
        
#         if len(time_data) > 0:
#             ln_h.set_data(time_data, h_data)
#             ln_mu.set_data(time_data, mu_data)
            
#             # Auto-scroll X-axis for time plots
#             t_current = time_data[-1]
#             t_window = 10.0 # Show last 10 seconds
#             t_min = max(0, t_current - t_window)
            
#             ax_h.set_xlim(t_min, t_current + 0.5)
#             ax_mu.set_xlim(t_min, t_current + 0.5)

#         return ln_t, ln_a, ln_h, ln_mu

#     ani = FuncAnimation(fig, update_plot, interval=100, blit=False) # Blit=False for subplots usually safer
#     plt.show()
#     node.stop_robot(); node.destroy_node(); rclpy.shutdown()

# if __name__ == '__main__':
#     main()



"""Without CBF constraints - Basic CLF-QP solver."""
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
# from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
# from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization

# # --- CONFIGURATIONS ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# # EE_NAME = "endeffector" 
# EE_NAMES = ["gear1_claw", "gear2_claw"]

# # === [SWITCH] CHANGE THIS VARIABLE TO ENABLE/DISABLE JOINT 1 ===
# USE_JOINT_1 = False  
# # False = Joint 1 Locked at 0.0 (PD Control)
# # True  = Joint 1 Active (QP Control)
# # ===============================================================

# ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# class ResclfNode(Node):
#     """
#     Main Control Node implementing RES-CLF-QP.
    
#     Control Law Structure:
#     τ = M(q) [ u_nom + a_d + μ ] + n(q,q̇)
    
#     where μ is the optimal correction from the QP solver.
#     """
#     def __init__(self):
#         super().__init__('resclf_modular_node')
        
#         # 1. Initialize Modules (Standard 7-DOF model)
#         self.robot = RobotDynamics(URDF_PATH, EE_NAMES, ALL_JOINTS)
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
#         # 2. ROS Setup
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         self.start_time = None
#         self.q = np.zeros(7)
#         self.dq = np.zeros(7)
        
#         # Limits for 7 Joints
#         self.tau_limits = np.array([80.0, 80.0, 60.0, 60.0, 55.0, 15.0, 15.0]) 

#         # PD Gains for Locking Joint 1
#         self.kp_lock = 150.0
#         self.kd_lock = 15.0

#         # Logging
#         self.log_t = {'x':[], 'y':[]} 
#         self.log_a = {'x':[], 'y':[]} 

#     def cb_joints(self, msg):
#         try:
#             q_buf = [0.0] * 7
#             dq_buf = [0.0] * 7
#             for i, name in enumerate(ALL_JOINTS):
#                 if name in msg.name:
#                     idx = msg.name.index(name)
#                     q_buf[i] = msg.position[idx]
#                     dq_buf[i] = msg.velocity[idx]
#             self.q = np.array(q_buf)
#             self.dq = np.array(dq_buf)
#         except ValueError:
#             pass

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # ---------------------------------------------------------
#         # A. Dynamics Computation
#         #    M(q)q̈ + n(q,q̇) = τ
#         #    ẋ = J(q)q̇
#         # ---------------------------------------------------------
#         M, nle, J, dJ, x, dx = self.robot.compute_dynamics(self.q, self.dq, use_joint1=USE_JOINT_1)
        
#         # ---------------------------------------------------------
#         # B. Trajectory Generation
#         #    x_d, v_d, a_d (Desired Pos, Vel, Acc)
#         # ---------------------------------------------------------
#         xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

#         # ---------------------------------------------------------
#         # C. Nominal Feedback Control (u_nom)
#         #    u_nom = -K_p e - K_d ė   (Eq. 7 in standard papers)
#         #    u_ref = a_d + u_nom      (Feedforward + Feedback)
#         # ---------------------------------------------------------
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
#         u_ref = ad + u_nominal
        
#         # ---------------------------------------------------------
#         # D. Lyapunov Constraints (Lie Derivatives)
#         #    V(η) = ηᵀ P η
#         #    Calculate LfV, LgV such that:
#         #    V̇ = LfV + LgV μ
#         # ---------------------------------------------------------
#         LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)

#         # ---------------------------------------------------------
#         # E. Quadratic Program Formulation (CLF-QP)
#         #
#         #    min(μ)  ½ ‖μ‖²
#         #    s.t.    LfV + LgV μ ≤ -γ V   (Stability)
#         #            τ_min ≤ τ ≤ τ_max    (Actuation Limits)
#         #
#         #    Where: τ = M J† (u_ref + μ - J̇q̇) + n
#         # ---------------------------------------------------------
#         J_pinv = self.robot.get_pseudo_inverse(J)
        
#         # Map task-space acceleration to joint torque:
#         # τ = (M J†) μ + [ (M J†)(u_ref - J̇q̇) + n ]
#         # τ = A_tau  μ + b_tau_bias
#         A_tau_base = M @ J_pinv
#         drift_acc = u_ref - (dJ @ self.dq)
#         b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
#         # Stack inequalities for solver: A_tau μ ≤ b_tau
#         # (Combined Upper and Lower limits)
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
#         b_tau = b_tau.reshape(-1, 1)

#         # Solve for μ (Optimal Correction)
#         mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau)

#         # ---------------------------------------------------------
#         # F. Compute Final Control Law
#         #    u_cmd = u_ref + μ
#         #    q̈_cmd = J† (u_cmd - J̇q̇)
#         #    τ_cmd = M q̈_cmd + n
#         # ---------------------------------------------------------
#         acc_cmd = u_ref + mu 
#         tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle

#         # ---------------------------------------------------------
#         # G. Switch Logic (Manual Override for Joint 1)
#         #    If OFF: Overwrite τ_1 with PD lock
#         # ---------------------------------------------------------
#         if not USE_JOINT_1:
#             # τ_lock = -k_p q_1 - k_d q̇_1
#             tau_lock = (-self.kp_lock * self.q[0]) - (self.kd_lock * self.dq[0])
#             tau_cmd[0] = np.clip(tau_lock, -80.0, 80.0)

#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)

#         # Publish
#         msg = Float64MultiArray()
#         msg.data = tau_cmd.tolist()
#         self.pub.publish(msg)

#         # --- Logging ---
#         if len(self.log_a['x']) > 2000: 
#             self.log_a['x'].pop(0); self.log_a['y'].pop(0)
#             self.log_t['x'].pop(0); self.log_t['y'].pop(0)
            
#         self.log_a['x'].append(x[0])
#         self.log_a['y'].append(x[1])
#         self.log_t['x'].append(xd[0])
#         self.log_t['y'].append(xd[1])

#     def stop_robot(self):
#         msg = Float64MultiArray(data=[0.0]*7)
#         self.pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ResclfNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     fig, ax = plt.subplots(figsize=(6, 8))
#     ln_a, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
#     ln_t, = ax.plot([], [], 'b--', linewidth=2, label='Target')
    
#     ax.set_xlim(-0.4, 0.4)
#     ax.set_ylim(-0.3, 0.3)
#     ax.set_xlabel('Robot Y [m] (Left/Right)')
#     ax.set_ylabel('Robot X [m] (Forward/Back)')
#     ax.set_title('Top-Down View')
#     ax.invert_xaxis()
#     ax.legend()
#     ax.grid(True)
#     ax.set_aspect('equal', 'box')

#     def update_plot(frame):
#         ln_t.set_data(node.log_t['y'], node.log_t['x'])
#         ln_a.set_data(node.log_a['y'], node.log_a['x'])
#         return ln_t, ln_a

#     ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
    
#     try:
#         plt.show() 
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

    


"""Without joint_1 (base joint) locking - 7 DOF arm dynamics."""
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- MODULAR IMPORTS ---
# # Ensure these match your actual folder structure
# from daadbot_clf_cbf.CLF_CBF.robot_dynamics import RobotDynamics
# from daadbot_clf_cbf.CLF_CBF.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CLF_CBF.resclf_controller import RESCLF_Controller
# from daadbot_clf_cbf.CLF_CBF.qp_solver import solve_optimization

# # --- CONFIGURATIONS ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# EE_NAME = "endeffector" 
# TARGET_JOINTS = ['joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# class ResclfNode(Node):
#     def __init__(self):
#         super().__init__('resclf_modular_node')
        
#         # 1. Initialize Modules
#         self.robot = RobotDynamics(URDF_PATH, EE_NAME, TARGET_JOINTS)
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=3)
        
#         # 2. ROS Setup
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) # 100Hz Control Loop
        
#         # State Variables
#         self.start_time = None
#         self.q = None
#         self.dq = None
#         self.tau_limits = np.array([80, 80, 60, 60, 55, 15, 15]) 

#         # Data Logging (Thread-safe lists)
#         # We log 'robot_x' and 'robot_y' here. We will flip them in the plot function.
#         self.log_t = {'x':[], 'y':[]} # Target (Robot Frame)
#         self.log_a = {'x':[], 'y':[]} # Actual (Robot Frame)

#     def cb_joints(self, msg):
#         try:
#             q_buf, dq_buf = [], []
#             for name in TARGET_JOINTS:
#                 idx = msg.name.index(name)
#                 q_buf.append(msg.position[idx])
#                 dq_buf.append(msg.velocity[idx])
#             self.q = np.array(q_buf)
#             self.dq = np.array(dq_buf)
#         except ValueError:
#             pass

#     def control_loop(self):
#         if self.q is None: return
#         if self.start_time is None: self.start_time = time.time()
        
#         t_clock = time.time() - self.start_time

#         # --- A. Get Dynamics & State ---
#         # Returns: Mass Matrix, Nonlinear Effects, Jacobian, dJ, Position(x), Velocity(dx)
#         M, nle, J, dJ, x, dx = self.robot.compute_dynamics(self.q, self.dq)
        
#         # --- B. Get Trajectory ---
#         # Pass current 'x' so Phase 1 knows where to start
#         xd, vd, ad = self.traj_gen.get_ref(t_clock, current_actual_pos=x)

#         # --- C. Control Logic ---
#         u_nominal = self.clf_ctrl.get_nominal_acceleration(x, dx, xd, vd)
#         u_ref = ad + u_nominal
        
#         LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x, dx, xd, vd)

#         # --- D. QP Solver ---
#         J_pinv = self.robot.get_pseudo_inverse(J)
#         A_tau_base = M @ J_pinv
#         drift_acc = u_ref - (dJ @ self.dq)
#         b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
#         A_tau = np.vstack([A_tau_base, -A_tau_base])
#         b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
#         b_tau = b_tau.reshape(-1, 1)

#         mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau)

#         # --- E. Publish Torque ---
#         acc_cmd = u_ref + mu 
#         tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)

#         msg = Float64MultiArray()
#         msg.data = tau_cmd.tolist()
#         self.pub.publish(msg)

#         # --- F. Log Data for Plotting ---
#         # Keep buffer size reasonable (e.g., 2000 points)
#         if len(self.log_a['x']) > 2000: 
#             self.log_a['x'].pop(0); self.log_a['y'].pop(0)
#             self.log_t['x'].pop(0); self.log_t['y'].pop(0)
            
#         self.log_a['x'].append(x[0])
#         self.log_a['y'].append(x[1])
#         self.log_t['x'].append(xd[0])
#         self.log_t['y'].append(xd[1])

#     def stop_robot(self):
#         msg = Float64MultiArray(data=[0.0]*7)
#         self.pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ResclfNode()
    
#     # 1. Start ROS Spin in a Separate Thread
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     # 2. Setup Matplotlib (Main Thread)
#     # We flip the axes here: X-axis = Robot Y, Y-axis = Robot X
#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     ln_a, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
#     ln_t, = ax.plot([], [], 'b--', linewidth=2, label='Target')
    
#     # Set limits (Robot Y is now horizontal, Robot X is vertical)
#     ax.set_xlim(-0.4, 0.4) # Robot Y range
#     ax.set_ylim(-0.3, 0.3) # Robot X range
    
#     ax.set_xlabel('Robot Y [m] (Left/Right)')
#     ax.set_ylabel('Robot X [m] (Forward/Back)')
#     ax.set_title('Top-Down View (Aligned with Gazebo)')
#     ax.invert_xaxis() # Optional: Flip if left/right is reversed
#     ax.legend()
#     ax.grid(True)
#     ax.set_aspect('equal', 'box')

#     # 3. Animation Update Function
#     def update_plot(frame):
#         # Fetch data safely
#         # Robot X data -> Plot Y
#         # Robot Y data -> Plot X
        
#         target_robot_x = node.log_t['x'][:]
#         target_robot_y = node.log_t['y'][:]
#         actual_robot_x = node.log_a['x'][:]
#         actual_robot_y = node.log_a['y'][:]
        
#         # PLOT SWAPPED: (Y, X)
#         ln_t.set_data(target_robot_y, target_robot_x)
#         ln_a.set_data(actual_robot_y, actual_robot_x)
        
#         return ln_t, ln_a

#     # 4. Start Animation
#     ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
    
#     try:
#         plt.show() 
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
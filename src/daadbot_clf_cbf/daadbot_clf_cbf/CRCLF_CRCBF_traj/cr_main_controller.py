import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.patches import Rectangle
from ament_index_python.packages import get_package_share_directory

# --- IMPORT NEW ROBUST MODULES ---
# 1. Robot Dynamics (Shared, unchanged)
from daadbot_clf_cbf.CRCLF_CRCBF_traj.robot_dynamics import RobotDynamics

# 2. CR-CLF (Stability with Robustness Term)
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crclf_formulation import CRCLF_Formulation

# 3. CR-CBF (Safety with Robustness Term)
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crcbf_formulation import CRCBF_Formulation

# 4. CR-QP Solver (Accepts robust_term)
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crqp_solver import solve_qp

import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot_noisy.urdf"
)

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class CR_Controller_Node(Node):
    def __init__(self):
        super().__init__('cr_modular_node')
        
        # --- 1. CONFORMAL QUANTILE ---
        self.q_cp = 0.29861  # <--- YOUR QUANTILE

        print(f"Initializing CR-Controller with Quantile (q_cp): {self.q_cp}")

        # --- 2. Initialize Dynamics ---
        self.robot = RobotDynamics(URDF_PATH, 'endeffector', TARGET_JOINTS)
        
        # --- 3. Initialize CR-CLF (Stability) ---
        self.crclf = CRCLF_Formulation(dim=3, q_cp=self.q_cp)
        
        # --- 4. Initialize CR-CBF (Safety) ---
        self.crcbf = CRCBF_Formulation(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.2, 0.2, 0.4], 
            power_n=4,
            q_cp=self.q_cp
        )
        
        self.cbf_active = False 

        # Trajectory Parameters
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.27 
        self.period = 12.0
        
        # ROS Communication
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(0.002, self.control_loop)
        
        self.start_time = None
        self.received_state = False
        self.start_approach_pos = None  
        
        # Data Logging
        self.log_target = {'x':[], 'y':[]}
        self.log_actual = {'x':[], 'y':[]}
        # NEW: Log h(x) and time
        self.log_h = []
        self.log_t = []

    def cb_joints(self, msg):
        msg_map = {name: i for i, name in enumerate(msg.name)}
        self.robot.update_state_from_ros(msg, msg_map, TARGET_JOINTS)
        self.received_state = True

    def get_trajectory(self, t):
        omega = 2 * math.pi / self.period
        angle = omega * t
        
        pd = self.center_pos + np.array([
            self.ellipse_a * math.cos(angle), 
            self.ellipse_b * math.sin(angle), 
            0.0
        ])
        vd = np.array([
            -self.ellipse_a * omega * math.sin(angle), 
             self.ellipse_b * omega * math.cos(angle), 
             0.0
        ])
        ad = np.array([
            -self.ellipse_a * (omega**2) * math.cos(angle), 
            -self.ellipse_b * (omega**2) * math.sin(angle), 
             0.0
        ])
        return pd, vd, ad

    def control_loop(self):
        if not self.received_state: return
        
        # 1. Dynamics
        M, nle, p, v, J, dJ_dq = self.robot.compute()
        
        # 2. Trajectory Generation
        if self.start_time is None: 
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        
        if t < 5.0: 
            start_pt = self.center_pos + np.array([self.ellipse_a, 0, 0])
            if self.start_approach_pos is None:
                self.start_approach_pos = p
            ratio = t / 5.0
            sm_ratio = (1 - math.cos(ratio * math.pi)) / 2
            pd = (1 - sm_ratio) * self.start_approach_pos + sm_ratio * start_pt
            vd, ad = np.zeros(3), np.zeros(3)
        else:
            pd, vd, ad = self.get_trajectory(t - 5.0)

        # 3. Logging
        self.log_actual['x'].append(p[0])
        self.log_actual['y'].append(p[1])
        self.log_target['x'].append(pd[0])
        self.log_target['y'].append(pd[1])
        
        # --- NEW: Calculate and Log h(x) ---
        h_val = self.crcbf.get_h(p)
        self.log_h.append(h_val)
        self.log_t.append(t)

        # 4. Formulate Errors
        e = p - pd
        de = v - vd
        
        # --- 5. Get Stability Constraints (CR-CLF) ---
        LfV, LgV, V_val, gamma, robust_term = self.crclf.get_qp_constraints(e, de)
        
        # --- 6. Get Safety Constraints (CR-CBF) ---
        cbf_L, cbf_b = None, None
        
        if self.cbf_active:
            cbf_L, cbf_b = self.crcbf.get_constraints(p, v, ad)
        
        # --- 7. Solve Unified QP ---
        mu = solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L, cbf_b, robust_term)
        
        # 8. Compute Torque
        J_pinv = np.linalg.pinv(J)
        x_ddot_command = ad + mu
        q_ddot_command = J_pinv @ (x_ddot_command - dJ_dq)
        tau = (M @ q_ddot_command) + nle
        
        tau = np.clip(tau[self.robot.v_indices], -45.0, 45.0)
        
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

    def stop(self):
        msg = Float64MultiArray(data=[0.0]*7)
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CR_Controller_Node()
    
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    
    # --- PLOTTING SETUP ---
    # Create two subplots: Left (Trajectory), Right (h vs Time)
    fig, (ax_traj, ax_h) = plt.subplots(1, 2, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25) # Make room for widgets
    
    # === Subplot 1: Trajectory ===
    ln_t, = ax_traj.plot([],[], 'b--', label='Target')
    ln_a, = ax_traj.plot([],[], 'r-', label='Actual')
    
    # Draw Safe Set
    rx = node.crcbf.dims[0]
    ry = node.crcbf.dims[1]
    safe_rect = Rectangle(
        (-ry, -rx), 2*ry, 2*rx, 
        linewidth=2, edgecolor='g', facecolor='none', 
        linestyle='-', label='Safe Set'
    )
    ax_traj.add_patch(safe_rect)
    
    limit_pad = 0.1
    ax_traj.set_xlim(-(ry + limit_pad), (ry + limit_pad))
    ax_traj.set_ylim(-(rx + limit_pad), (rx + limit_pad)) 
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('Y [m]') 
    ax_traj.set_ylabel('X [m]')  
    ax_traj.set_title('Trajectory Tracking')
    ax_traj.legend(loc='upper right')
    ax_traj.grid(True)

    # === Subplot 2: Barrier Function h(x) ===
    ln_h, = ax_h.plot([], [], 'g-', linewidth=2, label='Barrier h(x)')
    
    # Draw Safety Limits
    ax_h.axhline(0, color='r', linestyle='--', linewidth=2, label='Safety Limit (h=0)')
    ax_h.axhline(1.0, color='k', linestyle=':', label='Center (h=1)')
    
    ax_h.set_ylim(-0.2, 1.2)
    ax_h.set_title('Conformal Safety Metric')
    ax_h.set_xlabel('Time [s]')
    ax_h.set_ylabel('h(x)')
    ax_h.legend(loc='upper right')
    ax_h.grid(True)

    # --- WIDGETS ---
    ax_hist = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_history = Slider(ax_hist, 'Tail Length', 10, 5000, valinit=500, valstep=10)
    
    ax_check = plt.axes([0.05, 0.05, 0.15, 0.1])
    check = CheckButtons(ax_check, ['Activate\nCR-CBF'], [False])
    
    def toggle_cbf(label):
        node.cbf_active = not node.cbf_active
        print(f"Safety CR-CBF Active: {node.cbf_active}")
        
    check.on_clicked(toggle_cbf)

    def update(frame):
        # 1. Update Trajectory Plot
        ty = node.log_target['y']
        tx = node.log_target['x']
        ay = node.log_actual['y']
        ax_data = node.log_actual['x']
        
        hist_len = int(slider_history.val)
        min_len = min(len(ty), len(tx), len(ay), len(ax_data))
        start_idx = max(0, min_len - hist_len)
        
        ln_t.set_data(ty[start_idx:min_len], tx[start_idx:min_len])
        ln_a.set_data(ay[start_idx:min_len], ax_data[start_idx:min_len])
        
        # 2. Update h(x) Plot
        times = node.log_t
        h_vals = node.log_h
        min_h = min(len(times), len(h_vals))
        
        ln_h.set_data(times[:min_h], h_vals[:min_h])
        
        # Auto-scroll h(x) plot
        if min_h > 0:
            current_t = times[min_h-1]
            ax_h.set_xlim(max(0, current_t - 15), current_t + 1) # Show last 15s
        
        return ln_t, ln_a, ln_h
        
    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    
    node.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
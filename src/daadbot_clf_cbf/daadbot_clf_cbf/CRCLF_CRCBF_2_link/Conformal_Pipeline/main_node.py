#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RadioButtons
import os
import sys

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# --- SINDY PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq = dq[0]**2, dq[1]**2
        dq_cross = dq[0] * dq[1]
        
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

ALL_JOINTS = ["baseHinge", "interArm"]

class UniversalTrackingNode(Node):
    def __init__(self):
        super().__init__('universal_tracking_node')
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        
        try:
            self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
        except FileNotFoundError:
            self.get_logger().error("SINDy weight files not found at " + ws_path)
            sys.exit(1)
            
        self.q_quantile = self.sindy.q_quantile
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # --- SAFETY CAGE SETUP ---
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=20.0, k_vel=10.0
        )
        self.cbf_active = False 
        self.active_mode = 'Task Track'  

        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        self.start_time = None
        self.q = np.array([0.0, 0.1])
        self.dq = np.zeros(2)
        self.tau_limits = np.array([500.0, 300.0]) 
        
        # Added V_bound to logs
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[], 'V_bound':[]}
        
        # Stability tracking variables
        self.V0 = None
        self.stab_start_t = 0.0
        
        self.lock = threading.Lock()

    def cb_joints(self, msg):
        q_buf, dq_buf = [None]*2, [None]*2
        for i, name in enumerate(ALL_JOINTS):
            if name in msg.name:
                idx = msg.name.index(name)
                q_buf[i] = msg.position[idx]
                dq_buf[i] = msg.velocity[idx]
        
        if all(v is not None for v in q_buf):
            with self.lock:
                self.q = np.arctan2(np.sin(q_buf), np.cos(q_buf))
                self.dq = np.array(dq_buf)

    def compute_analytical_kinematics(self, q, dq):
        L1, L2 = 0.75, 1.0
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        x, y = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12
        J = np.array([[-L1*s1 - L2*s12, -L2*s12], [ L1*c1 + L2*c12,  L2*c12]])
        dJ = np.array([[-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])], 
                       [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]])
        return np.array([x, y]), J @ dq, J, dJ

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
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time
        
        with self.lock:
            q, dq = self.q.copy(), self.dq.copy()

        x_task, dx_task, J, dJ = self.compute_analytical_kinematics(q, dq)
        ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
        is_task_space = 'Task' in self.active_mode
        current_pos, current_vel, J_clf = (x_task, dx_task, J) if is_task_space else (q, dq, None)

        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        u_nom = self.clf_ctrl.get_nominal_acceleration(current_pos, current_vel, ref_pos, ref_vel)
        
        # 1. CLF Constraints
        LfV, LgV, V, gamma, robust_clf_term = self.clf_ctrl.get_lyapunov_constraints(
            current_pos, current_vel, ref_pos, ref_vel, u_nom, q_quantile=self.q_quantile, J=J_clf
        )

        # Theoretical Bound Calculation
        v_bound = np.nan
        if 'Stab' in self.active_mode:
            if self.V0 is None:
                self.V0 = V
                self.stab_start_t = t_clock
            # Theoretical bound: V(t) <= V(0) * e^(-gamma * t)
            v_bound = self.V0 * np.exp(-gamma * (t_clock - self.stab_start_t))
        else:
            self.V0 = None # Reset when not in stability mode

        # 2. CBF Constraints
        cbf_A, cbf_b = None, None
        if self.cbf_active:
            u_ref = ref_acc + u_nom
            A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_ref, 0.0), q_quantile=self.q_quantile)
            cbf_A, cbf_b = A_3d[:, :2], b_3d

        # 3. Solve QP
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=robust_clf_term, cbf_A=cbf_A, cbf_b=cbf_b)
        # mu = 0

        if feasible:
            b_inv = np.linalg.pinv(b_hat)
            if is_task_space:
                manipulability = np.sqrt(max(0, np.linalg.det(J @ J.T)))
                epsilon = 0.05 * (1 - (manipulability / 0.1)**2) if manipulability < 0.1 else 1e-6
                J_pinv = J.T @ np.linalg.inv(J @ J.T + epsilon * np.eye(2))
                tau_cmd = b_inv @ (J_pinv @ (ref_acc + u_nom + mu - dJ @ dq) - a_hat)

            else:
                tau_cmd = b_inv @ (ref_acc + u_nom + mu - a_hat)
        else:
            tau_cmd = -15.0 * dq 

        self.pub.publish(Float64MultiArray(data=np.clip(tau_cmd, -self.tau_limits, self.tau_limits).tolist()))
        
        with self.lock:
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock)
            self.log['V'].append(V)
            self.log['V_bound'].append(v_bound)
            self.log['mu'].append(np.linalg.norm(mu))
            self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])
            if is_task_space:
                self.log['xd'].append(ref_pos[0]); self.log['yd'].append(ref_pos[1])
            else: 
                self.log['xd'].append(0.75 * np.cos(ref_pos[0]) + 1.0 * np.cos(ref_pos[0] + ref_pos[1]))
                self.log['yd'].append(0.75 * np.sin(ref_pos[0]) + 1.0 * np.sin(ref_pos[0] + ref_pos[1]))

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]*2))

def main():
    rclpy.init()
    node = UniversalTrackingNode()
    
    t_ros = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    t_ros.start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    plt.subplots_adjust(bottom=0.25) 
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    # Safety Boundary Circle/Cage
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbf.radii[0], node.cbf.radii[1], node.cbf.power_n
    x_cage = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
    y_cage = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
    ax_traj.plot(x_cage, y_cage, 'k--', alpha=0.3, label='Cage Boundary')
    
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0); ax_traj.grid(True); ax_traj.legend()
    
    # Lyapunov Plot with Bound
    ln_v, = ax_v.plot([], [], 'g-', label='Actual $V(t)$')
    ln_vb, = ax_v.plot([], [], 'r--', alpha=0.8, label='$V(0)e^{-\gamma t}$')
    ax_v.set_title("Lyapunov Energy Stability"); ax_v.grid(True); ax_v.legend()
    
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction Magnitude $||\mu||$"); ax_mu.grid(True)

    # UI
    radio = RadioButtons(plt.axes([0.05, 0.05, 0.15, 0.12]), ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
    check = CheckButtons(plt.axes([0.22, 0.05, 0.15, 0.05]), ['Safety Cage'], [False])

    def switch_mode(label): 
        with node.lock:
            node.active_mode = label
            node.V0 = None # Trigger new bound calculation
            for k in node.log: node.log[k] = []
    radio.on_clicked(switch_mode)
    
    def toggle_safety(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle_safety)

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_t, ln_v, ln_vb, ln_mu
            t, x, y, xd, yd, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'xd', 'yd', 'V', 'V_bound', 'mu']]
        
        ln_a.set_data(x, y); ln_t.set_data(xd, yd)
        ln_v.set_data(t, v); ln_vb.set_data(t, vb)
        ln_mu.set_data(t, mu)
        
        ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
        v_max = np.nanmax(v) if not np.all(np.isnan(v)) else 0.1
        vb_max = np.nanmax(vb) if not np.all(np.isnan(vb)) else 0.0
        ax_v.set_ylim(0, max(v_max, vb_max) * 1.2)
        ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
        return ln_a, ln_t, ln_v, ln_vb, ln_mu

    try:
        ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt: pass
    finally:
        node.stop_robot()
        plt.close('all')
        rclpy.shutdown()

if __name__ == '__main__':
    main()




# """
# Main Node for 2-DOF Robot: Universal Tracking Framework
# Tests 4 Modes: Joint/Task Space x Stabilization/Tracking.
# Fully Data-Driven: No Pinocchio Physics Engine utilized in the control loop.
# """
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons, RadioButtons
# import os

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# # --- SINDY PREDICTOR (Data-Driven Dynamics) ---
# class SINDyPredictor:
#     def __init__(self, xi_path, q_val_path):
#         self.Xi = np.load(xi_path)
#         with open(q_val_path, "r") as f:
#             self.q_quantile = float(f.read())
            
#     def get_dynamics(self, q, dq):
#         s1, c1 = np.sin(q[0]), np.cos(q[0])
#         s2, c2 = np.sin(q[1]), np.cos(q[1])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        
#         c2_sq, s2_sq = c2**2, s2**2
#         dq0_sq, dq1_sq = dq[0]**2, dq[1]**2
#         dq_cross = dq[0] * dq[1]
        
#         # [UPDATED] 20 features: Removed q[0] and q[1]
#         H_x = np.array([
#             1.0, dq[0], dq[1], 
#             s1, c1, s2, c2, s12, c12,
#             dq0_sq, dq1_sq, dq_cross,
#             dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#             dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#             np.sign(dq[0]), np.sign(dq[1])
#         ])
        
#         # [UPDATED] 8 features: Added c2**3 and c2**4
#         G_basis = np.array([1.0, c2, s2, c12, c2_sq, s2_sq, c2**3, c2**4])
        
#         # [UPDATED] Matrix slicing for the 36-row Xi matrix
#         a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
        
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
        
#         return a_hat, b_hat


# ALL_JOINTS = ["baseHinge", "interArm"]

# class UniversalTrackingNode(Node):
#     def __init__(self):
#         super().__init__('universal_tracking_node')
        
#         # --- 1. MODEL & CONTROL SETUP ---
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.q_quantile = self.sindy.q_quantile
        
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=45.0, k_vel=30.0
#         )
#         self.cbf_active = False 
        
#         # --- 2. CONTROL MODE ---
#         self.active_mode = 'Task Track'  # Default Mode

#         # --- 3. ROS INTERFACE ---
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         self.start_time = None
#         self.q = np.array([0.0, 0.1])
#         self.dq = np.zeros(2)
#         self.tau_limits = np.array([500.0, 300.0]) # Matches URDF limits
        
#         # Logging structure
#         self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[]}

#     def cb_joints(self, msg):
#         q_buf, dq_buf = [0.0]*2, [0.0]*2
#         found = 0
#         for i, name in enumerate(ALL_JOINTS):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 q_buf[i] = msg.position[idx]
#                 dq_buf[i] = msg.velocity[idx]
#                 found += 1
#         if found == 2:
#             # [UPDATED] Wrap continuous angles to [-pi, pi]
#             raw_q = np.array(q_buf)
#             self.q = np.arctan2(np.sin(raw_q), np.cos(raw_q))
#             self.dq = np.array(dq_buf)

#     def compute_analytical_kinematics(self, q, dq):
#         """Pure geometric mapping (No Pinocchio dependency)"""
#         L1, L2 = 0.75, 1.0
#         s1, c1 = np.sin(q[0]), np.cos(q[0])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])

#         x = L1 * c1 + L2 * c12
#         y = L1 * s1 + L2 * s12
#         x_task = np.array([x, y])

#         J = np.array([
#             [-L1*s1 - L2*s12, -L2*s12],
#             [ L1*c1 + L2*c12,  L2*c12]
#         ])

#         dJ = np.array([
#             [-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])],
#             [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]
#         ])

#         dx_task = J @ dq
#         return x_task, dx_task, J, dJ

#     def generate_reference(self, t_clock, current_x_task):
#         """Generates references based on the active GUI mode"""
#         if self.active_mode == 'Joint Stab':
#             # Target: Hold Joint 0 at 45 deg, Joint 1 at -45 deg
#             ref_pos = np.array([np.pi/4, -np.pi/4])
#             ref_vel = np.zeros(2)
#             ref_acc = np.zeros(2)
            
#         elif self.active_mode == 'Joint Track':
#             # Target: Sine wave oscillation in joint space
#             ref_pos = np.array([np.sin(t_clock), np.cos(t_clock)])
#             ref_vel = np.array([np.cos(t_clock), -np.sin(t_clock)])
#             ref_acc = np.array([-np.sin(t_clock), -np.cos(t_clock)])
            
#         elif self.active_mode == 'Task Stab':
#             # Target: Hold End-Effector at Cartesian (X=0.8m, Y=0.2m)
#             ref_pos = np.array([0.8, 0.2])
#             ref_vel = np.zeros(2)
#             ref_acc = np.zeros(2)
            
#         elif self.active_mode == 'Task Track':
#             # Target: Custom path (e.g., flower/circle) from TrajectoryGenerator
#             xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(current_x_task, (0,1)))
#             ref_pos, ref_vel, ref_acc = xd_f[:2], vd_f[:2], ad_f[:2]
            
#         return ref_pos, ref_vel, ref_acc

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # A. KINEMATICS (Analytical)
#         x_task, dx_task, J, dJ = self.compute_analytical_kinematics(self.q, self.dq)

#         # B. MODE SWITCHING LOGIC
#         ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
#         is_task_space = 'Task' in self.active_mode
        
#         if is_task_space:
#             current_pos, current_vel = x_task, dx_task
#             J_clf = J # Pass Jacobian for robust formulation scaling
#         else:
#             current_pos, current_vel = self.q, self.dq
#             J_clf = None # Joint space, J is implicit Identity

#         # C. GET SINDy DYNAMICS (Fully Data-Driven)
#         a_hat, b_hat = self.sindy.get_dynamics(self.q, self.dq)

#         # D. GET NOMINAL ACCELERATION
#         u_nom = self.clf_ctrl.get_nominal_acceleration(current_pos, current_vel, ref_pos, ref_vel)
#         u_ref_total = ref_acc + u_nom
        
#         # E. CR-CLF FORMULATION
#         LfV, LgV, V, gamma, robust_clf_term = self.clf_ctrl.get_lyapunov_constraints(
#             current_pos, current_vel, ref_pos, ref_vel, u_nom, 
#             q_quantile=self.q_quantile, J=J_clf
#         )

#         # F. CR-CBF (Task Space Only)
#         cbf_A, cbf_b = None, None
#         if self.cbf_active and is_task_space:
#             x_3d = np.array([x_task[0], x_task[1], 0.0])
#             dx_3d = np.array([dx_task[0], dx_task[1], 0.0])
#             u_ref_3d = np.array([u_ref_total[0], u_ref_total[1], 0.0])
#             A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.q_quantile)
#             cbf_A = A_temp[:, :2] 
#             cbf_b = b_temp

#         # G. SOLVE CR-QP
#         mu, feasible = solve_optimization(
#             LfV, LgV, V, gamma, 
#             robust_clf_term=robust_clf_term, 
#             cbf_A=cbf_A, cbf_b=cbf_b
#         )

#         # H. FEEDBACK LINEARIZATION (mu -> tau)
#         if feasible:
#             b_inv = np.linalg.pinv(b_hat)
#             if is_task_space:
#                 # --- SR-IK: Singularity-Robust Pseudo-Inverse ---
#                 manipulability = np.sqrt(max(0, np.linalg.det(J @ J.T)))
#                 eps_0 = 0.05
#                 thresh = 0.1
#                 if manipulability < thresh:
#                     epsilon = eps_0 * (1 - (manipulability / thresh)**2)
#                 else:
#                     epsilon = 1e-6
                    
#                 J_pinv = J.T @ np.linalg.inv(J @ J.T + epsilon * np.eye(2))
#                 tau_cmd = b_inv @ (J_pinv @ (u_ref_total + mu - dJ @ self.dq) - a_hat)
#             else:
#                 tau_cmd = b_inv @ (u_ref_total + mu - a_hat)
#         else:
#             tau_cmd = -10.0 * self.dq # Fallback damping

#         # I. PUBLISH COMMANDS
#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
#         msg = Float64MultiArray(data=tau_cmd.tolist())
#         self.pub.publish(msg)

#         # J. LOGGING
#         if len(self.log['t']) > 500:
#             for k in self.log: self.log[k].pop(0)
#         self.log['t'].append(t_clock)
#         self.log['V'].append(V)
#         self.log['mu'].append(np.linalg.norm(mu))
        
#         self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])
#         if is_task_space:
#             self.log['xd'].append(ref_pos[0]); self.log['yd'].append(ref_pos[1])
#         else: 
#             # If in joint mode, FK the reference to task space for plotting visualization
#             xd_task = 0.75 * np.cos(ref_pos[0]) + 1.0 * np.cos(ref_pos[0] + ref_pos[1])
#             yd_task = 0.75 * np.sin(ref_pos[0]) + 1.0 * np.sin(ref_pos[0] + ref_pos[1])
#             self.log['xd'].append(xd_task); self.log['yd'].append(yd_task)

#     def stop_robot(self):
#         self.pub.publish(Float64MultiArray(data=[0.0]*2))

# def main(args=None):
#     rclpy.init(args=args)
#     node = UniversalTrackingNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     # --- PLOTTING SETUP ---
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
#     ax_traj = fig.add_subplot(gs[:, 0]) 
#     ax_v = fig.add_subplot(gs[0, 1])    
#     ax_mu = fig.add_subplot(gs[1, 1])   
#     plt.subplots_adjust(bottom=0.25) 
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual Path')
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target Path')
    
#     # Visualize CR-Safe Set (Super-ellipsoid)
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry = node.cbf.radii[0], node.cbf.radii[1]
#     n  = node.cbf.power_n
#     x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
#     ax_traj.plot(x_b, y_b, 'g-', label='Conformal Safe Set')
    
#     ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
#     ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True); ax_traj.legend()

#     # Lyapunov Decay Plot
#     ln_v, = ax_v.plot([], [], 'g-', linewidth=2, label='Actual V(eta)')
#     ln_v_bound, = ax_v.plot([], [], 'k--', linewidth=2, label='V_0 e^{-γt} (Theoretical Bound)')
#     ax_v.set_title("Lyapunov Energy V(eta)"); ax_v.grid(True); ax_v.legend()
    
#     ln_mu, = ax_mu.plot([], [], 'k-')
#     ax_mu.set_title("CR-Correction ||μ||"); ax_mu.grid(True)

#     # --- GUI WIDGETS ---
#     ax_radio = plt.axes([0.05, 0.05, 0.25, 0.15])
#     radio = RadioButtons(ax_radio, ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
#     def switch_mode(label): 
#         node.active_mode = label
#         node.log = {k: [] for k in node.log} # Clear log on switch for clean plot
#         node.start_time = None # Reset clock
#     radio.on_clicked(switch_mode)

#     ax_check = plt.axes([0.35, 0.05, 0.15, 0.15]) 
#     check = CheckButtons(ax_check, ['Activate CBF'], [False])
#     def toggle(label): node.cbf_active = not node.cbf_active
#     check.on_clicked(toggle)

#     def update(frame):
#         if len(node.log['t']) == 0: return ln_a, ln_t, ln_v, ln_v_bound, ln_mu
        
#         t_d = np.array(node.log['t'])
#         x_d, y_d = list(node.log['x']), list(node.log['y'])
#         xd_d, yd_d = list(node.log['xd']), list(node.log['yd'])
#         v_d = np.array(node.log['V'])
#         mu_d = list(node.log['mu'])

#         # Calculate Exponential Decay Bound
#         v0 = v_d[0]
#         gamma = node.clf_ctrl.gamma
#         v_bound_d = v0 * np.exp(-gamma * (t_d - t_d[0]))

#         ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
#         ln_v.set_data(t_d, v_d)
#         ln_v_bound.set_data(t_d, v_bound_d)
#         ln_mu.set_data(t_d, mu_d)
        
#         if len(t_d) > 0:
#             ax_v.set_xlim(t_d[0], t_d[-1]); ax_mu.set_xlim(t_d[0], t_d[-1])
#             max_v = max(v0 * 1.1, 0.1) 
#             ax_v.set_ylim(0.0, max_v)
#             ax_mu.set_ylim(-1.0, max(max(mu_d)*1.1, 5.0))
            
#         return ln_a, ln_t, ln_v, ln_v_bound, ln_mu

#     ani = FuncAnimation(fig, update, interval=50)
#     plt.show()

    
#     node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

# if __name__ == '__main__':
#     main()


















# """
# Main Node for 2-DOF Robot: Universal Tracking Framework
# Tests 4 Modes: Joint/Task Space x Stabilization/Tracking.
# Fully Data-Driven: No Pinocchio Physics Engine utilized in the control loop.
# """
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons, RadioButtons

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics

# import os
# from ament_index_python.packages import get_package_share_directory


# URDF_TRUE = os.path.join(
#     get_package_share_directory("daadbot_desc"),
#     "urdf",
#     "2_link_urdf",
#     "2link_robot.urdf"
# )

# # --- SINDY PREDICTOR (Data-Driven Dynamics) ---
# class SINDyPredictor:
#     def __init__(self, xi_path, q_val_path):
#         self.Xi = np.load(xi_path)
#         with open(q_val_path, "r") as f:
#             self.q_quantile = float(f.read())
            
#     def get_dynamics(self, q, dq):
#         s1, c1 = np.sin(q[0]), np.cos(q[0])
#         s2, c2 = np.sin(q[1]), np.cos(q[1])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        
#         c2_sq, s2_sq = c2**2, s2**2
#         dq0_sq, dq1_sq = dq[0]**2, dq[1]**2
#         dq_cross = dq[0] * dq[1]
        
#         H_x = np.array([
#             1.0, q[0], q[1], dq[0], dq[1], 
#             s1, c1, s2, c2, s12, c12,
#             dq0_sq, dq1_sq, dq_cross,
#             dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#             dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#             np.sign(dq[0]), np.sign(dq[1])
#         ])
        
#         G_basis = np.array([1.0, c2, s2, c12, c2_sq, s2_sq])
        
#         a_hat = (H_x @ self.Xi[:22, 2:4]).flatten()
        
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[22:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:34, 2:4]
        
#         return a_hat, b_hat


# ALL_JOINTS = ["baseHinge", "interArm"]

# class UniversalTrackingNode(Node):
#     def __init__(self):
#         super().__init__('universal_tracking_node')
        
#         # --- 1. MODEL & CONTROL SETUP ---
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.q_quantile = self.sindy.q_quantile*1.0
        
#         self.traj_gen = TrajectoryGenerator() 
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
#         self.robot_ctrl = RobotDynamics(URDF_TRUE, ["endEffector"], ["baseHinge", "interArm"], noise_level=0.0)
        
#         self.cbf = CBF_SuperEllipsoid(
#             center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=45.0, k_vel=30.0
#         )
#         self.cbf_active = False 
        
#         # --- 2. CONTROL MODE ---
#         self.active_mode = 'Task Track'  # Default Mode

#         # --- 3. ROS INTERFACE ---
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         self.start_time = None
#         self.q = np.array([0.0, 0.1])
#         self.dq = np.zeros(2)
#         self.tau_limits = np.array([50.0, 30.0]) # Matches URDF limits
        
#         # Logging structure
#         self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[]}

#     def cb_joints(self, msg):
#         q_buf, dq_buf = [0.0]*2, [0.0]*2
#         found = 0
#         for i, name in enumerate(ALL_JOINTS):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 q_buf[i] = msg.position[idx]
#                 dq_buf[i] = msg.velocity[idx]
#                 found += 1
#         if found == 2:
#             self.q, self.dq = np.array(q_buf), np.array(dq_buf)

#     def compute_analytical_kinematics(self, q, dq):
#         """Pure geometric mapping (No Pinocchio dependency)"""
#         L1, L2 = 0.75, 1.0
#         s1, c1 = np.sin(q[0]), np.cos(q[0])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])

#         x = L1 * c1 + L2 * c12
#         y = L1 * s1 + L2 * s12
#         x_task = np.array([x, y])

#         J = np.array([
#             [-L1*s1 - L2*s12, -L2*s12],
#             [ L1*c1 + L2*c12,  L2*c12]
#         ])

#         dJ = np.array([
#             [-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])],
#             [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]
#         ])

#         dx_task = J @ dq
#         return x_task, dx_task, J, dJ

#     def generate_reference(self, t_clock, current_x_task):
#         """Generates references based on the active GUI mode"""
#         if self.active_mode == 'Joint Stab':
#             # Target: Hold Joint 0 at 45 deg, Joint 1 at -45 deg
#             ref_pos = np.array([np.pi/4, -np.pi/4])
#             ref_vel = np.zeros(2)
#             ref_acc = np.zeros(2)
            
#         elif self.active_mode == 'Joint Track':
#             # Target: Sine wave oscillation in joint space
#             ref_pos = np.array([np.sin(t_clock), np.cos(t_clock)])
#             ref_vel = np.array([np.cos(t_clock), -np.sin(t_clock)])
#             ref_acc = np.array([-np.sin(t_clock), -np.cos(t_clock)])
            
#         elif self.active_mode == 'Task Stab':
#             # Target: Hold End-Effector at Cartesian (X=0.8m, Y=0.2m)
#             ref_pos = np.array([0.8, 0.2])
#             ref_vel = np.zeros(2)
#             ref_acc = np.zeros(2)
            
#         elif self.active_mode == 'Task Track':
#             # Target: Custom path (e.g., flower/circle) from TrajectoryGenerator
#             xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(current_x_task, (0,1)))
#             ref_pos, ref_vel, ref_acc = xd_f[:2], vd_f[:2], ad_f[:2]
            
#         return ref_pos, ref_vel, ref_acc

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         # A. KINEMATICS (Analytical)
#         x_task, dx_task, J, dJ = self.compute_analytical_kinematics(self.q, self.dq)

#         # B. MODE SWITCHING LOGIC
#         ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
#         is_task_space = 'Task' in self.active_mode
        
#         if is_task_space:
#             current_pos, current_vel = x_task, dx_task
#             J_clf = J # Pass Jacobian for robust formulation scaling
#         else:
#             current_pos, current_vel = self.q, self.dq
#             J_clf = None # Joint space, J is implicit Identity

#         # C. GET SINDy DYNAMICS (Fully Data-Driven)
#         a_hat, b_hat = self.sindy.get_dynamics(self.q, self.dq)
#         # Get perfect analytical physics from Pinocchio
#         # M, nle, _, _, _, _ = self.robot_ctrl.compute_dynamics(self.q, self.dq)
#         # M_inv = np.linalg.inv(M)

#         # # Fake the SINDy outputs using perfect math
#         # a_hat = -M_inv @ nle
#         # b_hat = M_inv

#         # D. GET NOMINAL ACCELERATION
#         u_nom = self.clf_ctrl.get_nominal_acceleration(current_pos, current_vel, ref_pos, ref_vel)
#         u_ref_total = ref_acc + u_nom
        
#         # E. CR-CLF FORMULATION (incorporating u_nom for closed-loop Lyapunov drift)
#         LfV, LgV, V, gamma, robust_clf_term = self.clf_ctrl.get_lyapunov_constraints(
#             current_pos, current_vel, ref_pos, ref_vel, u_nom, 
#             q_quantile=self.q_quantile, J=J_clf
#         )

#         # F. CR-CBF (Task Space Only)
#         cbf_A, cbf_b = None, None
#         if self.cbf_active and is_task_space:
#             x_3d = np.array([x_task[0], x_task[1], 0.0])
#             dx_3d = np.array([dx_task[0], dx_task[1], 0.0])
#             u_ref_3d = np.array([u_ref_total[0], u_ref_total[1], 0.0])
#             A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.q_quantile)
#             cbf_A = A_temp[:, :2] 
#             cbf_b = b_temp

#         # G. SOLVE CR-QP
#         mu, feasible = solve_optimization(
#             LfV, LgV, V, gamma, 
#             robust_clf_term=robust_clf_term, 
#             cbf_A=cbf_A, cbf_b=cbf_b
#         )

#         # H. FEEDBACK LINEARIZATION (mu -> tau)
#         if feasible:
#             b_inv = np.linalg.pinv(b_hat)
#             if is_task_space:
#                 # --- SR-IK: Singularity-Robust Pseudo-Inverse ---
#                 manipulability = np.sqrt(max(0, np.linalg.det(J @ J.T)))
#                 eps_0 = 0.05
#                 thresh = 0.1
#                 if manipulability < thresh:
#                     epsilon = eps_0 * (1 - (manipulability / thresh)**2)
#                 else:
#                     epsilon = 1e-6
                    
#                 J_pinv = J.T @ np.linalg.inv(J @ J.T + epsilon * np.eye(2))
#                 tau_cmd = b_inv @ (J_pinv @ (u_ref_total + mu - dJ @ self.dq) - a_hat)
#             else:
#                 tau_cmd = b_inv @ (u_ref_total + mu - a_hat)
#         else:
#             tau_cmd = -10.0 * self.dq # Fallback damping

#         # I. PUBLISH COMMANDS
#         tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
#         msg = Float64MultiArray(data=tau_cmd.tolist())
#         self.pub.publish(msg)

#         # J. LOGGING
#         if len(self.log['t']) > 500:
#             for k in self.log: self.log[k].pop(0)
#         self.log['t'].append(t_clock)
#         self.log['V'].append(V)
#         self.log['mu'].append(np.linalg.norm(mu))
        
#         self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])
#         if is_task_space:
#             self.log['xd'].append(ref_pos[0]); self.log['yd'].append(ref_pos[1])
#         else: 
#             # If in joint mode, FK the reference to task space for plotting visualization
#             xd_task = 0.75 * np.cos(ref_pos[0]) + 1.0 * np.cos(ref_pos[0] + ref_pos[1])
#             yd_task = 0.75 * np.sin(ref_pos[0]) + 1.0 * np.sin(ref_pos[0] + ref_pos[1])
#             self.log['xd'].append(xd_task); self.log['yd'].append(yd_task)

#     def stop_robot(self):
#         self.pub.publish(Float64MultiArray(data=[0.0]*2))

# def main(args=None):
#     rclpy.init(args=args)
#     node = UniversalTrackingNode()
    
#     t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     t_ros.start()
    
#     # --- PLOTTING SETUP ---
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
#     ax_traj = fig.add_subplot(gs[:, 0]) 
#     ax_v = fig.add_subplot(gs[0, 1])    
#     ax_mu = fig.add_subplot(gs[1, 1])   
#     plt.subplots_adjust(bottom=0.25) 
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual Path')
#     ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target Path')
    
#     # Visualize CR-Safe Set (Super-ellipsoid)
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry = node.cbf.radii[0], node.cbf.radii[1]
#     n  = node.cbf.power_n
#     x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
#     ax_traj.plot(x_b, y_b, 'g-', label='Conformal Safe Set')
    
#     ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
#     ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True); ax_traj.legend()

#     # Lyapunov Decay Plot
#     ln_v, = ax_v.plot([], [], 'g-', linewidth=2, label='Actual V(eta)')
#     ln_v_bound, = ax_v.plot([], [], 'k--', linewidth=2, label='V_0 e^{-γt} (Theoretical Bound)')
#     ax_v.set_title("Lyapunov Energy V(eta)"); ax_v.grid(True); ax_v.legend()
    
#     ln_mu, = ax_mu.plot([], [], 'k-')
#     ax_mu.set_title("CR-Correction ||μ||"); ax_mu.grid(True)

#     # --- GUI WIDGETS ---
#     ax_radio = plt.axes([0.05, 0.05, 0.25, 0.15])
#     radio = RadioButtons(ax_radio, ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
#     def switch_mode(label): 
#         node.active_mode = label
#         node.log = {k: [] for k in node.log} # Clear log on switch for clean plot
#         node.start_time = None # Reset clock
#     radio.on_clicked(switch_mode)

#     ax_check = plt.axes([0.35, 0.05, 0.15, 0.15]) 
#     check = CheckButtons(ax_check, ['Activate CBF'], [False])
#     def toggle(label): node.cbf_active = not node.cbf_active
#     check.on_clicked(toggle)

#     def update(frame):
#         if len(node.log['t']) == 0: return ln_a, ln_t, ln_v, ln_v_bound, ln_mu
        
#         t_d = np.array(node.log['t'])
#         x_d, y_d = list(node.log['x']), list(node.log['y'])
#         xd_d, yd_d = list(node.log['xd']), list(node.log['yd'])
#         v_d = np.array(node.log['V'])
#         mu_d = list(node.log['mu'])

#         # Calculate Exponential Decay Bound
#         v0 = v_d[0]
#         gamma = node.clf_ctrl.gamma
#         v_bound_d = v0 * np.exp(-gamma * (t_d - t_d[0]))

#         ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
#         ln_v.set_data(t_d, v_d)
#         ln_v_bound.set_data(t_d, v_bound_d)
#         ln_mu.set_data(t_d, mu_d)
        
#         if len(t_d) > 0:
#             ax_v.set_xlim(t_d[0], t_d[-1]); ax_mu.set_xlim(t_d[0], t_d[-1])
#             max_v = max(v0 * 1.1, 0.1) 
#             ax_v.set_ylim(0.0, max_v)
#             ax_mu.set_ylim(-1.0, max(max(mu_d)*1.1, 5.0))
            
#         return ln_a, ln_t, ln_v, ln_v_bound, ln_mu

#     ani = FuncAnimation(fig, update, interval=50)
#     plt.show()
    
#     node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

# if __name__ == '__main__':
#     main()
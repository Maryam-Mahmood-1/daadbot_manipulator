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
import os
import sys
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
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
        s1, c1 = np.sin(q[0]), np.cos(q[0]); s2, c2 = np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
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

class TaskStabilizationNode(Node):
    def __init__(self):
        super().__init__('task_stabilization_node')
        
        # 1. Setup Pinocchio for High-Precision Kinematics
        urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.ee_id = self.model.getFrameId("endEffector")
        
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.control_loop) 
        
        self.initialized = False
        self.start_time = None
        self.q = None
        self.dq = None
        
        # Integrator and Tweaks
        self.error_int = np.zeros(2)
        self.ki = 0.8  
        self.kd_extra = 3.5  
        
        self.tau_limits = np.array([500.0, 300.0]) 
        self.target_pos = np.array([0.8, 0.2])
        self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[]}
        
        # Bound variables
        self.V0 = None
        self.V_bound_running = None
        
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
                self.q = np.array(q_buf)
                self.dq = np.array(dq_buf)
                self.initialized = True

    def compute_kinematics(self, q, dq):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        x_task = self.data.oMf[self.ee_id].translation[:2]
        J_full = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:2, :]
        dx_task = J @ dq
        dj_dq = pin.getFrameAcceleration(self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
        return x_task, dx_task, J, dj_dq

    def control_loop(self):
        if not self.initialized: return

        with self.lock:
            q, dq = self.q.copy(), self.dq.copy()

        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        
        if self.start_time is None:
            tau_hold = np.linalg.pinv(b_hat) @ (-a_hat)
            self.pub.publish(Float64MultiArray(data=np.clip(tau_hold, -self.tau_limits, self.tau_limits).tolist()))
            self.start_time = time.time()
            return

        t_clock = time.time() - self.start_time
        x_task, dx_task, J, dj_dq = self.compute_kinematics(q, dq)
        ref_pos, ref_vel, ref_acc = self.target_pos, np.zeros(2), np.zeros(2)

        # 1. Nominal Acceleration with Integrator
        error = x_task - ref_pos
        self.error_int += error * 0.01  
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, ref_pos, ref_vel)
        u_nom -= (self.ki * self.error_int + self.kd_extra * dx_task)

        # 2. Get Constraints using LQR-optimal gamma
        LfV, LgV, V, gamma_clf, robust_term = self.clf_ctrl.get_lyapunov_constraints(
            x_task, dx_task, ref_pos, ref_vel, u_nom, q_quantile=self.sindy.q_quantile, J=J
        )

        # 3. Cumulative Bound Calculation (Prevents Jumps)
        if self.V0 is None: 
            self.V0 = V
            self.V_bound_running = V
        
        # Update bound incrementally: V_new = V_old * exp(-gamma * dt)
        self.V_bound_running *= np.exp(-gamma_clf * 0.01)
        v_bound = self.V_bound_running

        A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), q_quantile=self.sindy.q_quantile)
        mu, feasible = solve_optimization(LfV, LgV, V, gamma_clf, robust_clf_term=robust_term, cbf_A=A_3d[:, :2], cbf_b=b_3d)

        if feasible:
            manipulability = np.sqrt(max(0, np.linalg.det(J @ J.T)))
            eps = 0.01 if manipulability < 0.05 else 1e-6
            J_pinv = J.T @ np.linalg.inv(J @ J.T + eps * np.eye(2))
            ddq_des = J_pinv @ (u_nom + mu - dj_dq)
            tau_cmd = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
        else:
            tau_cmd = -20.0 * dq 

        self.pub.publish(Float64MultiArray(data=np.clip(tau_cmd, -self.tau_limits, self.tau_limits).tolist()))
        
        with self.lock:
            if len(self.log['t']) > 500:
                for k in self.log: self.log[k].pop(0)
            self.log['t'].append(t_clock); self.log['V'].append(V); self.log['V_bound'].append(v_bound)
            self.log['mu'].append(np.linalg.norm(mu)); self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]*2))

def main():
    rclpy.init()
    node = TaskStabilizationNode()
    t_ros = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    t_ros.start()
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual Path')
    ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target Goal')
    ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True); ax_traj.legend()
    
    ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$'); ln_vb, = ax_v.plot([], [], 'r--', label='Theoretical Bound')
    ax_v.set_title("Lyapunov Stability Comparison"); ax_v.grid(True); ax_v.legend()
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title(r"Robust Correction $||\mu||$"); ax_mu.grid(True)

    def update(frame):
        with node.lock:
            if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu
            t, x, y, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'V', 'V_bound', 'mu']]
        ln_a.set_data(x, y); ln_v.set_data(t, v); ln_vb.set_data(t, vb); ln_mu.set_data(t, mu)
        ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
        ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2); ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
        return ln_a, ln_v, ln_vb, ln_mu

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    node.stop_robot(); rclpy.shutdown()

if __name__ == '__main__': main()






# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import os
# import sys

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# # --- SINDY PREDICTOR ---
# class SINDyPredictor:
#     def __init__(self, xi_path, q_val_path):
#         self.Xi = np.load(xi_path)
#         with open(q_val_path, "r") as f:
#             self.q_quantile = float(f.read())
            
#     def get_dynamics(self, q, dq):
#         # SINDy sees raw joint values from sensors
#         s1, c1 = np.sin(q[0]), np.cos(q[0]); s2, c2 = np.sin(q[1]), np.cos(q[1])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
#         dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
#         H_x = np.array([
#             1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
#             dq0_sq, dq1_sq, dq_cross,
#             dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#             dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#             np.sign(dq[0]), np.sign(dq[1])
#         ])
        
#         G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
#         a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
#         return a_hat, b_hat

# ALL_JOINTS = ["baseHinge", "interArm"]

# class TaskStabilizationNode(Node):
#     def __init__(self):
#         super().__init__('task_stabilization_node')
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
#         self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
#         # --- URDF OFFSETS (Crucial for synchronization) ---
#         self.q1_offset = 0.5772411021609789
#         self.q2_offset = -1.0307324155578172
        
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         self.initialized = False
#         self.start_time = None
#         self.q = None
#         self.dq = None
        
#         self.tau_limits = np.array([500.0, 300.0]) 
#         self.target_pos = np.array([0.8, 0.2])
#         self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[]}
#         self.V0 = None
#         self.lock = threading.Lock()

#     def cb_joints(self, msg):
#         q_buf, dq_buf = [None]*2, [None]*2
#         for i, name in enumerate(ALL_JOINTS):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 q_buf[i] = msg.position[idx]
#                 dq_buf[i] = msg.velocity[idx]
        
#         if all(v is not None for v in q_buf):
#             with self.lock:
#                 self.q = np.array(q_buf) # Raw sensor data
#                 self.dq = np.array(dq_buf)
#                 self.initialized = True

#     def compute_kinematics(self, q, dq):
#         L1, L2 = 0.75, 1.0
#         # Incorporate URDF offsets so world (0,0,0) is correct
#         q_eff = [q[0] + self.q1_offset, q[1] + self.q2_offset]
        
#         s1, c1 = np.sin(q_eff[0]), np.cos(q_eff[0])
#         s12, c12 = np.sin(q_eff[0]+q_eff[1]), np.cos(q_eff[0]+q_eff[1])
#         x, y = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12
        
#         J = np.array([[-L1*s1 - L2*s12, -L2*s12], [ L1*c1 + L2*c12,  L2*c12]])
#         dJ = np.array([[-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])], 
#                        [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]])
#         return np.array([x, y]), J @ dq, J, dJ

#     def control_loop(self):
#         if not self.initialized: return

#         with self.lock:
#             q, dq = self.q.copy(), self.dq.copy()

#         a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        
#         if self.start_time is None:
#             tau_hold = np.linalg.pinv(b_hat) @ (-a_hat)
#             self.pub.publish(Float64MultiArray(data=np.clip(tau_hold, -self.tau_limits, self.tau_limits).tolist()))
#             self.start_time = time.time()
#             return

#         t_clock = time.time() - self.start_time
#         x_task, dx_task, J, dJ = self.compute_kinematics(q, dq)
#         ref_pos, ref_vel, ref_acc = self.target_pos, np.zeros(2), np.zeros(2)

#         u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, ref_pos, ref_vel)
#         LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#             x_task, dx_task, ref_pos, ref_vel, u_nom, q_quantile=self.sindy.q_quantile, J=J
#         )

#         if self.V0 is None: self.V0 = V
#         v_bound = self.V0 * np.exp(-gamma * t_clock)

#         A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), q_quantile=self.sindy.q_quantile)
#         mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=robust_term, cbf_A=A_3d[:, :2], cbf_b=b_3d)

#         if feasible:
#             J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#             ddq_des = J_pinv @ (u_nom + mu - dJ @ dq)
#             tau_cmd = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
#         else:
#             tau_cmd = -15.0 * dq 

#         self.pub.publish(Float64MultiArray(data=np.clip(tau_cmd, -self.tau_limits, self.tau_limits).tolist()))
        
#         with self.lock:
#             if len(self.log['t']) > 500:
#                 for k in self.log: self.log[k].pop(0)
#             self.log['t'].append(t_clock); self.log['V'].append(V); self.log['V_bound'].append(v_bound)
#             self.log['mu'].append(np.linalg.norm(mu)); self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])

#     def stop_robot(self):
#         self.pub.publish(Float64MultiArray(data=[0.0]*2))

# def main():
#     rclpy.init()
#     node = TaskStabilizationNode()
#     t_ros = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
#     t_ros.start()
    
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
#     ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
#     ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target Goal')
    
#     # Boundary (Relative to synchronized world origin)
#     theta = np.linspace(0, 2*np.pi, 200); rx, ry, n = 1.1, 1.1, 4
#     x_cage = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
#     y_cage = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
#     ax_traj.plot(x_cage, y_cage, 'k--', alpha=0.3)
#     ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True)
    
#     ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$'); ln_vb, = ax_v.plot([], [], 'r--', label='Theoretical Bound')
#     ax_v.set_title("Lyapunov Stability Comparison"); ax_v.grid(True); ax_v.legend()
#     ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction Magnitude $||\mu||$"); ax_mu.grid(True)

#     def update(frame):
#         with node.lock:
#             if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu
#             t, x, y, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'V', 'V_bound', 'mu']]
#         ln_a.set_data(x, y); ln_v.set_data(t, v); ln_vb.set_data(t, vb); ln_mu.set_data(t, mu)
#         ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
#         ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2); ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
#         return ln_a, ln_v, ln_vb, ln_mu

#     ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
#     plt.show()
#     node.stop_robot(); rclpy.shutdown()

# if __name__ == '__main__': main()







# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import numpy as np
# import threading
# import time
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import os
# import sys

# # --- MODULAR IMPORTS ---
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from daadbot_clf_cbf.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# # --- SINDY PREDICTOR ---
# class SINDyPredictor:
#     def __init__(self, xi_path, q_val_path):
#         self.Xi = np.load(xi_path)
#         with open(q_val_path, "r") as f:
#             self.q_quantile = float(f.read())
            
#     def get_dynamics(self, q, dq):
#         s1, c1 = np.sin(q[0]), np.cos(q[0]); s2, c2 = np.sin(q[1]), np.cos(q[1])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
#         dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
#         H_x = np.array([
#             1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
#             dq0_sq, dq1_sq, dq_cross,
#             dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#             dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#             np.sign(dq[0]), np.sign(dq[1])
#         ])
        
#         G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
#         a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
#         return a_hat, b_hat

# ALL_JOINTS = ["baseHinge", "interArm"]

# class TaskStabilizationNode(Node):
#     def __init__(self):
#         super().__init__('task_stabilization_node')
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.clf_ctrl = RESCLF_Controller(dim_task=2) 
#         self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
#         self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
#         self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
#         self.timer = self.create_timer(0.01, self.control_loop) 
        
#         # --- SYNC GATE ---
#         self.initialized = False
#         self.start_time = None
#         self.q = None
#         self.dq = None
        
#         self.tau_limits = np.array([500.0, 300.0]) 
#         self.target_pos = np.array([0.8, 0.2])
#         self.log = {'t':[], 'x':[], 'y':[], 'V':[], 'mu':[], 'V_bound':[]}
#         self.V0 = None
#         self.lock = threading.Lock()

#     def cb_joints(self, msg):
#         q_buf, dq_buf = [None]*2, [None]*2
#         for i, name in enumerate(ALL_JOINTS):
#             if name in msg.name:
#                 idx = msg.name.index(name)
#                 q_buf[i] = msg.position[idx]
#                 dq_buf[i] = msg.velocity[idx]
        
#         if all(v is not None for v in q_buf):
#             with self.lock:
#                 self.q = np.arctan2(np.sin(q_buf), np.cos(q_buf))
#                 self.dq = np.array(dq_buf)
#                 self.initialized = True

#     def compute_kinematics(self, q, dq):
#         L1, L2 = 0.75, 1.0
#         s1, c1 = np.sin(q[0]), np.cos(q[0]); s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
#         x, y = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12
#         J = np.array([[-L1*s1 - L2*s12, -L2*s12], [ L1*c1 + L2*c12,  L2*c12]])
#         dJ = np.array([[-L1*c1*dq[0] - L2*c12*(dq[0]+dq[1]), -L2*c12*(dq[0]+dq[1])], 
#                        [-L1*s1*dq[0] - L2*s12*(dq[0]+dq[1]), -L2*s12*(dq[0]+dq[1])]])
#         return np.array([x, y]), J @ dq, J, dJ

#     def control_loop(self):
#         if not self.initialized: return

#         with self.lock:
#             q, dq = self.q.copy(), self.dq.copy()

#         # 1. IMMEDIATE GRAVITY COMPENSATION WHILE WAITING
#         a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        
#         if self.start_time is None:
#             # Active Hold: Counteract gravity using learned SINDy model
#             tau_hold = np.linalg.pinv(b_hat) @ (-a_hat)
#             self.pub.publish(Float64MultiArray(data=np.clip(tau_hold, -self.tau_limits, self.tau_limits).tolist()))
#             self.start_time = time.time()
#             print("Initial synchronization achieved. Starting control loop...")
#             print(f"Start time: {self.start_time:.2f}, Initial a_hat: {a_hat}, Initial b_hat: {b_hat}")
#             return

#         t_clock = time.time() - self.start_time
#         x_task, dx_task, J, dJ = self.compute_kinematics(q, dq)
#         ref_pos, ref_vel, ref_acc = self.target_pos, np.zeros(2), np.zeros(2)

#         u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, ref_pos, ref_vel)
#         LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#             x_task, dx_task, ref_pos, ref_vel, u_nom, q_quantile=self.sindy.q_quantile, J=J
#         )

#         if self.V0 is None: self.V0 = V
#         v_bound = self.V0 * np.exp(-gamma * t_clock)

#         A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), q_quantile=self.sindy.q_quantile)
#         mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=robust_term, cbf_A=A_3d[:, :2], cbf_b=b_3d)

#         if feasible:
#             J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#             ddq_des = J_pinv @ (u_nom + mu - dJ @ dq)
#             tau_cmd = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
#         else:
#             tau_cmd = -15.0 * dq 

#         self.pub.publish(Float64MultiArray(data=np.clip(tau_cmd, -self.tau_limits, self.tau_limits).tolist()))
        
#         with self.lock:
#             if len(self.log['t']) > 500:
#                 for k in self.log: self.log[k].pop(0)
#             self.log['t'].append(t_clock); self.log['V'].append(V); self.log['V_bound'].append(v_bound)
#             self.log['mu'].append(np.linalg.norm(mu)); self.log['x'].append(x_task[0]); self.log['y'].append(x_task[1])

#     def stop_robot(self):
#         self.pub.publish(Float64MultiArray(data=[0.0]*2))

# def main():
#     rclpy.init()
#     node = TaskStabilizationNode()
#     t_ros = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
#     t_ros.start()
    
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
#     ax_traj, ax_v, ax_mu = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])   
    
#     ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
#     ax_traj.plot(node.target_pos[0], node.target_pos[1], 'bx', markersize=10, label='Target')
    
#     # Boundary
#     theta = np.linspace(0, 2*np.pi, 200); rx, ry, n = 1.1, 1.1, 4
#     x_cage = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**(2/n))
#     y_cage = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta))**(2/n))
#     ax_traj.plot(x_cage, y_cage, 'k--', alpha=0.3)
#     ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.grid(True)
    
#     ln_v, = ax_v.plot([], [], 'g-', label='Energy $V(t)$'); ln_vb, = ax_v.plot([], [], 'r--', label='Bound')
#     ax_v.set_title("Lyapunov Stability"); ax_v.grid(True); ax_v.legend()
#     ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction Magnitude $||\mu||$"); ax_mu.grid(True)

#     def update(frame):
#         with node.lock:
#             if not node.log['t'] or len(node.log['t']) < 2: return ln_a, ln_v, ln_vb, ln_mu
#             t, x, y, v, vb, mu = [np.array(node.log[k]) for k in ['t', 'x', 'y', 'V', 'V_bound', 'mu']]
#         ln_a.set_data(x, y); ln_v.set_data(t, v); ln_vb.set_data(t, vb); ln_mu.set_data(t, mu)
#         ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
#         ax_v.set_ylim(0, max(np.nanmax(v), np.nanmax(vb)) * 1.2); ax_mu.set_ylim(-0.1, np.nanmax(mu)*1.1 + 0.5)
#         return ln_a, ln_v, ln_vb, ln_mu

#     ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
#     plt.show()
#     node.stop_robot(); rclpy.shutdown()

# if __name__ == '__main__': main()
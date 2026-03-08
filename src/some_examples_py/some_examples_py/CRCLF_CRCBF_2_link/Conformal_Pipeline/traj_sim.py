#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

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

class MultiStateOfflineSim:
    def __init__(self, num_trials=15):
        urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.ee_id = self.model.getFrameId("endEffector")
        
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
        self.clf_ctrl = RESCLF_Controller(dim_task=2)
        self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
        self.target_pos = np.array([0.8, 0.2])
        self.dt = 0.001 
        self.sim_time = 7.0 
        self.tau_limits = np.array([20.0, 10.0]) 
        self.num_trials = num_trials

    def get_v_contour_points(self, target_v=150):
        samples = []
        P = self.clf_ctrl.P
        print(f"Sampling {self.num_trials} points on V = {target_v} contour...")
        while len(samples) < self.num_trials:
            x_rand = np.array([np.random.uniform(0.3, 1.2), np.random.uniform(-0.2, 1.2)])
            v_rand = np.random.uniform(-0.5, 0.5, 2) 
            eta = np.hstack((x_rand - self.target_pos, v_rand))
            V = eta.T @ P @ eta
            if np.abs(V - target_v) < (0.01 * target_v):
                samples.append({'pos': x_rand, 'vel': v_rand})
        return samples

    def run_simulation(self, init_state, use_robust=True):
        data = self.model.createData()
        q = np.array([0.0, 0.1]) 
        dq = init_state['vel'].copy()
        e_int = np.zeros(2)
        
        t_axis = np.arange(0, self.sim_time, self.dt)
        v_hist, q_hist, dq_hist = [], [], []
        quantile = self.sindy.q_quantile if use_robust else 0.0
        
        for t in t_axis:
            pin.forwardKinematics(self.model, data, q, dq)
            pin.updateFramePlacements(self.model, data)
            x_task = data.oMf[self.ee_id].translation[:2]
            J = pin.computeFrameJacobian(self.model, data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :]
            dj_dq = pin.getFrameAcceleration(self.model, data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
            dx_task = J @ dq

            err = x_task - self.target_pos
            e_int = np.clip(e_int + err * self.dt, -0.5, 0.5)
            u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, self.target_pos, np.zeros(2))
            u_nom -= (0.5 * e_int + 4.5 * dx_task)

            LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(x_task, dx_task, self.target_pos, np.zeros(2), u_nom, quantile, J)
            A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), quantile)
            mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_term, cbf_A=A_3d[:, :2], cbf_b=b_3d)

            if feasible:
                J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
                a_h, b_h = self.sindy.get_dynamics(q, dq)
                tau = np.linalg.pinv(b_h) @ (J_pinv @ (u_nom + mu - dj_dq) - a_h)
            else:
                tau = -50.0 * dq

            tau_applied = np.clip(tau, -self.tau_limits, self.tau_limits)
            ddq = pin.aba(self.model, data, q, dq, tau_applied - 3.5 * dq)
            dq += ddq * self.dt
            q = pin.integrate(self.model, q, dq * self.dt)
            
            v_hist.append(V); q_hist.append(q.copy()); dq_hist.append(dq.copy())
            
        return t_axis, np.array(v_hist), np.array(q_hist), np.array(dq_hist)

def main():
    sim = MultiStateOfflineSim(num_trials=15)
    init_states = sim.get_v_contour_points(target_v=150)
    
    fig, (ax_v, ax_p1, ax_p2) = plt.subplots(1, 3, figsize=(22, 7))
    r2d = 180 / np.pi 

    for i, state in enumerate(init_states):
        t, vr, qr, dqr = sim.run_simulation(state, use_robust=True)
        ax_v.plot(t, vr, 'dodgerblue', alpha=0.5, label='CR-CLF' if i==0 else "")
        ax_p1.plot(qr[:,0]*r2d, dqr[:,0]*r2d, 'dodgerblue', alpha=0.4)
        ax_p2.plot(qr[:,1]*r2d, dqr[:,1]*r2d, 'dodgerblue', alpha=0.4)

        _, vn, qn, dqn = sim.run_simulation(state, use_robust=False)
        ax_v.plot(t, vn, 'salmon', alpha=0.3, label='Regular CLF' if i==0 else "")
        ax_p1.plot(qn[:,0]*r2d, dqn[:,0]*r2d, 'salmon', alpha=0.3)
        ax_p2.plot(qn[:,1]*r2d, dqn[:,1]*r2d, 'salmon', alpha=0.3)
        
    # --- ADD ALPHA SHADING (Matching image_c87aa4.png) ---
    v_target_initial = 150
    vb = v_target_initial * np.exp(-sim.clf_ctrl.gamma * t)
    
    # Shading exponentially stable region
    ax_v.fill_between(t, 0, vb, color='lightblue', alpha=0.42)
    # Shading potentially unstable region
    ax_v.fill_between(t, vb, v_target_initial*1.2, color='mistyrose', alpha=0.78)
    
    # Add Text Annotations
    ax_v.text(t[len(t)//10], v_target_initial*0.1, "exp. stable", color='midnightblue', fontweight='bold', fontsize=12)
    ax_v.text(t[len(t)//2], v_target_initial*0.8, "not exp. stable\n(might be unstable)", color='darkred', fontweight='bold', fontsize=12)

    # Lyapunov Styling
    ax_v.plot(t, vb, 'k--', linewidth=3, label='Theoretical Bound')
    ax_v.set_title("Lyapunov Energy Decay with Stability Regions", fontsize=14)
    ax_v.set_xlabel("Time (s)"); ax_v.set_ylabel("V(x)")
    ax_v.set_ylim(0, v_target_initial*1.2)
    ax_v.grid(True, alpha=0.3); ax_v.legend()

    # Phase Portrait Styling
    ax_p1.set_title("Joint 1 Phase Portrait", fontsize=14); ax_p1.set_xlabel("Theta (deg)"); ax_p1.set_ylabel("dTheta (deg/s)")
    ax_p1.axhline(0, color='black', lw=1); ax_p1.axvline(0, color='black', lw=1); ax_p1.grid(True, linestyle='--', alpha=0.5)
    ax_p2.set_title("Joint 2 Phase Portrait", fontsize=14); ax_p2.set_xlabel("Theta (deg)"); ax_p2.set_ylabel("dTheta (deg/s)")
    ax_p2.axhline(0, color='black', lw=1); ax_p2.axvline(0, color='black', lw=1); ax_p2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__': main()









# #!/usr/bin/env python3

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import pinocchio as pin
# from ament_index_python.packages import get_package_share_directory

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

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

# class MultiStateOfflineSim:
#     def __init__(self, num_trials=15):
#         urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
#         self.model = pin.buildModelFromUrdf(urdf_path)
#         self.ee_id = self.model.getFrameId("endEffector")
        
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.clf_ctrl = RESCLF_Controller(dim_task=2)
#         self.cbf = CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4)
        
#         self.target_pos = np.array([0.8, 0.2])
#         self.dt = 0.001 
#         self.sim_time = 7.0 # Optimized for visualization
#         self.tau_limits = np.array([20.0, 10.0]) 
#         self.num_trials = num_trials

#     def get_v_contour_points(self, target_v=150):
#         samples = []
#         P = self.clf_ctrl.P
#         print(f"Sampling {self.num_trials} points on V = {target_v} contour...")
#         while len(samples) < self.num_trials:
#             x_rand = np.array([np.random.uniform(0.3, 1.2), np.random.uniform(-0.2, 1.2)])
#             # Super-tight velocity to match ideal plots
#             v_rand = np.random.uniform(-0.5, 0.5, 2) 
#             eta = np.hstack((x_rand - self.target_pos, v_rand))
#             V = eta.T @ P @ eta
#             if np.abs(V - target_v) < (0.01 * target_v):
#                 samples.append({'pos': x_rand, 'vel': v_rand})
#         return samples

#     def run_simulation(self, init_state, use_robust=True):
#         data = self.model.createData()
#         q = np.array([0.0, 0.1]) 
#         dq = init_state['vel'].copy()
#         e_int = np.zeros(2)
        
#         t_axis = np.arange(0, self.sim_time, self.dt)
#         v_history = []
#         quantile = self.sindy.q_quantile if use_robust else 0.0
        
#         for t in t_axis:
#             pin.forwardKinematics(self.model, data, q, dq)
#             pin.updateFramePlacements(self.model, data)
#             x_task = data.oMf[self.ee_id].translation[:2]
#             J = pin.computeFrameJacobian(self.model, data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :]
#             dj_dq = pin.getFrameAcceleration(self.model, data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
#             dx_task = J @ dq

#             # Control Logic Alignment
#             err = x_task - self.target_pos
#             e_int = np.clip(e_int + err * self.dt, -0.5, 0.5)
#             u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, self.target_pos, np.zeros(2))
#             u_nom -= (0.5 * e_int + 4.5 * dx_task) # Higher damping gain for stability

#             LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#                 x_task, dx_task, self.target_pos, np.zeros(2), u_nom, quantile, J
#             )
            
#             A_3d, b_3d = self.cbf.get_constraints(np.append(x_task, 0.0), np.append(dx_task, 0.0), np.append(u_nom, 0.0), quantile)
#             mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_term, cbf_A=A_3d[:, :2], cbf_b=b_3d)

#             if feasible:
#                 J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#                 a_h, b_h = self.sindy.get_dynamics(q, dq)
#                 tau = np.linalg.pinv(b_h) @ (J_pinv @ (u_nom + mu - dj_dq) - a_h)
#             else:
#                 tau = -50.0 * dq # Aggressive braking fallback

#             # Physics with enhanced damping parity
#             tau_applied = np.clip(tau, -self.tau_limits, self.tau_limits)
#             ddq = pin.aba(self.model, data, q, dq, tau_applied - 3.5 * dq)
#             dq += ddq * self.dt
#             q = pin.integrate(self.model, q, dq * self.dt)
#             v_history.append(V)
            
#         return t_axis, np.array(v_history)

# def main():
#     sim = MultiStateOfflineSim(num_trials=15)
#     init_states = sim.get_v_contour_points(target_v=150)
    
#     plt.figure(figsize=(12, 7))
#     for i, state in enumerate(init_states):
#         t, v_rob = sim.run_simulation(state, use_robust=True)
#         plt.plot(t, v_rob, 'dodgerblue', alpha=0.5, label='CR-CLF (Robust)' if i == 0 else "")
        
#         _, v_reg = sim.run_simulation(state, use_robust=False)
#         plt.plot(t, v_reg, 'salmon', alpha=0.3, label='Regular CLF (Q=0)' if i == 0 else "")
#         print(f"Trial {i+1}/15 complete.")

#     v_bound = 150 * np.exp(-sim.clf_ctrl.gamma * t)
#     plt.plot(t, v_bound, 'k--', linewidth=2.5, label=r'Theoretical Bound $V(x_0)e^{-\gamma t}$')

#     plt.title("Convergence Comparison: Multi-Trial Robustness Alignment", fontsize=14)
#     plt.xlabel("Time (s)", fontsize=12); plt.ylabel("Lyapunov Energy V(x)", fontsize=12)
#     plt.grid(True, which='both', linestyle='--', alpha=0.5); plt.legend(fontsize=11)
#     plt.show()

# if __name__ == '__main__': main()






# #!/usr/bin/env python3

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import pinocchio as pin
# from ament_index_python.packages import get_package_share_directory

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

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

# class OfflineSimulator:
#     def __init__(self):
#         # 1. Models and Data
#         urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
#         self.model = pin.buildModelFromUrdf(urdf_path)
#         self.ee_id = self.model.getFrameId("endEffector")
        
#         # 2. Controller & Predictor
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.clf_ctrl = RESCLF_Controller(dim_task=2)
        
#         # 3. Parameters
#         self.target_pos = np.array([0.8, 0.2])
#         self.dt = 0.001  # 1kHz resolution for accuracy
#         self.sim_time = 15.0
#         self.tau_limits = np.array([30.0, 20.0])

#     def run_simulation(self, use_robust=True):
#         """ Run a full offline simulation branch """
#         data = self.model.createData()
#         q = np.array([0.0, 0.1])
#         dq = np.zeros(self.model.nv)
#         e_int = np.zeros(2)
        
#         t_axis = np.arange(0, self.sim_time, self.dt)
#         v_history = []
#         x_history = []
        
#         # Determine robustness quantile
#         quantile = self.sindy.q_quantile if use_robust else 0.0
        
#         print(f"Running {'Robust (CR-CLF)' if use_robust else 'Regular (CLF)'} simulation...")
        
#         for t in t_axis:
#             # Kinematics
#             pin.forwardKinematics(self.model, data, q, dq)
#             pin.updateFramePlacements(self.model, data)
#             x_task = data.oMf[self.ee_id].translation[:2]
#             J = pin.computeFrameJacobian(self.model, data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :]
#             dj_dq = pin.getFrameAcceleration(self.model, data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
#             dx_task = J @ dq

#             # Control Logic
#             err = x_task - self.target_pos
#             e_int = np.clip(e_int + err * self.dt, -0.5, 0.5)
#             u_nom = self.clf_ctrl.get_nominal_acceleration(x_task, dx_task, self.target_pos, np.zeros(2))
#             u_nom -= (0.5 * e_int + 3.0 * dx_task) # Using your ki=0.5, kd=3.0

#             LfV, LgV, V, gamma, robust_term = self.clf_ctrl.get_lyapunov_constraints(
#                 x_task, dx_task, self.target_pos, np.zeros(2), u_nom, quantile, J
#             )
#             mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_term)

#             if feasible:
#                 J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#                 a_h, b_h = self.sindy.get_dynamics(q, dq)
#                 tau = np.linalg.pinv(b_h) @ (J_pinv @ (u_nom + mu - dj_dq) - a_h)
#             else:
#                 tau = -40.0 * dq

#             # Physics Integration (Semi-Implicit Euler)
#             tau_applied = np.clip(tau, -self.tau_limits, self.tau_limits)
#             ddq = pin.aba(self.model, data, q, dq, tau_applied - 5.0 * dq)
#             dq += ddq * self.dt
#             q = pin.integrate(self.model, q, dq * self.dt)

#             v_history.append(V)
#             x_history.append(x_task.copy())
            
#         return t_axis, np.array(v_history), np.array(x_history)

# def main():
#     sim = OfflineSimulator()
    
#     # 1. RUN BOTH BRANCHES
#     t, v_rob, x_rob = sim.run_simulation(use_robust=True)
#     _, v_reg, x_reg = sim.run_simulation(use_robust=False)
    
#     # 2. PLOTTING
#     fig, (ax_traj, ax_v) = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Trajectory Plot
#     ax_traj.plot(x_rob[:, 0], x_rob[:, 1], 'dodgerblue', label='CR-CLF (Robust)', linewidth=2)
#     ax_traj.plot(x_reg[:, 0], x_reg[:, 1], 'salmon', label='Regular CLF (Q=0)', linewidth=1.5)
#     ax_traj.plot(sim.target_pos[0], sim.target_pos[1], 'bx', markersize=10, label='Target')
#     ax_traj.set_title("Task Space Trajectory")
#     ax_traj.set_xlim(-1.2, 1.2); ax_traj.set_ylim(-1.2, 1.2)
#     ax_traj.grid(True); ax_traj.legend()
    
#     # Lyapunov Convergence Plot
#     ax_v.plot(t, v_rob, 'dodgerblue', label='Energy $V_{robust}$')
#     ax_v.plot(t, v_reg, 'salmon', label='Energy $V_{regular}$')
    
#     # Theoretical Bound
#     v0 = v_rob[0]
#     gamma = sim.clf_ctrl.gamma # Assuming gamma is stored in controller
#     v_bound = v0 * np.exp(-gamma * t)
#     ax_v.plot(t, v_bound, 'k--', alpha=0.6, label='Theoretical Bound')
    
#     ax_v.set_title("Lyapunov Energy Decay Comparison")
#     ax_v.set_xlabel("Time (s)"); ax_v.set_ylabel("Energy V(x)")
#     ax_v.grid(True); ax_v.legend()
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     main()





# #!/usr/bin/env python3

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# import pinocchio as pin
# from ament_index_python.packages import get_package_share_directory

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

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
#         H_x = np.array([1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12, dq0_sq, dq1_sq, dq_cross,
#                         dq0_sq*s2, dq1_sq*s2, dq_cross*s2, dq0_sq*c2, dq1_sq*c2, dq_cross*c2,
#                         np.sign(dq[0]), np.sign(dq[1])])
#         G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
#         a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
#         return a_hat, b_hat

# # --- STATE SAMPLER (V=150) ---
# def get_initial_states(P, target_pos, target_v=150, num=15):
#     samples = []
#     print(f"Sampling {num} points where initial energy V = {target_v}...")
#     while len(samples) < num:
#         x_rand = np.array([np.random.uniform(0.3, 1.2), np.random.uniform(-0.2, 1.2)])
#         v_rand = np.random.uniform(-0.65, 0.65, 2)
#         eta = np.hstack((x_rand - target_pos, v_rand))
#         V = eta.T @ P @ eta
#         if np.abs(V - target_v) < (0.01 * target_v):
#             samples.append({'pos': x_rand, 'vel': v_rand})
#     return samples

# class OfflinePhysicsSim:
#     def __init__(self):
#         urdf_path = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_fixed.urdf")
#         self.model = pin.buildModelFromUrdf(urdf_path)
#         self.data = self.model.createData()
#         self.ee_id = self.model.getFrameId("endEffector")
        
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
#         self.sindy = SINDyPredictor(ws_path + "sindy_Xi_state_space2.npy", ws_path + "q_quantile_state_space2.txt")
#         self.clf_ctrl = RESCLF_Controller(dim_task=2)
#         self.target_pos = np.array([0.8, 0.2])
#         self.dt = 0.01
#         self.sim_duration = 15.0

#     def run_trial(self, x0, use_robust=True):
#         q = np.array([0.0, 0.1]) 
#         dq = x0['vel'].copy()
        
#         V_log = []
#         t_axis = np.arange(0, self.sim_duration, self.dt)
#         quantile = self.sindy.q_quantile if use_robust else 0.0 #
#         error_int = np.zeros(2)

#         for _ in t_axis:
#             pin.forwardKinematics(self.model, self.data, q, dq)
#             pin.updateFramePlacements(self.model, self.data)
#             x_curr = self.data.oMf[self.ee_id].translation[:2]
#             J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :]
#             dj_dq = pin.getFrameAcceleration(self.model, self.data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
#             dx_curr = J @ dq

#             u_nom = self.clf_ctrl.get_nominal_acceleration(x_curr, dx_curr, self.target_pos, np.zeros(2))
#             error_int = np.clip(error_int + (x_curr - self.target_pos) * self.dt, -0.5, 0.5)
#             u_nom -= (0.5 * error_int + 25.0 * dx_curr)

#             LfV, LgV, V, gamma, robust = self.clf_ctrl.get_lyapunov_constraints(
#                 x_curr, dx_curr, self.target_pos, np.zeros(2), u_nom, quantile, J
#             )
#             mu, _ = solve_optimization(LfV, LgV, V, gamma, robust)
            
#             a_hat, b_hat = self.sindy.get_dynamics(q, dq)
#             J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#             tau = np.linalg.pinv(b_hat) @ (J_pinv @ (u_nom + mu - dj_dq) - a_hat)
            
#             # Physics loop with joint damping
#             ddq = pin.aba(self.model, self.data, q, dq, np.clip(tau, -20, 10) - 2.5*dq)
#             dq += ddq * self.dt
#             q = pin.integrate(self.model, q, dq * self.dt)
#             V_log.append(V)

#         return t_axis, V_log

# def main():
#     sim = OfflinePhysicsSim()
#     initial_states = get_initial_states(sim.clf_ctrl.P, sim.target_pos, 150, 15)
    
#     plt.figure(figsize=(10, 6))
    
#     print("\nRunning Offline Comparison...")
#     for i, state in enumerate(initial_states):
#         # CR-CLF (Robust)
#         t, V_robust = sim.run_trial(state, use_robust=True)
#         plt.plot(t, V_robust, color='dodgerblue', alpha=0.6, label='CR-CLF' if i==0 else "")
        
#         # Regular CLF (Non-Robust)
#         _, V_regular = sim.run_trial(state, use_robust=False)
#         plt.plot(t, V_regular, color='salmon', alpha=0.4, label='Regular CLF' if i==0 else "")
#         print(f"Trial {i+1}/15 complete.")

#     # Theoretical Decay Bound
#     V_bound = 150 * np.exp(-sim.clf_ctrl.gamma * t)
#     plt.plot(t, V_bound, 'k--', linewidth=3, label=r'$V(x_t) = V(x_0)e^{-\gamma t}$')

#     plt.title("Convergence Comparison: CR-CLF vs Regular CLF (15 Trials)")
#     plt.xlabel("Time (s)"); plt.ylabel("Energy V(x)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.show()

# if __name__ == '__main__':
#     main()
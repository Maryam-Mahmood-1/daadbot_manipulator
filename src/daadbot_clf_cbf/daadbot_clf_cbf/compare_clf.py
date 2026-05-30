import pinocchio as pin
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CLASSES
# ==========================================
class RobotSystem:
    def __init__(self, urdf_path, controlled_joints):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        full_model = pin.buildModelFromUrdf(urdf_path)
        joints_to_lock = [full_model.getJointId(n) for n in full_model.names 
                         if n not in controlled_joints and n != "universe"]
        self.model = pin.buildReducedModel(full_model, joints_to_lock, pin.neutral(full_model))
        self.data = self.model.createData()
        self.effort_limit = self.model.effortLimit

    def get_dynamics(self, q, dq):
        pin.computeAllTerms(self.model, self.data, q, dq)
        return self.data.M, self.data.nle

class RESCLF_Controller:
    def __init__(self, robot_model, num_joints, gamma=1.0, quantiles=None):
        self.robot = robot_model
        self.n = num_joints
        self.gamma = gamma 
        
        # Handle Quantiles (Scalar or Vector)
        if quantiles is None:
            self.quantiles = np.zeros(self.n)
        elif isinstance(quantiles, (list, np.ndarray)):
            self.quantiles = np.array(quantiles)
        else:
            self.quantiles = np.full(self.n, quantiles)

        # Linear System (Double Integrator)
        self.A = np.zeros((2*self.n, 2*self.n))
        self.A[:self.n, self.n:] = np.eye(self.n)
        self.B = np.zeros((2*self.n, self.n))
        self.B[self.n:, :] = np.eye(self.n)
        
        # Tuning Gains
        Kp = 2.0 * np.eye(self.n) 
        Kd = 3.0 * np.eye(self.n)
        K = np.hstack([Kp, Kd])
        
        A_cl = self.A - self.B @ K
        Q = np.eye(2 * self.n)
        self.P = solve_continuous_lyapunov(A_cl.T, -Q)

    def V_func(self, z):
        return z.T @ self.P @ z

    def get_control(self, q, dq, q_des):
        error = q - q_des
        z = np.hstack([error, dq])
        
        dV_dz = 2 * z.T @ self.P     # Gradient (Size 14)
        LfV = dV_dz @ (self.A @ z)
        LgV = dV_dz @ self.B
        V_val = self.V_func(z)
        
        # --- PER-JOINT ROBUSTNESS ---
        # The disturbance enters the velocity/acceleration channels (last n elements of z)
        # Robust Term = Sum( |dV_dz_velocity_i| * quantile_i )
        
        dV_dz_vel = dV_dz[self.n:] # Extract gradient w.r.t velocity
        uncertainty = np.abs(dV_dz_vel) @ self.quantiles
        
        upper_bound = -self.gamma * V_val - LfV - uncertainty
        
        # Constraints & Limits
        M_hat, nle_hat = self.robot.get_dynamics(q, dq)
        tau_lim = self.robot.effort_limit
        
        def constraint_clf(x): return upper_bound - (LgV @ x[:self.n] - x[self.n])
        def const_tau_max(x): return (tau_lim - nle_hat) - (M_hat @ x[:self.n])
        def const_tau_min(x): return (tau_lim + nle_hat) - (-M_hat @ x[:self.n])
        
        p_slack = 1.0e8
        def cost(x): return 0.5*np.sum(x[:self.n]**2) + p_slack*(x[self.n]**2)
        
        x0 = np.zeros(self.n + 1)
        cons = [{'type':'ineq', 'fun':c} for c in [constraint_clf, const_tau_max, const_tau_min]]
        
        res = minimize(cost, x0, constraints=cons, method='SLSQP')
        u_aux = res.x[:self.n] if res.success else np.zeros(self.n)
        
        tau = M_hat @ u_aux + nle_hat
        return np.clip(tau, -tau_lim, tau_lim), V_val

# ==========================================
# 2. SIMULATION
# ==========================================
def run_sim_loop(ctrl, sys_true, q0, dq0, q_des, dt, steps):
    q, dq = q0.copy(), dq0.copy()
    v_hist = []
    q_hist = [] # Store full position state
    
    for _ in range(steps):
        tau, V = ctrl.get_control(q, dq, q_des)
        
        v_hist.append(V)
        q_hist.append(q.copy())
        
        M, nle = sys_true.get_dynamics(q, dq)
        ddq = np.linalg.solve(M, tau - nle)
        dq += ddq * dt
        q += dq * dt
        
    return np.array(v_hist), np.array(q_hist)

# ==========================================
# 3. MAIN
# ==========================================
def main():
    base = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/"
    joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
    
    try:
        sys_true = RobotSystem(base+"daadbot.urdf", joints)
        sys_noisy = RobotSystem(base+"daadbot_noisy.urdf", joints)
    except Exception as e: print(e); return

    # --- DATA FROM YOUR LOGS ---
    # Specific mismatch per joint
    joint_quantiles = [0.1589, 0.0994, 0.3474, 0.3287, 0.2168, 0.6331, 0.3997]

    # --- CONTROLLERS ---
    # 1. Ideal: True Model, No Robustness
    c_ideal = RESCLF_Controller(sys_true, 7, gamma=1.0, quantiles=0.0)
    
    # 2. Standard: Noisy Model, No Robustness
    c_std = RESCLF_Controller(sys_noisy, 7, gamma=1.0, quantiles=0.0)
    
    # 3. Robust: Noisy Model, WITH Per-Joint Robustness
    c_rob = RESCLF_Controller(sys_noisy, 7, gamma=1.0, quantiles=joint_quantiles)

    # Setup
    q0 = np.random.uniform(-0.5, 0.5, 7)
    dq0, q_des = np.zeros(7), np.zeros(7)
    dt, steps = 0.002, 2000
    
    print("Running Simulations...")
    v1, q1 = run_sim_loop(c_ideal, sys_true, q0, dq0, q_des, dt, steps)
    v2, q2 = run_sim_loop(c_std, sys_true, q0, dq0, q_des, dt, steps)
    v3, q3 = run_sim_loop(c_rob, sys_true, q0, dq0, q_des, dt, steps)
    
    # ==========================================
    # 4. PLOTTING (Grid for all joints)
    # ==========================================
    t = np.arange(steps)*dt
    
    # Create figure: 4 rows, 2 columns
    fig, axs = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle("Robust Control Comparison (Per-Joint Quantiles)", fontsize=16)
    
    # Plot V(x) in first slot
    ax_vx = axs[0, 0]
    ax_vx.plot(t, v1, 'g:', label='Ideal')
    ax_vx.plot(t, v2, 'r--', label='Standard')
    ax_vx.plot(t, v3, 'b-', label='Robust (Per-Joint)')
    ax_vx.set_title("Global Energy V(x)")
    ax_vx.set_yscale('log')
    ax_vx.grid(True)
    ax_vx.legend()
    
    # Plot Joints 1-7 in remaining slots
    # We map linear index 0-6 to the subplot grid
    plot_indices = [(0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
    
    for j_idx, (r, c) in enumerate(plot_indices):
        ax = axs[r, c]
        
        # Plot Trajectories
        # Subtract q_des to show error, or just show raw position if q_des is 0
        # Here we show Error (q - q_des)
        ax.plot(t, q1[:, j_idx], 'g:', label='Ideal')
        ax.plot(t, q2[:, j_idx], 'r--', label='Standard')
        ax.plot(t, q3[:, j_idx], 'b-', label='Robust')
        
        # Annotate Quantile used
        q_val = joint_quantiles[j_idx]
        ax.set_title(f"Joint {j_idx+1} Error (Quantile={q_val})")
        ax.grid(True)
        
        if j_idx == 0: ax.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
    plt.show()

if __name__ == "__main__":
    main()

# import pinocchio as pin
# import numpy as np
# from scipy.linalg import solve_continuous_lyapunov
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# import os

# # ==========================================
# # 1. CLASSES (Robot & Controller)
# # ==========================================
# class RobotSystem:
#     def __init__(self, urdf_path, controlled_joints):
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF not found: {urdf_path}")
#         full_model = pin.buildModelFromUrdf(urdf_path)
#         joints_to_lock = [full_model.getJointId(n) for n in full_model.names 
#                          if n not in controlled_joints and n != "universe"]
#         self.model = pin.buildReducedModel(full_model, joints_to_lock, pin.neutral(full_model))
#         self.data = self.model.createData()
#         self.effort_limit = self.model.effortLimit

#     def get_dynamics(self, q, dq):
#         pin.computeAllTerms(self.model, self.data, q, dq)
#         return self.data.M, self.data.nle

# class RESCLF_Controller:
#     def __init__(self, robot_model, num_joints, gamma=1.0, quantile=0.0):
#         self.robot = robot_model
#         self.n = num_joints
#         self.gamma = gamma 
#         self.quantile = quantile 
        
#         # Linear System
#         self.A = np.zeros((2*self.n, 2*self.n))
#         self.A[:self.n, self.n:] = np.eye(self.n)
#         self.B = np.zeros((2*self.n, self.n))
#         self.B[self.n:, :] = np.eye(self.n)
        
#         # Tuning Gains
#         Kp = 2.0 * np.eye(self.n) 
#         Kd = 3.0 * np.eye(self.n)
#         K = np.hstack([Kp, Kd])
        
#         A_cl = self.A - self.B @ K
#         Q = np.eye(2 * self.n)
#         self.P = solve_continuous_lyapunov(A_cl.T, -Q)

#     def V_func(self, z):
#         return z.T @ self.P @ z

#     def get_control(self, q, dq, q_des):
#         error = q - q_des
#         z = np.hstack([error, dq])
        
#         dV_dz = 2 * z.T @ self.P
#         LfV = dV_dz @ (self.A @ z)
#         LgV = dV_dz @ self.B
#         V_val = self.V_func(z)
        
#         # Robustness
#         uncertainty = np.linalg.norm(dV_dz) * self.quantile
#         upper_bound = -self.gamma * V_val - LfV - uncertainty
        
#         # Dynamics & Limits
#         M_hat, nle_hat = self.robot.get_dynamics(q, dq)
#         tau_lim = self.robot.effort_limit
        
#         # QP Constraints
#         def constraint_clf(x): return upper_bound - (LgV @ x[:self.n] - x[self.n])
#         def const_tau_max(x): return (tau_lim - nle_hat) - (M_hat @ x[:self.n])
#         def const_tau_min(x): return (tau_lim + nle_hat) - (-M_hat @ x[:self.n])
        
#         # QP Cost
#         def cost(x): return 0.5*np.sum(x[:self.n]**2) + 1.0e8*(x[self.n]**2)
        
#         x0 = np.zeros(self.n + 1)
#         cons = [{'type':'ineq', 'fun':c} for c in [constraint_clf, const_tau_max, const_tau_min]]
        
#         res = minimize(cost, x0, constraints=cons, method='SLSQP')
#         u_aux = res.x[:self.n] if res.success else np.zeros(self.n)
        
#         tau = M_hat @ u_aux + nle_hat
#         return np.clip(tau, -tau_lim, tau_lim), V_val

# # ==========================================
# # 2. SIMULATION UTILS
# # ==========================================
# def run_sim_loop(ctrl, sys_true, q0, dq0, q_des, dt, steps):
#     q, dq = q0.copy(), dq0.copy()
#     v_hist, j2_err_hist = [], []
    
#     for _ in range(steps):
#         tau, V = ctrl.get_control(q, dq, q_des)
        
#         # Record Data
#         v_hist.append(V)
#         j2_err_hist.append(q[1] - q_des[1]) # Joint 2 is index 1
        
#         # Physics Step
#         M, nle = sys_true.get_dynamics(q, dq)
#         ddq = np.linalg.solve(M, tau - nle)
#         dq += ddq * dt
#         q += dq * dt
        
#     return v_hist, j2_err_hist

# # ==========================================
# # 3. MAIN
# # ==========================================
# def main():
#     base = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/"
#     joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
    
#     try:
#         sys_true = RobotSystem(base+"daadbot.urdf", joints)
#         sys_noisy = RobotSystem(base+"daadbot_noisy.urdf", joints)
#     except Exception as e: print(e); return

#     # --- CONTROLLERS ---
#     # 1. Ideal: Knows the TRUE model (Baseline)
#     c_ideal = RESCLF_Controller(sys_true, 7, gamma=1.0, quantile=0.0)
    
#     # 2. Standard: Uses NOISY model, trusts it 100%
#     c_std = RESCLF_Controller(sys_noisy, 7, gamma=1.0, quantile=0.0)
    
#     # 3. Robust: Uses NOISY model, but adds Uncertainty Buffer
#     c_rob = RESCLF_Controller(sys_noisy, 7, gamma=1.0, quantile=0.2)

#     # Setup
#     q0 = np.random.uniform(-0.5, 0.5, 7)
#     dq0, q_des = np.zeros(7), np.zeros(7)
#     dt, steps = 0.002, 2000
    
#     print("Simulating 3 Scenarios...")
#     v1, j2_1 = run_sim_loop(c_ideal, sys_true, q0, dq0, q_des, dt, steps)
#     v2, j2_2 = run_sim_loop(c_std, sys_true, q0, dq0, q_des, dt, steps)
#     v3, j2_3 = run_sim_loop(c_rob, sys_true, q0, dq0, q_des, dt, steps)
    
#     # --- PLOTTING ---
#     t = np.arange(steps)*dt
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
#     # Plot 1: Energy V(x)
#     ax1.plot(t, v1, 'g:', linewidth=2, label='Ideal (True Model)')
#     ax1.plot(t, v2, 'r--', linewidth=2, label='Standard (Noisy Model)')
#     ax1.plot(t, v3, 'b-', linewidth=2, label='Robust CR-CLF (Noisy Model)')
#     ax1.set_title("Global Energy Convergence V(x)")
#     ax1.set_yscale('log')
#     ax1.legend()
#     ax1.grid(True)
    
#     # Plot 2: Joint 2 Error
#     ax2.plot(t, j2_1, 'g:', linewidth=2, label='Ideal')
#     ax2.plot(t, j2_2, 'r--', linewidth=2, label='Standard')
#     ax2.plot(t, j2_3, 'b-', linewidth=2, label='Robust')
#     ax2.set_title("Joint 2 Position Error (q - q_des)")
#     ax2.set_ylabel("Error (rad)")
#     ax2.set_xlabel("Time (s)")
#     ax2.legend()
#     ax2.grid(True)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()



# import pinocchio as pin
# import numpy as np
# from scipy.linalg import solve_continuous_lyapunov
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# import os

# # ==========================================
# # 1. ROBOT SYSTEM
# # ==========================================
# class RobotSystem:
#     def __init__(self, urdf_path, controlled_joints):
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF not found at: {urdf_path}")
#         full_model = pin.buildModelFromUrdf(urdf_path)
#         joints_to_lock = [full_model.getJointId(n) for n in full_model.names 
#                          if n not in controlled_joints and n != "universe"]
#         q_ref = pin.neutral(full_model) 
#         self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
#         self.data = self.model.createData()
#         self.nq = self.model.nq
#         self.nv = self.model.nv

#     def get_dynamics(self, q, dq):
#         pin.computeAllTerms(self.model, self.data, q, dq)
#         return self.data.M, self.data.nle

# # ==========================================
# # 2. CONTROLLER (With Torque Saturation)
# # ==========================================
# class RESCLF_Controller:
#     def __init__(self, robot_model, num_joints, gamma=1.0, quantile=0.0):
#         self.robot = robot_model
#         self.n = num_joints
#         self.gamma = gamma 
#         self.quantile = quantile 
        
#         # Reduced Gains for Stability
#         self.A = np.zeros((2 * self.n, 2 * self.n))
#         self.A[:self.n, self.n:] = np.eye(self.n)
#         self.B = np.zeros((2 * self.n, self.n))
#         self.B[self.n:, :] = np.eye(self.n)
        
#         # TUNING: Lower Kp to prevent overshoot
#         Kp = 2.0 * np.eye(self.n) 
#         Kd = 3.0 * np.eye(self.n)
#         K = np.hstack([Kp, Kd])
        
#         A_cl = self.A - self.B @ K
#         Q = np.eye(2 * self.n)
#         self.P = solve_continuous_lyapunov(A_cl.T, -Q)

#     def V_func(self, z):
#         return z.T @ self.P @ z

#     def get_control(self, q, dq, q_des):
#         error = q - q_des
#         z = np.hstack([error, dq])
        
#         dV_dz = 2 * z.T @ self.P
#         LfV = dV_dz @ (self.A @ z)
#         LgV = dV_dz @ self.B
#         V_val = self.V_func(z)
        
#         norm_grad_V = np.linalg.norm(dV_dz)
#         uncertainty_buffer = norm_grad_V * self.quantile
#         upper_bound = -self.gamma * V_val - LfV - uncertainty_buffer
        
#         # TUNING: Add Acceleration Limits (u_max)
#         u_max = 5.0 # rad/s^2 limit
        
#         def cost(x):
#             u = x[:self.n]
#             delta = x[self.n]
#             return 0.5 * np.sum(u**2) + 1.0e6 * (delta**2)
        
#         def constraint(x):
#             u = x[:self.n]
#             delta = x[self.n]
#             return upper_bound - (LgV @ u - delta)
        
#         # Add Bounds to QP (Saturation)
#         bounds = [(-u_max, u_max) for _ in range(self.n)] + [(None, None)]
        
#         x0 = np.zeros(self.n + 1)
#         res = minimize(cost, x0, constraints={'type': 'ineq', 'fun': constraint}, 
#                        bounds=bounds, method='SLSQP')
        
#         u_aux = res.x[:self.n] if res.success else np.zeros(self.n)
        
#         # Calculate Torque
#         M_hat, nle_hat = self.robot.get_dynamics(q, dq)
#         tau = M_hat @ u_aux + nle_hat
        
#         # Optional: Clip Final Torque (e.g. 50Nm limit)
#         tau = np.clip(tau, -50, 50)
        
#         return tau, V_val

# # ==========================================
# # 3. MAIN
# # ==========================================
# def run_simulation(controller, physics_engine, q0, dq0, q_des, dt, steps):
#     q = q0.copy()
#     dq = dq0.copy()
#     V_history = []
    
#     for i in range(steps):
#         tau, V = controller.get_control(q, dq, q_des)
#         V_history.append(V)
        
#         M_true, nle_true = physics_engine.get_dynamics(q, dq)
#         ddq = np.linalg.solve(M_true, tau - nle_true)
        
#         dq += ddq * dt
#         q += dq * dt
        
#     return V_history

# def main():
#     base_path = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/"
#     urdf_true = base_path + "daadbot.urdf"
#     urdf_noisy = base_path + "daadbot_noisy.urdf"
#     target_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

#     try:
#         sys_true = RobotSystem(urdf_true, target_joints)
#         sys_learned = RobotSystem(urdf_noisy, target_joints)
#     except FileNotFoundError as e:
#         print(e)
#         return

#     # TUNING: Reduced Gamma (Decay Rate)
#     # Lower Gamma = Slower convergence, but MORE stable
#     ctrl_ideal = RESCLF_Controller(sys_true, 7, gamma=1.0, quantile=0.0)
#     ctrl_naive = RESCLF_Controller(sys_learned, 7, gamma=1.0, quantile=0.0)
    
#     # TUNING: Reduced Quantile
#     # 0.55 was too aggressive. Try 0.2
#     ctrl_robust = RESCLF_Controller(sys_learned, 7, gamma=1.0, quantile=0.2)

#     q_start = np.random.uniform(-0.5, 0.5, 7)
#     dq_start = np.zeros(7)
#     q_des = np.zeros(7)
    
#     # TUNING: Faster Simulation Step (Prevent Integration Errors)
#     dt = 0.001 
#     steps = 3000 # Increased steps since dt is smaller
    
#     print("Running Sims...")
#     v_ideal = run_simulation(ctrl_ideal, sys_true, q_start, dq_start, q_des, dt, steps)
#     v_naive = run_simulation(ctrl_naive, sys_true, q_start, dq_start, q_des, dt, steps)
#     v_robust = run_simulation(ctrl_robust, sys_true, q_start, dq_start, q_des, dt, steps)
    
#     time_axis = np.arange(steps) * dt
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_axis, v_ideal, 'g:', label='Ideal (True Model)', linewidth=3, alpha=0.6)
#     plt.plot(time_axis, v_naive, 'r--', label='Standard CLF (Noisy Model)', linewidth=2)
#     plt.plot(time_axis, v_robust, 'b-', label='Robust CR-CLF (Noisy Model)', linewidth=2)
    
#     plt.title("Comparison with Torque Limits & Tuned Gains")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Lyapunov Energy V(x)")
#     plt.yscale('log')
#     plt.grid(True, which="both", alpha=0.4)
#     plt.legend()
#     plt.show()

# if __name__ == '__main__':
#     main()
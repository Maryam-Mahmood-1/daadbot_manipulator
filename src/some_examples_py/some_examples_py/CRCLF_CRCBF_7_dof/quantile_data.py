"""
FINAL PIPELINE: High-Torque Excitation + Ellipse Validation + FULL PLOTTING
"""
import os
import sys
import numpy as np
import math
import pinocchio as pin
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from ament_index_python.packages import get_package_share_directory
import pickle

# --- 1. CONFIGURATIONS ---
URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf"
)
DT = 0.002              # 500 Hz
DURATION = 4500.0       # Long duration
NOISE_LEVEL = 0.002     # Low noise
CONFIDENCE = 0.1        # 90% Confidence

# Safety Limits
MAX_VEL = np.deg2rad(150)
MAX_TORQUE = 30.0

# --- 2. TRAJECTORY GENERATOR ---
class TrajectoryGenerator:
    def __init__(self):
        self.center_pos = np.array([0.0, 0.0, 0.0])
        self.ellipse_a = 1.6 
        self.ellipse_b = 0.9
        self.period = 12.0     
        self.omega = 2 * np.pi / self.period
        self.approach_duration = 5.0
        self.start_pos = None  
        self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])

    def get_ref(self, t, current_actual_pos=None):
        if t < self.approach_duration:
            if self.start_pos is None:
                if current_actual_pos is None: return np.zeros(3), np.zeros(3), np.zeros(3)
                self.start_pos = current_actual_pos
            tau = t / self.approach_duration
            s = (1.0 - math.cos(tau * math.pi)) / 2.0
            ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
            dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)
            vector_diff = self.orbit_start_pos - self.start_pos
            return self.start_pos + (vector_diff * s), vector_diff * ds, vector_diff * dds
        else:
            t_orbit = t - self.approach_duration
            x_des = self.center_pos.copy()
            x_des[0] += self.ellipse_a * np.cos(self.omega * t_orbit)
            x_des[1] += self.ellipse_b * np.sin(self.omega * t_orbit)
            dx_des = np.zeros(3)
            dx_des[0] = -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit)
            dx_des[1] =  self.ellipse_b * self.omega * np.cos(self.omega * t_orbit)
            ddx_des = np.zeros(3)
            ddx_des[0] = -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit)
            ddx_des[1] = -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit)
            return x_des, dx_des, ddx_des

# --- 3. EXCITATION GENERATOR ---
def generate_rich_safe_excitation(t, dofs=2):
    tau = np.zeros(dofs)
    amp_scale = (np.sin(0.1 * t) + 1.0) / 2.0 
    mag = 25.0 * amp_scale 

    for i in range(dofs):
        tau[i] = mag * (0.6 * np.sin(0.5 * t) + 
                        0.3 * np.cos(1.5 * t + i) +
                        0.1 * np.sin(3.0 * t))
    return tau

# --- 4. PHYSICS HELPERS ---
def build_feature_matrix(X):
    q1, q2 = X[:, 0], X[:, 1]
    dq1, dq2 = X[:, 2], X[:, 3]
    tau1, tau2 = X[:, 4], X[:, 5]
    
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    
    dq1_sq = dq1**2
    dq2_sq = dq2**2
    
    return np.stack([tau1, tau2, dq1, dq2, s1, c1, s2, c2, s12, c12, dq1_sq, dq2_sq], axis=1)

def save_model(model, q_hat_x, q_hat_y, filename="my_learned_robot2.pkl"):
    abs_path = os.path.abspath(filename)
    data = {"model": model, "safety_bounds": {"x": q_hat_x, "y": q_hat_y}}
    try:
        with open(abs_path, "wb") as f:
            pickle.dump(data, f)
        print(f"\n[SUCCESS] Model Saved to: {abs_path}")
        print(f"Safety Bounds -> X: {q_hat_x:.4f}, Y: {q_hat_y:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")

def run_pipeline():
    print("=== STEP 1: GENERATING RICH DATASET (High Torque) ===")
    
    try:
        model_phys = pin.buildModelFromUrdf(URDF_PATH)
        data_phys = model_phys.createData()
        ee_id = model_phys.getFrameId("endEffector")
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return

    # Add Mass Mismatch (-20%)
    for i in range(1, len(model_phys.inertias)):
        model_phys.inertias[i].mass *= 0.80 

    # --- SIMULATION LOOP ---
    q = pin.neutral(model_phys)
    v = np.zeros(model_phys.nv)
    
    steps = int(DURATION / DT)
    X_list, Y_list = [], []
    
    for step in range(steps):
        t = step * DT
        
        # 1. Control
        tau_cmd = generate_rich_safe_excitation(t)
        tau_cmd = np.clip(tau_cmd, -MAX_TORQUE, MAX_TORQUE)
        
        # 2. Physics
        friction = 0.1 * v
        tau_real = tau_cmd - friction
        ddq = pin.aba(model_phys, data_phys, q, v, tau_real)
        
        # 3. Ground Truth
        pin.computeJointJacobians(model_phys, data_phys, q)
        pin.forwardKinematics(model_phys, data_phys, q, v, ddq)
        pin.updateFramePlacements(model_phys, data_phys)
        J = pin.getFrameJacobian(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2]
        dJdq = pin.getFrameJacobianTimeVariation(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v[:2]
        ddx_true = (J @ ddq[:2]) + dJdq
        
        # 4. Record
        q_n = q[:2] + np.random.normal(0, NOISE_LEVEL, 2)
        v_n = v[:2] + np.random.normal(0, NOISE_LEVEL, 2)
        X_list.append(np.hstack([q_n, v_n, tau_cmd]))
        Y_list.append(ddx_true)
        
        # 5. Integrate
        v += ddq * DT
        v = np.clip(v, -MAX_VEL, MAX_VEL)
        q = pin.integrate(model_phys, q, v * DT)

    X = np.array(X_list)
    Y = np.array(Y_list)

    # --- SPLIT DATA ---
    n = len(X)
    n_train = int(0.6 * n)
    n_cal = int(0.8 * n)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_cal, Y_cal     = X[n_train:n_cal], Y[n_train:n_cal]
    X_test, Y_test   = X[n_cal:], Y[n_cal:]
    print(f"Data Shapes -> Train: {X_train.shape}, Cal: {X_cal.shape}, Test: {X_test.shape}")

    print("\n=== STEP 2: TRAINING MODEL ===")
    Phi_train = build_feature_matrix(X_train)
    
    # [CRITICAL FIX] Degree 2 prevents "cubic explosion"
    # [CRITICAL FIX] Alpha 0.5 increases stability against noise
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True), 
        StandardScaler(),
        Ridge(alpha=0.00001) 
    )
    model.fit(Phi_train, Y_train)
    print("   Model Trained (Degree=2, Alpha=0.5).")

    print("\n=== STEP 3: CONFORMAL PREDICTION (CALIBRATION) ===")
    Phi_cal = build_feature_matrix(X_cal)
    Y_pred_cal = model.predict(Phi_cal)
    residuals = np.abs(Y_cal - Y_pred_cal)
    
    n_cal_samples = residuals.shape[0]
    q_idx = int(np.ceil((n_cal_samples + 1) * (1 - CONFIDENCE)))
    q_hat_x = np.sort(residuals[:, 0])[min(q_idx, n_cal_samples-1)]
    q_hat_y = np.sort(residuals[:, 1])[min(q_idx, n_cal_samples-1)]
    
    print(f"Safety Bounds: X={q_hat_x:.3f}, Y={q_hat_y:.3f} m/s^2")
    
    save_model(model, q_hat_x, q_hat_y)

    print("\n=== STEP 4: VISUALIZATION (TEST SET) ===")
    Phi_test = build_feature_matrix(X_test)
    Y_pred_test = model.predict(Phi_test)
    t_test = np.arange(len(Y_test)) * DT
    
    rmse = np.sqrt(np.mean((Y_test - Y_pred_test)**2, axis=0))
    print(f"Random Test RMSE: X={rmse[0]:.4f}, Y={rmse[1]:.4f}")

    # [FIGURE 1] Standard Random Test Set
    PLOT_STEPS = 50000 
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8))
    fig1.canvas.manager.set_window_title('Figure 1: Standard Random Test Set')
    
    axs1[0].plot(t_test[:PLOT_STEPS], Y_test[:PLOT_STEPS, 0], 'k-', label='True Physics', lw=2, alpha=0.7)
    axs1[0].plot(t_test[:PLOT_STEPS], Y_pred_test[:PLOT_STEPS, 0], 'b--', label='Learned Model', lw=1.5)
    axs1[0].fill_between(t_test[:PLOT_STEPS], 
                    Y_pred_test[:PLOT_STEPS, 0] - q_hat_x, 
                    Y_pred_test[:PLOT_STEPS, 0] + q_hat_x, 
                    color='blue', alpha=0.2, label='Safety Tube')
    axs1[0].set_title(f"X-Axis Acceleration (Test RMSE: {rmse[0]:.3f})")
    axs1[0].legend(loc="upper right"); axs1[0].grid(True, alpha=0.5)
    
    axs1[1].plot(t_test[:PLOT_STEPS], Y_test[:PLOT_STEPS, 1], 'k-', label='True Physics', lw=2, alpha=0.7)
    axs1[1].plot(t_test[:PLOT_STEPS], Y_pred_test[:PLOT_STEPS, 1], 'r--', label='Learned Model', lw=1.5)
    axs1[1].fill_between(t_test[:PLOT_STEPS], 
                    Y_pred_test[:PLOT_STEPS, 1] - q_hat_y, 
                    Y_pred_test[:PLOT_STEPS, 1] + q_hat_y, 
                    color='red', alpha=0.2, label='Safety Tube')
    axs1[1].set_title(f"Y-Axis Acceleration (Test RMSE: {rmse[1]:.3f})")
    axs1[1].legend(loc="upper right"); axs1[1].grid(True, alpha=0.5)

    # === STEP 5: VERIFY ON ACTUAL ELLIPSE TRAJECTORY ===
    # === STEP 5: VERIFY ON ACTUAL ELLIPSE TRAJECTORY ===
    print("\n=== STEP 5: VERIFYING ON ELLIPTICAL TRAJECTORY (Closed-Loop) ===")
    
    traj_gen = TrajectoryGenerator()
    q = pin.neutral(model_phys)
    v = np.zeros(model_phys.nv)
    
    X_ell, Y_ell = [], []
    
    # [FIX 1] Create separate lists for Actual and Desired paths
    pos_actual = []
    pos_desired = []
    
    Kp, Kd = 100.0, 20.0
    
    for i in range(int(20.0 / DT)):
        t = i * DT
        
        pin.forwardKinematics(model_phys, data_phys, q, v)
        pin.updateFramePlacements(model_phys, data_phys)
        x_curr = data_phys.oMf[ee_id].translation.copy() # [FIX] Add .copy()
        
        J = pin.computeFrameJacobian(model_phys, data_phys, q, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :2]
        dx_curr = J @ v[:2]
        
        xd, vd, _ = traj_gen.get_ref(t, x_curr)
        
        # [FIX 2] Store position data for plotting
        pos_actual.append(x_curr[:2]) 
        pos_desired.append(xd[:2])

        ep = xd - x_curr
        ev = vd - dx_curr
        F_task = Kp * ep + Kd * ev
        tau_cmd = J.T @ F_task[:3] 
        
        tau_cmd = np.clip(tau_cmd, -MAX_TORQUE, MAX_TORQUE)
        
        friction = 0.1 * v 
        tau_real = tau_cmd[:2] - friction
        ddq = pin.aba(model_phys, data_phys, q, v, tau_real)
        
        dJdq = pin.getFrameJacobianTimeVariation(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v[:2]
        ddx_true = (J[:2, :2] @ ddq[:2]) + dJdq
        
        X_ell.append(np.hstack([q[:2], v[:2], tau_cmd[:2]]))
        Y_ell.append(ddx_true)
        
        v += ddq * DT
        q = pin.integrate(model_phys, q, v * DT)

    X_test_ell = np.array(X_ell)
    Y_test_ell = np.array(Y_ell)
    t_test_ell = np.arange(len(Y_test_ell)) * DT
    
    # [FIX 3] Convert lists to arrays for plotting
    path_act = np.array(pos_actual)
    path_des = np.array(pos_desired)

    Phi_test_ell = build_feature_matrix(X_test_ell)
    Y_pred_ell = model.predict(Phi_test_ell)
    
    rmse_ell = np.sqrt(np.mean((Y_test_ell - Y_pred_ell)**2, axis=0))
    print(f"Ellipse Test RMSE: X={rmse_ell[0]:.4f}, Y={rmse_ell[1]:.4f}")

    # --- PLOTTING ---
    
    # [FIGURE 1] Standard Random Test Set
    # ... (Keep existing code for Fig 1) ...

    # [FIGURE 2] Ellipse Acceleration
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8))
    fig2.canvas.manager.set_window_title('Figure 2: Ellipse Accelerations')
    
    axs2[0].plot(t_test_ell, Y_test_ell[:, 0], 'k-', label='True Physics', lw=2, alpha=0.7)
    axs2[0].plot(t_test_ell, Y_pred_ell[:, 0], 'b--', label='Learned Model', lw=1.5)
    axs2[0].fill_between(t_test_ell, Y_pred_ell[:, 0] - q_hat_x, Y_pred_ell[:, 0] + q_hat_x, color='blue', alpha=0.2)
    axs2[0].set_title(f"X-Axis Acceleration (RMSE: {rmse_ell[0]:.3f})")
    axs2[0].legend()
    
    axs2[1].plot(t_test_ell, Y_test_ell[:, 1], 'k-', label='True Physics', lw=2, alpha=0.7)
    axs2[1].plot(t_test_ell, Y_pred_ell[:, 1], 'r--', label='Learned Model', lw=1.5)
    axs2[1].fill_between(t_test_ell, Y_pred_ell[:, 1] - q_hat_y, Y_pred_ell[:, 1] + q_hat_y, color='red', alpha=0.2)
    axs2[1].set_title(f"Y-Axis Acceleration (RMSE: {rmse_ell[1]:.3f})")
    
    # [FIGURE 3] Trajectory (Position X-Y) - FIXED
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    fig3.canvas.manager.set_window_title('Figure 3: End-Effector Trajectory')
    
    # Plot Target (Red Dashed) vs Actual (Blue Solid)
    ax3.plot(path_des[:, 0], path_des[:, 1], 'r--', linewidth=2, label='Target Ellipse')
    ax3.plot(path_act[:, 0], path_act[:, 1], 'b-', linewidth=2, label='Actual Path', alpha=0.7)
    
    ax3.set_title("Robot End-Effector Trajectory")
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.grid(True)
    ax3.axis('equal') 
    ax3.legend()

    plt.show()

if __name__ == '__main__':
    run_pipeline()





# """
# FINAL PIPELINE: High-Torque Excitation + Ellipse Validation (FIXED DEGREE)
# """
# import os
# import sys
# import numpy as np
# import math
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from ament_index_python.packages import get_package_share_directory
# import pickle

# # --- 1. CONFIGURATIONS ---
# URDF_PATH = os.path.join(
#     get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf"
# )
# DT = 0.002              # 500 Hz
# DURATION = 4500.0       # Long duration
# NOISE_LEVEL = 0.002     # Low noise
# CONFIDENCE = 0.1        # 90% Confidence

# # Safety Limits
# MAX_VEL = np.deg2rad(150)
# MAX_TORQUE = 30.0

# # --- 2. TRAJECTORY GENERATOR ---
# class TrajectoryGenerator:
#     def __init__(self):
#         self.center_pos = np.array([0.0, 0.0, 0.0])
#         self.ellipse_a = 1.6 
#         self.ellipse_b = 0.9
#         self.period = 12.0     
#         self.omega = 2 * np.pi / self.period
#         self.approach_duration = 5.0
#         self.start_pos = None  
#         self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])

#     def get_ref(self, t, current_actual_pos=None):
#         if t < self.approach_duration:
#             if self.start_pos is None:
#                 if current_actual_pos is None: return np.zeros(3), np.zeros(3), np.zeros(3)
#                 self.start_pos = current_actual_pos
#             tau = t / self.approach_duration
#             s = (1.0 - math.cos(tau * math.pi)) / 2.0
#             ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
#             dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)
#             vector_diff = self.orbit_start_pos - self.start_pos
#             return self.start_pos + (vector_diff * s), vector_diff * ds, vector_diff * dds
#         else:
#             t_orbit = t - self.approach_duration
#             x_des = self.center_pos.copy()
#             x_des[0] += self.ellipse_a * np.cos(self.omega * t_orbit)
#             x_des[1] += self.ellipse_b * np.sin(self.omega * t_orbit)
#             dx_des = np.zeros(3)
#             dx_des[0] = -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit)
#             dx_des[1] =  self.ellipse_b * self.omega * np.cos(self.omega * t_orbit)
#             ddx_des = np.zeros(3)
#             ddx_des[0] = -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit)
#             ddx_des[1] = -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit)
#             return x_des, dx_des, ddx_des

# # --- 3. EXCITATION GENERATOR ---
# def generate_rich_safe_excitation(t, dofs=2):
#     tau = np.zeros(dofs)
#     amp_scale = (np.sin(0.1 * t) + 1.0) / 2.0 
#     mag = 25.0 * amp_scale 

#     for i in range(dofs):
#         tau[i] = mag * (0.6 * np.sin(0.5 * t) + 
#                         0.3 * np.cos(1.5 * t + i) +
#                         0.1 * np.sin(3.0 * t))
#     return tau

# # --- 4. PHYSICS HELPERS ---
# def build_feature_matrix(X):
#     q1, q2 = X[:, 0], X[:, 1]
#     dq1, dq2 = X[:, 2], X[:, 3]
#     tau1, tau2 = X[:, 4], X[:, 5]
    
#     s1, c1 = np.sin(q1), np.cos(q1)
#     s2, c2 = np.sin(q2), np.cos(q2)
#     s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    
#     dq1_sq = dq1**2
#     dq2_sq = dq2**2
    
#     return np.stack([tau1, tau2, dq1, dq2, s1, c1, s2, c2, s12, c12, dq1_sq, dq2_sq], axis=1)

# def save_model(model, q_hat_x, q_hat_y, filename="my_learned_robot.pkl"):
#     abs_path = os.path.abspath(filename)
#     data = {"model": model, "safety_bounds": {"x": q_hat_x, "y": q_hat_y}}
#     try:
#         with open(abs_path, "wb") as f:
#             pickle.dump(data, f)
#         print(f"\n[SUCCESS] Model Saved to: {abs_path}")
#         print(f"Safety Bounds -> X: {q_hat_x:.4f}, Y: {q_hat_y:.4f}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save model: {e}")

# def run_pipeline():
#     print("=== STEP 1: GENERATING RICH DATASET (High Torque) ===")
    
#     try:
#         model_phys = pin.buildModelFromUrdf(URDF_PATH)
#         data_phys = model_phys.createData()
#         ee_id = model_phys.getFrameId("endEffector")
#     except Exception as e:
#         print(f"Error loading URDF: {e}")
#         return

#     # Add Mass Mismatch (+20%)
#     for i in range(1, len(model_phys.inertias)):
#         model_phys.inertias[i].mass *= 1.20 

#     # --- SIMULATION LOOP ---
#     q = pin.neutral(model_phys)
#     v = np.zeros(model_phys.nv)
    
#     steps = int(DURATION / DT)
#     X_list, Y_list = [], []
    
#     for step in range(steps):
#         t = step * DT
        
#         # 1. Control
#         tau_cmd = generate_rich_safe_excitation(t)
#         tau_cmd = np.clip(tau_cmd, -MAX_TORQUE, MAX_TORQUE)
        
#         # 2. Physics
#         friction = 0.1 * v
#         tau_real = tau_cmd - friction
#         ddq = pin.aba(model_phys, data_phys, q, v, tau_real)
        
#         # 3. Ground Truth
#         pin.computeJointJacobians(model_phys, data_phys, q)
#         pin.forwardKinematics(model_phys, data_phys, q, v, ddq)
#         pin.updateFramePlacements(model_phys, data_phys)
#         J = pin.getFrameJacobian(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2]
#         dJdq = pin.getFrameJacobianTimeVariation(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v[:2]
#         ddx_true = (J @ ddq[:2]) + dJdq
        
#         # 4. Record
#         q_n = q[:2] + np.random.normal(0, NOISE_LEVEL, 2)
#         v_n = v[:2] + np.random.normal(0, NOISE_LEVEL, 2)
#         X_list.append(np.hstack([q_n, v_n, tau_cmd]))
#         Y_list.append(ddx_true)
        
#         # 5. Integrate
#         v += ddq * DT
#         v = np.clip(v, -MAX_VEL, MAX_VEL)
#         q = pin.integrate(model_phys, q, v * DT)

#     X = np.array(X_list)
#     Y = np.array(Y_list)

#     # --- SPLIT DATA ---
#     n = len(X)
#     n_train = int(0.6 * n)
#     n_cal = int(0.8 * n)
    
#     X_train, Y_train = X[:n_train], Y[:n_train]
#     X_cal, Y_cal     = X[n_train:n_cal], Y[n_train:n_cal]
    
#     print(f"Data Shapes -> Train: {X_train.shape}, Cal: {X_cal.shape}")

#     print("\n=== STEP 2: TRAINING MODEL ===")
#     Phi_train = build_feature_matrix(X_train)
    
#     # [CRITICAL FIX] Degree 2 prevents "cubic explosion"
#     # [CRITICAL FIX] Alpha 0.5 increases stability against noise
#     model = make_pipeline(
#         PolynomialFeatures(degree=2, include_bias=True), 
#         StandardScaler(),
#         Ridge(alpha=0.00001) 
#     )
#     model.fit(Phi_train, Y_train)
#     print("   Model Trained (Degree=2, Alpha=0.5).")

#     print("\n=== STEP 3: CONFORMAL PREDICTION (CALIBRATION) ===")
#     Phi_cal = build_feature_matrix(X_cal)
#     Y_pred_cal = model.predict(Phi_cal)
#     residuals = np.abs(Y_cal - Y_pred_cal)
    
#     n_cal_samples = residuals.shape[0]
#     q_idx = int(np.ceil((n_cal_samples + 1) * (1 - CONFIDENCE)))
#     q_hat_x = np.sort(residuals[:, 0])[min(q_idx, n_cal_samples-1)]
#     q_hat_y = np.sort(residuals[:, 1])[min(q_idx, n_cal_samples-1)]
    
#     print(f"Safety Bounds: X={q_hat_x:.3f}, Y={q_hat_y:.3f} m/s^2")
    
#     save_model(model, q_hat_x, q_hat_y)

#     # === STEP 5: VERIFY ON ACTUAL ELLIPSE TRAJECTORY ===
#     print("\n=== STEP 5: VERIFYING ON ELLIPTICAL TRAJECTORY (Closed-Loop) ===")
    
#     traj_gen = TrajectoryGenerator()
#     q = pin.neutral(model_phys)
#     v = np.zeros(model_phys.nv)
    
#     X_ell, Y_ell = [], []
#     Kp, Kd = 100.0, 20.0
    
#     for i in range(int(20.0 / DT)):
#         t = i * DT
        
#         pin.forwardKinematics(model_phys, data_phys, q, v)
#         pin.updateFramePlacements(model_phys, data_phys)
#         x_curr = data_phys.oMf[ee_id].translation
#         J = pin.computeFrameJacobian(model_phys, data_phys, q, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :2]
#         dx_curr = J @ v[:2]
        
#         xd, vd, _ = traj_gen.get_ref(t, x_curr)
#         ep = xd - x_curr
#         ev = vd - dx_curr
#         F_task = Kp * ep + Kd * ev
#         tau_cmd = J.T @ F_task[:3] 
        
#         tau_cmd = np.clip(tau_cmd, -MAX_TORQUE, MAX_TORQUE)
        
#         # friction = 0.1 * v
#         friction=0
#         tau_real = tau_cmd[:2] - friction
#         ddq = pin.aba(model_phys, data_phys, q, v, tau_real)
        
#         dJdq = pin.getFrameJacobianTimeVariation(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v[:2]
#         ddx_true = (J[:2, :2] @ ddq[:2]) + dJdq
        
#         X_ell.append(np.hstack([q[:2], v[:2], tau_cmd[:2]]))
#         Y_ell.append(ddx_true)
        
#         v += ddq * DT
#         q = pin.integrate(model_phys, q, v * DT)

#     X_test_ell = np.array(X_ell)
#     Y_test_ell = np.array(Y_ell)
#     t_test_ell = np.arange(len(Y_test_ell)) * DT

#     Phi_test_ell = build_feature_matrix(X_test_ell)
#     Y_pred_ell = model.predict(Phi_test_ell)
    
#     rmse_ell = np.sqrt(np.mean((Y_test_ell - Y_pred_ell)**2, axis=0))
#     print(f"Ellipse Test RMSE: X={rmse_ell[0]:.4f}, Y={rmse_ell[1]:.4f}")

#     # Plotting
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8))
#     fig.canvas.manager.set_window_title('Validation on Ellipse Trajectory')
    
#     axs[0].plot(t_test_ell, Y_test_ell[:, 0], 'k-', label='True Physics', lw=2, alpha=0.7)
#     axs[0].plot(t_test_ell, Y_pred_ell[:, 0], 'b--', label='Learned Model', lw=1.5)
#     axs[0].fill_between(t_test_ell, Y_pred_ell[:, 0] - q_hat_x, Y_pred_ell[:, 0] + q_hat_x, color='blue', alpha=0.2)
#     axs[0].set_title(f"X-Axis Acceleration (RMSE: {rmse_ell[0]:.3f})")
#     axs[0].legend()
    
#     axs[1].plot(t_test_ell, Y_test_ell[:, 1], 'k-', label='True Physics', lw=2, alpha=0.7)
#     axs[1].plot(t_test_ell, Y_pred_ell[:, 1], 'r--', label='Learned Model', lw=1.5)
#     axs[1].fill_between(t_test_ell, Y_pred_ell[:, 1] - q_hat_y, Y_pred_ell[:, 1] + q_hat_y, color='red', alpha=0.2)
#     axs[1].set_title(f"Y-Axis Acceleration (RMSE: {rmse_ell[1]:.3f})")
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     run_pipeline()




# """
# Unified Pipeline for Exact Symbolic Derivation, Data Collection, 
# SINDy Learning, and Conformal Quantile Calculation.
# """
# import os
# import sys
# import time
# import numpy as np
# import sympy as sp
# import pinocchio as pin
# import pysindy as ps
# import matplotlib.pyplot as plt
# from ament_index_python.packages import get_package_share_directory

# # --- 1. CONFIGURATIONS ---
# URDF_PATH = os.path.join(
#     get_package_share_directory("daadbot_desc"),
#     "urdf",
#     "2_link_urdf", 
#     "2link_robot.urdf" 
# )

# ALL_JOINTS = ["baseHinge", "interArm"]

# def generate_exact_equations():
#     """ Derives exact equations of motion symbolically using SymPy """
#     print("=== STEP 0: EXACT SYMBOLIC EQUATIONS DERIVATION ===")
    
#     # 1. Define Symbolic Variables
#     q1, q2 = sp.symbols('q1 q2')
#     dq1, dq2 = sp.symbols('dq1 dq2')
#     tau1, tau2 = sp.symbols('tau1 tau2')
    
#     # Robot Physical Parameters
#     m1, m2 = sp.symbols('m1 m2')
#     L1, L2 = sp.symbols('L1 L2')
#     lc1, lc2 = sp.symbols('lc1 lc2')
#     I1, I2 = sp.symbols('I1 I2')
#     g = sp.symbols('g') # Gravity

#     # 2. Forward Kinematics (End Effector Position)
#     x = L1 * sp.cos(q1) + L2 * sp.cos(q1 + q2)
#     y = L1 * sp.sin(q1) + L2 * sp.sin(q1 + q2)

#     # 3. Jacobians
#     J11 = sp.diff(x, q1); J12 = sp.diff(x, q2)
#     J21 = sp.diff(y, q1); J22 = sp.diff(y, q2)
#     J = sp.Matrix([[J11, J12], [J21, J22]])

#     # Jacobian Time Derivative (dJ/dt)
#     dJ11 = sp.diff(J11, q1)*dq1 + sp.diff(J11, q2)*dq2
#     dJ12 = sp.diff(J12, q1)*dq1 + sp.diff(J12, q2)*dq2
#     dJ21 = sp.diff(J21, q1)*dq1 + sp.diff(J21, q2)*dq2
#     dJ22 = sp.diff(J22, q1)*dq1 + sp.diff(J22, q2)*dq2
#     dJ = sp.Matrix([[dJ11, dJ12], [dJ21, dJ22]])

#     # 4. Joint Space Dynamics Matrices (M, C, G)
#     M11 = m1*lc1**2 + I1 + m2*(L1**2 + lc2**2 + 2*L1*lc2*sp.cos(q2)) + I2
#     M12 = m2*(lc2**2 + L1*lc2*sp.cos(q2)) + I2
#     M21 = M12
#     M22 = m2*lc2**2 + I2
#     M = sp.Matrix([[M11, M12], [M21, M22]])

#     h = -m2*L1*lc2*sp.sin(q2)
#     C = sp.Matrix([[h*dq2, h*(dq1 + dq2)], [-h*dq1, 0]])

#     G1 = (m1*lc1 + m2*L1)*g*sp.cos(q1) + m2*lc2*g*sp.cos(q1 + q2)
#     G2 = m2*lc2*g*sp.cos(q1 + q2)
#     G = sp.Matrix([[G1], [G2]])

#     Tau = sp.Matrix([[tau1], [tau2]])
#     dq = sp.Matrix([[dq1], [dq2]])

#     # 5. Solve for Joint Acceleration: ddq = M_inv * (Tau - C*dq - G)
#     M_inv = M.inv()
#     ddq = M_inv * (Tau - C*dq - G)

#     # 6. Task Space Acceleration: ddx = J*ddq + dJ*dq
#     ddx = J * ddq + dJ * dq

#     ddx_x = sp.simplify(ddx[0])
#     ddx_y = sp.simplify(ddx[1])

#     print("\nEquation for ddx_x (X-Axis Acceleration):")
#     sp.pprint(ddx_x) 
    
#     print("\nEquation for ddx_y (Y-Axis Acceleration):")
#     sp.pprint(ddx_y)
#     print("--------------------------------------------------\n")

# def generate_lipschitz_input(t):
#     """ Rich sum-of-sines torque signal to excite system modes. """
#     tau_1 = 15.0 * np.sin(1.3 * t) + 10.0 * np.sin(0.4 * t)
#     tau_2 = 12.0 * np.cos(2.2 * t) + 8.0 * np.sin(0.7 * t)
#     return np.array([tau_1, tau_2])

# def run_pipeline():
#     # Execute Symbolic Derivation First
#     generate_exact_equations()

#     print("=== STEP 1: COLLECTING DATA ===")
    
#     # Setup Pinocchio "True Reality" Model
#     model_phys = pin.buildModelFromUrdf(URDF_PATH)
    
#     # Inject hidden uncertainty (Mass +15%) to create a gap for SINDy to learn
#     for i in range(1, len(model_phys.inertias)):
#         model_phys.inertias[i].mass *= 1.15 
    
#     data_phys = model_phys.createData()
#     ee_id = model_phys.getFrameId("endEffector")

#     # Simulation setup
#     q = pin.neutral(model_phys); q[0] = 0.1
#     v = np.zeros(model_phys.nv)
    
#     dt = 0.002
#     duration = 30.0
#     steps = int(duration / dt)

#     X_data = [] # Features: [q1, q2, dq1, dq2, tau1, tau2]
#     Y_data = [] # Targets:  [ddx_x, ddx_y]

#     # Run Simulation
#     for step in range(steps):
#         t = step * dt
#         tau_cmd = generate_lipschitz_input(t)

#         # Apply hidden friction
#         damping = 0.5 * v + 0.1 * np.sign(v) 
#         tau_full = np.zeros(model_phys.nv)
#         tau_full[:2] = tau_cmd - damping[:2]

#         # True Physics Step
#         ddq = pin.aba(model_phys, data_phys, q, v, tau_full)
        
#         # Extract Task-Space Acceleration
#         pin.computeJointJacobians(model_phys, data_phys, q)
#         pin.forwardKinematics(model_phys, data_phys, q, v, ddq)
        
#         J = pin.getFrameJacobian(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2]
#         dJdq = pin.getFrameJacobianTimeVariation(model_phys, data_phys, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v[:2]
        
#         ddx = (J @ ddq[:2]) + dJdq

#         # Store Data
#         X_data.append(np.hstack([q[:2], v[:2], tau_cmd]))
#         Y_data.append(ddx)

#         # Integrate
#         v += ddq * dt
#         q = pin.integrate(model_phys, q, v * dt)

#     X = np.array(X_data)
#     Y = np.array(Y_data)
#     print(f"Data collected: {len(X)} samples.")

#     print("\n=== STEP 2: LEARNING DYNAMICS WITH SINDy ===")
    
#     # Using Polynomial + Trig features suitable for robot dynamics
#     feature_library = ps.PolynomialLibrary(degree=2) + ps.FourierLibrary(n_frequencies=1)
    
#     # STLSQ Optimizer enforces sparsity (filters out noise)
#     optimizer = ps.STLSQ(threshold=0.01, alpha=0.05)
#     model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)

#     # [FIX APPLIED]: Added t=dt argument to satisfy PySINDy's API requirements
#     model.fit(X, t=dt, x_dot=Y) 
#     print("SINDy Model Discovered Equations:")
#     model.print()

#     print("\n=== STEP 3: CONFORMAL PREDICTION (QUANTILE CALCULATION) ===")
    
#     # Predict over the dataset
#     Y_pred = model.predict(X)

#     # Calculate L2-norm residuals (m/s^2)
#     acc_errors = np.linalg.norm(Y - Y_pred, axis=1)

#     # Conformal Splitting (Train/Calibration)
#     n = len(acc_errors)
#     calib_errors = acc_errors[int(0.8*n):] # Use last 20% for calibration
#     n_calib = len(calib_errors)

#     # Calculate Quantile
#     alpha = 0.1 # 90% confidence
#     quantile_idx = int(np.ceil((n_calib + 1) * (1 - alpha)))
#     sorted_errors = np.sort(calib_errors)
#     q_hat = sorted_errors[min(quantile_idx, n_calib-1)]

#     print(f"Mean SINDy Error: {np.mean(acc_errors):.4f} m/s^2")
#     print(f"Conformal Quantile (q_hat): {q_hat:.4f} m/s^2")

#     # Visualize
#     plt.figure(figsize=(10,5))
#     plt.hist(calib_errors, bins=50, alpha=0.7, color='green')
#     plt.axvline(q_hat, color='red', linestyle='dashed', linewidth=2, label=f'q_hat = {q_hat:.3f}')
#     plt.title("Conformal Distribution of SINDy Residuals")
#     plt.xlabel("Acceleration Error (m/s²)")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()

# if __name__ == '__main__':
#     run_pipeline()
#     sys.exit(0)
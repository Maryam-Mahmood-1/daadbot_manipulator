import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PATHS
# =============================================================================
urdf_clean = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
urdf_noisy = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"

# =============================================================================
# 2. LOAD MODELS
# =============================================================================
model_nom = pin.buildModelFromUrdf(urdf_clean)
data_nom = model_nom.createData()

model_real = pin.buildModelFromUrdf(urdf_noisy)
data_real = model_real.createData()

# =============================================================================
# 3. SETTINGS & LIMITS
# =============================================================================
target_joint_names = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7"
]

# Get indices
joint_v_indices = []
for jname in target_joint_names:
    jid = model_nom.getJointId(jname)
    v_idx = model_nom.joints[jid].idx_v
    joint_v_indices.append(v_idx)

# Limits
acc_max_limit = 10.0  # User requested small constant accel limit
torque_limits = model_nom.effortLimit # Read from URDF

# Simulation
dt = 0.01
n_samples = 1000
total_time = n_samples * dt

# Gains for Computed Torque Control
kp_val = 100.0
kd_val = 2 * np.sqrt(kp_val)

# =============================================================================
# 4. TRAJECTORY GENERATION
# =============================================================================
def get_trajectory_point(t):
    """
    Returns q_des, v_des, a_des for a smooth sine wave.
    q = A * sin(w*t)
    v = A * w * cos(w*t)
    a = -A * w^2 * sin(w*t)
    """
    freq = 0.5  # rad/s
    amp = 0.5   # Amplitude (radians)
    
    # Offset phases so joints don't all move identically
    phases = np.arange(model_nom.nq) * 0.2
    
    q_des = amp * np.sin(freq * t + phases)
    v_des = amp * freq * np.cos(freq * t + phases)
    a_des = -amp * (freq**2) * np.sin(freq * t + phases)
    
    return q_des, v_des, a_des

# =============================================================================
# 5. SIMULATION LOOP (INTEGRATION)
# =============================================================================
# Lists to store history
errors = []
acc_nom_hist = []
acc_real_hist = []
time_hist = []

# INITIALIZATION: Start robot exactly at the first trajectory point
q_current, _, _ = get_trajectory_point(0)
v_current = np.zeros(model_nom.nv)

print(f"\nRunning {n_samples} steps of continuous integration...")

for k in range(n_samples):
    t = k * dt
    time_hist.append(t)

    # A. Get Target (Smooth Ramp/Sine)
    q_des, v_des, a_des_ref = get_trajectory_point(t)

    # B. Computed Torque Control Law
    # 1. Error terms
    e_q = q_current - q_des
    e_v = v_current - v_des
    
    # 2. Desired Acceleration (Feedback Linearization)
    #    a_cmd = a_ref - Kp*error - Kd*error_dot
    a_cmd = a_des_ref - kp_val * e_q - kd_val * e_v
    
    # 3. CLIP ACCELERATION (User Request)
    a_cmd = np.clip(a_cmd, -acc_max_limit, acc_max_limit)
    
    # 4. Inverse Dynamics (calculate required torque for Nominal Model)
    #    tau = M(q)*a_cmd + C(q,v)*v + g(q)
    tau = pin.rnea(model_nom, data_nom, q_current, v_current, a_cmd)
    
    # 5. CLIP TORQUE (User Request - URDF Limits)
    tau = np.clip(tau, -torque_limits, torque_limits)

    # C. Forward Dynamics (What actually happens?)
    # Apply the SAME torque to both models to see the mismatch
    
    # 1. Nominal (Prediction)
    pin.aba(model_nom, data_nom, q_current, v_current, tau)
    acc_nom = data_nom.ddq
    
    # 2. Real (Ground Truth / Noisy)
    pin.aba(model_real, data_real, q_current, v_current, tau)
    acc_real = data_real.ddq

    # D. Store Data
    acc_nom_vec = acc_nom[joint_v_indices]
    acc_real_vec = acc_real[joint_v_indices]
    diff_vec = acc_real_vec - acc_nom_vec
    
    acc_nom_hist.append(acc_nom_vec)
    acc_real_hist.append(acc_real_vec)
    errors.append(diff_vec)
    
    # E. INTEGRATION (Move the "Real" robot forward)
    # This ensures the next step relies on the physical result of this step
    # v_next = v + a_real * dt
    # q_next = q + v_mean * dt
    v_current += acc_real * dt
    q_current = pin.integrate(model_real, q_current, v_current * dt)

# Format Data
errors = np.array(errors)
acc_nom_hist = np.array(acc_nom_hist)
acc_real_hist = np.array(acc_real_hist)

# =============================================================================
# 6. GLOBAL STATISTICS
# =============================================================================
error_norms = np.linalg.norm(errors, axis=1)

print("\n" + "=" * 70)
print("--- GLOBAL DYNAMICS MISMATCH (Continuous Trajectory) ---")
print("=" * 70)
print(f"Mean ‖Δddq‖        : {np.mean(error_norms):.4f} rad/s²")
print(f"Max  ‖Δddq‖        : {np.max(error_norms):.4f} rad/s²")

# =============================================================================
# 7. PER-JOINT STATISTICS
# =============================================================================
per_joint_error_q95 = np.quantile(np.abs(errors), 0.95, axis=0)
per_joint_nom_q95   = np.quantile(np.abs(acc_nom_hist), 0.95, axis=0)
per_joint_real_q95  = np.quantile(np.abs(acc_real_hist), 0.95, axis=0)

print("\n" + "=" * 90)
print("--- PER-JOINT 95% ACCELERATION STATISTICS (Absolute Values) ---")
print("=" * 90)
print(f"{'JOINT NAME':<15} | {'Nominal 95%':<15} | {'Noisy 95%':<15} | {'Mismatch 95%':<15}")
print(f"{'':<15} | {'(rad/s²)':<15} | {'(rad/s²)':<15} | {'(rad/s²)':<15}")
print("-" * 90)

for i, jname in enumerate(target_joint_names):
    print(f"{jname:<15} | "
          f"{per_joint_nom_q95[i]:<15.4f} | "
          f"{per_joint_real_q95[i]:<15.4f} | "
          f"{per_joint_error_q95[i]:<15.4f}")

# =============================================================================
# 8. PLOTS
# =============================================================================
# Plot 1: Acceleration Tracking (Joint 2 Example)
plt.figure(figsize=(10, 4))
plt.plot(time_hist, acc_nom_hist[:, 1], label='Nominal Acc (J2)', linestyle='--')
plt.plot(time_hist, acc_real_hist[:, 1], label='Real Acc (J2)', alpha=0.7)
plt.title("Acceleration Profile (Joint 2) - Smooth Trajectory")
plt.xlabel("Time [s]")
plt.ylabel("Acc [rad/s²]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Error Histogram
plt.figure(figsize=(10, 4))
plt.hist(error_norms, bins=50, color='gray', alpha=0.7, label='Error Norm')
plt.title("Global Acceleration Error Norm")
plt.xlabel("‖Δddq‖ [rad/s²]")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 3: Bar Chart Comparison
x = np.arange(len(target_joint_names))
width = 0.25
plt.figure(figsize=(12, 6))
plt.bar(x - width, per_joint_nom_q95, width, label='Nominal 95%', alpha=0.8)
plt.bar(x, per_joint_real_q95, width, label='Noisy 95%', alpha=0.8)
plt.bar(x + width, per_joint_error_q95, width, label='Mismatch 95%', color='red', alpha=0.8)
plt.ylabel("Acceleration [rad/s²]")
plt.title("Per-Joint 95% Quantile: Nominal vs Noisy vs Mismatch")
plt.xticks(x, target_joint_names)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.show()




# import pinocchio as pin
# import numpy as np
# import matplotlib.pyplot as plt

# # =============================================================================
# # 1. SETUP & SIMULATION (Same as before)
# # =============================================================================
# urdf_clean = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
# urdf_noisy = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"

# model_nom = pin.buildModelFromUrdf(urdf_clean)
# data_nom = model_nom.createData()
# model_real = pin.buildModelFromUrdf(urdf_noisy)
# data_real = model_real.createData()

# target_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]

# # Indices
# joint_v_indices = []
# for jname in target_joint_names:
#     jid = model_nom.getJointId(jname)
#     joint_v_indices.append(model_nom.joints[jid].idx_v)

# # Simulation Params
# dt = 0.01
# n_samples = 1500 # More samples for better histogram
# acc_max_limit = 10.0
# torque_limits = model_nom.effortLimit
# kp_val, kd_val = 100.0, 20.0

# # Trajectory Generator
# def get_trajectory_point(t):
#     freq, amp = 0.5, 0.5
#     phases = np.arange(model_nom.nq) * 0.2
#     q = amp * np.sin(freq * t + phases)
#     v = amp * freq * np.cos(freq * t + phases)
#     a = -amp * (freq**2) * np.sin(freq * t + phases)
#     return q, v, a

# # Run Simulation
# errors = []
# q_curr, _, _ = get_trajectory_point(0)
# v_curr = np.zeros(model_nom.nv)

# print(f"Collecting {n_samples} samples of dynamics mismatch...")

# for k in range(n_samples):
#     t = k * dt
#     q_des, v_des, a_des_ref = get_trajectory_point(t)

#     # Control
#     e_q = q_curr - q_des
#     e_v = v_curr - v_des
#     a_cmd = np.clip(a_des_ref - kp_val*e_q - kd_val*e_v, -acc_max_limit, acc_max_limit)
    
#     # Dynamics Mismatch
#     tau = np.clip(pin.rnea(model_nom, data_nom, q_curr, v_curr, a_cmd), -torque_limits, torque_limits)
    
#     pin.aba(model_nom, data_nom, q_curr, v_curr, tau)
#     acc_nom = data_nom.ddq[joint_v_indices]
    
#     pin.aba(model_real, data_real, q_curr, v_curr, tau)
#     acc_real = data_real.ddq[joint_v_indices]
    
#     errors.append(acc_real - acc_nom)
    
#     # Integrate
#     v_curr += data_real.ddq * dt
#     q_curr = pin.integrate(model_real, q_curr, v_curr * dt)

# abs_errors = np.abs(np.array(errors))
# per_joint_q95 = np.quantile(abs_errors, 0.95, axis=0)

# # =============================================================================
# # 2. PLOTTING: HISTOGRAM WITH 95% LINE
# # =============================================================================
# fig, axs = plt.subplots(4, 2, figsize=(15, 12))
# fig.suptitle("Dynamics Mismatch Distribution with 95% Quantile Cutoff", fontsize=16)
# axs = axs.flatten()

# for i, jname in enumerate(target_joint_names):
#     ax = axs[i]
    
#     # 1. Histogram (The "Bars of Error")
#     data = abs_errors[:, i]
#     ax.hist(data, bins=40, color='skyblue', edgecolor='black', alpha=0.7, label='Error Freq')
    
#     # 2. Vertical Red Line (The 95% Spot)
#     q95 = per_joint_q95[i]
#     ax.axvline(q95, color='red', linestyle='--', linewidth=2, label=f'95% ({q95:.3f})')
    
#     # Styling
#     ax.set_title(f"{jname}")
#     ax.set_xlabel("|Δddq| [rad/s²]")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# # Hide empty plot
# axs[7].axis('off')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
import numpy as np
import pinocchio as pin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# ---------------------------
# 1. Setup & Data Collection
# ---------------------------
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
nv = model.nv
ee_id = model.getFrameId("endEffector")

# Parameters optimized for 250s simulation stability
DT, DURATION = 0.01, 250.0  
steps = int(DURATION / DT)
q, v = pin.neutral(model), np.zeros(nv)

DAMPING = 1.5 
MAX_V = 20.0  # Limit to prevent numerical explosion at 0.01s DT

def tau_excitation(t):
    return np.array([6.0 * np.sin(1.2*t) + 2.0 * np.cos(3.0*t),
                     4.0 * np.sin(1.5*t) + 1.0 * np.cos(2.5*t)])

# Data containers for regression and conformal calibration
X_M, X_C, M_target, C_target, Accel_target = [], [], [], [], []

print(f"Collecting data for {DURATION}s with DT={DT}...")

for step in range(steps):
    t = step * DT
    tau = tau_excitation(t) - DAMPING * v
    
    # 1. Physics Step
    ddq = pin.aba(model, data, q, v, tau)
    M_q = pin.crba(model, data, q)
    C_q = pin.computeCoriolisMatrix(model, data, q, v)
    
    # 2. Task Space Acceleration (Ground Truth for Calibration)
    pin.forwardKinematics(model, data, q, v, ddq)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2]
    dJdq = pin.getFrameJacobianTimeVariation(model, data, ee_id, pin.LOCAL_WORLD_ALIGNED)[:2, :2] @ v
    ddx_true = J @ ddq + dJdq 
    
    # 3. Integration
    v += ddq * DT
    v = np.clip(v, -MAX_V, MAX_V)
    q = pin.integrate(model, q, v*DT)
    
    if not np.all(np.isfinite(q)):
        print(f"Divergence at {t}s!")
        break

    # 4. Features (Physics-Informed)
    feat_M = np.array([1.0, np.cos(q[1])])
    feat_C = np.array([v[0]*np.sin(q[1]), v[1]*np.sin(q[1]), 
                       v[0]*np.cos(q[1]), v[1]*np.cos(q[1])])

    X_M.append(feat_M)
    X_C.append(feat_C)
    M_target.append(M_q.flatten())
    C_target.append(C_q.flatten())
    Accel_target.append(ddx_true)

# Convert to Numpy Arrays
X_M, X_C = np.array(X_M), np.array(X_C)
M_target, C_target = np.array(M_target), np.array(C_target)
Y_accel = np.array(Accel_target)

# ---------------------------------------------------------
# 2. Training & Conformal Calibration
# ---------------------------------------------------------
# Split: 60% Train, 20% Calibrate, 20% Test
X_comb = np.hstack([X_M, X_C])
X_train, X_temp, Y_accel_train, Y_accel_temp = train_test_split(X_comb, Y_accel, test_size=0.4, random_state=42)
X_cal, X_test, Y_accel_cal, Y_accel_test = train_test_split(X_temp, Y_accel_temp, test_size=0.5, random_state=42)

# Train Dynamics Matrix Models
M_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(X_M, M_target)
C_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(X_C, C_target)

# Train Acceleration Model specifically for Conformal Calibration
accel_model = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(X_train, Y_accel_train)

# --- QUANTILE CALCULATION ---
delta = 0.1 # 90% Confidence bound
n_cal = len(X_cal)
Y_pred_cal = accel_model.predict(X_cal)
cal_scores = np.linalg.norm(Y_accel_cal - Y_pred_cal, axis=1) # Residuals

q_idx = int(np.ceil((n_cal + 1) * (1 - delta)))
q_hat = np.sort(cal_scores)[min(q_idx, n_cal - 1)] # Formal q_quantile

print(f"\n--- Conformal Results ---")
print(f"Task-Space Acceleration Quantile (q_hat): {q_hat:.4f} m/s^2")

# ---------------------------
# 3. Save to .pkl
# ---------------------------
model_payload = {
    "M_model": M_model,
    "C_model": C_model,
    "q_quantile": q_hat,
    "metadata": {"urdf": URDF_PATH, "damping": DAMPING}
}

with open("robot_dynamics_model.pkl", "wb") as f:
    pickle.dump(model_payload, f)
print(f"[SUCCESS] Model saved as robot_dynamics_model.pkl")

# ---------------------------
# 4. Visualization
# ---------------------------
M_preds = M_model.predict(X_M)
C_preds = C_model.predict(X_C)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for idx in range(4):
    r, c = idx // 2, idx % 2
    axs[0, c].plot(M_target[:, idx], 'k', alpha=0.3, label='True')
    axs[0, c].plot(M_preds[:, idx], 'b--', label='Pred')
    axs[0, c].set_title(f'M[{idx}] Reconstruction')
    
    axs[1, c].plot(C_target[:, idx], 'k', alpha=0.3, label='True')
    axs[1, c].plot(C_preds[:, idx], 'r--', label='Pred')
    axs[1, c].set_title(f'C[{idx}] Reconstruction')

plt.tight_layout()
plt.show()
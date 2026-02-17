import os
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory

# --- 1. CONFIGURATION ---
URDF_TRUE = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
URDF_NOISY = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_noisy_4.urdf")

DT = 0.002              # 500 Hz
DURATION = 20.0         # Seconds of data
CONFIDENCE = 0.99        # 90%
MAX_TORQUE = 25.0

# --- 2. SETUP ---
def load_model(path):
    model = pin.buildModelFromUrdf(path)
    data = model.createData()
    if model.existFrame("endEffector"):
        ee_id = model.getFrameId("endEffector")
    else:
        ee_id = model.nframes - 1 
    return model, data, ee_id

model_true, data_true, ee_true = load_model(URDF_TRUE)
model_noisy, data_noisy, ee_noisy = load_model(URDF_NOISY)

# --- 3. EXCITATION ---
def generate_excitation(t):
    tau = np.zeros(2)
    tau[0] = 12.0 * np.sin(2.0 * t) + 8.0 * np.cos(5.0 * t)
    tau[1] = 8.0 * np.sin(3.0 * t) + 4.0 * np.cos(7.0 * t)
    return np.clip(tau, -MAX_TORQUE, MAX_TORQUE)

def main():
    # --- 4. DATA COLLECTION LOOP ---
    print("Collecting Calibration Dataset...")
    q = pin.neutral(model_true)
    v = np.zeros(model_true.nv)

    residuals_x = []
    residuals_y = []
    true_acc_log = []
    pred_acc_log = []

    steps = int(DURATION / DT)

    for k in range(steps):
        t = k * DT
        tau = generate_excitation(t)
        
        # A. GROUND TRUTH
        ddq_true = pin.aba(model_true, data_true, q, v, tau)
        pin.forwardKinematics(model_true, data_true, q, v, ddq_true)
        pin.updateFramePlacements(model_true, data_true)
        acc_true = pin.getFrameAcceleration(model_true, data_true, ee_true, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        ddx_true = acc_true.linear[:2]
        
        # B. PREDICTION
        ddq_pred = pin.aba(model_noisy, data_noisy, q, v, tau)
        pin.forwardKinematics(model_noisy, data_noisy, q, v, ddq_pred)
        pin.updateFramePlacements(model_noisy, data_noisy)
        acc_pred = pin.getFrameAcceleration(model_noisy, data_noisy, ee_noisy, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        ddx_pred = acc_pred.linear[:2]
        
        # C. SCORES
        res = np.abs(ddx_true - ddx_pred)
        residuals_x.append(res[0])
        residuals_y.append(res[1])
        
        true_acc_log.append(ddx_true)
        pred_acc_log.append(ddx_pred)
        
        # D. INTEGRATE
        v += ddq_true * DT
        v = np.clip(v, -5.0, 5.0)
        q = pin.integrate(model_true, q, v * DT)

    # --- 5. QUANTILE CALCULATION ---
    residuals_x = np.array(residuals_x)
    residuals_y = np.array(residuals_y)
    n_samples = len(residuals_x)

    q_idx = int(np.ceil((n_samples + 1) * CONFIDENCE))
    q_idx = min(q_idx, n_samples - 1)

    q_hat_x = np.sort(residuals_x)[q_idx]
    q_hat_y = np.sort(residuals_y)[q_idx]

    print(f"\n=== RESULTS (Confidence: {CONFIDENCE*100}%) ===")
    print(f"X-Axis Quantile: {q_hat_x:.4f} m/s²")
    print(f"Y-Axis Quantile: {q_hat_y:.4f} m/s²")

    # --- 6. VISUALIZATION (PDF + CDF) ---
    # We use a 3-row layout: 
    # Row 1: X-Axis Stats (Hist vs CDF)
    # Row 2: Y-Axis Stats (Hist vs CDF)
    # Row 3: Tracking Performance
    
    fig = plt.figure(figsize=(14, 12))
    
    # Helper to calculate CDF
    def get_cdf(data):
        sorted_data = np.sort(data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, y_vals

    # --- ROW 1: X-Axis ---
    # 1.1 Histogram
    ax_hx = fig.add_subplot(3, 2, 1)
    ax_hx.hist(residuals_x, bins=50, color='skyblue', alpha=0.7, density=True, label='Error PDF')
    ax_hx.axvline(q_hat_x, color='red', linestyle='--', linewidth=2, label=f'Quantile: {q_hat_x:.2f}')
    ax_hx.set_title("X-Axis Error Distribution (PDF)")
    ax_hx.set_xlabel("Error (m/s²)"); ax_hx.legend()

    # 1.2 CDF
    ax_cx = fig.add_subplot(3, 2, 2)
    sx, sy = get_cdf(residuals_x)
    ax_cx.plot(sx, sy, linewidth=2, color='blue', label='Empirical CDF')
    ax_cx.axhline(CONFIDENCE, color='green', linestyle='--', label=f'{CONFIDENCE*100}% Confidence')
    ax_cx.axvline(q_hat_x, color='red', linestyle='--', label=f'Quantile: {q_hat_x:.2f}')
    ax_cx.set_title("X-Axis Reliability (CDF)")
    ax_cx.set_ylabel("Probability"); ax_cx.set_xlabel("Error (m/s²)"); ax_cx.legend()
    ax_cx.grid(True, alpha=0.3)

    # --- ROW 2: Y-Axis ---
    # 2.1 Histogram
    ax_hy = fig.add_subplot(3, 2, 3)
    ax_hy.hist(residuals_y, bins=50, color='orange', alpha=0.7, density=True, label='Error PDF')
    ax_hy.axvline(q_hat_y, color='red', linestyle='--', linewidth=2, label=f'Quantile: {q_hat_y:.2f}')
    ax_hy.set_title("Y-Axis Error Distribution (PDF)")
    ax_hy.set_xlabel("Error (m/s²)"); ax_hy.legend()

    # 2.2 CDF
    ax_cy = fig.add_subplot(3, 2, 4)
    sx_y, sy_y = get_cdf(residuals_y)
    ax_cy.plot(sx_y, sy_y, linewidth=2, color='darkorange', label='Empirical CDF')
    ax_cy.axhline(CONFIDENCE, color='green', linestyle='--', label=f'{CONFIDENCE*100}% Confidence')
    ax_cy.axvline(q_hat_y, color='red', linestyle='--', label=f'Quantile: {q_hat_y:.2f}')
    ax_cy.set_title("Y-Axis Reliability (CDF)")
    ax_cy.set_ylabel("Probability"); ax_cy.set_xlabel("Error (m/s²)"); ax_cy.legend()
    ax_cy.grid(True, alpha=0.3)

    # --- ROW 3: Tracking (Spans full width) ---
    ax_track = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
    true_acc_log = np.array(true_acc_log)
    pred_acc_log = np.array(pred_acc_log)
    subset = np.random.choice(n_samples, 200)

    ax_track.plot(true_acc_log[subset, 0], 'k-', alpha=0.6, label='True Physics')
    ax_track.plot(pred_acc_log[subset, 0], 'r--', alpha=0.8, label='Noisy Model Pred')
    ax_track.fill_between(range(len(subset)), 
                    pred_acc_log[subset, 0] - q_hat_x, 
                    pred_acc_log[subset, 0] + q_hat_x, 
                    color='red', alpha=0.2, label='90% Safety Tube')
    ax_track.set_title("Tracking Performance Subset (X-Axis)")
    ax_track.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
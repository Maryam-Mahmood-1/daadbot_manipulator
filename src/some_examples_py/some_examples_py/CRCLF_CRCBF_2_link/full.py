import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Set plotting style for compatibility
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')

np.set_printoptions(suppress=True, precision=4)

# --- URDF PARAMETERS FOR TASK SPACE PLOTTING ---
L1 = 0.75  # Length of Link 1 (meters)
L2 = 1.0   # Length of Link 2 (meters)

# ==========================================
# 1. ENHANCED ROBOTIC LIBRARY (Determinant Expansion)
# ==========================================
def build_library(X, U):
    q0 = X[:, 0:1]; q1 = X[:, 1:2]
    dq0 = X[:, 2:3]; dq1 = X[:, 3:4]
    u0 = U[:, 0:1]; u1 = U[:, 1:2]
    
    s1 = np.sin(q0); c1 = np.cos(q0)
    s2 = np.sin(q1); c2 = np.cos(q1)
    s12 = np.sin(q0 + q1); c12 = np.cos(q0 + q1)
    
    c2_sq = c2**2; s2_sq = s2**2
    dq0_sq = dq0**2; dq1_sq = dq1**2
    dq_cross = dq0 * dq1
    
    H_x = np.hstack([
        np.ones((X.shape[0], 1)), 
        q0, q1, dq0, dq1, 
        s1, c1, s2, c2, s12, c12,
        dq0_sq, dq1_sq, dq_cross,
        dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
        dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
        np.sign(dq0), np.sign(dq1)
    ])
    
    Theta_g0 = np.hstack([u0, u0 * c2, u0 * s2, u0 * c12, u0 * c2_sq, u0 * s2_sq])
    Theta_g1 = np.hstack([u1, u1 * c2, u1 * s2, u1 * c12, u1 * c2_sq, u1 * s2_sq])
    
    return np.hstack([H_x, Theta_g0, Theta_g1])

# ==========================================
# 2. SINDY TRAINING (Normalized STLSQ)
# ==========================================
def sparse_regression(Theta, Y, threshold=0.01, alpha=1e-3):
    std_theta = np.std(Theta, axis=0) + 1e-6
    Theta_n = Theta / std_theta
    std_y = np.std(Y, axis=0) + 1e-6
    Y_n = Y / std_y
    
    def ridge_ls(A, b):
        return np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ A.T @ b

    Xi_n = ridge_ls(Theta_n, Y_n)
    
    for _ in range(15):
        Xi_phys = Xi_n * (std_y[np.newaxis, :] / std_theta[:, np.newaxis])
        small_idx = np.abs(Xi_phys) < threshold
        Xi_n[small_idx] = 0
        for j in range(Y.shape[1]):
            big_idx = ~small_idx[:, j]
            if np.sum(big_idx) > 0:
                Xi_n[big_idx, j] = ridge_ls(Theta_n[:, big_idx], Y_n[:, j])
                
    return Xi_n * (std_y[np.newaxis, :] / std_theta[:, np.newaxis])

# ==========================================
# 3. HELPER: DASHBOARD GENERATOR (100Hz 1-Step Prediction)
# ==========================================
def plot_dashboard(q_true, dq_true, Y_true, Y_pred, title, filename):
    t = np.arange(len(q_true)) * 0.01
    dt = 0.01

    # 1-Step Ahead Prediction (Simulating the 100Hz controller loop)
    q_pred = np.zeros_like(q_true)
    dq_pred = np.zeros_like(dq_true)
    q_pred[0, :] = q_true[0, :]
    dq_pred[0, :] = dq_true[0, :]
    
    for k in range(1, len(t)):
        # The controller measures the TRUE state at k-1, and predicts state at k
        dq_pred[k, :] = dq_true[k-1, :] + Y_pred[k-1, :] * dt
        # Kinematic integration for position
        q_pred[k, :] = q_true[k-1, :] + dq_true[k-1, :] * dt + 0.5 * Y_pred[k-1, :] * dt**2

    # Calculate Errors (Standard shortest path wrap)
    raw_q_error = q_true - q_pred
    q_error = (raw_q_error + np.pi) % (2 * np.pi) - np.pi
    dq_error = dq_true - dq_pred

    # Forward Kinematics
    x1_t = L1 * np.cos(q_true[:, 0])
    y1_t = L1 * np.sin(q_true[:, 0])
    x2_t = x1_t + L2 * np.cos(q_true[:, 0] + q_true[:, 1])
    y2_t = y1_t + L2 * np.sin(q_true[:, 0] + q_true[:, 1])

    x1_p = L1 * np.cos(q_pred[:, 0])
    y1_p = L1 * np.sin(q_pred[:, 0])
    x2_p = x1_p + L2 * np.cos(q_pred[:, 0] + q_pred[:, 1])
    y2_p = y1_p + L2 * np.sin(q_pred[:, 0] + q_pred[:, 1])

    # Plotting
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.5])

    # Left: Task Space
    ax_task = fig.add_subplot(gs[:, 0])
    ax_task.plot(x2_t, y2_t, 'k-', lw=3, label='True GT Path', alpha=0.5)
    ax_task.plot(x2_p, y2_p, 'r--', lw=1.5, label='1-Step Predicted Path')
    ax_task.scatter(x2_t[0], y2_t[0], color='green', s=100, label='Start', zorder=5)
    ax_task.plot(0, 0, 'ko', markersize=10, label='Base Origin')
    ax_task.set_aspect('equal')
    ax_task.set_xlabel('X Position (m)')
    ax_task.set_ylabel('Y Position (m)')
    ax_task.set_title('Task Space Path (100Hz Prediction)')
    ax_task.legend()
    ax_task.grid(True, linestyle='--', alpha=0.5)

    # Right 1: Position Error
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.plot(t, q_error[:, 0], 'b-', label='Joint 0 Pos Error')
    ax_pos.plot(t, q_error[:, 1], 'g-', label='Joint 1 Pos Error')
    ax_pos.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_pos.set_ylabel('Position Error (rad)')
    ax_pos.set_title('Joint State 1-Step Prediction Errors')
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)

    # Right 2: Velocity Error
    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)
    ax_vel.plot(t, dq_error[:, 0], 'b-', label='Joint 0 Vel Error')
    ax_vel.plot(t, dq_error[:, 1], 'g-', label='Joint 1 Vel Error')
    ax_vel.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_vel.set_ylabel('Velocity Error (rad/s)')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)

    # Right 3: Accelerations
    ax_acc = fig.add_subplot(gs[2, 1], sharex=ax_pos)
    ax_acc.plot(t, Y_true[:, 0], 'k-', lw=2, label='J0 True')
    ax_acc.plot(t, Y_pred[:, 0], 'r--', lw=1.5, label='J0 Pred')
    ax_acc.plot(t, Y_true[:, 1], 'k-', lw=2, alpha=0.3, label='J1 True')
    ax_acc.plot(t, Y_pred[:, 1], 'm--', lw=1.5, label='J1 Pred')
    ax_acc.set_ylabel('Accel (rad/s^2)')
    ax_acc.set_xlabel('Time (s)')
    ax_acc.legend(ncol=2)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(filename)

# ==========================================
# 4. PLOTTING FUNCTION MASTER
# ==========================================
def generate_detailed_plots(plot_df, Xi, q_val):
    print("Generating detailed plots...")
    if len(plot_df) == 0:
        return

    plot_df = plot_df.copy()
    
    X_plot = plot_df[['q0', 'q1', 'dq0', 'dq1']].values
    U_plot = plot_df[['tau0', 'tau1']].values
    Y_true = plot_df[['target_ddq0', 'target_ddq1']].values
    
    Theta_plot = build_library(X_plot, U_plot)
    Y_pred = Theta_plot @ Xi
    errors = np.linalg.norm(Y_true - Y_pred, axis=1)
    plot_df['l2_error'] = errors

    # --- Plot 1: Standard Time Series ---
    sample_id = plot_df['traj_id'].unique()[0] if 'traj_id' in plot_df.columns else np.arange(min(500, len(plot_df)))
    mask1 = plot_df['traj_id'] == sample_id if 'traj_id' in plot_df.columns else mask1
    
    t1 = np.arange(np.sum(mask1)) * 0.01 
    fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(2):
        ax1[i].plot(t1, Y_true[mask1, i], 'k-', lw=1.5, label='True Dynamics')
        ax1[i].plot(t1, Y_pred[mask1, i], 'r--', lw=1.5, label='SINDy Prediction')
        ax1[i].set_ylabel(f'Joint {i} Accel (rad/s^2)')
        ax1[i].legend(loc='upper right')
        ax1[i].grid(True, alpha=0.3)
    ax1[1].set_xlabel('Time (s)')
    fig1.suptitle('Learned vs True Joint Acceleration (Standard Trajectory)')
    fig1.tight_layout()
    fig1.savefig('1_acceleration_tracking.png')

    # --- Plot 2: Conformal Histogram ---
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.axvline(q_val, color='darkred', linestyle='--', linewidth=2.5, label=f'Quantile q={q_val:.2f}')
    plt.xlabel('L2 Error Norm')
    plt.ylabel('Density')
    plt.title('Error Distribution & Conformal Bound')
    plt.legend()
    fig2.savefig('2_conformal_distribution.png')

    # --- Plot 3: Spatial Error Heatmap ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    scatter = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=errors, cmap='inferno', s=10, alpha=0.6, vmax=q_val*1.5)
    ax3.grid(False) 
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('L2 Acceleration Error', rotation=270, labelpad=15)
    ax3.set_xlabel('Joint 0 Position (rad)')
    ax3.set_ylabel('Joint 1 Position (rad)')
    ax3.set_title('Spatial Error Distribution Across Full Workspace')
    fig3.tight_layout()
    fig3.savefig('3_spatial_error_heatmap.png')

    # --- Plot 4: Global Parity Plot ---
    fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6))
    for i in range(2):
        hb = ax4[i].hexbin(Y_true[:, i], Y_pred[:, i], gridsize=50, cmap='Blues', mincnt=1)
        cb = fig4.colorbar(hb, ax=ax4[i])
        cb.set_label('Density (Count)')
        min_val = np.min([Y_true[:, i].min(), Y_pred[:, i].min()])
        max_val = np.max([Y_true[:, i].max(), Y_pred[:, i].max()])
        ax4[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax4[i].set_xlabel(f'True Joint {i} Accel (rad/s^2)')
        ax4[i].set_ylabel(f'Predicted Joint {i} Accel (rad/s^2)')
        ax4[i].set_title(f'Global Fit: Joint {i}')
        ax4[i].legend()
        ax4[i].grid(True, alpha=0.3)
    fig4.suptitle('Global Model Performance Across Entire Test Set')
    fig4.tight_layout()
    fig4.savefig('4_global_parity_plot.png')

    # Ensure Trajectory Data is Available
    if 'traj_id' not in plot_df.columns:
        print("Warning: 'traj_id' column missing. Cannot plot trajectory dashboards.")
        return

    # --- Plot 5: WORST-CASE Trajectory Dashboard ---
    print("Plotting Worst-Case Trajectory...")
    worst_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmax()
    worst_mask = plot_df['traj_id'] == worst_traj_id
    plot_dashboard(
        X_plot[worst_mask, :2], X_plot[worst_mask, 2:4], 
        Y_true[worst_mask], Y_pred[worst_mask],
        f'Diagnostic Dashboard: WORST-CASE Trajectory (ID: {worst_traj_id})', 
        '5_worst_case_dashboard.png'
    )

    # --- Plot 6: BEST-CASE Trajectory Dashboard ---
    print("Plotting Best-Case Trajectory...")
    best_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmin()
    best_mask = plot_df['traj_id'] == best_traj_id
    plot_dashboard(
        X_plot[best_mask, :2], X_plot[best_mask, 2:4], 
        Y_true[best_mask], Y_pred[best_mask],
        f'Diagnostic Dashboard: BEST-CASE Trajectory (ID: {best_traj_id})', 
        '6_best_case_dashboard.png'
    )
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
df = pd.read_csv(csv_path)

train_df = df[df['split'] == 'train'].copy()
cal_full = df[df['split'] == 'cal'].copy()

if cal_full.empty:
    print("Warning: 'cal' split not found. Using random split.")
    train_df = df.sample(frac=0.7)
    cal_full = df.drop(train_df.index)

split_idx = int(len(cal_full) * 0.5)
cal_df = cal_full.iloc[:split_idx]
test_df = cal_full.iloc[split_idx:] 

print(f"Data Splits: Train={len(train_df)}, Cal={len(cal_df)}, Plotting={len(test_df)}")

# 1. Training
X_train, U_train, Y_train = train_df[['q0', 'q1', 'dq0', 'dq1']].values, \
                            train_df[['tau0', 'tau1']].values, \
                            train_df[['target_ddq0', 'target_ddq1']].values

print("Training SINDy...")
Theta_train = build_library(X_train, U_train)
Xi = sparse_regression(Theta_train, Y_train, threshold=0.01)

# 2. Calibration
X_cal, U_cal, Y_cal = cal_df[['q0', 'q1', 'dq0', 'dq1']].values, \
                      cal_df[['tau0', 'tau1']].values, \
                      cal_df[['target_ddq0', 'target_ddq1']].values

Theta_cal = build_library(X_cal, U_cal)
Y_pred_cal = Theta_cal @ Xi
scores = np.linalg.norm(Y_cal - Y_pred_cal, axis=1)

delta = 0.1
n = len(scores)
q_1_delta = np.quantile(scores, (1 - delta) * (n + 1) / n, method='higher')

print(f"Safety Bound (q_1-delta): {q_1_delta:.6f}")

# 3. Save
np.save("/home/maryammahmood/xdaadbot_ws/sindy_Xi_joint.npy", Xi)
with open("/home/maryammahmood/xdaadbot_ws/q_quantile.txt", "w") as f:
    f.write(str(q_1_delta))

# 4. Plot
generate_detailed_plots(test_df, Xi, q_1_delta)
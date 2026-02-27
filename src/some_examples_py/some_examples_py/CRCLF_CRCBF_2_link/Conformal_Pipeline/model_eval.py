"""
model_evaluation.py
Generates comprehensive diagnostic plots for the 4D state-space SINDy model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_collection import load_and_split_data
from model_training import build_library
import os

try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')

np.set_printoptions(suppress=True, precision=4)

L1 = 0.75  
L2 = 1.0   

def plot_dashboard(q_true, dq_true, Y_true, Y_pred, title, filename):
    t = np.arange(len(q_true)) * 0.01
    dt = 0.01

    q_pred = np.zeros_like(q_true)
    dq_pred = np.zeros_like(dq_true)
    q_pred[0, :] = q_true[0, :]
    dq_pred[0, :] = dq_true[0, :]
    
    for k in range(1, len(t)):
        # Using Y_pred[k-1, 2:4] (ddq) to integrate velocity
        dq_pred[k, :] = dq_true[k-1, :] + Y_pred[k-1, 2:4] * dt
        # Using SINDy's direct velocity prediction Y_pred[k-1, 0:2] to integrate position
        q_pred[k, :] = q_true[k-1, :] + Y_pred[k-1, 0:2] * dt + 0.5 * Y_pred[k-1, 2:4] * dt**2

    raw_q_error = q_true - q_pred
    q_error = (raw_q_error + np.pi) % (2 * np.pi) - np.pi
    dq_error = dq_true - dq_pred

    x1_t = L1 * np.cos(q_true[:, 0])
    y1_t = L1 * np.sin(q_true[:, 0])
    x2_t = x1_t + L2 * np.cos(q_true[:, 0] + q_true[:, 1])
    y2_t = y1_t + L2 * np.sin(q_true[:, 0] + q_true[:, 1])

    x1_p = L1 * np.cos(q_pred[:, 0])
    y1_p = L1 * np.sin(q_pred[:, 0])
    x2_p = x1_p + L2 * np.cos(q_pred[:, 0] + q_pred[:, 1])
    y2_p = y1_p + L2 * np.sin(q_pred[:, 0] + q_pred[:, 1])

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.5])

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

    ax_pos = fig.add_subplot(gs[0, 1])
    ax_pos.plot(t, q_error[:, 0], 'b-', label='Joint 0 Pos Error')
    ax_pos.plot(t, q_error[:, 1], 'g-', label='Joint 1 Pos Error')
    ax_pos.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_pos.set_ylabel('Position Error (rad)')
    ax_pos.set_title('Joint State 1-Step Prediction Errors')
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)

    ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)
    ax_vel.plot(t, dq_error[:, 0], 'b-', label='Joint 0 Vel Error')
    ax_vel.plot(t, dq_error[:, 1], 'g-', label='Joint 1 Vel Error')
    ax_vel.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax_vel.set_ylabel('Velocity Error (rad/s)')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)

    ax_acc = fig.add_subplot(gs[2, 1], sharex=ax_pos)
    ax_acc.plot(t, Y_true[:, 2], 'k-', lw=2, label='J0 ddq True')
    ax_acc.plot(t, Y_pred[:, 2], 'r--', lw=1.5, label='J0 ddq Pred')
    ax_acc.plot(t, Y_true[:, 3], 'k-', lw=2, alpha=0.3, label='J1 ddq True')
    ax_acc.plot(t, Y_pred[:, 3], 'm--', lw=1.5, label='J1 ddq Pred')
    ax_acc.set_ylabel('Accel (rad/s^2)')
    ax_acc.set_xlabel('Time (s)')
    ax_acc.legend(ncol=2)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(filename)

def generate_detailed_plots(plot_df, Xi, q_val):
    print("Generating detailed plots...")
    if len(plot_df) == 0:
        return

    plot_df = plot_df.copy()
    
    X_plot = plot_df[['q0', 'q1', 'dq0', 'dq1']].values
    U_plot = plot_df[['tau0', 'tau1']].values
    
    Y_true = plot_df[['dq0', 'dq1', 'target_ddq0', 'target_ddq1']].values
    
    Theta_plot = build_library(X_plot, U_plot)
    Y_pred = Theta_plot @ Xi
    errors = np.linalg.norm(Y_true - Y_pred, axis=1)
    plot_df['l2_error'] = errors

    # --- Plot 1: Standard Time Series (4 Panels) ---
    sample_id = plot_df['traj_id'].unique()[0] if 'traj_id' in plot_df.columns else np.arange(min(500, len(plot_df)))
    mask1 = plot_df['traj_id'] == sample_id if 'traj_id' in plot_df.columns else mask1
    t1 = np.arange(np.sum(mask1)) * 0.01 
    
    fig1, ax1 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    labels = ['J0 Vel (rad/s)', 'J1 Vel (rad/s)', 'J0 Accel (rad/s^2)', 'J1 Accel (rad/s^2)']
    for i in range(4):
        ax1[i].plot(t1, Y_true[mask1, i], 'k-', lw=1.5, label='True Dynamics')
        ax1[i].plot(t1, Y_pred[mask1, i], 'r--', lw=1.5, label='SINDy Prediction')
        ax1[i].set_ylabel(labels[i])
        ax1[i].legend(loc='upper right')
        ax1[i].grid(True, alpha=0.3)
    ax1[3].set_xlabel('Time (s)')
    fig1.suptitle('Learned vs True State Derivatives x_dot')
    fig1.tight_layout()
    fig1.savefig('1_state_derivative_tracking.png')

    # --- Plot 2: Conformal Histogram & CDF ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Histogram (PDF)
    ax2a.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue', label='Error Frequency')
    ax2a.axvline(q_val, color='darkred', linestyle='--', linewidth=2.5, label=f'Quantile q={q_val:.2f}')
    ax2a.set_xlabel('L2 Error Norm (Full State Derivative)')
    ax2a.set_ylabel('Density')
    ax2a.set_title('Probability Density Function (PDF)')
    ax2a.legend()

    # Right: Cumulative Distribution Function (CDF)
    sorted_errors = np.sort(errors)
    cdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Calculate empirical probability (1-delta) represented by the quantile
    empirical_prob = np.searchsorted(sorted_errors, q_val) / len(sorted_errors)

    ax2b.plot(sorted_errors, cdf_values, color='darkorange', lw=2.5, label='Error CDF')
    ax2b.axvline(q_val, color='darkred', linestyle='--', linewidth=2.5, label=f'Quantile q={q_val:.2f}')
    ax2b.axhline(empirical_prob, color='grey', linestyle=':', linewidth=2, label=f'1-δ ≈ {empirical_prob:.2f}')
    
    ax2b.set_xlabel('L2 Error Norm (Full State Derivative)')
    ax2b.set_ylabel('Cumulative Probability')
    ax2b.set_title('Cumulative Distribution Function (CDF)')
    ax2b.grid(True, alpha=0.3)
    ax2b.legend()

    fig2.suptitle('Error Distribution & Conformal Calibration Bound', fontsize=16)
    fig2.tight_layout()
    fig2.savefig('2_conformal_distribution.png')

    # --- Plot 3: Spatial Error Heatmap ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    scatter = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=errors, cmap='inferno', s=10, alpha=0.6, vmax=q_val*1.5)
    ax3.grid(False) 
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('L2 Mismatch (x_dot)', rotation=270, labelpad=15)
    ax3.set_xlabel('Joint 0 Position (rad)')
    ax3.set_ylabel('Joint 1 Position (rad)')
    ax3.set_title('Spatial Error Distribution Across Full Workspace')
    fig3.tight_layout()
    fig3.savefig('3_spatial_error_heatmap.png')

    # --- Plot 4: Global Parity Plot (2x2 Grid) ---
    fig4, ax4 = plt.subplots(2, 2, figsize=(14, 12))
    ax4 = ax4.flatten()
    for i in range(4):
        hb = ax4[i].hexbin(Y_true[:, i], Y_pred[:, i], gridsize=50, cmap='Blues', mincnt=1)
        cb = fig4.colorbar(hb, ax=ax4[i])
        cb.set_label('Density (Count)')
        min_val = np.min([Y_true[:, i].min(), Y_pred[:, i].min()])
        max_val = np.max([Y_true[:, i].max(), Y_pred[:, i].max()])
        ax4[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax4[i].set_xlabel(f'True {labels[i]}')
        ax4[i].set_ylabel(f'Pred {labels[i]}')
        ax4[i].set_title(f'Global Fit: {labels[i]}')
        ax4[i].legend()
        ax4[i].grid(True, alpha=0.3)
    fig4.suptitle('Global Model Performance (All State Derivatives)')
    fig4.tight_layout()
    fig4.savefig('4_global_parity_plot.png')

    if 'traj_id' not in plot_df.columns:
        print("Warning: 'traj_id' column missing. Cannot plot trajectory dashboards.")
        return

    # --- Plot 5 & 6: Dashboards ---
    print("Plotting Dashboards...")
    worst_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmax()
    worst_mask = plot_df['traj_id'] == worst_traj_id
    plot_dashboard(
        X_plot[worst_mask, :2], X_plot[worst_mask, 2:4], 
        Y_true[worst_mask], Y_pred[worst_mask],
        f'WORST-CASE Trajectory Dashboard (ID: {worst_traj_id})', 
        '5_worst_case_dashboard.png'
    )

    best_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmin()
    best_mask = plot_df['traj_id'] == best_traj_id
    plot_dashboard(
        X_plot[best_mask, :2], X_plot[best_mask, 2:4], 
        Y_true[best_mask], Y_pred[best_mask],
        f'BEST-CASE Trajectory Dashboard (ID: {best_traj_id})', 
        '6_best_case_dashboard.png'
    )
    plt.show()

if __name__ == "__main__":
    csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
    
    try:
        _, _, test_df = load_and_split_data(csv_path)
        Xi = np.load("/home/maryammahmood/xdaadbot_ws/sindy_Xi_state_space.npy")
        with open("/home/maryammahmood/xdaadbot_ws/q_quantile_state_space.txt", "r") as f:
            q_val = float(f.read())
            
        generate_detailed_plots(test_df, Xi, q_val)
    except Exception as e:
        print(f"Error loading files: {e}\nPlease ensure data and model files exist.")



# """
# model_evaluation.py
# Generates comprehensive diagnostic plots for the learned SINDy model.
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from data_collection import load_and_split_data
# from model_training import build_library
# import os

# # Set plotting style
# try:
#     plt.style.use('seaborn-v0_8')
# except:
#     plt.style.use('ggplot')

# np.set_printoptions(suppress=True, precision=4)

# # URDF PARAMETERS
# L1 = 0.75  
# L2 = 1.0   

# def plot_dashboard(q_true, dq_true, Y_true, Y_pred, title, filename):
#     t = np.arange(len(q_true)) * 0.01
#     dt = 0.01

#     q_pred = np.zeros_like(q_true)
#     dq_pred = np.zeros_like(dq_true)
#     q_pred[0, :] = q_true[0, :]
#     dq_pred[0, :] = dq_true[0, :]
    
#     for k in range(1, len(t)):
#         dq_pred[k, :] = dq_true[k-1, :] + Y_pred[k-1, :] * dt
#         q_pred[k, :] = q_true[k-1, :] + dq_true[k-1, :] * dt + 0.5 * Y_pred[k-1, :] * dt**2

#     raw_q_error = q_true - q_pred
#     q_error = (raw_q_error + np.pi) % (2 * np.pi) - np.pi
#     dq_error = dq_true - dq_pred

#     x1_t = L1 * np.cos(q_true[:, 0])
#     y1_t = L1 * np.sin(q_true[:, 0])
#     x2_t = x1_t + L2 * np.cos(q_true[:, 0] + q_true[:, 1])
#     y2_t = y1_t + L2 * np.sin(q_true[:, 0] + q_true[:, 1])

#     x1_p = L1 * np.cos(q_pred[:, 0])
#     y1_p = L1 * np.sin(q_pred[:, 0])
#     x2_p = x1_p + L2 * np.cos(q_pred[:, 0] + q_pred[:, 1])
#     y2_p = y1_p + L2 * np.sin(q_pred[:, 0] + q_pred[:, 1])

#     fig = plt.figure(figsize=(16, 10))
#     gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.5])

#     ax_task = fig.add_subplot(gs[:, 0])
#     ax_task.plot(x2_t, y2_t, 'k-', lw=3, label='True GT Path', alpha=0.5)
#     ax_task.plot(x2_p, y2_p, 'r--', lw=1.5, label='1-Step Predicted Path')
#     ax_task.scatter(x2_t[0], y2_t[0], color='green', s=100, label='Start', zorder=5)
#     ax_task.plot(0, 0, 'ko', markersize=10, label='Base Origin')
#     ax_task.set_aspect('equal')
#     ax_task.set_xlabel('X Position (m)')
#     ax_task.set_ylabel('Y Position (m)')
#     ax_task.set_title('Task Space Path (100Hz Prediction)')
#     ax_task.legend()
#     ax_task.grid(True, linestyle='--', alpha=0.5)

#     ax_pos = fig.add_subplot(gs[0, 1])
#     ax_pos.plot(t, q_error[:, 0], 'b-', label='Joint 0 Pos Error')
#     ax_pos.plot(t, q_error[:, 1], 'g-', label='Joint 1 Pos Error')
#     ax_pos.axhline(0, color='grey', linestyle='--', alpha=0.5)
#     ax_pos.set_ylabel('Position Error (rad)')
#     ax_pos.set_title('Joint State 1-Step Prediction Errors')
#     ax_pos.legend()
#     ax_pos.grid(True, alpha=0.3)

#     ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)
#     ax_vel.plot(t, dq_error[:, 0], 'b-', label='Joint 0 Vel Error')
#     ax_vel.plot(t, dq_error[:, 1], 'g-', label='Joint 1 Vel Error')
#     ax_vel.axhline(0, color='grey', linestyle='--', alpha=0.5)
#     ax_vel.set_ylabel('Velocity Error (rad/s)')
#     ax_vel.legend()
#     ax_vel.grid(True, alpha=0.3)

#     ax_acc = fig.add_subplot(gs[2, 1], sharex=ax_pos)
#     ax_acc.plot(t, Y_true[:, 0], 'k-', lw=2, label='J0 True')
#     ax_acc.plot(t, Y_pred[:, 0], 'r--', lw=1.5, label='J0 Pred')
#     ax_acc.plot(t, Y_true[:, 1], 'k-', lw=2, alpha=0.3, label='J1 True')
#     ax_acc.plot(t, Y_pred[:, 1], 'm--', lw=1.5, label='J1 Pred')
#     ax_acc.set_ylabel('Accel (rad/s^2)')
#     ax_acc.set_xlabel('Time (s)')
#     ax_acc.legend(ncol=2)
#     ax_acc.grid(True, alpha=0.3)

#     fig.suptitle(title, fontsize=16)
#     fig.tight_layout()
#     fig.savefig(filename)

# def generate_detailed_plots(plot_df, Xi, q_val):
#     print("Generating detailed plots...")
#     if len(plot_df) == 0:
#         return

#     plot_df = plot_df.copy()
    
#     X_plot = plot_df[['q0', 'q1', 'dq0', 'dq1']].values
#     U_plot = plot_df[['tau0', 'tau1']].values
#     Y_true = plot_df[['target_ddq0', 'target_ddq1']].values
    
#     Theta_plot = build_library(X_plot, U_plot)
#     Y_pred = Theta_plot @ Xi
#     errors = np.linalg.norm(Y_true - Y_pred, axis=1)
#     plot_df['l2_error'] = errors

#     # --- Plot 1: Standard Time Series ---
#     sample_id = plot_df['traj_id'].unique()[0] if 'traj_id' in plot_df.columns else np.arange(min(500, len(plot_df)))
#     mask1 = plot_df['traj_id'] == sample_id if 'traj_id' in plot_df.columns else mask1
    
#     t1 = np.arange(np.sum(mask1)) * 0.01 
#     fig1, ax1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     for i in range(2):
#         ax1[i].plot(t1, Y_true[mask1, i], 'k-', lw=1.5, label='True Dynamics')
#         ax1[i].plot(t1, Y_pred[mask1, i], 'r--', lw=1.5, label='SINDy Prediction')
#         ax1[i].set_ylabel(f'Joint {i} Accel (rad/s^2)')
#         ax1[i].legend(loc='upper right')
#         ax1[i].grid(True, alpha=0.3)
#     ax1[1].set_xlabel('Time (s)')
#     fig1.suptitle('Learned vs True Joint Acceleration (Standard Trajectory)')
#     fig1.tight_layout()
#     fig1.savefig('1_acceleration_tracking.png')

#     # --- Plot 2: Conformal Histogram ---
#     fig2 = plt.figure(figsize=(10, 6))
#     plt.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue')
#     plt.axvline(q_val, color='darkred', linestyle='--', linewidth=2.5, label=f'Quantile q={q_val:.2f}')
#     plt.xlabel('L2 Error Norm')
#     plt.ylabel('Density')
#     plt.title('Error Distribution & Conformal Bound')
#     plt.legend()
#     fig2.savefig('2_conformal_distribution.png')

#     # --- Plot 3: Spatial Error Heatmap ---
#     fig3, ax3 = plt.subplots(figsize=(10, 8))
#     scatter = ax3.scatter(X_plot[:, 0], X_plot[:, 1], c=errors, cmap='inferno', s=10, alpha=0.6, vmax=q_val*1.5)
#     ax3.grid(False) 
#     cbar = plt.colorbar(scatter, ax=ax3)
#     cbar.set_label('L2 Acceleration Error', rotation=270, labelpad=15)
#     ax3.set_xlabel('Joint 0 Position (rad)')
#     ax3.set_ylabel('Joint 1 Position (rad)')
#     ax3.set_title('Spatial Error Distribution Across Full Workspace')
#     fig3.tight_layout()
#     fig3.savefig('3_spatial_error_heatmap.png')

#     # --- Plot 4: Global Parity Plot ---
#     fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6))
#     for i in range(2):
#         hb = ax4[i].hexbin(Y_true[:, i], Y_pred[:, i], gridsize=50, cmap='Blues', mincnt=1)
#         cb = fig4.colorbar(hb, ax=ax4[i])
#         cb.set_label('Density (Count)')
#         min_val = np.min([Y_true[:, i].min(), Y_pred[:, i].min()])
#         max_val = np.max([Y_true[:, i].max(), Y_pred[:, i].max()])
#         ax4[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
#         ax4[i].set_xlabel(f'True Joint {i} Accel (rad/s^2)')
#         ax4[i].set_ylabel(f'Predicted Joint {i} Accel (rad/s^2)')
#         ax4[i].set_title(f'Global Fit: Joint {i}')
#         ax4[i].legend()
#         ax4[i].grid(True, alpha=0.3)
#     fig4.suptitle('Global Model Performance Across Entire Test Set')
#     fig4.tight_layout()
#     fig4.savefig('4_global_parity_plot.png')

#     if 'traj_id' not in plot_df.columns:
#         print("Warning: 'traj_id' column missing. Cannot plot trajectory dashboards.")
#         return

#     # --- Plot 5: WORST-CASE Trajectory Dashboard ---
#     print("Plotting Worst-Case Trajectory...")
#     worst_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmax()
#     worst_mask = plot_df['traj_id'] == worst_traj_id
#     plot_dashboard(
#         X_plot[worst_mask, :2], X_plot[worst_mask, 2:4], 
#         Y_true[worst_mask], Y_pred[worst_mask],
#         f'Diagnostic Dashboard: WORST-CASE Trajectory (ID: {worst_traj_id})', 
#         '5_worst_case_dashboard.png'
#     )

#     # --- Plot 6: BEST-CASE Trajectory Dashboard ---
#     print("Plotting Best-Case Trajectory...")
#     best_traj_id = plot_df.groupby('traj_id')['l2_error'].mean().idxmin()
#     best_mask = plot_df['traj_id'] == best_traj_id
#     plot_dashboard(
#         X_plot[best_mask, :2], X_plot[best_mask, 2:4], 
#         Y_true[best_mask], Y_pred[best_mask],
#         f'Diagnostic Dashboard: BEST-CASE Trajectory (ID: {best_traj_id})', 
#         '6_best_case_dashboard.png'
#     )
#     plt.show()

# if __name__ == "__main__":
#     csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
    
#     # Load test data and the saved model
#     _, _, test_df = load_and_split_data(csv_path)
    
#     try:
#         Xi = np.load("/home/maryammahmood/xdaadbot_ws/sindy_Xi_joint.npy")
#         with open("/home/maryammahmood/xdaadbot_ws/q_quantile.txt", "r") as f:
#             q_val = float(f.read())
            
#         generate_detailed_plots(test_df, Xi, q_val)
#     except FileNotFoundError:
#         print("Model files not found. Please run model_training.py first.")
"""
model_training.py
Trains the SINDy model for the full state derivative x_dot = [dq, ddq].
"""
import numpy as np
import pandas as pd
import os
from data_collection import load_and_split_data

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

def train_and_calibrate(train_df, cal_df, delta=0.1, threshold=0.01):
    X_train = train_df[['q0', 'q1', 'dq0', 'dq1']].values
    U_train = train_df[['tau0', 'tau1']].values
    
    # NEW: Target is now 4D (dq0, dq1, ddq0, ddq1)
    Y_train = train_df[['dq0', 'dq1', 'target_ddq0', 'target_ddq1']].values

    print("Training State-Space SINDy Model...")
    Theta_train = build_library(X_train, U_train)
    Xi = sparse_regression(Theta_train, Y_train, threshold=threshold)

    X_cal = cal_df[['q0', 'q1', 'dq0', 'dq1']].values
    U_cal = cal_df[['tau0', 'tau1']].values
    
    # NEW: Target is now 4D
    Y_cal = cal_df[['dq0', 'dq1', 'target_ddq0', 'target_ddq1']].values

    print("Calibrating Conformal Bound...")
    Theta_cal = build_library(X_cal, U_cal)
    Y_pred_cal = Theta_cal @ Xi
    
    # Bound calculated over the full 4D state derivative mismatch
    scores = np.linalg.norm(Y_cal - Y_pred_cal, axis=1)
    n = len(scores)
    q_1_delta = np.quantile(scores, (1 - delta) * (n + 1) / n, method='higher')
    
    return Xi, q_1_delta

if __name__ == "__main__":
    csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
    train_df, cal_df, _ = load_and_split_data(csv_path)
    
    Xi, q_val = train_and_calibrate(train_df, cal_df)
    
    print(f"\nModel Shape (Features x Targets): {Xi.shape}")
    print(f"Safety Bound (q_1-delta): {q_val:.6f}")
    
    np.save("/home/maryammahmood/xdaadbot_ws/sindy_Xi_state_space.npy", Xi)
    with open("/home/maryammahmood/xdaadbot_ws/q_quantile_state_space.txt", "w") as f:
        f.write(str(q_val))
    print("Model and Quantile Saved.")


# """
# model_training.py
# Trains the SINDy model and calibrates the Conformal Prediction bound.
# """
# import numpy as np
# import os
# from data_collection import load_and_split_data

# # ==========================================
# # 1. ENHANCED ROBOTIC LIBRARY (Determinant Expansion)
# # ==========================================
# def build_library(X, U):
#     q0 = X[:, 0:1]; q1 = X[:, 1:2]
#     dq0 = X[:, 2:3]; dq1 = X[:, 3:4]
#     u0 = U[:, 0:1]; u1 = U[:, 1:2]
    
#     s1 = np.sin(q0); c1 = np.cos(q0)
#     s2 = np.sin(q1); c2 = np.cos(q1)
#     s12 = np.sin(q0 + q1); c12 = np.cos(q0 + q1)
    
#     c2_sq = c2**2; s2_sq = s2**2
#     dq0_sq = dq0**2; dq1_sq = dq1**2
#     dq_cross = dq0 * dq1
    
#     H_x = np.hstack([
#         np.ones((X.shape[0], 1)), 
#         q0, q1, dq0, dq1, 
#         s1, c1, s2, c2, s12, c12,
#         dq0_sq, dq1_sq, dq_cross,
#         dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#         dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#         np.sign(dq0), np.sign(dq1)
#     ])
    
#     Theta_g0 = np.hstack([u0, u0 * c2, u0 * s2, u0 * c12, u0 * c2_sq, u0 * s2_sq])
#     Theta_g1 = np.hstack([u1, u1 * c2, u1 * s2, u1 * c12, u1 * c2_sq, u1 * s2_sq])
    
#     return np.hstack([H_x, Theta_g0, Theta_g1])

# # ==========================================
# # 2. SINDY TRAINING (Normalized STLSQ)
# # ==========================================
# def sparse_regression(Theta, Y, threshold=0.01, alpha=1e-3):
#     std_theta = np.std(Theta, axis=0) + 1e-6
#     Theta_n = Theta / std_theta
#     std_y = np.std(Y, axis=0) + 1e-6
#     Y_n = Y / std_y
    
#     def ridge_ls(A, b):
#         return np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ A.T @ b

#     Xi_n = ridge_ls(Theta_n, Y_n)
    
#     for _ in range(15):
#         Xi_phys = Xi_n * (std_y[np.newaxis, :] / std_theta[:, np.newaxis])
#         small_idx = np.abs(Xi_phys) < threshold
#         Xi_n[small_idx] = 0
#         for j in range(Y.shape[1]):
#             big_idx = ~small_idx[:, j]
#             if np.sum(big_idx) > 0:
#                 Xi_n[big_idx, j] = ridge_ls(Theta_n[:, big_idx], Y_n[:, j])
                
#     return Xi_n * (std_y[np.newaxis, :] / std_theta[:, np.newaxis])

# def train_and_calibrate(train_df, cal_df, delta=0.1, threshold=0.01):
#     # 1. Training
#     X_train = train_df[['q0', 'q1', 'dq0', 'dq1']].values
#     U_train = train_df[['tau0', 'tau1']].values
#     Y_train = train_df[['target_ddq0', 'target_ddq1']].values

#     print("Training SINDy Model...")
#     Theta_train = build_library(X_train, U_train)
#     Xi = sparse_regression(Theta_train, Y_train, threshold=threshold)

#     # 2. Calibration
#     X_cal = cal_df[['q0', 'q1', 'dq0', 'dq1']].values
#     U_cal = cal_df[['tau0', 'tau1']].values
#     Y_cal = cal_df[['target_ddq0', 'target_ddq1']].values

#     print("Calibrating Conformal Bound...")
#     Theta_cal = build_library(X_cal, U_cal)
#     Y_pred_cal = Theta_cal @ Xi
#     scores = np.linalg.norm(Y_cal - Y_pred_cal, axis=1)

#     n = len(scores)
#     q_1_delta = np.quantile(scores, (1 - delta) * (n + 1) / n, method='higher')
    
#     return Xi, q_1_delta

# if __name__ == "__main__":
#     csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
#     train_df, cal_df, _ = load_and_split_data(csv_path)
    
#     Xi, q_val = train_and_calibrate(train_df, cal_df)
#     print(f"Safety Bound (q_1-delta): {q_val:.6f}")
    
#     np.save("/home/maryammahmood/xdaadbot_ws/sindy_Xi_joint.npy", Xi)
#     with open("/home/maryammahmood/xdaadbot_ws/q_quantile.txt", "w") as f:
#         f.write(str(q_val))
#     print("Model and Quantile Saved.")


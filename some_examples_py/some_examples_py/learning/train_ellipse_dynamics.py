#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import pinocchio as pin  # <--- CRITICAL IMPORT FOR PHYSICS

# --- CONFIGURATION ---
from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

DEFAULT_DATA_PATH = "/home/maryammahmood/xdaadbot_ws/robot_data_sum_sines_20251231_215405.csv"
MODEL_SAVE_PATH = "/home/maryammahmood/xdaadbot_ws/dynamics_model_residual.pth"

EPOCHS = 1000
LEARNING_RATE = 0.001 
BATCH_SIZE = 128  # Keep this small (64-256) for best generalization!

# Joints to model (Must match URDF and CSV)
JOINT_NAMES = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

# --- 1. DEFINE RESIDUAL NETWORK ---
class ResidualDynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualDynamicsNet, self).__init__()
        
        # LeakyReLU is industry standard for regression
        # Network is "Bottlenecked" to force feature compression
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, output_dim)
        )
        
        # Initialize weights properly (He initialization)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.net(x)

def compute_pinocchio_dynamics(model, data, q, v, tau):
    """
    Uses Pinocchio (ABA) to compute Forward Dynamics:
    Given q, v, tau -> Returns acc (acceleration)
    """
    pin.aba(model, data, q, v, tau)
    return data.ddq

def main():
    parser = argparse.ArgumentParser(description="Train Residual Dynamics Model")
    parser.add_argument("--file", type=str, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device.type.upper()}")

    # --- A. LOAD PINOCCHIO MODEL ---
    print(f"🤖 Loading URDF: {URDF_PATH}")
    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()
    
    # Map joint names to Pinocchio IDs
    # (Subtract 1 because Pinocchio Universe frame is 0)
    pin_joint_ids = [pin_model.getJointId(name) for name in JOINT_NAMES]
    q_indices = [pin_model.joints[i].idx_q for i in pin_joint_ids]
    v_indices = [pin_model.joints[i].idx_v for i in pin_joint_ids]

    # --- B. LOAD DATA ---
    if not os.path.exists(args.file):
        print(f"❌ Error: File not found at {args.file}")
        sys.exit(1)
        
    print(f"📂 Loading data from: {args.file}")
    df = pd.read_csv(args.file)

    # --- C. COMPUTE PHYSICS BASELINE (RBD) ---
    print("🧠 Computing Rigid Body Dynamics (Pinocchio) for all samples...")
    # Arrays to hold sorted data for Pinocchio
    n_samples = len(df)
    
    # Extract columns in correct order
    q_data = df[[f'q_{n}' for n in JOINT_NAMES]].values
    v_data = df[[f'v_{n}' for n in JOINT_NAMES]].values
    tau_data = df[[f'tau_{n}' for n in JOINT_NAMES]].values
    acc_true = df[[f'acc_{n}' for n in JOINT_NAMES]].values
    
    acc_rbd = np.zeros_like(acc_true)

    # Loop through data to compute theoretical acceleration
    # (This takes a few seconds but is worth it)
    for i in range(n_samples):
        q_vec = np.zeros(pin_model.nq)
        v_vec = np.zeros(pin_model.nv)
        tau_vec = np.zeros(pin_model.nv)
        
        # Fill vectors based on mapping
        for j, (q_idx, v_idx) in enumerate(zip(q_indices, v_indices)):
            q_vec[q_idx] = q_data[i, j]
            v_vec[v_idx] = v_data[i, j]
            tau_vec[v_idx] = tau_data[i, j] # torque index maps to velocity index
            
        ddq = compute_pinocchio_dynamics(pin_model, pin_data, q_vec, v_vec, tau_vec)
        
        # Extract relevant accelerations
        for j, v_idx in enumerate(v_indices):
            acc_rbd[i, j] = ddq[v_idx]

    # --- D. COMPUTE RESIDUAL TARGET ---
    # The AI only learns: Actual - Theoretical
    acc_residual = acc_true - acc_rbd
    
    print("   - Physics Calculation Complete.")
    print(f"   - Mean Residual Error (Physics Gap): {np.mean(np.abs(acc_residual)):.4f}")

    # --- E. SPLIT DATA ---
    split_idx = int(df['episode_id'].max() * 0.8)
    train_mask = df['episode_id'] <= split_idx
    test_mask = df['episode_id'] > split_idx
    
    # Inputs: State + Torque
    X_raw = np.hstack([q_data, v_data, tau_data])
    # Target: The Residual Error
    y_raw = acc_residual 

    X_train, y_train = X_raw[train_mask], y_raw[train_mask]
    X_test, y_test = X_raw[test_mask], y_raw[test_mask]
    
    # Keep track of the BASELINE (Physics only) accuracy for comparison
    acc_true_test = acc_true[test_mask]
    acc_rbd_test = acc_rbd[test_mask]
    baseline_mse = np.mean((acc_true_test - acc_rbd_test)**2)
    print(f"   - Baseline (Physics Only) MSE: {baseline_mse:.5f}")

    # --- F. NORMALIZE ---
    print("⚖️  Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train)
    
    X_test_s = scaler_X.transform(X_test)
    y_test_s = scaler_y.transform(y_test)

    # To Tensors
    X_train_t = torch.FloatTensor(X_train_s).to(device)
    y_train_t = torch.FloatTensor(y_train_s).to(device)
    X_test_t = torch.FloatTensor(X_test_s).to(device)
    y_test_t = torch.FloatTensor(y_test_s).to(device)

    # --- G. TRAINING SETUP ---
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = ResidualDynamicsNet(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight decay
    
    # Huber Loss is better for outliers than MSE
    criterion = nn.HuberLoss(delta=1.0) 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)

    # --- H. TRAINING LOOP ---
    print("💪 Starting Residual Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t)
        
        scheduler.step(val_loss)
        
        if epoch % 50 == 0:
            print(f"   - Epoch {epoch}: Train Loss {avg_train_loss:.6f} | Val Loss {val_loss.item():.6f}")

    # --- I. FINAL EVALUATION ---
    print("🧪 Final Evaluation...")
    model.eval()
    with torch.no_grad():
        # Predict Residuals
        pred_resid_s = model(X_test_t).cpu().numpy()
        pred_resid = scaler_y.inverse_transform(pred_resid_s)
        
        # Combine: Physics + AI
        final_prediction = acc_rbd_test + pred_resid
        
        # Calculate Final MSE
        final_mse = np.mean((acc_true_test - final_prediction)**2)
        improvement = (baseline_mse - final_mse) / baseline_mse * 100

    print(f"   - Physics Only MSE: {baseline_mse:.5f}")
    print(f"   - Physics + AI MSE: {final_mse:.5f}")
    print(f"   - Improvement:      {improvement:.2f}%")

    # --- J. SAVE ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open("scaler_X_resid.pkl", "wb") as f: pickle.dump(scaler_X, f)
    with open("scaler_y_resid.pkl", "wb") as f: pickle.dump(scaler_y, f)
    print(f"💾 Model saved to '{MODEL_SAVE_PATH}'")

    # --- K. PLOT ---
    plt.figure(figsize=(12, 6))
    
    # Joint 1
    plt.subplot(1, 2, 1)
    plt.plot(acc_true_test[:300, 0], 'k', alpha=0.5, label='Actual')
    plt.plot(acc_rbd_test[:300, 0], 'b--', alpha=0.5, label='Physics Only')
    plt.plot(final_prediction[:300, 0], 'r', label='Physics + AI')
    plt.title("Joint 1 Accel Prediction")
    plt.legend()
    
    # Joint 4
    plt.subplot(1, 2, 2)
    plt.plot(acc_true_test[:300, 3], 'k', alpha=0.5, label='Actual')
    plt.plot(acc_rbd_test[:300, 3], 'b--', alpha=0.5, label='Physics Only')
    plt.plot(final_prediction[:300, 3], 'r', label='Physics + AI')
    plt.title("Joint 4 Accel Prediction")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
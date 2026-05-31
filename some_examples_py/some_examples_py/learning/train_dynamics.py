#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# --- CONFIGURATION ---
DEFAULT_DATA_PATH = "/home/maryammahmood/xdaadbot_ws/robot_data_dynamic_20251226_110730.csv"
MODEL_SAVE_PATH = "/home/maryammahmood/xdaadbot_ws/dynamics_model_deep.pth"
EPOCHS = 9000
# Lowered LR slightly because larger networks can be unstable with high LR
LEARNING_RATE = 0.001 

# --- 1. DEFINE DEEPER MODEL ---
class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicsNet, self).__init__()
        # Deeper and Wider Architecture
        self.net = nn.Sequential(
            # Layer 1: Expand features
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            
            # Layer 2: Deep processing
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 3: Deep processing
            nn.Linear(256, 256),
            nn.ReLU(),
            
            # Layer 4: Bottleneck before output
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    # --- A. SETUP ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Train Deep Forward Dynamics Model")
    parser.add_argument("--file", type=str, default=DEFAULT_DATA_PATH, help="Path to the CSV data file")
    args = parser.parse_args()

    # --- B. DEVICE SETUP (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device.type.upper()}")

    # --- C. LOAD DATA ---
    if not os.path.exists(args.file):
        print(f"❌ Error: File not found at {args.file}")
        sys.exit(1)
        
    print(f"📂 Loading data from: {args.file}")
    df = pd.read_csv(args.file)
    print(f"   - Loaded {len(df)} samples.")

    # --- D. PREPARE FEATURES ---
    q_cols = [c for c in df.columns if 'q_' in c]
    v_cols = [c for c in df.columns if 'v_' in c]
    tau_cols = [c for c in df.columns if 'tau_' in c]
    acc_cols = [c for c in df.columns if 'acc_' in c]

    X_raw = df[q_cols + v_cols + tau_cols].values
    y_raw = df[acc_cols].values

    # --- E. SPLIT BY EPISODE ---
    train_mask = df['episode_id'] <= 80
    test_mask = df['episode_id'] > 80

    X_train = X_raw[train_mask]
    y_train = y_raw[train_mask]
    X_test = X_raw[test_mask]
    y_test = y_raw[test_mask]

    print(f"   - Training Set: {len(X_train)} samples")
    print(f"   - Testing Set:  {len(X_test)} samples")

    # --- F. NORMALIZE ---
    print("⚖️  Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train)
    
    X_test_s = scaler_X.transform(X_test)
    y_test_s = scaler_y.transform(y_test)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_s).to(device)
    y_train_t = torch.FloatTensor(y_train_s).to(device)
    X_test_t = torch.FloatTensor(X_test_s).to(device)
    y_test_t = torch.FloatTensor(y_test_s).to(device)

    # --- G. INITIALIZE TRAINING ---
    model = DynamicsNet(input_dim=21, output_dim=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- H. TRAINING LOOP ---
    print("💪 Starting Deep Training Loop...")
    losses = []
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 500 == 0:  # Print less frequently
            print(f"   - Epoch {epoch}: Loss {loss.item():.6f}")

    # --- I. EVALUATION ---
    print("🧪 Evaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        y_test_pred_s = model(X_test_t)
        test_loss = criterion(y_test_pred_s, y_test_t)
        print(f"   - Final Test Loss (MSE): {test_loss.item():.6f}")
        
        y_test_pred_s = y_test_pred_s.cpu().numpy()
        y_test_pred = scaler_y.inverse_transform(y_test_pred_s)

    # --- J. SAVE MODEL & SCALERS ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    with open("scaler_X_deep.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_y_deep.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
        
    print(f"💾 Model saved to '{MODEL_SAVE_PATH}'")

    # --- K. PLOT ---
    print("📊 Plotting results...")
    plt.figure(figsize=(10, 5))
    # Joint 1
    plt.subplot(1, 2, 1)
    plt.plot(y_test[:300, 0], label='Real', color='black', alpha=0.6)
    plt.plot(y_test_pred[:300, 0], label='AI', color='red', linestyle='--')
    plt.title("Joint 1 Accel")
    plt.legend()
    
    # Joint 4 (Usually more complex dynamics)
    plt.subplot(1, 2, 2)
    plt.plot(y_test[:300, 3], label='Real', color='black', alpha=0.6)
    plt.plot(y_test_pred[:300, 3], label='AI', color='blue', linestyle='--')
    plt.title("Joint 4 Accel")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
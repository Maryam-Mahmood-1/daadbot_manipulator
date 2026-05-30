"""
data_generation.py
Generates or handles the processing of the dataset S for system identification.
"""
import pandas as pd
import numpy as np
import os

def load_and_split_data(csv_path, val_split=0.5):
    """
    Loads the dataset and ensures train/cal/test splits exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Robust Splitting: Use 'test' if exists, otherwise split 'cal'
    train_df = df[df['split'] == 'train'].copy()
    cal_full = df[df['split'] == 'cal'].copy()

    if cal_full.empty:
        print("Warning: 'cal' split not found. Using random split.")
        train_df = df.sample(frac=0.7)
        cal_full = df.drop(train_df.index)

    # Split calibration into actual calibration and a test set for plotting
    split_idx = int(len(cal_full) * val_split)
    cal_df = cal_full.iloc[:split_idx]
    test_df = cal_full.iloc[split_idx:] 
    
    return train_df, cal_df, test_df

if __name__ == "__main__":
    csv_path = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset.csv"
    train, cal, test = load_and_split_data(csv_path)
    print(f"Data Splits: Train={len(train)}, Cal={len(cal)}, Test={len(test)}")






# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pinocchio as pin
# from tqdm import trange

# # ==============================
# # SETTINGS
# # ==============================
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
# DT = 0.01           # simulation timestep
# N_SAMPLES = 200000    # number of data points

# # ==============================
# # LOAD ROBOT MODEL
# # ==============================
# model = pin.buildModelFromUrdf(URDF_PATH)
# data = model.createData()
# nq = model.nq
# nv = model.nv

# # ==============================
# # DATA GENERATION
# # ==============================
# q_data = []
# dq_data = []
# ddq_data = []
# tau_data = []

# for _ in trange(N_SAMPLES):
#     # Random joint positions and velocities within reasonable limits
#     q = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(nq,))
#     dq = np.random.uniform(low=-1.0, high=1.0, size=(nv,))

#     # Random torques for excitation
#     tau = np.random.uniform(low=-5.0, high=5.0, size=(nv,))

#     # Forward dynamics: compute joint acceleration
#     ddq = pin.aba(model, data, q, dq, tau)

#     # Store
#     q_data.append(q)
#     dq_data.append(dq)
#     ddq_data.append(ddq)
#     tau_data.append(tau)

# # Convert to arrays
# q_data = np.array(q_data)
# dq_data = np.array(dq_data)
# ddq_data = np.array(ddq_data)
# tau_data = np.array(tau_data)

# # Concatenate features: [q, dq, ddq]
# X = np.concatenate([q_data, dq_data, ddq_data], axis=1)
# Y = tau_data

# print("Data shapes:", X.shape, Y.shape)
# # X: [N_SAMPLES, nq+nv+nv], Y: [N_SAMPLES, nv]

# # ==============================
# # DATA-DRIVEN MODEL (PyTorch)
# # ==============================
# class TorqueNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(TorqueNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model_nn = TorqueNet(input_dim=X.shape[1], output_dim=Y.shape[1]).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)

# # Convert data to torch tensors
# X_t = torch.from_numpy(X).float().to(device)
# Y_t = torch.from_numpy(Y).float().to(device)

# # ==============================
# # TRAINING LOOP
# # ==============================
# N_EPOCHS = 50
# BATCH_SIZE = 256

# dataset = torch.utils.data.TensorDataset(X_t, Y_t)
# loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# for epoch in range(N_EPOCHS):
#     running_loss = 0.0
#     for xb, yb in loader:
#         optimizer.zero_grad()
#         y_pred = model_nn(xb)
#         loss = criterion(y_pred, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#     epoch_loss = running_loss / len(dataset)
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

# print("Training complete!")
# # ==============================
# # SAVE DATA AND MODEL
# # ==============================
# np.savez("/home/maryammahmood/xdaadbot_ws/src/some_examples_py/some_examples_py/CRCLF_CRCBF_2_link/Conformal_Pipeline/data.npz",
#          X=X, Y=Y)
# print("Data saved to data.npz")

# # Save the trained PyTorch model
# torch.save(model_nn.state_dict(),
#            "/home/maryammahmood/xdaadbot_ws/src/some_examples_py/some_examples_py/CRCLF_CRCBF_2_link/Conformal_Pipeline/torque_model.pth")
# print("Model saved to torque_model.pth")
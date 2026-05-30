import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
import math
from daadbot_clf_cbf.CRCLF_CRCBF_7_dof.trajectory_generator import TrajectoryGenerator

# --- 1. CONFIGURATIONS ---
URDF_PATH = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "urdf_inverted_torque", "daadbot.urdf")
DT, TOTAL_STEPS = 0.005, 120000 
BATCH_SIZE, LEARNING_RATE = 256, 5e-4
EE_FRAME_NAME = "gear2_claw"  # Ensure this frame has non-zero mass in URDF

class TaskSpaceNet(nn.Module):
    def __init__(self, input_dim=28, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.network(x)

def build_features(q, v, tau):
    return np.hstack([np.sin(q), np.cos(q), v, tau])

# --- 2. PIPELINE ---
def run_pipeline():
    # Setup Robot Model
    full_model = pin.buildModelFromUrdf(URDF_PATH)
    active_joints = [f"joint_{i}" for i in range(1, 8)]
    lock_ids = [full_model.getJointId(n) for n in full_model.names if n != "universe" and n not in active_joints]
    model = pin.buildReducedModel(full_model, lock_ids, pin.neutral(full_model))
    data_pin, ee_id = model.createData(), model.getFrameId(EE_FRAME_NAME)

    # A. Collection (Lipschitz Continuous Persistent Excitation)
    X_raw, Y_raw = [], []
    q, v = pin.neutral(model), np.zeros(model.nv)
    print("Collecting physics-consistent dataset...")
    for i in range(TOTAL_STEPS):
        t = i * DT
        tau = np.zeros(7)
        for j in range(7):
            for s in range(4):
                w = np.random.uniform(0.5, 3.5)
                tau[j] += ((4.0/4)/(w**2)) * np.sin(w*t + np.random.uniform(0, 2*np.pi))
        tau *= 45.0 # High torque to break noise floor

        # PHYSICS FIX: Propagate acceleration to the frame
        ddq = pin.aba(model, data_pin, q, v, tau)
        pin.forwardKinematics(model, data_pin, q, v, ddq) # Must pass ddq!
        pin.updateFramePlacements(model, data_pin)
        
        # Spatial acceleration query
        acc = pin.getFrameAcceleration(model, data_pin, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        X_raw.append(build_features(q, v, tau)); Y_raw.append(acc.linear)
        v = np.clip(v + ddq*DT, -1.5, 1.5); q = pin.integrate(model, q, v*DT)

    X_raw, Y_raw = np.array(X_raw), np.array(Y_raw)
    print(f"Max Ground Truth Accel: {np.max(np.abs(Y_raw)):.4f} m/s²") # Verify scale
    
    X_m, X_s = X_raw.mean(0), X_raw.std(0) + 1e-6
    Y_m, Y_s = Y_raw.mean(0), Y_raw.std(0) + 1e-6

    # B. Training (Double Normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TaskSpaceNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(torch.FloatTensor((X_raw - X_m)/X_s), torch.FloatTensor((Y_raw - Y_m)/Y_s)), batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training dynamics model...")
    for _ in range(100):
        for bx, by in loader:
            optimizer.zero_grad(); nn.HuberLoss()(net(bx.to(device)), by.to(device)).backward(); optimizer.step()

    # C. MODEL-BASED CTC EVALUATION
    traj_gen = TrajectoryGenerator()
    qv, vv = pin.neutral(model), np.zeros(model.nv)
    p_act, p_des, true_acc, pred_acc = [], [], [], []
    
    print("Evaluating with CTC Controller...")
    net.eval()
    for i in range(2500):
        t = i * DT
        pin.forwardKinematics(model, data_pin, qv, vv); pin.updateFramePlacements(model, data_pin)
        xc = data_pin.oMf[ee_id].translation.copy()
        J = pin.computeFrameJacobian(model, data_pin, qv, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        
        xd, vd, ad = traj_gen.get_ref(t, current_actual_pos=xc)
        
        # Model-Based Control Law
        # ad is feedforward; (xd-xc) is feedback
        u_fb = 40.0 * (xd - xc) + 8.0 * (vd - J @ vv)
        u_total = ad + u_fb
        
        # Mapping task-space command to joint torque
        tau = J.T @ (1.8 * u_total) 
        
        ddq = pin.aba(model, data_pin, qv, vv, tau)
        pin.forwardKinematics(model, data_pin, qv, vv, ddq)
        pin.updateFramePlacements(model, data_pin)
        acc_real = pin.getFrameAcceleration(model, data_pin, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        
        feat = (build_features(qv, vv, tau) - X_m) / X_s
        with torch.no_grad():
            out = net(torch.FloatTensor(feat).to(device).unsqueeze(0)).cpu().detach().numpy()[0]
            ddx_pred = (out * Y_s) + Y_m
            
        true_acc.append(acc_real); pred_acc.append(ddx_pred)
        p_act.append(xc); p_des.append(xd)
        vv += ddq*DT; qv = pin.integrate(model, qv, vv*DT)

    # D. VISUALIZATION
    p_act, p_des, true_acc, pred_acc = np.array(p_act), np.array(p_des), np.array(true_acc), np.array(pred_acc)
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(p_des[:,0], p_des[:,1], p_des[:,2], 'r--', label='Target')
    ax1.plot(p_act[:,0], p_act[:,1], p_act[:,2], 'b-', label='CTC Path')
    ax1.set_title("3D Trajectory Tracking"); ax1.legend()

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, i+4)
        ax.plot(true_acc[:, i], 'k', alpha=0.5, label='Physics')
        ax.plot(pred_acc[:, i], 'r--', label='NN Pred')
        ax.set_title(f"{label}-Axis Accel (m/s²)"); ax.legend()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    run_pipeline()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from ament_index_python.packages import get_package_share_directory
# import os
# import math

# # --- 1. CONFIGURATION ---
# URDF_PATH = os.path.join(
#     get_package_share_directory("daadbot_desc"), "urdf", "urdf_inverted_torque", "daadbot.urdf"
# )
# DT = 0.005
# TOTAL_STEPS = 150000 
# BATCH_SIZE = 128
# LEARNING_RATE = 1e-3
# CONFIDENCE_LEVEL = 0.9
# EE_FRAME_NAME = "gear2_claw"

# # --- 2. TRAJECTORY GENERATOR ---
# class TrajectoryGenerator:
#     def __init__(self, approach_duration=5.0):
#         self.center_pos = np.array([0.0, 0.0, 0.72])
#         self.ellipse_a, self.ellipse_b = 0.15, 0.36
#         self.period = 12.0     
#         self.omega = 2 * np.pi / self.period
#         self.approach_duration = approach_duration  
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
#             dx_des = np.array([-self.ellipse_a * self.omega * np.sin(self.omega * t_orbit),
#                                 self.ellipse_b * self.omega * np.cos(self.omega * t_orbit), 0.0])
#             ddx_des = np.array([-self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit),
#                                  -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit), 0.0])
#             return x_des, dx_des, ddx_des

# # --- 3. ROBOT & MODEL SETUP ---
# def setup_robot_model():
#     full_model = pin.buildModelFromUrdf(URDF_PATH)
#     joints_to_keep = [f"joint_{i}" for i in range(1, 8)]
#     lock_ids = [full_model.getJointId(n) for n in full_model.names if n != "universe" and n not in joints_to_keep]
#     return pin.buildReducedModel(full_model, lock_ids, pin.neutral(full_model))

# class DaadbotTaskSpaceNet(nn.Module):
#     def __init__(self, input_dim=21, output_dim=3):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 512), nn.ReLU(),
#             nn.Linear(512, 512), nn.ReLU(),
#             nn.Linear(512, 256), nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )
#     def forward(self, x): return self.network(x)

# # --- 4. MAIN PIPELINE ---
# def run_pipeline():
#     model_pin = setup_robot_model()
#     data_pin = model_pin.createData()
#     ee_id = model_pin.getFrameId(EE_FRAME_NAME)

#     print("Collecting Task-Space dataset...")
#     X_raw, Y_raw = [], []
#     q, v = pin.neutral(model_pin), np.zeros(model_pin.nv)
    
#     for _ in range(TOTAL_STEPS):
#         tau = np.random.uniform(-7.0, 7.0, 7)
#         ddq = pin.aba(model_pin, data_pin, q, v, tau)
#         pin.forwardKinematics(model_pin, data_pin, q, v, ddq)
#         pin.updateFramePlacements(model_pin, data_pin)
        
#         J = pin.getFrameJacobian(model_pin, data_pin, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#         dJdq = pin.getFrameJacobianTimeVariation(model_pin, data_pin, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :] @ v
#         ddx = (J @ ddq) + dJdq 
        
#         X_raw.append(np.hstack([q, v, tau]))
#         Y_raw.append(ddx)
        
#         v += ddq * DT; v *= 0.98; q = pin.integrate(model_pin, q, v * DT)
#         if np.any(np.abs(v) > 25.0): q, v = pin.neutral(model_pin), np.zeros(7)
    
#     X_raw, Y_raw = np.array(X_raw), np.array(Y_raw)
#     indices = np.random.permutation(len(X_raw))
#     train_end, cal_end = int(0.7 * len(X_raw)), int(0.85 * len(X_raw))
#     X_train, Y_train = X_raw[indices[:train_end]], Y_raw[indices[:train_end]]
#     X_cal, Y_cal     = X_raw[indices[train_end:cal_end]], Y_raw[indices[train_end:cal_end]]

#     X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
#     def scale(x): return (x - X_mean) / X_std

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = DaadbotTaskSpaceNet().to(device)
#     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
#     criterion = nn.MSELoss()

#     train_loader = DataLoader(TensorDataset(torch.FloatTensor(scale(X_train)), torch.FloatTensor(Y_train)), batch_size=BATCH_SIZE, shuffle=True)

#     print(f"Training Task-Space Model on {device}...")
#     net.train()
#     for epoch in range(150):
#         for bx, by in train_loader:
#             bx, by = bx.to(device), by.to(device)
#             optimizer.zero_grad(); criterion(net(bx), by).backward(); optimizer.step()
#         if epoch % 30 == 0: print(f"Epoch {epoch} complete.")

#     print("Calculating Cartesian Quantiles (X, Y, Z)...")
#     net.eval()
#     with torch.no_grad():
#         cal_preds = net(torch.FloatTensor(scale(X_cal)).to(device)).cpu().numpy()
#         residuals = np.abs(Y_cal - cal_preds)
#         q_idx = int(np.ceil((len(X_cal) + 1) * CONFIDENCE_LEVEL))
#         cartesian_quantiles = [np.sort(residuals[:, i])[min(q_idx, len(residuals)-1)] for i in range(3)]

#     # --- D. TRAJECTORY VERIFICATION ---
#     print("\n--- Verifying Predicted Cartesian Acceleration ---")
#     traj_gen = TrajectoryGenerator()
#     q_sim, v_sim = pin.neutral(model_pin), np.zeros(model_pin.nv)
    
#     actual_pos, target_pos = [], []
#     pred_ddx_list, true_ddx_list = [], []
    
#     for i in range(int(20.0 / DT)):
#         t = i * DT
#         pin.forwardKinematics(model_pin, data_pin, q_sim, v_sim)
#         pin.updateFramePlacements(model_pin, data_pin)
#         x_curr = data_pin.oMf[ee_id].translation.copy()
#         J = pin.computeFrameJacobian(model_pin, data_pin, q_sim, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#         dx_curr = J @ v_sim
        
#         xd, vd, ad = traj_gen.get_ref(t, x_curr)
#         F_task = 150.0 * (xd - x_curr) + 30.0 * (vd - dx_curr)
#         tau = J.T @ F_task
        
#         ddq_true = pin.aba(model_pin, data_pin, q_sim, v_sim, tau)
#         dJdq = pin.getFrameJacobianTimeVariation(model_pin, data_pin, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :] @ v_sim
#         ddx_true = (J @ ddq_true) + dJdq
        
#         x_nn = scale(np.hstack([q_sim, v_sim, tau]))
#         with torch.no_grad():
#             ddx_pred = net(torch.FloatTensor(x_nn).to(device).unsqueeze(0)).cpu().numpy()[0]
            
#         actual_pos.append(x_curr); target_pos.append(xd)
#         true_ddx_list.append(ddx_true); pred_ddx_list.append(ddx_pred)
        
#         v_sim += ddq_true * DT
#         q_sim = pin.integrate(model_pin, q_sim, v_sim * DT)

#     # --- E. FINAL VISUALIZATION (Fixed Plotting Logic) ---
#     actual_pos = np.array(actual_pos)
#     target_pos = np.array(target_pos)
#     true_ddx_list = np.array(true_ddx_list)
#     pred_ddx_list = np.array(pred_ddx_list)
    
#     fig = plt.figure(figsize=(15, 6))
    
#     # Left Plot: 3D Workspace Path
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.plot(target_pos[:,0], target_pos[:,1], target_pos[:,2], 'r--', label='Target Path')
#     ax1.plot(actual_pos[:,0], actual_pos[:,1], actual_pos[:,2], 'b-', label='Actual Robot Path', alpha=0.7)
#     ax1.set_title("3D Trajectory Tracking")
#     ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
#     ax1.legend()

#     # Right Plot: X-Acceleration Accuracy
#     ax2 = fig.add_subplot(122)
#     time_steps = np.arange(len(true_ddx_list)) * DT
#     ax2.plot(time_steps, true_ddx_list[:, 0], 'k-', alpha=0.5, label='True X-Accel')
#     ax2.plot(time_steps, pred_ddx_list[:, 0], 'r--', label='NN Predicted X-Accel')
    
#     # Confidence Interval Shading
#     ax2.fill_between(time_steps, 
#                      pred_ddx_list[:, 0] - cartesian_quantiles[0], 
#                      pred_ddx_list[:, 0] + cartesian_quantiles[0], 
#                      color='red', alpha=0.2, label=f'{int(CONFIDENCE_LEVEL*100)}% Confidence')
    
#     ax2.set_title("X-Axis Acceleration Prediction Accuracy")
#     ax2.set_xlabel("Time (s)")
#     ax2.set_ylabel("Acceleration (m/s²)")
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     run_pipeline()













# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from ament_index_python.packages import get_package_share_directory
# import os
# import math

# # --- 1. CONFIGURATION ---
# URDF_PATH = os.path.join(
#     get_package_share_directory("daadbot_desc"), "urdf", "urdf_inverted_torque", "daadbot.urdf"
# )
# DT = 0.005
# TOTAL_STEPS = 60000 
# BATCH_SIZE = 128
# LEARNING_RATE = 1e-3
# CONFIDENCE_LEVEL = 0.9
# EE_FRAME_NAME = "gear2_claw"

# # --- 2. TRAJECTORY GENERATOR ---
# class TrajectoryGenerator:
#     def __init__(self, approach_duration=5.0):
#         self.center_pos = np.array([0.0, 0.0, 0.72])
#         self.ellipse_a, self.ellipse_b = 0.15, 0.36
#         self.period = 12.0     
#         self.omega = 2 * np.pi / self.period
#         self.approach_duration = approach_duration  
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
#             dx_des = np.array([-self.ellipse_a * self.omega * np.sin(self.omega * t_orbit),
#                                 self.ellipse_b * self.omega * np.cos(self.omega * t_orbit), 0.0])
#             ddx_des = np.array([-self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit),
#                                  -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit), 0.0])
#             return x_des, dx_des, ddx_des

# # --- 3. ROBOT & MODEL SETUP ---
# def setup_robot_model():
#     full_model = pin.buildModelFromUrdf(URDF_PATH)
#     joints_to_keep = [f"joint_{i}" for i in range(1, 8)]
#     lock_ids = [full_model.getJointId(n) for n in full_model.names if n != "universe" and n not in joints_to_keep]
#     return pin.buildReducedModel(full_model, lock_ids, pin.neutral(full_model))

# class DaadbotDynamicsNet(nn.Module):
#     def __init__(self, input_dim=21, output_dim=7):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 512), nn.ReLU(),
#             nn.Linear(512, 256), nn.ReLU(),
#             nn.Linear(256, 128), nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
#     def forward(self, x): return self.network(x)

# # --- 4. MAIN PIPELINE ---
# def run_pipeline():
#     # A. Setup and Data Collection
#     model_pin = setup_robot_model()
#     data_pin = model_pin.createData()
#     ee_id = model_pin.getFrameId(EE_FRAME_NAME)

#     print("Collecting dataset...")
#     X_raw, Y_raw = [], []
#     q, v = pin.neutral(model_pin), np.zeros(model_pin.nv)
#     for _ in range(TOTAL_STEPS):
#         tau = np.random.uniform(-5.0, 5.0, 7)
#         ddq = pin.aba(model_pin, model_pin.createData(), q, v, tau)
#         X_raw.append(np.hstack([q, v, tau])); Y_raw.append(ddq)
#         v += ddq * DT; v *= 0.98; q = pin.integrate(model_pin, q, v * DT)
#         if np.any(np.abs(v) > 25.0): q, v = pin.neutral(model_pin), np.zeros(7)
    
#     X_raw, Y_raw = np.array(X_raw), np.array(Y_raw)
#     indices = np.random.permutation(len(X_raw))
#     train_end, cal_end = int(0.7 * len(X_raw)), int(0.85 * len(X_raw))
#     X_train, Y_train = X_raw[indices[:train_end]], Y_raw[indices[:train_end]]
#     X_cal, Y_cal     = X_raw[indices[train_end:cal_end]], Y_raw[indices[train_end:cal_end]]

#     # Normalization
#     X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
#     def scale(x): return (x - X_mean) / X_std

#     # B. Training
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = DaadbotDynamicsNet().to(device)
#     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
#     criterion = nn.MSELoss()

#     train_loader = DataLoader(TensorDataset(torch.FloatTensor(scale(X_train)), torch.FloatTensor(Y_train)), batch_size=BATCH_SIZE, shuffle=True)

#     print(f"Training on {device}...")
#     net.train()
#     for epoch in range(100):
#         for bx, by in train_loader:
#             bx, by = bx.to(device), by.to(device)
#             optimizer.zero_grad(); criterion(net(bx), by).backward(); optimizer.step()
#         if epoch % 20 == 0: print(f"Epoch {epoch} complete.")

#     # C. Conformal Quantiles
#     print("Calculating Quantiles...")
#     net.eval()
#     with torch.no_grad():
#         cal_preds = net(torch.FloatTensor(scale(X_cal)).to(device)).cpu().numpy()
#         residuals = np.abs(Y_cal - cal_preds)
#         q_idx = int(np.ceil((len(X_cal) + 1) * CONFIDENCE_LEVEL))
#         joint_quantiles = [np.sort(residuals[:, i])[min(q_idx, len(residuals)-1)] for i in range(7)]

#     # --- D. TRAJECTORY VERIFICATION ---
#     print("\n--- Starting Ellipse Trajectory Verification ---")
#     traj_gen = TrajectoryGenerator()
#     q_sim, v_sim = pin.neutral(model_pin), np.zeros(model_pin.nv)
    
#     actual_pos, target_pos = [], []
#     pred_accel_x, true_accel_x = [], []
    
#     for i in range(int(20.0 / DT)):
#         t = i * DT
#         pin.forwardKinematics(model_pin, data_pin, q_sim, v_sim)
#         pin.updateFramePlacements(model_pin, data_pin)
#         x_curr = data_pin.oMf[ee_id].translation
#         J = pin.computeFrameJacobian(model_pin, data_pin, q_sim, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
#         dx_curr = J @ v_sim
        
#         # PD Control
#         xd, vd, ad = traj_gen.get_ref(t, x_curr)
#         F_task = 150.0 * (xd - x_curr) + 30.0 * (vd - dx_curr)
#         tau = J.T @ F_task
        
#         # Ground Truth Physics
#         ddq_true = pin.aba(model_pin, data_pin, q_sim, v_sim, tau)
#         dJdq = pin.getFrameJacobianTimeVariation(model_pin, data_pin, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :] @ v_sim
#         ddx_true = (J @ ddq_true) + dJdq
        
#         # NN Prediction
#         x_nn = scale(np.hstack([q_sim, v_sim, tau]))
#         with torch.no_grad():
#             ddq_pred = net(torch.FloatTensor(x_nn).to(device).unsqueeze(0)).cpu().numpy()[0]
#             ddx_pred = (J @ ddq_pred) + dJdq
            
#         # Logging
#         actual_pos.append(x_curr.copy()); target_pos.append(xd.copy())
#         true_accel_x.append(ddx_true[0]); pred_accel_x.append(ddx_pred[0])
        
#         # Simulation Step
#         v_sim += ddq_true * DT
#         q_sim = pin.integrate(model_pin, q_sim, v_sim * DT)

#     # --- E. FINAL VISUALIZATION ---
#     actual_pos, target_pos = np.array(actual_pos), np.array(target_pos)
    
#     fig = plt.figure(figsize=(12, 5))
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.plot(target_pos[:,0], target_pos[:,1], target_pos[:,2], 'r--', label='Target')
#     ax1.plot(actual_pos[:,0], actual_pos[:,1], actual_pos[:,2], 'b-', label='Actual')
#     ax1.set_title("3D Trajectory Verification")
#     ax1.legend()

#     ax2 = fig.add_subplot(122)
#     ax2.plot(true_accel_x, 'k-', alpha=0.5, label='True X-Accel')
#     ax2.plot(pred_accel_x, 'r--', label='NN Pred X-Accel')
#     ax2.set_title("Acceleration Prediction (X-Axis)")
#     ax2.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     run_pipeline()
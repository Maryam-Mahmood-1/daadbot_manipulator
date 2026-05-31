import pinocchio as pin
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 1. MODEL DEFINITION (Must match your training exactly) ---
class StructuredDynamicsNet(nn.Module):
    def __init__(self, n_dof=2):
        super().__init__()
        self.n_dof = n_dof
        self.backbone = nn.Sequential(
            nn.Linear(n_dof, 128), nn.Softplus(), 
            nn.Linear(128, 256), nn.Softplus(),
            nn.Linear(256, 128), nn.Softplus()
        )
        self.mass_head = nn.Linear(128, (n_dof * n_dof + n_dof) // 2)
        self.h_head = nn.Sequential(
            nn.Linear(128 + n_dof, 128), nn.Softplus(), 
            nn.Linear(128, n_dof)
        )

    def get_matrices(self, q, dq):
        phi = self.backbone(q)
        l_params = self.mass_head(phi)
        batch_size = q.shape[0]
        L = torch.zeros((batch_size, self.n_dof, self.n_dof), device=q.device)
        L[:, 0, 0] = torch.exp(l_params[:, 0]) + 1e-2 
        L[:, 1, 0] = l_params[:, 1]
        L[:, 1, 1] = torch.exp(l_params[:, 2]) + 1e-2
        M = torch.bmm(L, L.transpose(1, 2))
        H = self.h_head(torch.cat([phi, dq], dim=-1))
        return M, H

# --- 2. COMPONENT DIAGNOSTIC VISUALIZATION ---
def run_component_diagnostic():
    URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
    MODEL_PATH = "2dof_dynamics_model_best.pth" 

    model = StructuredDynamicsNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()

    # Vary q2 from -pi to pi while keeping q1=0 to isolate Joint 2 behavior
    q2_range = np.linspace(-np.pi, np.pi, 200)
    
    # Storage for comparison
    m22_gt, m22_pred = [], []
    h2_gt, h2_pred = [], []

    for q2 in q2_range:
        q = np.array([0.0, q2])
        dq = np.array([0.0, 1.0]) # Add some velocity to see Coriolis/Centripetal effects
        
        # Ground Truth
        M_gt = pin.crba(pin_model, pin_data, q)
        H_gt = pin.nonLinearEffects(pin_model, pin_data, q, dq)
        
        # Neural Prediction
        with torch.no_grad():
            q_t = torch.tensor(q).float().unsqueeze(0)
            dq_t = torch.tensor(dq).float().unsqueeze(0)
            M_p, H_p = model.get_matrices(q_t, dq_t)
            
        m22_gt.append(M_gt[1, 1])
        m22_pred.append(M_p[0, 1, 1].item())
        h2_gt.append(H_gt[1])
        h2_pred.append(H_p[0, 1].item())

    # --- PLOTTING ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    # Mass Matrix M22 Comparison
    ax[0].plot(q2_range, m22_gt, 'k-', label='Pinocchio $M_{22}$')
    ax[0].plot(q2_range, m22_pred, 'r--', label='Learned $\hat{M}_{22}$')
    ax[0].set_title("Joint 2 Inertia Element ($M_{22}$)")
    ax[0].set_ylabel("Inertia ($kg \cdot m^2$)")
    ax[0].legend()

    # Non-linear Vector H2 Comparison
    ax[1].plot(q2_range, h2_gt, 'k-', label='Pinocchio $H_{2}$')
    ax[1].plot(q2_range, h2_pred, 'b--', label='Learned $\hat{H}_{2}$')
    ax[1].set_title("Joint 2 Non-linear effects ($H_{2}$)")
    ax[1].set_ylabel("Torque (Nm)")
    ax[1].set_xlabel("q2 (rad)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_component_diagnostic()





# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import pinocchio as pin
# import matplotlib.pyplot as plt
# import os

# # --- 1. MODEL ARCHITECTURE (Must match training script) ---
# class StructuredDynamicsNet(nn.Module):
#     def __init__(self, n_dof=2):
#         super().__init__()
#         self.n_dof = n_dof
#         self.backbone = nn.Sequential(nn.Linear(n_dof, 128), nn.Softplus(), nn.Linear(128, 128), nn.Softplus())
#         self.mass_head = nn.Linear(128, (n_dof * n_dof + n_dof) // 2)
#         self.coriolis_head = nn.Sequential(nn.Linear(128 + n_dof, 64), nn.Softplus(), nn.Linear(64, n_dof))
#         self.gravity_head = nn.Linear(128, n_dof)

#     def get_matrices(self, q, dq):
#         phi = self.backbone(q)
#         l_params = self.mass_head(phi)
#         L = torch.zeros((q.shape[0], self.n_dof, self.n_dof), device=q.device)
#         L[:, 0, 0], L[:, 1, 0], L[:, 1, 1] = torch.exp(l_params[:, 0]), l_params[:, 1], torch.exp(l_params[:, 2])
#         M = torch.bmm(L, L.transpose(1, 2))
#         return M, self.coriolis_head(torch.cat([phi, dq], dim=-1)), self.gravity_head(phi)

# # --- 2. TRAJECTORY GENERATOR ---
# class TrajectoryGenerator:
#     def __init__(self, approach_duration=2.0):
#         self.center_pos = np.array([0.0, 0.0, 0.0])
#         self.ellipse_a, self.ellipse_b = 1.6, 0.9
#         self.period = 12.0     
#         self.omega = 2 * np.pi / self.period
#         self.approach_duration = approach_duration
#         self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])
#         self.orbit_start_vel = np.array([0.0, self.ellipse_b * self.omega, 0.0])
#         self.orbit_start_acc = np.array([-self.ellipse_a * (self.omega**2), 0.0, 0.0])

#     def get_ref(self, t, current_actual_pos=None):
#         if t < self.approach_duration:
#             # For verification, we assume a simple linear ramp if not using splines
#             return self.orbit_start_pos, self.orbit_start_vel, self.orbit_start_acc
#         else:
#             t_orbit = t - self.approach_duration
#             x_des = self.center_pos + np.array([self.ellipse_a * np.cos(self.omega * t_orbit), self.ellipse_b * np.sin(self.omega * t_orbit), 0.0])
#             dx_des = np.array([-self.ellipse_a * self.omega * np.sin(self.omega * t_orbit), self.ellipse_b * self.omega * np.cos(self.omega * t_orbit), 0.0])
#             ddx_des = np.array([-self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit), -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit), 0.0])
#             return x_des, dx_des, ddx_des

# # --- 3. VERIFICATION LOGIC ---
# MODEL_PATH = "2dof_dynamics_model.pth"
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"

# # Load Learned Model
# model = StructuredDynamicsNet()
# model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
# model.eval()

# # Setup Pinocchio
# pin_model = pin.buildModelFromUrdf(URDF_PATH)
# pin_data = pin_model.createData()
# ee_id = pin_model.getFrameId("endEffector")

# # Load Ground Truth Data (from your CSV)
# df = pd.read_csv("2dof_trajectory_dataset.csv")
# # Filter a segment where the robot was performing the ellipse or similar task
# eval_data = df.iloc[1000:1300] 

# gt_ddx, pred_ddx = [], []

# for _, row in eval_data.iterrows():
#     q = np.array([row['q1'], row['q2']])
#     dq = np.array([row['dq1'], row['dq2']])
#     tau = np.array([row['tau1'], row['tau2']])
    
#     # Predict with Learned Model
#     q_t = torch.from_numpy(q).float().unsqueeze(0)
#     dq_t = torch.from_numpy(dq).float().unsqueeze(0)
#     with torch.no_grad():
#         M_hat, C_hat, G_hat = model.get_matrices(q_t, dq_t)
    
#     M_inv = np.linalg.inv(M_hat.squeeze().numpy())
#     nle = C_hat.squeeze().numpy() + G_hat.squeeze().numpy()
    
#     # Kinematics via Pinocchio
#     pin.forwardKinematics(pin_model, pin_data, q, dq)
#     pin.updateFramePlacements(pin_model, pin_data)
#     J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_id, pin.ReferenceFrame.WORLD)[:2, :]
#     dJ = pin.getFrameJacobianTimeVariation(pin_model, pin_data, ee_id, pin.ReferenceFrame.WORLD)[:2, :]
    
#     # Task Acceleration Prediction: ddx = J * M_inv * (tau - nle) + dJ * dq
#     ddx_p = J @ M_inv @ (tau - nle) + dJ @ dq
    
#     gt_ddx.append([row['ddx'], row['ddy']])
#     pred_ddx.append(ddx_p)

# # Plotting
# gt_ddx, pred_ddx = np.array(gt_ddx), np.array(pred_ddx)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(gt_ddx[:, 0], 'k--', label="GT Acceleration X")
# plt.plot(pred_ddx[:, 0], 'r', label="Predicted Acceleration X")
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(gt_ddx[:, 1], 'k--', label="GT Acceleration Y")
# plt.plot(pred_ddx[:, 1], 'g', label="Predicted Acceleration Y")
# plt.legend()
# plt.suptitle("Model Verification on Elliptical Path Dynamics")
# plt.show()



# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import pinocchio as pin
# import matplotlib.pyplot as plt
# import os

# # --- 1. Model Architecture (Must match training) ---
# class StructuredDynamicsNet(nn.Module):
#     def __init__(self, n_dof=2):
#         super().__init__()
#         self.n_dof = n_dof
#         self.backbone = nn.Sequential(nn.Linear(n_dof, 128), nn.Softplus(), nn.Linear(128, 128), nn.Softplus())
#         self.mass_head = nn.Linear(128, (n_dof * n_dof + n_dof) // 2)
#         self.coriolis_head = nn.Sequential(nn.Linear(128 + n_dof, 64), nn.Softplus(), nn.Linear(64, n_dof))
#         self.gravity_head = nn.Linear(128, n_dof)

#     def get_matrices(self, q, dq):
#         phi = self.backbone(q)
#         l_params = self.mass_head(phi)
#         L = torch.zeros((q.shape[0], self.n_dof, self.n_dof), device=q.device)
#         L[:, 0, 0], L[:, 1, 0], L[:, 1, 1] = torch.exp(l_params[:, 0]), l_params[:, 1], torch.exp(l_params[:, 2])
#         M = torch.bmm(L, L.transpose(1, 2))
#         return M, self.coriolis_head(torch.cat([phi, dq], dim=-1)), self.gravity_head(phi)

# # --- 2. Setup and Loading ---
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
# MODEL_PATH = "2dof_dynamics_model.pth"

# device = torch.device("cpu")
# model = StructuredDynamicsNet()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# pin_model = pin.buildModelFromUrdf(URDF_PATH)
# pin_data = pin_model.createData()

# # --- 3. Create Joint Space Meshgrid for Heatmap ---
# # Define the range for q1 and q2 (e.g., -pi to pi)
# res = 50  # Resolution of the heatmap
# q1_range = np.linspace(-np.pi, np.pi, res)
# q2_range = np.linspace(-np.pi, np.pi, res)
# Q1, Q2 = np.meshgrid(q1_range, q2_range) #

# # Initialize error matrices
# m_error_map = np.zeros((res, res))
# g_error_map = np.zeros((res, res))

# print("Generating heatmaps... this may take a moment.")
# for i in range(res):
#     for j in range(res):
#         q = np.array([Q1[i, j], Q2[i, j]])
#         dq = np.zeros(2) # Evaluating static properties for heatmap
        
#         # Ground Truth from Pinocchio
#         M_pin = pin.crba(pin_model, pin_data, q)
#         G_pin = pin.computeGeneralizedGravity(pin_model, pin_data, q)
        
#         # NN Prediction
#         q_tensor = torch.tensor([q], dtype=torch.float32)
#         dq_tensor = torch.tensor([dq], dtype=torch.float32)
#         with torch.no_grad():
#             M_hat, _, G_hat = model.get_matrices(q_tensor, dq_tensor)
        
#         # Absolute Errors
#         m_error_map[i, j] = np.linalg.norm(M_pin - M_hat[0].numpy())
#         g_error_map[i, j] = np.linalg.norm(G_pin - G_hat[0].numpy())

# # --- 4. Plot Heatmaps ---
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# # Mass Matrix Error Heatmap
# im1 = ax[0].imshow(m_error_map, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='viridis')
# ax[0].set_title("Mass Matrix Absolute Error $||M - \hat{M}||$")
# ax[0].set_xlabel("$q_1$ (rad)")
# ax[0].set_ylabel("$q_2$ (rad)")
# fig.colorbar(im1, ax=ax[0], label="Error (kg·m²)")

# # Gravity Vector Error Heatmap
# im2 = ax[1].imshow(g_error_map, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='magma')
# ax[1].set_title("Gravity Vector Absolute Error $||G - \hat{G}||$")
# ax[1].set_xlabel("$q_1$ (rad)")
# ax[1].set_ylabel("$q_2$ (rad)")
# fig.colorbar(im2, ax=ax[1], label="Error (N·m)")

# plt.tight_layout()
# plt.show()
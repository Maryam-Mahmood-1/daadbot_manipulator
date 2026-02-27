import numpy as np
import torch
import pinocchio as pin
import cvxpy as cp
import matplotlib.pyplot as plt

# ==============================
# SETTINGS
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
MODEL_PATH = "/home/maryammahmood/xdaadbot_ws/src/some_examples_py/some_examples_py/CRCLF_CRCBF_2_link/Conformal_Pipeline/torque_model.pth"
DATA_PATH = "/home/maryammahmood/xdaadbot_ws/src/some_examples_py/some_examples_py/CRCLF_CRCBF_2_link/Conformal_Pipeline/data.npz"

# CR-CLF gains
Kp = np.diag([10.0, 10.0])
Kv = np.diag([5.0, 5.0])
c3 = 0.5
dt = 0.001

# ==============================
# LOAD DATA
# ==============================
data_np = np.load(DATA_PATH)
X = data_np['X']
Y = data_np['Y']

# ==============================
# LOAD TORQUE MODEL
# ==============================
import torch.nn as nn

class TorqueNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TorqueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model_nn = TorqueNet(input_dim=X.shape[1], output_dim=Y.shape[1]).to(DEVICE)
model_nn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_nn.eval()

# ==============================
# COMPUTE RESIDUAL QUANTILE
# ==============================
with torch.no_grad():
    X_t = torch.from_numpy(X).float().to(DEVICE)
    Y_pred = model_nn(X_t).cpu().numpy()

residuals = Y - Y_pred
residual_norms = np.linalg.norm(residuals, axis=1)
delta = 0.1  # 90% conformal prediction
q_1_delta = np.quantile(residual_norms, 1 - delta)
print("Conformal quantile q1-delta:", q_1_delta)

# ==============================
# PINOCCHIO ROBOT MODEL
# ==============================
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
ee_frame = model.njoints - 1  # end-effector frame

# ==============================
# DEFINE TRAJECTORY
# ==============================
T_total = 2.0
steps = int(T_total / dt)
t = np.linspace(0, T_total, steps)

q_des_traj = np.vstack([0.5*np.sin(2*np.pi*t/T_total),
                        0.5*np.cos(2*np.pi*t/T_total)]).T
dq_des_traj = np.gradient(q_des_traj, dt, axis=0)
ddq_des_traj = np.gradient(dq_des_traj, dt, axis=0)

# Convert to task-space
x_des_traj, dx_des_traj, ddx_des_traj = [], [], []
for i in range(steps):
    q_des = q_des_traj[i]
    dq_des = dq_des_traj[i]
    ddq_des = ddq_des_traj[i]
    
    pin.forwardKinematics(model, data, q_des, dq_des)
    x = data.oMi[ee_frame].translation[:2]   # EE pos
    J = pin.computeJointJacobian(model, data, q_des, ee_frame)[:2,:2]
    dx = J @ dq_des
    ddx = J @ ddq_des  # ignore Jdot*dq for simplicity
    
    x_des_traj.append(x)
    dx_des_traj.append(dx)
    ddx_des_traj.append(ddx)

x_des_traj = np.array(x_des_traj)
dx_des_traj = np.array(dx_des_traj)
ddx_des_traj = np.array(ddx_des_traj)

# ==============================
# INITIALIZE SIMULATION
# ==============================
q = np.zeros(2)
dq = np.zeros(2)
Q_sim, Tau_sim, error_sim = [], [], []

# ==============================
# SIMULATION LOOP
# ==============================
for i in range(steps):
    # Forward kinematics
    pin.forwardKinematics(model, data, q, dq)
    x_cur = data.oMi[ee_frame].translation[:2]
    J = pin.computeJointJacobian(model, data, q, ee_frame)[:2,:2]
    dx_cur = J @ dq
    
    # Task-space error
    e = x_cur - x_des_traj[i]
    de = dx_cur - dx_des_traj[i]
    
    # Desired task-space acceleration
    ddx_cmd = ddx_des_traj[i] - Kp @ e - Kv @ de
    
    # Learned torque prediction
    input_tensor = torch.from_numpy(np.hstack([q, dq, ddx_cmd])).float().unsqueeze(0).to(DEVICE)
    tau_hat = model_nn(input_tensor).detach().cpu().numpy().flatten()
    
    # CR-CLF QP
    mu = cp.Variable(2)
    eta = np.hstack([e, de])
    V = eta.T @ eta
    dVdeta = 2 * eta
    LfV = dVdeta[:2] @ de
    LgV = dVdeta[2:] @ J
    constraints = [LfV + LgV @ mu + c3*V + np.linalg.norm(dVdeta[2:] @ J)*q_1_delta <= 0]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(mu)), constraints)
    prob.solve(solver=cp.OSQP)
    
    tau = tau_hat + mu.value
    Tau_sim.append(tau)
    
    # Forward integration (simplified)
    ddq = ddx_cmd  # approximate mapping
    dq += ddq * dt
    q += dq * dt
    Q_sim.append(q.copy())
    error_sim.append(np.linalg.norm(e))

# ==============================
# PLOTS
# ==============================
Q_sim = np.array(Q_sim)
Tau_sim = np.array(Tau_sim)
error_sim = np.array(error_sim)

plt.figure(figsize=(10,4))
plt.plot(t, Q_sim[:,0], label='q1')
plt.plot(t, q_des_traj[:,0], '--', label='q1_des')
plt.plot(t, Q_sim[:,1], label='q2')
plt.plot(t, q_des_traj[:,1], '--', label='q2_des')
plt.title("Joint Positions")
plt.xlabel("Time [s]")
plt.ylabel("q [rad]")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, Tau_sim[:,0], label='Tau1')
plt.plot(t, Tau_sim[:,1], label='Tau2')
plt.title("Torque Commands")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, error_sim)
plt.title("Task-Space Tracking Error Norm")
plt.xlabel("Time [s]")
plt.ylabel("||e|| [m]")
plt.grid(True)
plt.show()
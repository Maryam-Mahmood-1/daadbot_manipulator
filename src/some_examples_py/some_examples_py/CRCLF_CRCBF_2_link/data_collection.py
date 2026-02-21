import pinocchio as pin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Configuration ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
EE_FRAME_NAME = "endEffector" 
NUM_TRAJECTORIES = 300
TRAJ_DURATION = 10.0  # 10 seconds per snippet
HZ = 100
DT = 1.0 / HZ

# --- 2. Load Model ---
if not os.path.exists(URDF_PATH):
    print(f"Error: URDF not found at {URDF_PATH}. Please ensure xacro is converted.")
    exit()

model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
nq, nv = model.nq, model.nv
ee_frame_id = model.getFrameId(EE_FRAME_NAME)

# --- 3. Persistent Excitation Trajectory Generator ---
def generate_sos_params(num_joints):
    # Prime-ish frequencies to avoid periodic overlap
    freqs = np.random.uniform(0.5, 5.0, (num_joints, 4))
    phases = np.random.uniform(0, 2*np.pi, (num_joints, 4))
    # Base amplitudes
    amps = np.random.uniform(0.1, 0.4, (num_joints, 4))
    return freqs, phases, amps

def collect_data():
    all_rows = []
    print(f"Starting collection of {NUM_TRAJECTORIES} trajectories...")

    for traj_id in range(NUM_TRAJECTORIES):
        freqs, phases, amps = generate_sos_params(nq)
        
        for step in range(int(TRAJ_DURATION * HZ)):
            t = step * DT
            q, dq, ddq = np.zeros(nq), np.zeros(nq), np.zeros(nq)
            
            for j in range(nq):
                for f_idx in range(4):
                    w = freqs[j, f_idx]
                    p = phases[j, f_idx]
                    # Lipschitz Scaling: Amplitude / w ensures bounded velocity
                    a = amps[j, f_idx] / w 
                    
                    q[j]   += a * np.sin(w * t + p)
                    dq[j]  += a * w * np.cos(w * t + p)
                    ddq[j] -= a * (w**2) * np.sin(w * t + p)
            
            # Dynamics (RNEA)
            tau = pin.rnea(model, data, q, dq, ddq)
            
            # Task Space Kinematics
            pin.forwardKinematics(model, data, q, dq, ddq)
            pin.updateFramePlacements(model, data)
            pos = data.oMf[ee_frame_id].translation
            acc_vec = pin.getFrameAcceleration(model, data, ee_frame_id, pin.ReferenceFrame.WORLD)
            
            all_rows.append(np.concatenate([
                [traj_id], q, dq, ddq, tau, pos, acc_vec.linear
            ]))
            
    return all_rows

# --- 4. Execution and Verification ---
raw_data = collect_data()
cols = ['traj_id', 'q1', 'q2', 'dq1', 'dq2', 'ddq1', 'ddq2', 
        'tau1', 'tau2', 'x', 'y', 'z', 'ddx', 'ddy', 'ddz']
df = pd.DataFrame(raw_data, columns=cols)

# Verification 1: Lipschitz Continuity Check
max_vel = df[['dq1', 'dq2']].abs().max().max()
print(f"--- Verification ---")
print(f"Max Joint Velocity: {max_vel:.2f} rad/s (Lipschitz Check)")

# Verification 2: Acceleration Richness
std_acc = df[['ddq1', 'ddq2']].std().mean()
print(f"Acceleration Excitation (Std Dev): {std_acc:.2f}")
if std_acc < 0.5:
    print("Warning: Low acceleration richness. Consider increasing amplitudes.")

# Verification 3: Histogram Distribution Plot
def plot_verifications(dataframe):
    plt.figure(figsize=(12, 5))
    
    # Histogram of Joint Accelerations
    plt.subplot(1, 2, 1)
    plt.hist(dataframe['ddq1'], bins=50, alpha=0.5, label='Joint 1')
    plt.hist(dataframe['ddq2'], bins=50, alpha=0.5, label='Joint 2')
    plt.title("Acceleration Distribution")
    plt.xlabel("rad/s²")
    plt.legend()
    
    # Sample Trajectory Plot (First 5 seconds)
    plt.subplot(1, 2, 2)
    sample_traj = dataframe[dataframe['traj_id'] == 0]
    plt.plot(sample_traj['q1'], label='q1')
    plt.plot(sample_traj['q2'], label='q2')
    plt.title("Sample Trajectory Snippet (5s)")
    plt.xlabel("Samples")
    plt.ylabel("Radians")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_verifications(df)

# Save
df.to_csv("2dof_trajectory_dataset.csv", index=False)
print("Dataset saved to 2dof_trajectory_dataset.csv")
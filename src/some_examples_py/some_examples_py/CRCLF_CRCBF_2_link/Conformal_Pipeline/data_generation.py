"""
data_generation.py
Generates a Persistently Exciting (PE) dataset for System Identification.
Uses a Sum-of-Sines (SoS) trajectory and Pinocchio for ground truth physics.
"""
import numpy as np
import pandas as pd
import pinocchio as pin
import os
from tqdm import trange

# ==============================
# SETTINGS
# ==============================
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
CSV_OUT_PATH = "/home/maryammahmood/xdaadbot_ws/2link_pe_dataset2.csv"

# Simulation parameters
DT = 0.01             # Timestep
TRAJ_DURATION = 7.5   # Length of each trajectory in seconds
NUM_TRAJECTORIES = 900 # Number of independent trajectories to simulate

# Train/Cal/Test Split Ratios
TRAIN_RATIO = 0.6
CAL_RATIO = 0.2
# Test ratio is the remaining 0.2

# ==============================
# SUM OF SINES TRAJECTORY GENERATOR
# ==============================
class SoSTrajectory:
    def __init__(self, num_joints, base_freq=0.2, num_harmonics=5):
        self.num_joints = num_joints
        self.base_freq = base_freq
        self.num_harmonics = num_harmonics
        
        # Randomize amplitudes and phases for each joint and harmonic
        self.a = np.random.uniform(-1.0, 1.0, size=(num_joints, num_harmonics))
        self.b = np.random.uniform(-1.0, 1.0, size=(num_joints, num_harmonics))

    def get_state(self, t):
        q = np.zeros(self.num_joints)
        dq = np.zeros(self.num_joints)
        ddq = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            for k in range(1, self.num_harmonics + 1):
                omega = 2 * np.pi * self.base_freq * k
                
                # Position
                q[i] += self.a[i, k-1] * np.sin(omega * t) + self.b[i, k-1] * np.cos(omega * t)
                # Velocity
                dq[i] += omega * (self.a[i, k-1] * np.cos(omega * t) - self.b[i, k-1] * np.sin(omega * t))
                # Acceleration
                ddq[i] += -(omega**2) * (self.a[i, k-1] * np.sin(omega * t) + self.b[i, k-1] * np.cos(omega * t))
                
        return q, dq, ddq

# ==============================
# DATA GENERATION LOOP
# ==============================
def generate_dataset():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    nq = model.nq
    nv = model.nv

    all_data = []
    time_steps = int(TRAJ_DURATION / DT)

    print(f"Generating {NUM_TRAJECTORIES} trajectories...")
    
    for traj_idx in trange(NUM_TRAJECTORIES):
        # Create a unique Sum-of-Sines generator for this trajectory
        sos_gen = SoSTrajectory(num_joints=nv)
        
        # Determine split (Train, Cal, Test)
        rand_val = np.random.rand()
        if rand_val < TRAIN_RATIO:
            split_label = 'train'
        elif rand_val < (TRAIN_RATIO + CAL_RATIO):
            split_label = 'cal'
        else:
            split_label = 'test'

        for step in range(time_steps):
            t = step * DT
            
            # 1. Get smooth kinematic targets from the SoS generator
            q, dq, ddq = sos_gen.get_state(t)
            
            # Wrap q to [-pi, pi] to prevent continuous rotation math explosions
            q = np.arctan2(np.sin(q), np.cos(q))
            
            # 2. Calculate true required torque using Recursive Newton-Euler (RNEA)
            tau = pin.rnea(model, data, q, dq, ddq)
            
            # 3. Store row
            row = {
                't': t,
                'traj_id': traj_idx,
                'split': split_label,
                'q0': q[0], 'q1': q[1],
                'dq0': dq[0], 'dq1': dq[1],
                'tau0': tau[0], 'tau1': tau[1],
                'target_ddq0': ddq[0], 'target_ddq1': ddq[1]
            }
            all_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    df.to_csv(CSV_OUT_PATH, index=False)
    print(f"\nDataset saved to {CSV_OUT_PATH}")
    
    # Print summary statistics
    train_count = len(df[df['split'] == 'train'])
    cal_count = len(df[df['split'] == 'cal'])
    test_count = len(df[df['split'] == 'test'])
    print(f"Total Frames: {len(df)}")
    print(f"Splits: Train={train_count}, Cal={cal_count}, Test={test_count}")

if __name__ == "__main__":
    generate_dataset()
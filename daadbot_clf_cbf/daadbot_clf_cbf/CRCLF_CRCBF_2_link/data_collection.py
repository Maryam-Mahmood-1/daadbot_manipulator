import pinocchio as pin
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
DATA_DIR = "robot_data" # Folder to store splits
NUM_TRAJECTORIES = 10000
TRAJ_DURATION = 5.0  

def collect_robot_data():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF not found at {URDF_PATH}")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    nq, ee_id = model.nq, model.getFrameId("endEffector")
    all_rows = []

    print(f"Collecting {NUM_TRAJECTORIES} trajectories...")
    for traj_id in range(NUM_TRAJECTORIES):
        freqs = np.random.uniform(0.5, 5.0, (nq, 3))
        phases = np.random.uniform(0, 2*np.pi, (nq, 3))
        amps = np.random.uniform(0.1, 0.5, (nq, 3))
        
        for t in np.arange(0, TRAJ_DURATION, 0.01):
            q, dq, ddq = np.zeros(nq), np.zeros(nq), np.zeros(nq)
            for j in range(nq):
                for f_idx in range(3):
                    w, p, a = freqs[j, f_idx], phases[j, f_idx], amps[j, f_idx]
                    q[j]   += a * np.sin(w * t + p)
                    dq[j]  += a * w * np.cos(w * t + p)
                    ddq[j] -= a * (w**2) * np.sin(w * t + p)
            
            tau = pin.rnea(model, data, q, dq, ddq)
            pin.forwardKinematics(model, data, q, dq, ddq)
            pin.updateFramePlacements(model, data)
            pin.computeJointJacobiansTimeVariation(model, data, q, dq)
            
            pos = data.oMf[ee_id].translation
            acc = pin.getFrameClassicalAcceleration(model, data, ee_id, pin.ReferenceFrame.WORLD).linear
            all_rows.append(np.concatenate([[traj_id], q, dq, ddq, tau, pos, acc]))
            
    df = pd.DataFrame(all_rows, columns=['traj_id', 'q1', 'q2', 'dq1', 'dq2', 'ddq1', 'ddq2', 
                                         'tau1', 'tau2', 'x', 'y', 'z', 'ddx', 'ddy', 'ddz'])
    
    # --- STRATEGIC SPLITTING ---
    traj_ids = df['traj_id'].unique()
    train_ids, test_ids = train_test_split(traj_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

    df[df['traj_id'].isin(train_ids)].to_csv(f"{DATA_DIR}/train_data.csv", index=False)
    df[df['traj_id'].isin(val_ids)].to_csv(f"{DATA_DIR}/val_data.csv", index=False)
    df[df['traj_id'].isin(test_ids)].to_csv(f"{DATA_DIR}/test_data.csv", index=False)
    
    print(f"Datasets saved in {DATA_DIR}/ (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)} trajectories)")

if __name__ == "__main__":
    collect_robot_data()




# import pinocchio as pin
# import numpy as np
# import pandas as pd
# import os

# # --- CONFIGURATION ---
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"
# CSV_FILE = "2dof_trajectory_dataset.csv"
# NUM_TRAJECTORIES = 5000
# TRAJ_DURATION = 10.0  # seconds

# def collect_robot_data():
#     if not os.path.exists(URDF_PATH):
#         raise FileNotFoundError(f"URDF not found at {URDF_PATH}")

#     model = pin.buildModelFromUrdf(URDF_PATH)
#     data = model.createData()
#     nq, ee_id = model.nq, model.getFrameId("endEffector")
#     all_rows = []

#     print(f"Collecting {NUM_TRAJECTORIES} trajectories for rich excitation...")
#     for traj_id in range(NUM_TRAJECTORIES):
#         # High-frequency frequencies to capture Coriolis effects
#         freqs = np.random.uniform(0.5, 6.0, (nq, 4))
#         phases = np.random.uniform(0, 2*np.pi, (nq, 4))
#         amps = np.random.uniform(0.1, 0.4, (nq, 4))
        
#         for t in np.arange(0, TRAJ_DURATION, 0.01):
#             q, dq, ddq = np.zeros(nq), np.zeros(nq), np.zeros(nq)
#             for j in range(nq):
#                 for f_idx in range(4):
#                     w, p, a = freqs[j, f_idx], phases[j, f_idx], amps[j, f_idx] / freqs[j, f_idx]
#                     q[j]   += a * np.sin(w * t + p)
#                     dq[j]  += a * w * np.cos(w * t + p)
#                     ddq[j] -= a * (w**2) * np.sin(w * t + p)
            
#             tau = pin.rnea(model, data, q, dq, ddq)
#             pin.forwardKinematics(model, data, q, dq, ddq)
#             pin.updateFramePlacements(model, data)
#             pos = data.oMf[ee_id].translation
#             acc = pin.getFrameAcceleration(model, data, ee_id, pin.ReferenceFrame.WORLD).linear
#             all_rows.append(np.concatenate([[traj_id], q, dq, ddq, tau, pos, acc]))
            
#     df = pd.DataFrame(all_rows, columns=['traj_id', 'q1', 'q2', 'dq1', 'dq2', 'ddq1', 'ddq2', 
#                                          'tau1', 'tau2', 'x', 'y', 'z', 'ddx', 'ddy', 'ddz'])
#     df.to_csv(CSV_FILE, index=False)
#     print(f"Dataset saved to {CSV_FILE}")

# if __name__ == "__main__":
#     collect_robot_data()
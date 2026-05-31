import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from tqdm import tqdm  # pip install tqdm

# --- 1. USER CONFIGURATION ---
# Update these paths to match your system
URDF_CLEAN = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy_.urdf"

# Robot Definitions
ALL_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
TAU_LIMITS = np.array([10.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0]) 

# Calibration Settings
DT = 0.01           # 100Hz Sampling
T_HORIZON = 10.0     # Duration per trajectory
NUM_TRAJS = 150      # Total valid trajectories to collect
CONFIDENCE = 0.9   # 90% Confidence (delta = 0.1)

# --- 2. DYNAMICS WRAPPER ---
class ModelWrapper:
    """ Helper to load Pinocchio models and compute dynamics """
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # Map joint names to IDs
        self.joint_ids = []
        for name in ALL_JOINTS:
            if self.model.existJointName(name):
                self.joint_ids.append(self.model.getJointId(name))
            else:
                print(f"Warning: Joint {name} not found in {urdf_path}")
        
        self.nv = self.model.nv # Should be 13 in your case

    def compute_acceleration(self, q_full, dq_full, tau_full):
        """ Forward Dynamics: ddq = M^-1 (tau - nle) """
        return pin.aba(self.model, self.data, q_full, dq_full, tau_full)

    def compute_inverse_dynamics(self, q_full, dq_full, ddq_full):
        """ Inverse Dynamics: tau = M * ddq + nle """
        # Ensure all inputs are full size (13)
        return pin.rnea(self.model, self.data, q_full, dq_full, ddq_full)

# --- 3. TRAJECTORY GENERATOR ---
class RandomTrajectoryGenerator:
    """ Generates smooth random Lissajous trajectories in Joint Space (7 DOF) """
    def __init__(self, num_joints=7):
        self.n = num_joints

    def generate_new_trajectory(self):
        # q_i(t) = A * sin(omega * t + phi) + Offset
        # Amplitudes reduced to prevent instant torque saturation
        self.Amps = np.random.uniform(0.05, 0.4, self.n)   # Rad
        self.Omegas = np.random.uniform(0.5, 1.5, self.n)  # Rad/s
        self.Phases = np.random.uniform(0, 2*np.pi, self.n)
        self.Offsets = np.random.uniform(-0.2, 0.2, self.n)

    def get_state(self, t):
        # Position
        q = self.Offsets + self.Amps * np.sin(self.Omegas * t + self.Phases)
        # Velocity
        dq = self.Amps * self.Omegas * np.cos(self.Omegas * t + self.Phases)
        # Acceleration
        ddq = -self.Amps * (self.Omegas**2) * np.sin(self.Omegas * t + self.Phases)
        return q, dq, ddq

# --- 4. MAIN CALIBRATION LOOP ---
def main():
    print(f"--- STARTING CONFORMAL CALIBRATION ---")
    print(f"Clean (Real) Model: {URDF_CLEAN}")
    print(f"Noisy (Ctrl) Model: {URDF_NOISY}")
    print(f"Target: {NUM_TRAJS} Valid Trajectories | Confidence: {CONFIDENCE*100}%")

    # A. Load Models
    model_real = ModelWrapper(URDF_CLEAN) # Represents Physics (13 DOF)
    model_pred = ModelWrapper(URDF_NOISY) # Represents Controller internal model (13 DOF)

    traj_gen = RandomTrajectoryGenerator(num_joints=7)
    
    # Storage for Non-Conformity Scores
    all_scores = []
    
    valid_count = 0
    pbar = tqdm(total=NUM_TRAJS, desc="Collecting Data")
    
    while valid_count < NUM_TRAJS:
        # 1. Generate Candidate Trajectory
        traj_gen.generate_new_trajectory()
        
        # Temp buffers
        traj_scores = []
        is_feasible = True
        
        # Reset State
        t = 0.0
        
        # Prepare Full State Vectors (Size 13)
        q_full = pin.neutral(model_real.model)
        dq_full = np.zeros(model_real.nv)
        ddq_full = np.zeros(model_real.nv) # <--- NEW: Full Acceleration Vector
        
        steps = int(T_HORIZON / DT)
        
        for _ in range(steps):
            # 2. Sample 7-DOF Trajectory
            q_des, dq_des, ddq_des = traj_gen.get_state(t)
            
            # Update Full State Vectors
            # Map the 7 controlled joints into the 13-DOF vectors
            for k, jid in enumerate(model_real.joint_ids):
                idx_q = model_real.model.joints[jid].idx_q
                idx_v = model_real.model.joints[jid].idx_v
                
                q_full[idx_q] = q_des[k]
                dq_full[idx_v] = dq_des[k]
                ddq_full[idx_v] = ddq_des[k] # <--- FIX: Map acceleration too

            # 3. Compute Required Torque (Inverse Dynamics via NOISY Model)
            # We pass the full (13-DOF) acceleration vector now
            tau_cmd_full = model_pred.compute_inverse_dynamics(q_full, dq_full, ddq_full)
            
            # 4. CHECK TORQUE LIMITS (Rejection Sampling)
            # Extract 7-DOF torques for checking limits
            tau_7dof = np.zeros(7)
            for k, jid in enumerate(model_real.joint_ids):
                idx_v = model_real.model.joints[jid].idx_v
                tau_7dof[k] = tau_cmd_full[idx_v]
            
            # If violates limit, discard entire trajectory immediately
            if np.any(np.abs(tau_7dof) > TAU_LIMITS):
                is_feasible = False
                break 
            
            # 5. Forward Dynamics (via REAL Model)
            # "Physics says this actually happens"
            ddq_real = model_real.compute_acceleration(q_full, dq_full, tau_cmd_full)
            
            # 6. Predict Acceleration (via NOISY Model)
            # "Controller thought this would happen"
            ddq_pred = model_pred.compute_acceleration(q_full, dq_full, tau_cmd_full)
            
            # 7. Compute Error (Score)
            # We only care about the error on the 7 active joints
            acc_error = np.zeros(7)
            for k, jid in enumerate(model_real.joint_ids):
                idx_v = model_real.model.joints[jid].idx_v
                acc_error[k] = ddq_real[idx_v] - ddq_pred[idx_v]
            
            # Score = Norm of acceleration error vector
            traj_scores.append(np.linalg.norm(acc_error))
            t += DT

        # 8. Store Valid Data
        if is_feasible:
            all_scores.extend(traj_scores)
            valid_count += 1
            pbar.update(1)
        else:
            # If rejecting too much, reduce amplitude here
            # traj_gen.Amps *= 0.9 
            pass
            
    pbar.close()

    # --- 5. COMPUTE QUANTILE ---
    if not all_scores:
        print("No valid trajectories found! Check torque limits.")
        return

    all_scores = np.array(all_scores)
    n_samples = len(all_scores)
    
    # Sort
    sorted_scores = np.sort(all_scores)
    
    # Calculate Quantile Index (Paper Eq. 3)
    # ceil( (1-delta) * (N+1) )
    q_index = int(np.ceil(CONFIDENCE * (n_samples + 1))) - 1
    q_index = min(q_index, n_samples - 1) # Safety clamp
    
    quantile_val = sorted_scores[q_index]
    
    # --- 6. OUTPUT RESULTS ---
    print("\n" + "="*40)
    print("       CALIBRATION RESULTS       ")
    print("="*40)
    print(f"Total Time Steps:   {n_samples}")
    print(f"Max Accel Error:    {np.max(all_scores):.4f} rad/s²")
    print(f"Mean Accel Error:   {np.mean(all_scores):.4f} rad/s²")
    print(f"Std Dev Error:      {np.std(all_scores):.4f}")
    print("-" * 40)
    print(f"ROBUST QUANTILE (q_{{{CONFIDENCE}}}) : {quantile_val:.5f}")
    print("-" * 40)
    print(f"ACTION: Set 'self.q_quantile = {quantile_val:.5f}' in your Controller.")

    # --- 7. PLOT ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=60, color='skyblue', edgecolor='black', alpha=0.7, label='Error Samples')
    plt.axvline(quantile_val, color='r', linestyle='--', linewidth=3, label=f'Quantile ({CONFIDENCE*100}%)')
    plt.title(f"Model Uncertainty Distribution (Acceleration Error)\nRobust Quantile = {quantile_val:.4f} rad/s²")
    plt.xlabel("Acceleration Error Norm ||ddq_real - ddq_pred|| [rad/s²]")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()


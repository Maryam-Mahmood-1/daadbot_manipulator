import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# 1. GROUND TRUTH PHYSICS (The "Real Robot Plant")
# We ALWAYS use the Original URDF for the physics simulation step (aba)
URDF_PHYSICS = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"

# 2. CONTROLLER MODELS (The "Brains" doing CTC calculations (rnea))
# Baseline Brain: Has perfect knowledge of physics
URDF_MODEL_PERFECT = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
# Learned Brain: Has noisy knowledge of physics
URDF_MODEL_LEARNED = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"

# Simulation params
DURATION = 4.0      # Seconds for move
HOLD_TIME = 1.0     # Seconds to hold at end
DT = 0.001          # 1kHz loop
KP = 150.0          # Stiffness (increased slightly for tighter tracking)
KD = 25.0           # Damping

# Target Trajectory Definition
# Let's move Joint 2 (Shoulder) and Joint 4 (Elbow)
TARGET_JOINT_INDICES = [1, 3] # 0-based indices corresponding to q vector
TARGET_POSITIONS = [0.8, -0.8] # Radians

# ==========================================
# SETUP
# ==========================================

# Load Physics (The Plant)
model_plant = pin.buildModelFromUrdf(URDF_PHYSICS)
data_plant = model_plant.createData()

# Load Controllers
model_perfect = pin.buildModelFromUrdf(URDF_MODEL_PERFECT)
data_perfect = model_perfect.createData()

model_learned = pin.buildModelFromUrdf(URDF_MODEL_LEARNED)
data_learned = model_learned.createData()

nq = model_plant.nq
nv = model_plant.nv

# Helper: Trajectory Generator (Smooth Cosine Interpolation)
def get_trajectory(t):
    q_des = np.zeros(nq)
    v_des = np.zeros(nv)
    a_des = np.zeros(nv)
    
    move_duration = DURATION
    
    if t < move_duration:
        # Smooth S-curve scaling factor from 0 to 1
        u = t / move_duration
        s = 0.5 * (1 - np.cos(np.pi * u))
        ds = 0.5 * np.pi * np.sin(np.pi * u) / move_duration
        dds = 0.5 * np.pi**2 * np.cos(np.pi * u) / (move_duration**2)
    else:
        # Hold final position
        s, ds, dds = 1.0, 0.0, 0.0
    
    # Apply targets to specific joints
    for i, joint_idx in enumerate(TARGET_JOINT_INDICES):
        target_val = TARGET_POSITIONS[i]
        q_des[joint_idx] = target_val * s
        v_des[joint_idx] = target_val * ds
        a_des[joint_idx] = target_val * dds
        
    return q_des, v_des, a_des

# Helper: Run One Simulation Loop
def run_simulation(model_ctrl, data_ctrl):
    # Reset Plant State
    q = np.zeros(nq)
    v = np.zeros(nv)
    
    # Logs for plotting
    t_log = []
    q_actual_log = []
    q_target_log = []
    
    total_time = DURATION + HOLD_TIME
    
    for t in np.arange(0, total_time, DT):
        # 1. Get Reference Trajectory
        q_ref, v_ref, a_ref = get_trajectory(t)
        
        # 2. Controller Step (Uses the specific model_ctrl passed in)
        e = q - q_ref
        de = v - v_ref
        
        # CTC Law: tau = M_hat*(a_ref - Kp*e - Kd*de) + C_hat*v + g_hat
        # Pinocchio RNEA computes this efficiently
        a_cmd = a_ref - KP*e - KD*de
        tau = pin.rnea(model_ctrl, data_ctrl, q, v, a_cmd)
        
        # 3. Physics Step (ALWAYS uses model_plant / Original URDF)
        # Apply controller torque to real plant
        pin.aba(model_plant, data_plant, q, v, tau)
        
        # Integrate forward one step
        v += data_plant.ddq * DT
        q = pin.integrate(model_plant, q, v * DT)
        
        # Logging
        t_log.append(t)
        q_actual_log.append(q.copy())
        q_target_log.append(q_ref.copy())
        
    return np.array(t_log), np.array(q_actual_log), np.array(q_target_log)

# ==========================================
# EXECUTION & PLOTTING
# ==========================================
print(f"Moving joints {TARGET_JOINT_INDICES} to {TARGET_POSITIONS} rad.")

print("1. Simulating with Perfect Model Controller...")
time_vec, q_perf_act, q_target_ref = run_simulation(model_perfect, data_perfect)

print("2. Simulating with Learned (Noisy) Model Controller...")
# We only need the actual q from the second run, time and target are same
_, q_learn_act, _ = run_simulation(model_learned, data_learned)


# --- PLOTTING ---
joints_to_plot = TARGET_JOINT_INDICES
num_plots = len(joints_to_plot)

fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
if num_plots == 1: axes = [axes] # Handle single plot case

for i, ax in enumerate(axes):
    joint_idx = joints_to_plot[i]
    joint_name = f"Joint Index {joint_idx} (e.g., Joint {joint_idx+1})"

    # 1. Plot Target (Reference)
    ax.plot(time_vec, q_target_ref[:, joint_idx], 'k--', linewidth=2.5, label='Target Reference', alpha=0.6)

    # 2. Plot Perfect Controller Result
    ax.plot(time_vec, q_perf_act[:, joint_idx], 'g-', linewidth=2, label='Perfect Model Ctrl')

    # 3. Plot Learned Controller Result
    ax.plot(time_vec, q_learn_act[:, joint_idx], 'r-', linewidth=2, label='Learned Model Ctrl')

    ax.set_ylabel(f'Position (rad)')
    ax.set_title(joint_name)
    ax.grid(True, alpha=0.3)
    
    # Only show legend on the first plot so it doesn't get cluttered
    if i == 0:
        ax.legend(loc='best')

axes[-1].set_xlabel('Time (s)')
plt.suptitle('CTC Trajectory Tracking: Perfect vs Learned Model Physics', y=0.99)
plt.tight_layout()
plt.show()
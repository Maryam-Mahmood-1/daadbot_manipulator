import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pinocchio as pin

# --- PATH SETUP ---
sys.path.append(os.path.expanduser("~/xdaadbot_ws/src"))

# --- IMPORT MODULES ---
from daadbot_clf_cbf.CLF_CBF_traj.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crclf_formulation import CRCLF_Formulation
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crcbf_formulation import CRCBF_Formulation
from daadbot_clf_cbf.CRCLF_CRCBF_traj.crqp_solver import solve_qp

# --- CONFIGURATION ---
URDF_CLEAN = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy2.urdf"

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

# --- TRAJECTORY GENERATION ---
class TrajectoryGenerator:
    def __init__(self):
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.27 
        self.period = 12.0
        self.start_approach_pos = None

    def get_ref(self, t, current_pos=None):
        if t < 5.0:
            if self.start_approach_pos is None:
                self.start_approach_pos = current_pos
            start_pt = self.center_pos + np.array([self.ellipse_a, 0, 0])
            ratio = t / 5.0
            sm_ratio = (1 - math.cos(ratio * math.pi)) / 2
            pd = (1 - sm_ratio) * self.start_approach_pos + sm_ratio * start_pt
            vd = np.zeros(3)
            ad = np.zeros(3)
            return pd, vd, ad
        
        t_traj = t - 5.0
        omega = 2 * math.pi / self.period
        angle = omega * t_traj
        pd = self.center_pos + np.array([
            self.ellipse_a * math.cos(angle), 
            self.ellipse_b * math.sin(angle), 0.0])
        vd = np.array([
            -self.ellipse_a * omega * math.sin(angle), 
             self.ellipse_b * omega * math.cos(angle), 0.0])
        ad = np.array([
            -self.ellipse_a * (omega**2) * math.cos(angle), 
            -self.ellipse_b * (omega**2) * math.sin(angle), 0.0])
        return pd, vd, ad

# --- SIMULATION ENGINE ---
def run_simulation(mode_name, q_cp_val, ctrl_urdf, duration=20.0, dt=0.002):
    """
    Runs a simulation.
    mode_name: Label for print
    q_cp_val: Conformal quantile (0 for naive)
    ctrl_urdf: Which URDF the CONTROLLER uses (Clean vs Noisy)
    """
    print(f"--- Starting Simulation: {mode_name} (q={q_cp_val}) ---")
    
    # 1. Setup Models
    # Physics ALWAYS uses Clean URDF (Ground Truth)
    model_phys = pin.buildModelFromUrdf(URDF_CLEAN)
    data_phys = model_phys.createData()
    
    # Controller uses the specified URDF (Clean for Ideal, Noisy for Robust tests)
    robot_ctrl = RobotDynamics(ctrl_urdf, 'endeffector', TARGET_JOINTS)
    
    # 2. Setup Robust Math
    crclf = CRCLF_Formulation(dim=3, q_cp=q_cp_val)
    crcbf = CRCBF_Formulation(center=[0.0, 0.0, 0.72], lengths=[0.2, 0.2, 0.4], power_n=4, q_cp=q_cp_val)
    traj_gen = TrajectoryGenerator()
    
    # 3. State Init
    q_sim = pin.neutral(model_phys)
    v_sim = np.zeros(model_phys.nv)
    
    logs = {
        't': [], 'V': [], 'h': [], 
        'x_act': [], 'y_act': [], 'x_ref': [], 'y_ref': [],
        'err_pos': [], 'err_vel': []
    }
    
    steps = int(duration / dt)
    
    for i in range(steps):
        t = i * dt
        
        # --- A. PHYSICS UPDATE (Manually inject state into Controller) ---
        q_7dof = np.zeros(7)
        v_7dof = np.zeros(7)
        for k, name in enumerate(TARGET_JOINTS):
            jid = model_phys.getJointId(name)
            q_7dof[k] = q_sim[model_phys.joints[jid].idx_q]
            v_7dof[k] = v_sim[model_phys.joints[jid].idx_v]

        robot_ctrl.q[robot_ctrl.q_indices] = q_7dof
        robot_ctrl.dq[robot_ctrl.v_indices] = v_7dof
        
        # --- SAFETY CHECK (Prevent Crash) ---
        if np.any(np.isnan(q_7dof)) or np.any(np.abs(q_7dof) > 100):
            print("!!! SIMULATION DIVERGED (Explosion detected) !!!")
            break

        # --- B. CONTROL COMPUTATION ---
        M_est, nle_est, p, v, J, dJ_dq = robot_ctrl.compute()
        
        if i == 0: traj_gen.start_approach_pos = p
        pd, vd, ad = traj_gen.get_ref(t, p)
        
        e = p - pd
        de = v - vd
        
        LfV, LgV, V_val, gamma, robust_term = crclf.get_qp_constraints(e, de)
        cbf_L, cbf_b = crcbf.get_constraints(p, v, ad)
        
        mu = solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L, cbf_b, robust_term)
        
        J_pinv = np.linalg.pinv(J)
        tau_cmd = (M_est @ (J_pinv @ (ad + mu - dJ_dq))) + nle_est
        tau_cmd = np.clip(tau_cmd, -45.0, 45.0) 
        
        # --- C. INTEGRATION (Clean Physics) ---
        tau_full = np.zeros(model_phys.nv)
        for k, name in enumerate(TARGET_JOINTS):
            tau_full[model_phys.joints[model_phys.getJointId(name)].idx_v] = tau_cmd[k]
            
        ddq_true = pin.aba(model_phys, data_phys, q_sim, v_sim, tau_full)
        v_sim += ddq_true * dt
        q_sim = pin.integrate(model_phys, q_sim, v_sim * dt)
        
        # --- D. LOGGING ---
        logs['t'].append(t)
        logs['V'].append(V_val)
        logs['h'].append(crcbf.get_h(p))
        logs['x_act'].append(p[0]); logs['y_act'].append(p[1])
        logs['x_ref'].append(pd[0]); logs['y_ref'].append(pd[1])
        logs['err_pos'].append(np.linalg.norm(e))
        logs['err_vel'].append(np.linalg.norm(de))
        
        if i % 1000 == 0: print(f"  t={t:.2f} s | V={V_val:.4f}")

    return logs

# --- PLOTTING ---
def main():
    # 1. Run Ideal Simulation (Clean Physics + Clean Controller)
    # This generates the "perfect" baseline V(x) line
    logs_ideal = run_simulation("Ideal (Clean/Clean)", q_cp_val=0.0, ctrl_urdf=URDF_CLEAN)

    # 2. Run Naive Simulation (Clean Physics + Noisy Controller)
    logs_naive = run_simulation("Naive (Noisy/Naive)", q_cp_val=0.0, ctrl_urdf=URDF_NOISY)
    
    # 3. Run Robust Simulation (Clean Physics + Noisy Controller + Robust Math)
    logs_robust = run_simulation("Robust (Noisy/CR-CLF)", q_cp_val=0.29861, ctrl_urdf=URDF_NOISY)
    
    print("Generating Plots in 4 separate windows...")
    
    # --- FIGURE 1: TRAJECTORY ---
    plt.figure(figsize=(8, 8))
    plt.plot(logs_naive['y_ref'], logs_naive['x_ref'], 'k--', label='Reference', alpha=0.6)
    plt.plot(logs_naive['y_act'], logs_naive['x_act'], 'r', label='Naive (q=0)', alpha=0.8)
    plt.plot(logs_robust['y_act'], logs_robust['x_act'], 'g', label='Robust (q=0.29)', linewidth=2.0)
    
    # Safe Set Box
    safe_rect = Rectangle((-0.2, -0.2), 0.4, 0.4, linewidth=2, edgecolor='b', facecolor='none', linestyle=':', label='Safe Set')
    plt.gca().add_patch(safe_rect)
    
    plt.title("1. Trajectory Tracking (XY)")
    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # --- FIGURE 2: LYAPUNOV V(x) ---
    plt.figure(figsize=(10, 6))
    
    # Plot Actual Performance
    plt.plot(logs_naive['t'], logs_naive['V'], 'r', label='Naive (Noisy)', alpha=0.6)
    plt.plot(logs_robust['t'], logs_robust['V'], 'g', label='Robust (Noisy)', linewidth=1.5)
    
    # Plot Ideal Baseline (The new requirement)
    # This comes from the simulation where controller knew perfectly about physics
    plt.plot(logs_ideal['t'], logs_ideal['V'], 'k--', linewidth=2.5, label='Ideal Baseline (Clean Physics)')
    
    plt.title("2. Stability: Lyapunov Function V(x)")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy V(x)")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    
    # --- FIGURE 3: ERROR NORMS (Split Graphs) ---
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graph 1: Position Error
    ax1.plot(logs_naive['t'], logs_naive['err_pos'], 'r--', label='Naive ||e||', alpha=0.6)
    ax1.plot(logs_robust['t'], logs_robust['err_pos'], 'g-', label='Robust ||e||', linewidth=1.5)
    ax1.set_title("Position Error Norm ||e||")
    ax1.set_ylabel("Error [m]")
    ax1.legend()
    ax1.grid(True)
    
    # Graph 2: Velocity Error
    ax2.plot(logs_naive['t'], logs_naive['err_vel'], 'r--', label='Naive ||de||', alpha=0.6)
    ax2.plot(logs_robust['t'], logs_robust['err_vel'], 'g-', label='Robust ||de||', linewidth=1.5)
    ax2.set_title("Velocity Error Norm ||de||")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [m/s]")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()

    # --- FIGURE 4: SAFETY BARRIER ---
    plt.figure(figsize=(10, 6))
    plt.plot(logs_naive['t'], logs_naive['h'], 'r', label='Naive h(x)', alpha=0.7)
    plt.plot(logs_robust['t'], logs_robust['h'], 'g', label='Robust h(x)', linewidth=1.5)
    
    # Also plot Ideal Safety for context
    plt.plot(logs_ideal['t'], logs_ideal['h'], 'k:', label='Ideal h(x)', alpha=0.5)
    
    plt.axhline(0, color='k', linestyle='--', label='Safety Boundary')
    
    plt.title("4. Safety: Barrier Function h(x)")
    plt.xlabel("Time [s]")
    plt.ylabel("h(x) [>0 is Safe]")
    plt.grid(True)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()






# import sys
# import os
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import pinocchio as pin
# import time

# # --- PATH SETUP ---
# sys.path.append(os.path.expanduser("~/xdaadbot_ws/src"))

# # --- IMPORT MODULES ---
# from daadbot_clf_cbf.CLF_CBF_traj.robot_dynamics import RobotDynamics
# from daadbot_clf_cbf.CRCLF_CRCBF_traj.crclf_formulation import CRCLF_Formulation
# from daadbot_clf_cbf.CRCLF_CRCBF_traj.crcbf_formulation import CRCBF_Formulation
# from daadbot_clf_cbf.CRCLF_CRCBF_traj.crqp_solver import solve_qp

# # --- CONFIGURATION ---
# URDF_CLEAN = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
# URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy2.urdf"

# TARGET_JOINTS = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# # --- HELPER CLASSES ---

# class TrajectoryGenerator:
#     def __init__(self):
#         self.center_pos = np.array([0.0, 0.0, 0.72])
#         self.ellipse_a = 0.15 
#         self.ellipse_b = 0.27 
#         self.period = 12.0
#         self.start_approach_pos = None

#     def get_ref(self, t, current_pos=None):
#         if t < 5.0:
#             if self.start_approach_pos is None:
#                 self.start_approach_pos = current_pos
#             start_pt = self.center_pos + np.array([self.ellipse_a, 0, 0])
#             ratio = t / 5.0
#             sm_ratio = (1 - math.cos(ratio * math.pi)) / 2
#             pd = (1 - sm_ratio) * self.start_approach_pos + sm_ratio * start_pt
#             vd = np.zeros(3)
#             ad = np.zeros(3)
#             return pd, vd, ad
        
#         t_traj = t - 5.0
#         omega = 2 * math.pi / self.period
#         angle = omega * t_traj
#         pd = self.center_pos + np.array([
#             self.ellipse_a * math.cos(angle), 
#             self.ellipse_b * math.sin(angle), 0.0])
#         vd = np.array([
#             -self.ellipse_a * omega * math.sin(angle), 
#              self.ellipse_b * omega * math.cos(angle), 0.0])
#         ad = np.array([
#             -self.ellipse_a * (omega**2) * math.cos(angle), 
#             -self.ellipse_b * (omega**2) * math.sin(angle), 0.0])
#         return pd, vd, ad

# class SimInstance:
#     """
#     Holds the state for a single robot simulation instance (e.g. Robust, Naive, or Ideal)
#     """
#     def __init__(self, name, ctrl_urdf, q_cp_val, color, linestyle='-'):
#         self.name = name
#         self.color = color
#         self.ls = linestyle
        
#         # 1. Physics (Always Clean)
#         self.model_phys = pin.buildModelFromUrdf(URDF_CLEAN)
#         self.data_phys = self.model_phys.createData()
#         self.q_sim = pin.neutral(self.model_phys)
#         self.v_sim = np.zeros(self.model_phys.nv)
        
#         # 2. Controller (Specified URDF)
#         self.robot_ctrl = RobotDynamics(ctrl_urdf, 'endeffector', TARGET_JOINTS)
#         self.crclf = CRCLF_Formulation(dim=3, q_cp=q_cp_val)
#         self.crcbf = CRCBF_Formulation(center=[0.0, 0.0, 0.72], lengths=[0.2, 0.2, 0.4], power_n=4, q_cp=q_cp_val)
        
#         # 3. Data Buffers for Plotting
#         self.log_t = []
#         self.log_x = []
#         self.log_y = []
#         self.log_V = []
#         self.log_h = []
#         self.log_err_pos = []
#         self.log_err_vel = []

#     def step(self, t, dt, traj_gen):
#         # A. Update Controller State from Physics
#         q_7dof = np.zeros(7)
#         v_7dof = np.zeros(7)
#         for k, name in enumerate(TARGET_JOINTS):
#             jid = self.model_phys.getJointId(name)
#             q_7dof[k] = self.q_sim[self.model_phys.joints[jid].idx_q]
#             v_7dof[k] = self.v_sim[self.model_phys.joints[jid].idx_v]

#         self.robot_ctrl.q[self.robot_ctrl.q_indices] = q_7dof
#         self.robot_ctrl.dq[self.robot_ctrl.v_indices] = v_7dof
        
#         # B. Compute Control
#         M_est, nle_est, p, v, J, dJ_dq = self.robot_ctrl.compute()
        
#         if traj_gen.start_approach_pos is None: 
#             traj_gen.start_approach_pos = p # Init start pos if not set
            
#         pd, vd, ad = traj_gen.get_ref(t, p)
        
#         e = p - pd
#         de = v - vd
        
#         LfV, LgV, V_val, gamma, robust_term = self.crclf.get_qp_constraints(e, de)
#         cbf_L, cbf_b = self.crcbf.get_constraints(p, v, ad)
        
#         mu = solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L, cbf_b, robust_term)
        
#         J_pinv = np.linalg.pinv(J)
#         tau_cmd = (M_est @ (J_pinv @ (ad + mu - dJ_dq))) + nle_est
#         tau_cmd = np.clip(tau_cmd, -45.0, 45.0)
        
#         # C. Step Physics
#         tau_full = np.zeros(self.model_phys.nv)
#         for k, name in enumerate(TARGET_JOINTS):
#             tau_full[self.model_phys.joints[self.model_phys.getJointId(name)].idx_v] = tau_cmd[k]
            
#         ddq_true = pin.aba(self.model_phys, self.data_phys, self.q_sim, self.v_sim, tau_full)
#         self.v_sim += ddq_true * dt
#         self.q_sim = pin.integrate(self.model_phys, self.q_sim, self.v_sim * dt)
        
#         # D. Logging
#         self.log_t.append(t)
#         self.log_x.append(p[0])
#         self.log_y.append(p[1])
#         self.log_V.append(V_val)
#         self.log_h.append(self.crcbf.get_h(p))
#         self.log_err_pos.append(np.linalg.norm(e))
#         self.log_err_vel.append(np.linalg.norm(de))
        
#         return pd # Return ref for plotting

# # --- REAL-TIME VISUALIZER ---

# def run_realtime_comparison():
#     duration = 20.0
#     dt = 0.002
#     render_every = 25  # Update plots every N steps (to keep it fast)
    
#     # 1. Init Instances
#     # Ideal: Clean Physics + Clean Model (Baseline)
#     sim_ideal = SimInstance("Ideal", URDF_CLEAN, q_cp_val=0.0, color='k', linestyle='--')
#     # Naive: Clean Physics + Noisy Model (q=0)
#     sim_naive = SimInstance("Naive", URDF_NOISY, q_cp_val=0.0, color='r')
#     # Robust: Clean Physics + Noisy Model (q=0.29)
#     sim_robust = SimInstance("Robust", URDF_NOISY, q_cp_val=0.29861, color='g')
    
#     sims = [sim_ideal, sim_naive, sim_robust]
#     traj_gen = TrajectoryGenerator()

#     # 2. Setup Plots
#     plt.ion() # Interactive mode ON
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(2, 3)

#     # Ax1: Trajectory (XY)
#     ax_traj = fig.add_subplot(gs[:, 0])
#     ax_traj.set_title("Trajectory (XY)")
#     ax_traj.set_xlabel("Y [m]"); ax_traj.set_ylabel("X [m]")
#     ax_traj.axis('equal')
#     ax_traj.grid(True)
    
#     # Safe Set Box
#     safe_rect = Rectangle((-0.2, -0.2), 0.4, 0.4, lw=2, ec='b', fc='none', ls=':', label='Safe Set')
#     ax_traj.add_patch(safe_rect)

#     # Ax2: Lyapunov
#     ax_lyap = fig.add_subplot(gs[0, 1])
#     ax_lyap.set_title("Lyapunov V(x)")
#     ax_lyap.set_yscale('log')
#     ax_lyap.grid(True)

#     # Ax3: Safety Barrier
#     ax_h = fig.add_subplot(gs[1, 1])
#     ax_h.set_title("Barrier h(x)")
#     ax_h.axhline(0, color='k', ls='-', lw=1)
#     ax_h.grid(True)

#     # Ax4: Pos Error
#     ax_err = fig.add_subplot(gs[0, 2])
#     ax_err.set_title("Position Error ||e||")
#     ax_err.grid(True)

#     # Ax5: Vel Error
#     ax_vel = fig.add_subplot(gs[1, 2])
#     ax_vel.set_title("Velocity Error ||de||")
#     ax_vel.grid(True)
    
#     # Initialize Line Objects
#     lines = {}
#     ref_line, = ax_traj.plot([], [], 'b:', label='Reference', alpha=0.5)
    
#     for sim in sims:
#         lines[sim.name] = {
#             'traj': ax_traj.plot([], [], color=sim.color, ls=sim.ls, label=sim.name, lw=1.5)[0],
#             'V':    ax_lyap.plot([], [], color=sim.color, ls=sim.ls, label=sim.name)[0],
#             'h':    ax_h.plot([], [], color=sim.color, ls=sim.ls, label=sim.name)[0],
#             'err':  ax_err.plot([], [], color=sim.color, ls=sim.ls, label=sim.name)[0],
#             'vel':  ax_vel.plot([], [], color=sim.color, ls=sim.ls, label=sim.name)[0]
#         }

#     # Add legends once
#     ax_traj.legend(loc='upper right', fontsize='small')
#     ax_lyap.legend(loc='upper right', fontsize='small')

#     # 3. Main Loop
#     steps = int(duration / dt)
#     log_ref_x, log_ref_y = [], []
    
#     print("--- Starting Real-Time Simulation ---")
    
#     for i in range(steps):
#         t = i * dt
        
#         # Step all simulations
#         pd = None
#         for sim in sims:
#             # We pass a shared traj_gen so they chase the same target
#             pd = sim.step(t, dt, traj_gen)
            
#         # Log Reference (just once)
#         if pd is not None:
#             log_ref_x.append(pd[0])
#             log_ref_y.append(pd[1])

#         # Render Update
#         if i % render_every == 0:
#             # Update Reference Track
#             ref_line.set_data(log_ref_y, log_ref_x)
            
#             for sim in sims:
#                 # Trajectory
#                 lines[sim.name]['traj'].set_data(sim.log_y, sim.log_x)
#                 # V
#                 lines[sim.name]['V'].set_data(sim.log_t, sim.log_V)
#                 # h
#                 lines[sim.name]['h'].set_data(sim.log_t, sim.log_h)
#                 # Errors
#                 lines[sim.name]['err'].set_data(sim.log_t, sim.log_err_pos)
#                 lines[sim.name]['vel'].set_data(sim.log_t, sim.log_err_vel)
            
#             # Dynamic Rescaling (Auto-scale axes)
#             for ax in [ax_traj, ax_lyap, ax_h, ax_err, ax_vel]:
#                 ax.relim()
#                 ax.autoscale_view()
            
#             fig.canvas.draw()
#             fig.canvas.flush_events()
            
#         # Optional: Print status
#         if i % 500 == 0:
#             print(f"Time: {t:.2f}s | Robust V: {sim_robust.log_V[-1]:.4f}")

#     print("Simulation Complete. Keeping plot open.")
#     plt.ioff()
#     plt.show()

# if __name__ == "__main__":
#     run_realtime_comparison()
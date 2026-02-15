#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

# --- PATHS ---
import os
from ament_index_python.packages import get_package_share_directory

URDF_NOISY = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot_noisy_4.urdf")
URDF_TRUE = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")

EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class ParallelComparisonNode(Node):
    def __init__(self):
        super().__init__('parallel_cr_comparison')
        
        self.q_robust_val = 1000.0  
        self.dt_phys = 0.001
        self.control_rate = 100.0
        self.cbf_active = False
        self.lock = threading.Lock()

        self.labels = ['F. Lin (No CLF)', 'Nominal (q=0)', 'Robust (q=high)']
        self.quantiles = [0.0, 0.0, self.q_robust_val]
        
        self.models_phys = []
        self.data_phys = []
        self.q_sims = []
        self.v_sims = []
        self.tau_cmds = []
        self.robot_ctrls = []
        self.clf_ctrls = []
        self.cbfs = []
        self.traj_gen = TrajectoryGenerator() 

        for q_val in self.quantiles:
            m = pin.buildModelFromUrdf(URDF_TRUE)
            self.models_phys.append(m)
            self.data_phys.append(m.createData())
            
            qs = pin.neutral(m)
            jid1 = m.getJointId(ALL_JOINTS[1])
            qs[m.joints[jid1].idx_q] = 0.1
            self.q_sims.append(qs)
            self.v_sims.append(np.zeros(m.nv))
            self.tau_cmds.append(np.zeros(2))

            self.robot_ctrls.append(RobotDynamics(URDF_NOISY, EE_NAMES, ALL_JOINTS, noise_level=0.0))
            self.clf_ctrls.append(RESCLF_Controller(dim_task=2))
            self.cbfs.append(CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=30.0, k_vel=21.0))

        self.tau_limits = np.array([40.0, 30.0])
        self.phys_joint_ids = [self.models_phys[0].getJointId(name) for name in ALL_JOINTS]

        self.log = {lbl: {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'h':[], 'mu':[], 'tau1':[], 'tau2':[]} for lbl in self.labels}

        self.running = True
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None

    def physics_loop(self):
        next_tick = time.time()
        while self.running:
            with self.lock:
                for i in range(len(self.labels)):
                    tau_full = np.zeros(self.models_phys[i].nv)
                    damping = 0.15 * self.v_sims[i]
                    for idx, jid in enumerate(self.phys_joint_ids):
                        idx_v = self.models_phys[i].joints[jid].idx_v
                        tau_full[idx_v] = self.tau_cmds[i][idx] - damping[idx_v]

                    ddq = pin.aba(self.models_phys[i], self.data_phys[i], self.q_sims[i], self.v_sims[i], tau_full)
                    self.v_sims[i] += ddq * self.dt_phys
                    self.q_sims[i] = pin.integrate(self.models_phys[i], self.q_sims[i], self.v_sims[i] * self.dt_phys)

            next_tick += self.dt_phys
            sleep_time = next_tick - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        for i in range(len(self.labels)):
            with self.lock:
                q_curr = np.array([self.q_sims[i][self.models_phys[i].joints[jid].idx_q] for jid in self.phys_joint_ids])
                dq_curr = np.array([self.v_sims[i][self.models_phys[i].joints[jid].idx_v] for jid in self.phys_joint_ids])

            M, nle, J, dJ, x, dx = self.robot_ctrls[i].compute_dynamics(q_curr, dq_curr)
            J_2d, dJ_2d, x_2d, dx_2d = J[0:2, :], dJ[0:2, :], x[0:2], dx[0:2]
            xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
            xd, vd = xd_f[:2], vd_f[:2]

            u_nom = self.clf_ctrls[i].get_nominal_acceleration(x_2d, dx_2d, xd, vd)
            u_ref = ad_f[:2] + u_nom
            
            LfV, LgV, V_val, gamma, robust_clf_term = self.clf_ctrls[i].get_lyapunov_constraints(
                x_2d, dx_2d, xd, vd, q_quantile=self.quantiles[i]
            )

            J_pinv = np.linalg.pinv(J_2d)

            if i == 0:
                mu = np.zeros(2)
                feasible = True
                h_val = self.cbfs[i].get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
            else:
                cbf_A, cbf_b = None, None
                h_val = self.cbfs[i].get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
                
                if self.cbf_active:
                    x_3d, dx_3d = np.array([x_2d[0], x_2d[1], 0.0]), np.array([dx_2d[0], dx_2d[1], 0.0])
                    u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
                    A_temp, b_temp = self.cbfs[i].get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=(2.4 if i==2 else self.quantiles[i]))
                    cbf_A, cbf_b = A_temp[:, :2], b_temp

                b_tau_bias = (M @ J_pinv @ (u_ref - (dJ_2d @ dq_curr))) + nle
                A_tau = np.vstack([M @ J_pinv, -M @ J_pinv])
                b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

                mu, feasible = solve_optimization(LfV, LgV, V_val, gamma, robust_clf_term, A_tau, b_tau, cbf_A, cbf_b)

            if feasible:
                self.tau_cmds[i] = np.clip((M @ J_pinv @ (u_ref + mu - (dJ_2d @ dq_curr))) + nle, -self.tau_limits, self.tau_limits)
            else:
                self.tau_cmds[i] = -8.0 * dq_curr + nle

            lbl = self.labels[i]
            with self.lock:
                if len(self.log[lbl]['t']) > 400:
                    for k in self.log[lbl]: self.log[lbl][k].pop(0)
                self.log[lbl]['t'].append(t_clock)
                self.log[lbl]['x'].append(x_2d[0]); self.log[lbl]['y'].append(x_2d[1])
                self.log[lbl]['xd'].append(xd[0]); self.log[lbl]['yd'].append(xd[1])
                self.log[lbl]['V'].append(V_val); self.log[lbl]['h'].append(h_val)
                self.log[lbl]['mu'].append(np.linalg.norm(mu))
                self.log[lbl]['tau1'].append(self.tau_cmds[i][0])
                self.log[lbl]['tau2'].append(self.tau_cmds[i][1])

def main():
    rclpy.init()
    node = ParallelComparisonNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    fig = plt.figure(figsize=(18, 11))
    
    # --- MAXIMIZING SPACE ---
    # Reducing wspace and hspace to use blue-highlighted areas
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.2)
    
    # Adjust subplots to fill figure edges (left, bottom, right, top)
    fig.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.95)
    
    ax_traj_list = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_v = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])
    ax_mu = fig.add_subplot(gs[1, 2])
    ax_tau1 = fig.add_subplot(gs[2, 0:2]) 
    ax_tau2 = fig.add_subplot(gs[2, 2])

    colors = ['#9467bd', '#ff7f0e', '#1f77b4'] 
    lns_traj, lns_ref = [], []
    
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbfs[0].radii[0], node.cbfs[0].radii[1], node.cbfs[0].power_n
    xb = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    yb = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))

    for i in range(3):
        ax = ax_traj_list[i]
        ln_r, = ax.plot([], [], 'k--', lw=1, alpha=0.7, label='Reference')
        lns_ref.append(ln_r); ln_t, = ax.plot([], [], color=colors[i], label='Actual', lw=2)
        lns_traj.append(ln_t); ax.plot(xb, yb, 'g--', alpha=0.4, label='Safe Boundary')
        ax.set_title(node.labels[i]); ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right', fontsize='small')

    lns_v = [ax_v.plot([], [], color=colors[i], label=node.labels[i])[0] for i in range(3)]
    lns_h = [ax_h.plot([], [], color=colors[i], label=node.labels[i])[0] for i in range(3)]
    lns_mu = [ax_mu.plot([], [], color=colors[i], label=node.labels[i])[0] for i in range(3)]
    lns_tau1 = [ax_tau1.plot([], [], color=colors[i], label=node.labels[i])[0] for i in range(3)]
    lns_tau2 = [ax_tau2.plot([], [], color=colors[i], label=node.labels[i])[0] for i in range(3)]

    ax_v.set_title("Lyapunov V(x)"); ax_h.set_title("Safety h(x)"); ax_mu.set_title("Correction ||μ||")
    ax_tau1.set_title("Joint 1 Torque (Base)"); ax_tau2.set_title("Joint 2 Torque (Elbow)")
    
    for a in [ax_v, ax_h, ax_mu, ax_tau1, ax_tau2]: 
        a.grid(True); a.legend(fontsize='x-small', loc='upper right')
    
    ax_h.axhline(0, color='red', linestyle='--', alpha=0.6)
    
    ax_tau1.axhline(node.tau_limits[0], color='r', linestyle=':', alpha=0.5, label='Limit')
    ax_tau1.axhline(-node.tau_limits[0], color='r', linestyle=':', alpha=0.5)
    ax_tau2.axhline(node.tau_limits[1], color='r', linestyle=':', alpha=0.5)
    ax_tau2.axhline(-node.tau_limits[1], color='r', linestyle=':', alpha=0.5)

    # Move check button slightly to avoid overlapping larger plots
    check_ax = plt.axes([0.01, 0.45, 0.08, 0.04]) 
    check = CheckButtons(check_ax, ['Activate CBF'], [False])
    check.on_clicked(lambda l: setattr(node, 'cbf_active', not node.cbf_active))

    def update(frame):
        with node.lock:
            for i, lbl in enumerate(node.labels):
                if not node.log[lbl]['t']: continue
                d = node.log[lbl]
                lns_traj[i].set_data(d['x'], d['y']); lns_ref[i].set_data(d['xd'], d['yd'])
                lns_v[i].set_data(d['t'], d['V']); lns_h[i].set_data(d['t'], d['h'])
                lns_mu[i].set_data(d['t'], d['mu'])
                lns_tau1[i].set_data(d['t'], d['tau1']); lns_tau2[i].set_data(d['t'], d['tau2'])
                
                if i == 0:
                    for a in [ax_v, ax_h, ax_mu, ax_tau1, ax_tau2]: a.set_xlim(d['t'][0], d['t'][-1])

            ax_v.set_ylim(0, 30); ax_h.set_ylim(-0.6, 1.2); ax_mu.set_ylim(0, 30)
            ax_tau1.set_ylim(-45, 45); ax_tau2.set_ylim(-35, 35)
        
        return lns_traj + lns_ref + lns_v + lns_h + lns_mu + lns_tau1 + lns_tau2

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    node.running = False
    rclpy.shutdown()

if __name__ == '__main__':
    main()





# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# import numpy as np
# import threading
# import time
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons 

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.robot_dynamics import RobotDynamics
# from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from some_examples_py.CRCLF_CRCBF_2_link.resclf_controller import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
# from some_examples_py.CRCLF_CRCBF_2_link.qp_solver import solve_optimization

# # --- PATHS ---
# URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot_noisy_3.urdf"
# URDF_TRUE = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"

# EE_NAMES = ["endEffector"]
# ALL_JOINTS = ["baseHinge", "interArm"]

# class ParallelComparisonNode(Node):
#     def __init__(self):
#         super().__init__('parallel_cr_comparison')
        
#         self.q_robust_val = 30000.0  
#         self.dt_phys = 0.001
#         self.control_rate = 100.0
#         self.cbf_active = False
#         self.lock = threading.Lock()

#         # UPDATED: 3 Comparisons
#         self.labels = ['F. Lin (No CLF)', 'Nominal (q=0)', 'Robust (q=high)']
#         self.quantiles = [0.0, 0.0, self.q_robust_val]
        
#         self.models_phys = []
#         self.data_phys = []
#         self.q_sims = []
#         self.v_sims = []
#         self.tau_cmds = []
#         self.robot_ctrls = []
#         self.clf_ctrls = []
#         self.cbfs = []
#         self.traj_gen = TrajectoryGenerator() 

#         # Initialize 3 environments
#         for q_val in self.quantiles:
#             m = pin.buildModelFromUrdf(URDF_TRUE)
#             self.models_phys.append(m)
#             self.data_phys.append(m.createData())
            
#             qs = pin.neutral(m)
#             jid1 = m.getJointId(ALL_JOINTS[1])
#             qs[m.joints[jid1].idx_q] = 0.1
#             self.q_sims.append(qs)
#             self.v_sims.append(np.zeros(m.nv))
#             self.tau_cmds.append(np.zeros(2))

#             self.robot_ctrls.append(RobotDynamics(URDF_NOISY, EE_NAMES, ALL_JOINTS, noise_level=0.0))
#             self.clf_ctrls.append(RESCLF_Controller(dim_task=2))
#             self.cbfs.append(CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=21.0, k_vel=12.0))

#         self.tau_limits = np.array([40.0, 30.0])
#         self.phys_joint_ids = [self.models_phys[0].getJointId(name) for name in ALL_JOINTS]

#         self.log = {lbl: {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'h':[], 'mu':[]} for lbl in self.labels}

#         self.running = True
#         self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
#         self.phys_thread.start()
#         self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
#         self.start_time = None

#     def physics_loop(self):
#         next_tick = time.time()
#         while self.running:
#             with self.lock:
#                 for i in range(len(self.labels)):
#                     tau_full = np.zeros(self.models_phys[i].nv)
#                     damping = 0.15 * self.v_sims[i]
#                     for idx, jid in enumerate(self.phys_joint_ids):
#                         idx_v = self.models_phys[i].joints[jid].idx_v
#                         tau_full[idx_v] = self.tau_cmds[i][idx] - damping[idx_v]

#                     ddq = pin.aba(self.models_phys[i], self.data_phys[i], self.q_sims[i], self.v_sims[i], tau_full)
#                     self.v_sims[i] += ddq * self.dt_phys
#                     self.q_sims[i] = pin.integrate(self.models_phys[i], self.q_sims[i], self.v_sims[i] * self.dt_phys)

#             next_tick += self.dt_phys
#             sleep_time = next_tick - time.time()
#             if sleep_time > 0: time.sleep(sleep_time)

#     def control_loop(self):
#         if self.start_time is None: self.start_time = time.time()
#         t_clock = time.time() - self.start_time

#         for i in range(len(self.labels)):
#             with self.lock:
#                 q_curr = np.array([self.q_sims[i][self.models_phys[i].joints[jid].idx_q] for jid in self.phys_joint_ids])
#                 dq_curr = np.array([self.v_sims[i][self.models_phys[i].joints[jid].idx_v] for jid in self.phys_joint_ids])

#             M, nle, J, dJ, x, dx = self.robot_ctrls[i].compute_dynamics(q_curr, dq_curr)
#             J_2d, dJ_2d, x_2d, dx_2d = J[0:2, :], dJ[0:2, :], x[0:2], dx[0:2]
            
#             xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
#             xd, vd = xd_f[:2], vd_f[:2]

#             u_nom = self.clf_ctrls[i].get_nominal_acceleration(x_2d, dx_2d, xd, vd)
#             u_ref = ad_f[:2] + u_nom
            
#             LfV, LgV, V_val, gamma, robust_clf_term = self.clf_ctrls[i].get_lyapunov_constraints(
#                 x_2d, dx_2d, xd, vd, q_quantile=self.quantiles[i]
#             )

#             # --- FIX: Calculate Pseudo-Inverse HERE so it exists for all cases ---
#             J_pinv = np.linalg.pinv(J_2d)

#             # --- STRATEGY SELECTION ---
#             if i == 0:
#                 # Case 1: Pure Feedback Linearization (No QP)
#                 mu = np.zeros(2)
#                 feasible = True
#                 # Dummy barrier value for logging
#                 h_val = self.cbfs[i].get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
#             else:
#                 # Case 2 & 3: CLF-QP (Nominal or Robust)
#                 cbf_A, cbf_b = None, None
#                 h_val = self.cbfs[i].get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
                
#                 if self.cbf_active:
#                     x_3d, dx_3d = np.array([x_2d[0], x_2d[1], 0.0]), np.array([dx_2d[0], dx_2d[1], 0.0])
#                     u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
#                     # Update index check: i=2 is Robust case
#                     if i == 2:
#                         A_temp, b_temp = self.cbfs[i].get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=2.4)
#                     else:
#                         A_temp, b_temp = self.cbfs[i].get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.quantiles[i])
#                     cbf_A, cbf_b = A_temp[:, :2], b_temp

#                 b_tau_bias = (M @ J_pinv @ (u_ref - (dJ_2d @ dq_curr))) + nle
#                 A_tau = np.vstack([M @ J_pinv, -M @ J_pinv])
#                 b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

#                 mu, feasible = solve_optimization(LfV, LgV, V_val, gamma, robust_clf_term, A_tau, b_tau, cbf_A, cbf_b)

#             # Apply Control
#             if feasible:
#                 # J_pinv is now guaranteed to be bound
#                 self.tau_cmds[i] = np.clip((M @ J_pinv @ (u_ref + mu - (dJ_2d @ dq_curr))) + nle, -self.tau_limits, self.tau_limits)
#             else:
#                 self.tau_cmds[i] = -8.0 * dq_curr + nle

#             lbl = self.labels[i]
#             with self.lock:
#                 if len(self.log[lbl]['t']) > 400:
#                     for k in self.log[lbl]: self.log[lbl][k].pop(0)
#                 self.log[lbl]['t'].append(t_clock)
#                 self.log[lbl]['x'].append(x_2d[0]); self.log[lbl]['y'].append(x_2d[1])
#                 self.log[lbl]['xd'].append(xd[0]); self.log[lbl]['yd'].append(xd[1])
#                 self.log[lbl]['V'].append(V_val); self.log[lbl]['h'].append(h_val)
#                 self.log[lbl]['mu'].append(np.linalg.norm(mu))

# def main():
#     rclpy.init()
#     node = ParallelComparisonNode()
#     threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

#     fig = plt.figure(figsize=(15, 10))
#     gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
#     ax_traj = fig.add_subplot(gs[:, 0])
#     ax_v = fig.add_subplot(gs[0, 1]); ax_h = fig.add_subplot(gs[1, 1]); ax_mu = fig.add_subplot(gs[2, 1])
#     plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3)

#     # UPDATED: 3 Colors
#     colors = ['#9467bd', '#ff7f0e', '#1f77b4'] # Purple (FBL), Orange (Nominal), Blue (Robust)
    
#     # Initialize lists for 3 items
#     lns_traj = [ax_traj.plot([], [], color=colors[i], label=node.labels[i], lw=2)[0] for i in range(3)]
    
#     ln_ref, = ax_traj.plot([], [], 'k--', lw=1, alpha=0.7, label='Reference')
    
#     lns_v = [ax_v.plot([], [], color=colors[i])[0] for i in range(3)]
#     lns_h = [ax_h.plot([], [], color=colors[i])[0] for i in range(3)]
#     lns_mu = [ax_mu.plot([], [], color=colors[i])[0] for i in range(3)]

#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry, n = node.cbfs[0].radii[0], node.cbfs[0].radii[1], node.cbfs[0].power_n
#     xb = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
#     yb = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
#     ax_traj.plot(xb, yb, 'g--', alpha=0.4, label='Safe Boundary')
    
#     ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.set_aspect('equal')
#     ax_traj.legend(loc='upper right'); ax_traj.set_title("Task Space Trajectory")
#     ax_v.set_title("Lyapunov V(x)"); ax_h.set_title("Safety h(x)"); ax_mu.set_title("Correction ||μ||")
#     for a in [ax_v, ax_h, ax_mu]: a.grid(True)
#     ax_h.axhline(0, color='red', linestyle='--', alpha=0.6)

#     check_ax = plt.axes([0.05, 0.02, 0.15, 0.05])
#     check = CheckButtons(check_ax, ['Activate CBF'], [False])
#     check.on_clicked(lambda l: setattr(node, 'cbf_active', not node.cbf_active))

#     def update(frame):
#         with node.lock:
#             # UPDATED: Loop over all labels
#             for i, lbl in enumerate(node.labels):
#                 if not node.log[lbl]['t']: continue
#                 d = node.log[lbl]
#                 lns_traj[i].set_data(d['x'], d['y'])
#                 lns_v[i].set_data(d['t'], d['V'])
#                 lns_h[i].set_data(d['t'], d['h'])
#                 lns_mu[i].set_data(d['t'], d['mu'])
#                 if i == 0:
#                     ln_ref.set_data(d['xd'], d['yd']) 
#                     for a in [ax_v, ax_h, ax_mu]: a.set_xlim(d['t'][0], d['t'][-1])
#             ax_v.set_ylim(0, 30); ax_h.set_ylim(-0.6, 1.2); ax_mu.set_ylim(0, 30)
#         return lns_traj + [ln_ref] + lns_v + lns_h + lns_mu

#     ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
#     plt.show()
#     node.running = False
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
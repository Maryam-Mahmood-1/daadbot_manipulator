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
URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot_noisy_3.urdf"
URDF_TRUE = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf"

EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class ParallelComparisonNode(Node):
    def __init__(self):
        super().__init__('parallel_cr_comparison')
        
        self.q_robust_val = 30000.0  
        self.dt_phys = 0.001
        self.control_rate = 100.0
        self.cbf_active = False
        self.lock = threading.Lock()

        self.labels = ['Nominal (q=0)', 'Robust (q=0.8)']
        self.quantiles = [0.0, self.q_robust_val]
        
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
            self.cbfs.append(CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4, k_pos=21.0, k_vel=12.0))

        self.tau_limits = np.array([40.0, 30.0])
        self.phys_joint_ids = [self.models_phys[0].getJointId(name) for name in ALL_JOINTS]

        # Logging including reference trajectory
        self.log = {lbl: {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'h':[], 'mu':[]} for lbl in self.labels}

        self.running = True
        self.phys_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.phys_thread.start()
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        self.start_time = None

    def physics_loop(self):
        next_tick = time.time()
        while self.running:
            with self.lock:
                for i in range(2):
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

        for i in range(2):
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

            cbf_A, cbf_b = None, None
            h_val = self.cbfs[i].get_h_value(np.array([x_2d[0], x_2d[1], 0.0]))
            if self.cbf_active:
                x_3d, dx_3d = np.array([x_2d[0], x_2d[1], 0.0]), np.array([dx_2d[0], dx_2d[1], 0.0])
                u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
                if i==1:
                    A_temp, b_temp = self.cbfs[i].get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=2.4)
                else:
                    A_temp, b_temp = self.cbfs[i].get_constraints(x_3d, dx_3d, u_ref_3d, q_quantile=self.quantiles[i])
                cbf_A, cbf_b = A_temp[:, :2], b_temp

            J_pinv = np.linalg.pinv(J_2d)
            b_tau_bias = (M @ J_pinv @ (u_ref - (dJ_2d @ dq_curr))) + nle
            A_tau = np.vstack([M @ J_pinv, -M @ J_pinv])
            b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

            mu, feasible = solve_optimization(LfV, LgV, V_val, gamma, robust_clf_term, A_tau, b_tau, cbf_A, cbf_b)

            if feasible:
                self.tau_cmds[i] = np.clip((M @ J_pinv @ (u_ref + mu - (dJ_2d @ dq_curr))) + nle, -self.tau_limits, self.tau_limits)
                #self.tau_cmds[i] = np.clip((M @ J_pinv @ (u_ref + 0 - (dJ_2d @ dq_curr))) + nle, -self.tau_limits, self.tau_limits)
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

def main():
    rclpy.init()
    node = ParallelComparisonNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1]); ax_h = fig.add_subplot(gs[1, 1]); ax_mu = fig.add_subplot(gs[2, 1])
    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3)

    colors = ['#ff7f0e', '#1f77b4'] # Nominal: Blue, Robust: Orange
    lns_traj = [ax_traj.plot([], [], color=colors[i], label=node.labels[i], lw=2)[0] for i in range(2)]
    # Reference Trajectory (shared/dashed)
    ln_ref, = ax_traj.plot([], [], 'k--', lw=1, alpha=0.7, label='Reference')
    
    lns_v = [ax_v.plot([], [], color=colors[i])[0] for i in range(2)]
    lns_h = [ax_h.plot([], [], color=colors[i])[0] for i in range(2)]
    lns_mu = [ax_mu.plot([], [], color=colors[i])[0] for i in range(2)]

    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbfs[0].radii[0], node.cbfs[0].radii[1], node.cbfs[0].power_n
    xb = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    yb = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(xb, yb, 'g--', alpha=0.4, label='Safe Boundary')
    
    ax_traj.set_xlim(-1.95, 1.95); ax_traj.set_ylim(-1.5, 1.5); ax_traj.set_aspect('equal')
    ax_traj.legend(loc='upper right'); ax_traj.set_title("Task Space Trajectory")
    ax_v.set_title("Lyapunov V(x)"); ax_h.set_title("Safety h(x)"); ax_mu.set_title("Correction ||μ||")
    for a in [ax_v, ax_h, ax_mu]: a.grid(True)
    ax_h.axhline(0, color='red', linestyle='--', alpha=0.6)

    check_ax = plt.axes([0.05, 0.02, 0.15, 0.05])
    check = CheckButtons(check_ax, ['Activate CBF'], [False])
    check.on_clicked(lambda l: setattr(node, 'cbf_active', not node.cbf_active))

    def update(frame):
        with node.lock:
            for i, lbl in enumerate(node.labels):
                if not node.log[lbl]['t']: continue
                d = node.log[lbl]
                lns_traj[i].set_data(d['x'], d['y'])
                lns_v[i].set_data(d['t'], d['V'])
                lns_h[i].set_data(d['t'], d['h'])
                lns_mu[i].set_data(d['t'], d['mu'])
                if i == 0:
                    ln_ref.set_data(d['xd'], d['yd']) # Update reference from log
                    for a in [ax_v, ax_h, ax_mu]: a.set_xlim(d['t'][0], d['t'][-1])
            ax_v.set_ylim(0, 30); ax_h.set_ylim(-0.6, 1.2); ax_mu.set_ylim(0, 30)
        return lns_traj + [ln_ref] + lns_v + lns_h + lns_mu

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    node.running = False
    rclpy.shutdown()

if __name__ == '__main__':
    main()
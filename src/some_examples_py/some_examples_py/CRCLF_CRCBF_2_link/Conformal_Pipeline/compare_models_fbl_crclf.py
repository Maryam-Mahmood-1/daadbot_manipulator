#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RadioButtons 
import os
import sys
from ament_index_python.packages import get_package_share_directory

# --- SINDY PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1, s2, c2 = np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
        H_x = np.array([
            1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
            dq0_sq, dq1_sq, dq_cross,
            dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
            dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
            np.sign(dq[0]), np.sign(dq[1])
        ])
        
        G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
        a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
        b_hat = np.zeros((2, 2))
        b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
        b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
        return a_hat, b_hat

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

class SINDyComparisonNode(Node):
    def __init__(self):
        super().__init__('sindy_clf_comparison')
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        
        self.sindy = SINDyPredictor(os.path.join(ws_path, "sindy_Xi_state_space2.npy"), 
                                    os.path.join(ws_path, "q_quantile_state_space2.txt"))
        
        urdf_true = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
        
        self.labels = ['Feedback Lin.', 'Regular CLF', 'CR-CLF (Robust)']
        self.quantiles = [0.0, 0.0, self.sindy.q_quantile] 
        
        self.models_phys, self.data_phys = [], []
        self.models_ctrl, self.data_ctrl = [], []
        self.q_sims, self.v_sims, self.tau_cmds = [], [], []
        
        self.clf_ctrls = [RESCLF_Controller(dim_task=2) for _ in range(3)]
        self.cbfs = [CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4) for _ in range(3)]
        self.traj_gen = TrajectoryGenerator()
        
        for _ in range(3):
            m = pin.buildModelFromUrdf(urdf_true)
            self.models_phys.append(m)
            self.data_phys.append(m.createData())
            mc = pin.buildModelFromUrdf(urdf_true)
            self.models_ctrl.append(mc)
            self.data_ctrl.append(mc.createData())
            self.q_sims.append(np.array([0.5, 0.5]))
            self.v_sims.append(np.zeros(2))
            self.tau_cmds.append(np.zeros(2))

        self.ee_id = self.models_ctrl[0].getFrameId("endEffector")
        self.tau_limits = np.array([50.0, 40.0])
        self.cbf_active = False
        self.active_mode = 'Task Track'
        self.lock = threading.Lock()
        
        self.log = {lbl: {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[]} for lbl in self.labels}

        threading.Thread(target=self.physics_loop, daemon=True).start()
        self.timer = self.create_timer(0.01, self.control_loop)

    def generate_reference(self, t_clock, current_x_task):
        if self.active_mode == 'Joint Stab':
            return np.array([np.pi/4, -np.pi/4]), np.zeros(2), np.zeros(2)
        elif self.active_mode == 'Joint Track':
            return np.array([np.sin(t_clock), np.cos(t_clock)]), np.array([np.cos(t_clock), -np.sin(t_clock)]), np.array([-np.sin(t_clock), -np.cos(t_clock)])
        elif self.active_mode == 'Task Stab':
            return np.array([0.8, 0.2]), np.zeros(2), np.zeros(2)
        elif self.active_mode == 'Task Track':
            xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(current_x_task, (0,1)))
            return xd_f[:2], vd_f[:2], ad_f[:2]

    def physics_loop(self):
        dt_p = 0.001
        while rclpy.ok():
            with self.lock:
                for i in range(3):
                    ddq = pin.aba(self.models_phys[i], self.data_phys[i], self.q_sims[i], self.v_sims[i], self.tau_cmds[i])
                    self.v_sims[i] += ddq * dt_p
                    self.q_sims[i] = pin.integrate(self.models_phys[i], self.q_sims[i], self.v_sims[i] * dt_p)
            time.sleep(dt_p)

    def control_loop(self):
        t_now = time.time()
        if not hasattr(self, 'start_t'): self.start_t = t_now
        t_clock = t_now - self.start_t

        for i in range(3):
            with self.lock:
                q, v = self.q_sims[i].copy(), self.v_sims[i].copy()
                active_safety = self.cbf_active
                mode = self.active_mode

            pin.forwardKinematics(self.models_ctrl[i], self.data_ctrl[i], q, v)
            pin.updateFramePlacements(self.models_ctrl[i], self.data_ctrl[i])
            x_task = self.data_ctrl[i].oMf[self.ee_id].translation[:2]
            J_full = pin.computeFrameJacobian(self.models_ctrl[i], self.data_ctrl[i], q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J = J_full[:2, :]
            dx_task = J @ v
            dj_dq = pin.getFrameAcceleration(self.models_ctrl[i], self.data_ctrl[i], self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]

            ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
            is_task_space = 'Task' in mode
            curr_p, curr_v, J_clf = (x_task, dx_task, J) if is_task_space else (q, v, None)

            a_hat, b_hat = self.sindy.get_dynamics(q, v)
            u_nom = self.clf_ctrls[i].get_nominal_acceleration(curr_p, curr_v, ref_pos, ref_vel)
            
            LfV, LgV, V, gamma, robust_term = self.clf_ctrls[i].get_lyapunov_constraints(
                curr_p, curr_v, ref_pos, ref_vel, u_nom, q_quantile=self.quantiles[i], J=J_clf
            )

            cbf_A, cbf_b = None, None
            if active_safety and i > 0: 
                A_t, b_t = self.cbfs[i].get_constraints(np.append(x_task,0), np.append(dx_task,0), np.append(ref_acc+u_nom,0), q_quantile=self.quantiles[i])
                cbf_A, cbf_b = A_t[:, :2], b_t
            
            mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=(0.0 if i < 2 else robust_term), cbf_A=cbf_A, cbf_b=cbf_b)
            
            if i == 0: mu = np.zeros(2) # Feedback Linearization has no correction

            if feasible or i == 0:
                if is_task_space:
                    J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
                    ddq_des = J_pinv @ (ref_acc + u_nom + mu - dj_dq)
                else:
                    ddq_des = ref_acc + u_nom + mu
                tau_raw = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
                self.tau_cmds[i] = np.clip(tau_raw, -self.tau_limits, self.tau_limits)
            else:
                self.tau_cmds[i] = -20.0 * v

            lbl = self.labels[i]
            with self.lock:
                if len(self.log[lbl]['t']) > 400:
                    for key in self.log[lbl]: self.log[lbl][key].pop(0)

                self.log[lbl]['t'].append(t_clock)
                self.log[lbl]['x'].append(x_task[0])
                self.log[lbl]['y'].append(x_task[1])
                self.log[lbl]['V'].append(V)
                self.log[lbl]['mu'].append(np.linalg.norm(mu))
                
                if is_task_space:
                    self.log[lbl]['xd'].append(ref_pos[0])
                    self.log[lbl]['yd'].append(ref_pos[1])
                else:
                    L1, L2 = 0.75, 1.0 
                    self.log[lbl]['xd'].append(L1*np.cos(ref_pos[0]) + L2*np.cos(ref_pos[0]+ref_pos[1]))
                    self.log[lbl]['yd'].append(L1*np.sin(ref_pos[0]) + L2*np.sin(ref_pos[0]+ref_pos[1]))

def main():
    rclpy.init()
    node = SINDyComparisonNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4) 
    ax_traj_list = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_combined = fig.add_subplot(gs[0, 3]) 
    ax_v, ax_mu = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # --- UI ---
    rax_check = plt.axes([0.45, 0.05, 0.12, 0.08], frameon=False)
    check = CheckButtons(rax_check, ['Safety Active'], [False])
    check.on_clicked(lambda l: setattr(node, 'cbf_active', not node.cbf_active))

    rax_radio = plt.axes([0.1, 0.05, 0.3, 0.08], frameon=False)
    radio = RadioButtons(rax_radio, ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
    def switch_mode(label):
        with node.lock:
            node.active_mode = label
            for lbl in node.labels:
                for k in node.log[lbl]: node.log[lbl][k] = []
    radio.on_clicked(switch_mode)

    # Use distinct colors and line thicknesses
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4'] 
    lns_traj, lns_ref, lns_v, lns_mu, lns_combined = [], [], [], [], []

    # super-ellipse cage
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = 1.1, 1.1, 4
    x_cage = rx * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/n)
    y_cage = ry * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/n)

    for ax in ax_traj_list + [ax_combined]:
        ax.plot(x_cage, y_cage, '--', color='gray', alpha=0.3)
        ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.95, 1.95); ax.grid(True)

    for i in range(3):
        # Top Row: Individual
        lr, = ax_traj_list[i].plot([], [], 'k--', alpha=0.4, label='Ref')
        lt, = ax_traj_list[i].plot([], [], color=colors[i], lw=2.5)
        lns_ref.append(lr); lns_traj.append(lt)
        ax_traj_list[i].set_title(node.labels[i])
        
        # Top Row: Combined (Vary widths so they don't hide each other)
        lc, = ax_combined.plot([], [], color=colors[i], lw=(4.5 - i), label=node.labels[i], alpha=0.8)
        lns_combined.append(lc)
        
        # Bottom Row: Metrics
        lv, = ax_v.plot([], [], color=colors[i], label=node.labels[i], lw=1.5)
        lm, = ax_mu.plot([], [], color=colors[i], label=node.labels[i], lw=1.5)
        lns_v.append(lv); lns_mu.append(lm)

    ax_combined.set_title("Comparison Overlay")
    ax_combined.legend(loc='upper right', fontsize='x-small')
    ax_v.set_title("Lyapunov Energy V(x)")
    ax_mu.set_title("Correction Force ||mu||")
    ax_v.legend(); ax_mu.legend()
    lr_comb, = ax_combined.plot([], [], 'k--', alpha=0.5, zorder=0)

    def update(frame):
        with node.lock:
            # 1. Update all controller logs
            for i, lbl in enumerate(node.labels):
                d = node.log[lbl]
                if len(d['t']) < 2: continue
                
                lns_traj[i].set_data(d['x'], d['y'])
                lns_ref[i].set_data(d['xd'], d['yd'])
                
                # UPDATE THE COMBINED LINE HERE EXPLICITLY
                lns_combined[i].set_data(d['x'], d['y'])
                
                lns_v[i].set_data(d['t'], d['V'])
                lns_mu[i].set_data(d['t'], d['mu'])
                
                # Shared ref on combined plot
                if i == 2: lr_comb.set_data(d['xd'], d['yd'])

            # 2. Re-scale X-Axis based on the MOST RECENT time available across all logs
            all_times = [node.log[l]['t'] for l in node.labels if node.log[l]['t']]
            if all_times:
                t_min = min(t[0] for t in all_times)
                t_max = max(t[-1] for t in all_times)
                ax_v.set_xlim(t_min, t_max)
                ax_mu.set_xlim(t_min, t_max)

            # 3. Dynamic Y-scaling
            v_data = [v for l in node.labels for v in node.log[l]['V'][-30:] if node.log[l]['V']]
            if v_data: ax_v.set_ylim(0, max(v_data) * 1.2)
            
            mu_data = [m for l in node.labels for m in node.log[l]['mu'][-30:] if node.log[l]['mu']]
            if mu_data: ax_mu.set_ylim(0, max(max(mu_data) * 1.2, 2.0))
                    
        return lns_traj + lns_ref + lns_combined + [lr_comb] + lns_v + lns_mu

    ani = FuncAnimation(fig, update, interval=40, cache_frame_data=False, blit=True)
    plt.show()
    rclpy.shutdown()

if __name__ == '__main__': main()






# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# import numpy as np
# import threading
# import time
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import CheckButtons, RadioButtons 
# import os
# import sys
# from ament_index_python.packages import get_package_share_directory

# # --- SINDY PREDICTOR ---
# class SINDyPredictor:
#     def __init__(self, xi_path, q_val_path):
#         self.Xi = np.load(xi_path)
#         with open(q_val_path, "r") as f:
#             self.q_quantile = float(f.read())
            
#     def get_dynamics(self, q, dq):
#         s1, c1, s2, c2 = np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])
#         s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
#         dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
#         H_x = np.array([
#             1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
#             dq0_sq, dq1_sq, dq_cross,
#             dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
#             dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
#             np.sign(dq[0]), np.sign(dq[1])
#         ])
        
#         G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
#         a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
#         b_hat = np.zeros((2, 2))
#         b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
#         b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
#         return a_hat, b_hat

# # --- MODULAR IMPORTS ---
# from some_examples_py.CRCLF_CRCBF_2_link.trajectory_generator import TrajectoryGenerator
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.cbf_formulation import CBF_SuperEllipsoid 
# from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

# class SINDyComparisonNode(Node):
#     def __init__(self):
#         super().__init__('sindy_clf_comparison')
#         ws_path = "/home/maryammahmood/xdaadbot_ws/"
        
#         self.sindy = SINDyPredictor(os.path.join(ws_path, "sindy_Xi_state_space2.npy"), 
#                                     os.path.join(ws_path, "q_quantile_state_space2.txt"))
        
#         urdf_true = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
        
#         self.labels = ['Regular CLF (q=0)', 'CR-CLF (Robust)']
#         self.quantiles = [0.0, self.sindy.q_quantile] 
        
#         self.models_phys, self.data_phys = [], []
#         self.models_ctrl, self.data_ctrl = [], []
#         self.q_sims, self.v_sims, self.tau_cmds = [], [], []
        
#         self.clf_ctrls = [RESCLF_Controller(dim_task=2) for _ in range(2)]
#         self.cbfs = [CBF_SuperEllipsoid(center=[0.0, 0.0, 0.0], lengths=[1.1, 1.1, 3.0], power_n=4) for _ in range(2)]
#         self.traj_gen = TrajectoryGenerator()
        
#         for _ in range(2):
#             m = pin.buildModelFromUrdf(urdf_true)
#             self.models_phys.append(m)
#             self.data_phys.append(m.createData())
#             mc = pin.buildModelFromUrdf(urdf_true)
#             self.models_ctrl.append(mc)
#             self.data_ctrl.append(mc.createData())
#             self.q_sims.append(np.array([0.5, 0.5]))
#             self.v_sims.append(np.zeros(2))
#             self.tau_cmds.append(np.zeros(2))

#         self.ee_id = self.models_ctrl[0].getFrameId("endEffector")
#         self.tau_limits = np.array([50.0, 40.0])
#         self.cbf_active = False
#         self.active_mode = 'Task Track'
#         self.lock = threading.Lock()
        
#         self.log = {lbl: {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'V':[], 'mu':[]} for lbl in self.labels}

#         threading.Thread(target=self.physics_loop, daemon=True).start()
#         self.timer = self.create_timer(0.01, self.control_loop)

#     def generate_reference(self, t_clock, current_x_task):
#         if self.active_mode == 'Joint Stab':
#             return np.array([np.pi/4, -np.pi/4]), np.zeros(2), np.zeros(2)
#         elif self.active_mode == 'Joint Track':
#             return np.array([np.sin(t_clock), np.cos(t_clock)]), np.array([np.cos(t_clock), -np.sin(t_clock)]), np.array([-np.sin(t_clock), -np.cos(t_clock)])
#         elif self.active_mode == 'Task Stab':
#             return np.array([0.8, 0.2]), np.zeros(2), np.zeros(2)
#         elif self.active_mode == 'Task Track':
#             xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(current_x_task, (0,1)))
#             return xd_f[:2], vd_f[:2], ad_f[:2]

#     def physics_loop(self):
#         dt_p = 0.001
#         while rclpy.ok():
#             with self.lock:
#                 for i in range(2):
#                     ddq = pin.aba(self.models_phys[i], self.data_phys[i], self.q_sims[i], self.v_sims[i], self.tau_cmds[i])
#                     self.v_sims[i] += ddq * dt_p
#                     self.q_sims[i] = pin.integrate(self.models_phys[i], self.q_sims[i], self.v_sims[i] * dt_p)
#             time.sleep(dt_p)

#     def control_loop(self):
#         t_now = time.time()
#         if not hasattr(self, 'start_t'): self.start_t = t_now
#         t_clock = t_now - self.start_t

#         for i in range(2):
#             with self.lock:
#                 q, v = self.q_sims[i].copy(), self.v_sims[i].copy()
#                 active_safety = self.cbf_active
#                 mode = self.active_mode

#             pin.forwardKinematics(self.models_ctrl[i], self.data_ctrl[i], q, v)
#             pin.updateFramePlacements(self.models_ctrl[i], self.data_ctrl[i])
#             x_task = self.data_ctrl[i].oMf[self.ee_id].translation[:2]
#             J_full = pin.computeFrameJacobian(self.models_ctrl[i], self.data_ctrl[i], q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
#             J = J_full[:2, :]
#             dx_task = J @ v
#             dj_dq = pin.getFrameAcceleration(self.models_ctrl[i], self.data_ctrl[i], self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]

#             ref_pos, ref_vel, ref_acc = self.generate_reference(t_clock, x_task)
#             is_task_space = 'Task' in mode
#             curr_p, curr_v, J_clf = (x_task, dx_task, J) if is_task_space else (q, v, None)

#             a_hat, b_hat = self.sindy.get_dynamics(q, v)
#             u_nom = self.clf_ctrls[i].get_nominal_acceleration(curr_p, curr_v, ref_pos, ref_vel)
            
#             LfV, LgV, V, gamma, robust_term = self.clf_ctrls[i].get_lyapunov_constraints(
#                 curr_p, curr_v, ref_pos, ref_vel, u_nom, q_quantile=self.quantiles[i], J=J_clf
#             )

#             cbf_A, cbf_b = None, None
#             if active_safety:
#                 A_t, b_t = self.cbfs[i].get_constraints(np.append(x_task,0), np.append(dx_task,0), np.append(ref_acc+u_nom,0), q_quantile=self.quantiles[i])
#                 cbf_A, cbf_b = A_t[:, :2], b_t
            
#             mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=(0.0 if i==0 else robust_term), cbf_A=cbf_A, cbf_b=cbf_b)

#             if feasible:
#                 if is_task_space:
#                     J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(2))
#                     ddq_des = J_pinv @ (ref_acc + u_nom + mu - dj_dq)
#                 else:
#                     ddq_des = ref_acc + u_nom + mu
#                 tau_raw = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
#                 self.tau_cmds[i] = np.clip(tau_raw, -self.tau_limits, self.tau_limits)
#             else:
#                 self.tau_cmds[i] = -20.0 * v

#             lbl = self.labels[i]
#             with self.lock:
#                 if len(self.log[lbl]['t']) > 400:
#                     for key in self.log[lbl]: self.log[lbl][key].pop(0)

#                 self.log[lbl]['t'].append(t_clock)
#                 self.log[lbl]['x'].append(x_task[0]); self.log[lbl]['y'].append(x_task[1])
#                 self.log[lbl]['V'].append(V); self.log[lbl]['mu'].append(np.linalg.norm(mu))
                
#                 # For visualization, if in joint space, convert reference to approximate workspace coordinates
#                 if is_task_space:
#                     self.log[lbl]['xd'].append(ref_pos[0]); self.log[lbl]['yd'].append(ref_pos[1])
#                 else:
#                     L1, L2 = 0.75, 1.0 # Approximated for visualization
#                     self.log[lbl]['xd'].append(L1*np.cos(ref_pos[0]) + L2*np.cos(ref_pos[0]+ref_pos[1]))
#                     self.log[lbl]['yd'].append(L1*np.sin(ref_pos[0]) + L2*np.sin(ref_pos[0]+ref_pos[1]))

# def main():
#     rclpy.init()
#     node = SINDyComparisonNode()
#     threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

#     fig = plt.figure(figsize=(15, 10))
#     gs = fig.add_gridspec(2, 3) # Changed to 3 columns
#     ax_traj_list = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
#     ax_combined = fig.add_subplot(gs[0, 2]) # New combined plot
#     ax_v, ax_mu = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2]) # Adjusted lower plots
#     plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
#     # --- UI WIDGETS ---
#     rax_check = plt.axes([0.45, 0.05, 0.12, 0.08], frameon=False)
#     check = CheckButtons(rax_check, ['Safety Active'], [False])
#     def toggle_safety(l): node.cbf_active = not node.cbf_active
#     check.on_clicked(toggle_safety)

#     rax_radio = plt.axes([0.1, 0.05, 0.3, 0.08], frameon=False)
#     radio = RadioButtons(rax_radio, ('Joint Stab', 'Joint Track', 'Task Stab', 'Task Track'), active=3)
#     def switch_mode(label):
#         with node.lock:
#             node.active_mode = label
#             for lbl in node.labels:
#                 for k in node.log[lbl]: node.log[lbl][k] = []
#     radio.on_clicked(switch_mode)

#     colors = ['#ff7f0e', '#1f77b4']
#     lns_traj, lns_ref, lns_v, lns_mu = [], [], [], []
#     lns_combined = [] # To store lines for the combined plot

#     # Super-Ellipse Cage (n=4)
#     theta = np.linspace(0, 2*np.pi, 200)
#     rx, ry, n = 1.1, 1.1, 4
#     x_cage = rx * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/n)
#     y_cage = ry * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/n)

#     # Setup individual and Combined axes
#     for ax in [ax_traj_list[0], ax_traj_list[1], ax_combined]:
#         ax.plot(x_cage, y_cage, '--', color='gray', alpha=0.5, label='Cage')
#         ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.95, 1.95); ax.grid(True)

#     for i in range(2):
#         # Individual Plots
#         lr, = ax_traj_list[i].plot([], [], 'k--', alpha=0.5, label='Ref')
#         lt, = ax_traj_list[i].plot([], [], color=colors[i], lw=2, label='Actual')
#         lns_ref.append(lr); lns_traj.append(lt)
#         ax_traj_list[i].set_title(node.labels[i])
        
#         # Combined Plot Lines
#         lc, = ax_combined.plot([], [], color=colors[i], lw=2, label=node.labels[i])
#         lns_combined.append(lc)
        
#         # Metrics
#         lns_v.append(ax_v.plot([], [], color=colors[i], label=node.labels[i])[0])
#         lns_mu.append(ax_mu.plot([], [], color=colors[i], label=node.labels[i])[0])

#     ax_combined.set_title("Combined Comparison")
#     ax_combined.legend(loc='upper right', fontsize='small')
#     ax_v.set_title("Lyapunov Energy V(x)"); ax_mu.set_title("Correction Force ||mu||")
#     ax_v.legend(); ax_mu.legend()

#     # Shared reference line for combined plot
#     lr_comb, = ax_combined.plot([], [], 'k--', alpha=0.5)

#     def update(frame):
#         with node.lock:
#             for i, lbl in enumerate(node.labels):
#                 d = node.log[lbl]
#                 if not d['t'] or len(d['t']) < 2: continue
#                 lns_traj[i].set_data(d['x'], d['y'])
#                 lns_ref[i].set_data(d['xd'], d['yd'])
#                 lns_combined[i].set_data(d['x'], d['y']) # Update combined
#                 lns_v[i].set_data(d['t'], d['V'])
#                 lns_mu[i].set_data(d['t'], d['mu'])
#                 if i == 0: lr_comb.set_data(d['xd'], d['yd']) # Reference for combined
            
#             ref_lbl = node.labels[1]
#             if node.log[ref_lbl]['t']:
#                 t = node.log[ref_lbl]['t']
#                 ax_v.set_xlim(t[0], t[-1]); ax_mu.set_xlim(t[0], t[-1])
#                 v_max = max(max(node.log[node.labels[0]]['V'][-10:] if node.log[node.labels[0]]['V'] else [1]), 1.0) * 1.5
#                 ax_v.set_ylim(0, v_max)
#                 ax_mu.set_ylim(0, 15)
#         return lns_traj + lns_ref + lns_combined + [lr_comb] + lns_v + lns_mu

#     ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
#     plt.show()
#     rclpy.shutdown()

# if __name__ == '__main__': main()
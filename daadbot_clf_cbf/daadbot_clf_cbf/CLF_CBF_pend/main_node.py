#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons 

# --- IMPORT YOUR HELPER CLASSES ---
# (Ensure these are in your PYTHONPATH or the same directory)
# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CLF_CBF_pend.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CLF_CBF_pend.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CLF_CBF_pend.resclf_controller import RESCLF_Controller
from daadbot_clf_cbf.CLF_CBF_pend.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CLF_CBF_pend.qp_solver import solve_optimization
# --- CONFIGURATIONS ---
URDF_CTRL = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_lab_inv_pendulum/invp.urdf.xacro"
EE_NAMES = ["end_effector"]
ALL_JOINTS = ["joint1"] 

class PendulumResclfNode(Node):
    def __init__(self):
        super().__init__('pendulum_resclf_node')
        
        # --- 1. CONTROLLER SETUP ---
        # Initialize Robot Dynamics for 1-DOF
        self.robot_ctrl = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        
        # CLF Controller: Task dimension is 2 (X, Y tracking)
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # Safety Barrier: Ellipse in XY plane
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], 
            lengths=[1.2, 1.2, 3.0], 
            power_n=4, k_pos=21.0, k_vel=12.0
        )
        self.cbf_active = False 

        # --- 2. ROS INTERFACE ---
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(1.0/100.0, self.control_loop) # 100Hz
        
        # --- 3. STATE ---
        self.start_time = None
        self.q = np.array([0.1]) # Single joint
        self.dq = np.zeros(1)
        self.tau_limits = np.array([13.0]) # From URDF

        # --- 4. LOGGING ---
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def cb_joints(self, msg):
        """ Update 1-DOF joint state """
        try:
            for i, name in enumerate(ALL_JOINTS):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self.q[i] = msg.position[idx]
                    self.dq[i] = msg.velocity[idx]
        except (ValueError, IndexError):
            pass

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # A. DYNAMICS (From your RobotDynamics class)
        # Returns M(1,1), nle(1,), J(3,1), dJ(3,1), x(3,), dx(3,)
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(self.q, self.dq)
        
        # Extract X-Y task space (2x1 Jacobian)
        J_task = J[0:2, :] 
        dJ_task = dJ[0:2, :]
        x_2d = x[0:2]
        dx_2d = dx[0:2]

        # B. TRAJECTORY
        xd_f, vd_f, ad_f = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
        xd, vd, ad = xd_f[:2], vd_f[:2], ad_f[:2]

        # C. NOMINAL ACCELERATION & CLF CONSTRAINTS (From RESCLF_Controller)
        u_nom = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
        u_ref = ad + u_nom
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x_2d, dx_2d, xd, vd)

        # D. CBF CONSTRAINTS (From CBF_SuperEllipsoid)
        cbf_A, cbf_b = None, None
        x_3d = np.array([x_2d[0], x_2d[1], 0.0])
        h_val = self.cbf.get_h_value(x_3d)

        if self.cbf_active:
            dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
            u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
            A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d)
            cbf_A = A_temp[:, :2] 
            cbf_b = b_temp

        # E. QP SETUP (Input Mapping for 1-DOF)
        # We need to map acceleration mu to joint torque tau:
        # tau = M * J_inv * (u_ref + mu - dJ*dq) + nle
        J_pinv = np.linalg.pinv(J_task)
        drift_acc = u_ref - (dJ_task @ self.dq)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias]).reshape(-1, 1)

        # F. SOLVE OPTIMIZATION (Using your provided solve_optimization function)
        mu, feasible = solve_optimization(
            LfV, LgV, V, gamma, 
            torque_A=A_tau, torque_b=b_tau, 
            cbf_A=cbf_A, cbf_b=cbf_b
        )

        # G. CALCULATE & PUBLISH FINAL TORQUE
        if feasible:
            acc_cmd = u_ref + mu 
            tau_out = (M @ J_pinv @ (acc_cmd - (dJ_task @ self.dq))) + nle
        else:
            tau_out = -2.0 * self.dq + nle # Fallback damping

        tau_out = np.clip(tau_out, -self.tau_limits, self.tau_limits)
        msg = Float64MultiArray(data=tau_out.tolist())
        self.pub.publish(msg)

        # H. LOGGING
        if len(self.log['t']) > 500:
            for k in self.log: self.log[k].pop(0)
        self.log['t'].append(t_clock)
        self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
        self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
        self.log['h'].append(h_val)
        self.log['mu'].append(np.linalg.norm(mu))

    def stop_robot(self):
        self.pub.publish(Float64MultiArray(data=[0.0]))

def main(args=None):
    rclpy.init(args=args)
    node = PendulumResclfNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1]); ax_mu = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(bottom=0.2)
    
    ln_a, = ax_traj.plot([], [], 'r-', label='EE Actual')
    ln_t, = ax_traj.plot([], [], 'b--', label='EE Target')
    
    # Boundary visualization
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry, n = node.cbf.radii[0], node.cbf.radii[1], node.cbf.power_n
    x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(x_b, y_b, 'g-', label='Safe Region')
    
    ax_traj.set_xlim(-1.5, 1.5); ax_traj.set_ylim(-1.5, 1.5); ax_traj.set_aspect('equal')
    ax_traj.grid(True); ax_traj.legend()

    ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', ls='--'); ax_h.set_title("Safety h(x)")
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Robust Correction ||μ||")

    ax_check = plt.axes([0.05, 0.05, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Enable Safety'], [False])
    check.on_clicked(lambda label: setattr(node, 'cbf_active', not node.cbf_active))

    def update(frame):
        if not node.log['t']: return ln_a, ln_t, ln_h, ln_mu
        t_d, x_d, y_d = node.log['t'], node.log['x'], node.log['y']
        xd_d, yd_d, h_d, mu_d = node.log['xd'], node.log['yd'], node.log['h'], node.log['mu']

        ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
        ax_h.set_xlim(t_d[0], t_d[-1]); ax_mu.set_xlim(t_d[0], t_d[-1])
        ax_h.set_ylim(-1.2, 1.2); ax_mu.set_ylim(-1.0, 20.0)
        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    
    node.stop_robot(); rclpy.shutdown(); t_ros.join()

if __name__ == '__main__':
    main()
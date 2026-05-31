"""
Main Node for 2-DOF Robot using GAZEBO Simulation
Updated with Sliding Window Plotting & Higher Torque Limits
"""
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

# --- CONFIGURATIONS ---
from ament_index_python.packages import get_package_share_directory
import os

# --- MODULAR IMPORTS ---
from daadbot_clf_cbf.CLF_CBF_2_link.robot_dynamics import RobotDynamics
from daadbot_clf_cbf.CLF_CBF_2_link.trajectory_generator import TrajectoryGenerator
from daadbot_clf_cbf.CLF_CBF_2_link.resclf_controller import RESCLF_Controller
from daadbot_clf_cbf.CLF_CBF_2_link.cbf_formulation import CBF_SuperEllipsoid 
from daadbot_clf_cbf.CLF_CBF_2_link.qp_solver import solve_optimization

URDF_CTRL = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "2_link_urdf", 
    "2link_robot.urdf.xacro" 
)

EE_NAMES = ["endEffector"]
ALL_JOINTS = ["baseHinge", "interArm"]

class GazeboResclfNode(Node):
    def __init__(self):
        super().__init__('gazebo_resclf_node')
        
        # --- 1. CONTROLLER SETUP ---
        self.robot_ctrl = RobotDynamics(URDF_CTRL, EE_NAMES, ALL_JOINTS, noise_level=0.0)
        self.traj_gen = TrajectoryGenerator() 
        self.clf_ctrl = RESCLF_Controller(dim_task=2) 
        
        # Safety Barrier (Planar Ellipse)
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.0], 
            lengths=[1.2, 1.2, 3.0], # Matched with previous good values
            power_n=4, k_pos=21.0, k_vel=12.0
        )
        self.cbf_active = False # Default OFF 

        # --- 2. ROS INTERFACE ---
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(1.0/100.0, self.control_loop) # 100Hz Control Rate
        
        # --- 3. STATE ---
        self.start_time = None
        self.q = np.array([0.0, 0.1]) # Avoiding singularity assumption
        self.dq = np.zeros(2)
        
        # INCREASED LIMITS to prevent QP failure
        self.tau_limits = np.array([50.0, 30.0]) 

        # --- 4. LOGGING (Sliding Window Configured) ---
        self.log = {'t':[], 'x':[], 'y':[], 'xd':[], 'yd':[], 'h':[], 'mu':[]}

    def cb_joints(self, msg):
        try:
            q_buf = [0.0] * 2
            dq_buf = [0.0] * 2
            found_count = 0
            for i, name in enumerate(ALL_JOINTS):
                if name in msg.name:
                    idx = msg.name.index(name)
                    q_buf[i] = msg.position[idx]
                    dq_buf[i] = msg.velocity[idx]
                    found_count += 1
            if found_count == 2:
                self.q = np.array(q_buf)
                self.dq = np.array(dq_buf)
        except ValueError:
            pass

    def control_loop(self):
        if self.start_time is None: self.start_time = time.time()
        t_clock = time.time() - self.start_time

        # A. DYNAMICS
        M, nle, J, dJ, x, dx = self.robot_ctrl.compute_dynamics(self.q, self.dq)
        J = J[0:2, :]; dJ = dJ[0:2, :]; x_2d = x[0:2]; dx_2d = dx[0:2]

        # B. TRAJECTORY
        xd_full, vd_full, ad_full = self.traj_gen.get_ref(t_clock, current_actual_pos=np.pad(x_2d, (0,1)))
        xd, vd, ad = xd_full[:2], vd_full[:2], ad_full[:2]

        # C. NOMINAL CONTROL
        u_nominal = self.clf_ctrl.get_nominal_acceleration(x_2d, dx_2d, xd, vd)
        u_ref = ad + u_nominal 
        LfV, LgV, V, gamma = self.clf_ctrl.get_lyapunov_constraints(x_2d, dx_2d, xd, vd)

        # D. CBF
        cbf_A, cbf_b = None, None
        x_3d = np.array([x_2d[0], x_2d[1], 0.0])
        h_val = self.cbf.get_h_value(x_3d)

        # Safety is OFF by default until you click the button
        if self.cbf_active:
            dx_3d = np.array([dx_2d[0], dx_2d[1], 0.0])
            u_ref_3d = np.array([u_ref[0], u_ref[1], 0.0])
            A_temp, b_temp = self.cbf.get_constraints(x_3d, dx_3d, u_ref_3d)
            cbf_A = A_temp[:, :2] 
            cbf_b = b_temp

        # E. QP SETUP
        J_pinv = np.linalg.pinv(J)
        drift_acc = u_ref - (dJ @ self.dq)
        b_tau_bias = (M @ J_pinv @ drift_acc) + nle
        
        A_tau_base = M @ J_pinv
        A_tau = np.vstack([A_tau_base, -A_tau_base])
        b_tau = np.hstack([self.tau_limits - b_tau_bias, self.tau_limits + b_tau_bias])
        b_tau = b_tau.reshape(-1, 1)

        # F. SOLVE
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, torque_A=A_tau, torque_b=b_tau, cbf_A=cbf_A, cbf_b=cbf_b)

        if feasible:
            acc_cmd = u_ref + mu 
            #acc_cmd = u_ref + 0.0 
            tau_cmd = (M @ J_pinv @ (acc_cmd - (dJ @ self.dq))) + nle
        else:
            # Fallback to damping if QP fails
            tau_cmd = -5.0 * self.dq + nle 

        tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)
        msg = Float64MultiArray(); msg.data = tau_cmd.tolist(); self.pub.publish(msg)

        # G. LOGGING (Optimized Sliding Window)
        if len(self.log['t']) > 500: # Maintain a 500 point history buffer
            for k in self.log: self.log[k].pop(0)
            
        self.log['t'].append(t_clock)
        self.log['x'].append(x_2d[0]); self.log['y'].append(x_2d[1])
        self.log['xd'].append(xd[0]); self.log['yd'].append(xd[1])
        self.log['h'].append(h_val)
        self.log['mu'].append(np.linalg.norm(mu))

    def stop_robot(self):
        msg = Float64MultiArray(data=[0.0]*2); self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboResclfNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    
    ax_traj = fig.add_subplot(gs[:, 0]) 
    ax_h = fig.add_subplot(gs[0, 1])    
    ax_mu = fig.add_subplot(gs[1, 1])   
    plt.subplots_adjust(bottom=0.15)
    
    # Init Lines
    ln_a, = ax_traj.plot([], [], 'r-', linewidth=2, label='Actual')
    ln_t, = ax_traj.plot([], [], 'b--', linewidth=1, label='Target')
    
    # Safe Set
    theta = np.linspace(0, 2*np.pi, 200)
    rx, ry = node.cbf.radii[0], node.cbf.radii[1]
    n  = node.cbf.power_n
    x_b = rx * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** (2/n))
    y_b = ry * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** (2/n))
    ax_traj.plot(x_b, y_b, 'g-', label='Safe Set')
    
    ax_traj.set_xlim(-2.0, 2.0); ax_traj.set_ylim(-2.0, 2.0)
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.grid(True)
    ax_traj.legend()

    ln_h, = ax_h.plot([], [], 'g-'); ax_h.axhline(0, color='r', linestyle='--'); ax_h.set_title("Safety h(x)"); ax_h.grid(True)
    ln_mu, = ax_mu.plot([], [], 'k-'); ax_mu.set_title("Correction ||μ||"); ax_mu.grid(True)

    # Checkbutton
    ax_check = plt.axes([0.05, 0.02, 0.15, 0.05]) 
    check = CheckButtons(ax_check, ['Safety On'], [False])
    def toggle(label): node.cbf_active = not node.cbf_active
    check.on_clicked(toggle)

    def update(frame):
        # Retrieve the unified log
        if len(node.log['t']) == 0: return ln_a, ln_t, ln_h, ln_mu
        t_d = list(node.log['t'])
        x_d = list(node.log['x']); y_d = list(node.log['y'])
        xd_d = list(node.log['xd']); yd_d = list(node.log['yd'])
        h_d = list(node.log['h']); mu_d = list(node.log['mu'])

        # UPDATE LINES
        ln_a.set_data(x_d, y_d); ln_t.set_data(xd_d, yd_d)
        ln_h.set_data(t_d, h_d); ln_mu.set_data(t_d, mu_d)
        
        # SCROLL X-AXIS
        if len(t_d) > 0:
            ax_h.set_xlim(t_d[0], t_d[-1])
            ax_mu.set_xlim(t_d[0], t_d[-1])
            ax_h.set_ylim(-1.0, 1.0)
            ax_mu.set_ylim(-10.0, 20.0)

        return ln_a, ln_t, ln_h, ln_mu

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
    
    node.stop_robot(); node.destroy_node(); rclpy.shutdown(); t_ros.join()

if __name__ == '__main__':
    main()
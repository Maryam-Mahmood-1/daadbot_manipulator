import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

# --- CONFIGURATION ---
from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class CBF_SuperEllipsoid:
    """
    Implements the Super-Ellipsoid Safe Set defined in Section III.
    h(x) = 1 - [|x-xc|^n]^T A [|x-xc|^n]  (Eq 24)
    """
    def __init__(self, center, lengths, power_n=4):
        self.xc = np.array(center)
        self.dims = np.array(lengths) # [length, width, height]
        self.n = int(power_n) # "n" in the paper (e.g., 4, 8)
        
        # Matrix A (Eq 22/23 implies A has 1/dim^(2n) on diagonal)
        # The paper writes |(x-xc)/length|^(2n). 
        # Eq 24 uses |x-xc|^n. So A must scale by 1/dim^(2n).
        self.A_diag = 1.0 / (self.dims ** (2 * self.n))
        self.A = np.diag(self.A_diag)
        
        # Pole Placement for CBF (Eq 19 text)
        # "placed at lambda = -[p0, p1]"
        self.p0 = 10.0
        self.p1 = 10.0
        self.K = np.array([self.p0 * self.p1, self.p0 + self.p1]) # [k2, k1] for h + k1*h_dot + k2*h_ddot ?? 
        # Actually standard form is h_ddot + K*[h, h_dot] >= 0
        # If poles are p0, p1, characteristic eq is (s+p0)(s+p1) = s^2 + (p0+p1)s + p0p1
        # So h_ddot + (p0+p1)h_dot + (p0*p1)h >= 0

    def get_h(self, x):
        """ Eq 24 """
        diff = x - self.xc
        abs_diff_n = np.abs(diff) ** self.n
        term = abs_diff_n.T @ self.A @ abs_diff_n
        return 1.0 - term

    def get_h_dot(self, x, dx):
        """ Eq 25 """
        diff = x - self.xc
        # element-wise operations
        term1 = np.abs(diff)**self.n
        
        # "X_tilde" in paper (Eq 26 text definition)
        # sign(x-xc) * n * (x-xc)^(n-1) * dx
        # Note: (x-xc)^(n-1) must handle sign carefully if n is odd, 
        # but paper says n is even positive integer.
        # However, standard derivative of |x|^n is n*|x|^(n-1)*sgn(x).
        # The paper uses notation: sign(...) o n(x-xc)^(n-1). 
        # Assuming (x-xc) is the raw difference.
        
        # Let's strictly follow the math structure of derivative of (x^n)^T A (x^n)
        # Chain rule on |x_i - xc_i|^n -> n * |x_i - xc_i|^(n-1) * sign(x_i - xc_i) * dx_i
        
        # Precompute vector terms
        sign_diff = np.sign(diff)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        
        # Vector X_tilde (part of chain rule)
        # Corresponds to d/dt( |x-xc|^n )
        X_tilde = sign_diff * self.n * pow_n_1 * dx 
        
        # h_dot = -2 * [|x-xc|^n]^T * A * X_tilde
        # (Assuming A is constant, so A_dot is 0)
        h_dot = -2 * term1.T @ self.A @ X_tilde
        return h_dot, X_tilde

    def get_h_ddot_terms(self, x, dx):
        """
        Eq 26.
        Returns terms A_cbf, b_cbf such that h_ddot = A_cbf * x_ddot + b_cbf
        """
        diff = x - self.xc
        sign_diff = np.sign(diff)
        
        # Recompute X_tilde just in case
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        X_tilde = sign_diff * self.n * pow_n_1 * dx
        
        term_abs_n = np.abs(diff) ** self.n
        term_abs_n_half = np.abs(diff) ** (self.n / 2) # Only appears in Eq 26 visual? 
        # Wait, Eq 26 middle term: -2 [|x-xc|^(n/2)]^T A [...]
        # This seems like a typo in the paper or specific factorization.
        # Let's derive d/dt(X_tilde) directly, it's safer.
        
        # X_tilde = n * sign * |diff|^(n-1) * dx
        # d(X_tilde)/dt = n * sign * [ (n-1)|diff|^(n-2)*sign*dx ] * dx + n * sign * |diff|^(n-1) * x_ddot
        #               = n(n-1)|diff|^(n-2) * dx^2 + n * sign * |diff|^(n-1) * x_ddot
        
        # h_ddot = -2 * [ d/dt(|x|^n) @ A @ X_tilde + |x|^n @ A @ d/dt(X_tilde) ]
        # We know d/dt(|x|^n) is X_tilde.
        # So term 1: -2 * X_tilde.T @ A @ X_tilde
        
        term1 = -2 * X_tilde.T @ self.A @ X_tilde
        
        # Term 2: -2 * |x|^n @ A @ [ ... ]
        # Let's break d/dt(X_tilde) into drift (velocity) and control (acceleration) parts
        
        pow_n_2 = np.abs(diff) ** (self.n - 2)
        
        # Part dependent on velocity (dx^2)
        dX_tilde_drift = self.n * (self.n - 1) * pow_n_2 * (dx * dx) # dx*dx elementwise
        
        # Part dependent on acceleration (x_ddot)
        # Coeff = n * sign * |diff|^(n-1)
        dX_tilde_acc_coeff = self.n * sign_diff * pow_n_1
        
        # Combine into h_ddot structure
        # h_ddot = term1 - 2 * term_abs_n.T @ A @ (dX_tilde_drift + dX_tilde_acc_coeff * x_ddot)
        
        # Constant scalar part (drift)
        h_ddot_drift = term1 - 2 * term_abs_n.T @ self.A @ dX_tilde_drift
        
        # Gradient vector w.r.t x_ddot
        # Shape (3,)
        h_ddot_grad_x_ddot = -2 * term_abs_n.T @ self.A @ np.diag(dX_tilde_acc_coeff)
        
        return h_ddot_drift, h_ddot_grad_x_ddot

class RESCLF_CBF_Controller(Node):
    def __init__(self):
        super().__init__('resclf_cbf_node')

        self.urdf_path = URDF_PATH
        self.ee_frame_name = 'endeffector'
        
        # --- SAFE SET DEFINITION (Eq 22/24) ---
        # Defining a virtual box centered at [0, 0, 0.72]
        # Dimensions: +/- 0.3m in X and Y, +/- 0.4m in Z
        self.cbf = CBF_SuperEllipsoid(
            center=[0.0, 0.0, 0.72], 
            lengths=[0.3, 0.3, 0.4], 
            power_n=4
        )

        # --- RESCLF MATRICES ---
        self.dim = 3 
        self.F_mat = np.zeros((2*self.dim, 2*self.dim))
        self.F_mat[:self.dim, self.dim:] = np.eye(self.dim)
        self.G_mat = np.zeros((2*self.dim, self.dim))
        self.G_mat[self.dim:, :] = np.eye(self.dim)

        self.Q_are = np.eye(2*self.dim) * 1000.0
        self.R_are = np.eye(self.dim)   * 0.1
        self.P = solve_continuous_are(self.F_mat, self.G_mat, self.Q_are, self.R_are)
        
        min_eig_Q = np.min(np.linalg.eigvals(self.Q_are).real)
        max_eig_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma_clf = 1.0 * (min_eig_Q / max_eig_P)

        # --- ROBOT SETUP ---
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

        self.joint_indices_q = []
        self.joint_indices_v = []
        for name in TARGET_JOINTS:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.joint_indices_q.append(self.model.joints[joint_id].idx_q)
                self.joint_indices_v.append(self.model.joints[joint_id].idx_v)

        self.q = pin.neutral(self.model) 
        self.dq = np.zeros(self.model.nv) 
        self.received_first_state = False
        
        # Trajectory
        self.trajectory_period = 7.5
        self.center_z = 0.72
        self.center_pos = np.array([0.0, 0.0, self.center_z]) 
        self.ellipse_a = 0.25  # Increased to push boundary
        self.ellipse_b = 0.35  # Increased to VIOLATE safe set (0.3 limit)

        # --- ROS & PLOTTING ---
        self.dt = 0.002
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/effort_arm_controller/commands', 10)

        self.actual_x, self.actual_y = [], []
        self.target_x, self.target_y = [], []
        self.h_values = [] # To plot barrier value

        self.start_time = None
        self.control_timer = self.create_timer(self.dt, self.control_loop)

    def joint_state_callback(self, msg):
        msg_map = {name: i for i, name in enumerate(msg.name)}
        try:
            for i, joint_name in enumerate(TARGET_JOINTS):
                if joint_name in msg_map:
                    idx = msg_map[joint_name]
                    self.q[self.joint_indices_q[i]] = msg.position[idx]
                    self.dq[self.joint_indices_v[i]] = msg.velocity[idx]
            self.received_first_state = True
        except IndexError: pass

    def get_desired_state(self, t_rel):
        omega = 2 * math.pi / self.trajectory_period
        angle = omega * t_rel
        # Desired Trajectory (Ellipse)
        p_des = self.center_pos + np.array([
            self.ellipse_a * math.cos(angle),
            self.ellipse_b * math.sin(angle),
            0.0
        ])
        v_des = np.array([
            -self.ellipse_a * omega * math.sin(angle),
             self.ellipse_b * omega * math.cos(angle),
             0.0
        ])
        a_des = np.array([
            -self.ellipse_a * omega**2 * math.cos(angle),
            -self.ellipse_b * omega**2 * math.sin(angle),
             0.0
        ])
        return p_des, v_des, a_des

    def control_loop(self):
        if not self.received_first_state: return

        # 1. Update Dynamics
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        
        M = self.data.M
        nle = self.data.nle
        
        ee_pose = self.data.oMf[self.ee_frame_id]
        p_curr = ee_pose.translation
        
        J_full = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_linear = J_full[:3, :]
        v_curr = (J_full @ self.dq)[:3]
        
        drift_accel = pin.getFrameClassicalAcceleration(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ_dq = drift_accel.linear

        # 2. Reference Trajectory
        if self.start_time is None: self.start_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        p_des, v_des, a_des = self.get_desired_state(elapsed)

        # 3. CLF Formulation
        e = p_curr - p_des
        de = v_curr - v_des
        eta = np.hstack([e, de])
        
        V_val = eta.T @ self.P @ eta
        LfV = (2 * eta.T @ self.P) @ (self.F_mat @ eta)
        LgV = (2 * eta.T @ self.P) @ self.G_mat

        # 4. CBF Formulation (Super Ellipsoid)
        # h >= 0 is safe
        h_val = self.cbf.get_h(p_curr)
        h_dot, _ = self.cbf.get_h_dot(p_curr, v_curr)
        h_ddot_drift, h_ddot_grad = self.cbf.get_h_ddot_terms(p_curr, v_curr)

        # Store for plotting
        self.actual_x.append(p_curr[0])
        self.actual_y.append(p_curr[1])
        self.target_x.append(p_des[0])
        self.target_y.append(p_des[1])
        self.h_values.append(h_val)

        # 5. UNIFIED QP (Eq 29)
        # Variables: x = [mu_x, mu_y, mu_z, delta]
        # Minimize: mu^T mu + p * delta^2
        # Subject to:
        #   1. CLF: LfV + LgV*mu - delta <= -gamma*V  ->  LgV*mu - delta <= -gamma*V - LfV
        #   2. CBF: h_ddot + K*[h, h_dot] >= 0
        #      (drift + grad*(a_des + mu)) + k1*h_dot + k2*h >= 0
        #      grad*mu >= -drift - grad*a_des - k1*h_dot - k2*h
        
        p_weight = 1000.0 # Weight for relaxation delta

        def cost_func(x):
            mu = x[:3]
            delta = x[3]
            return 0.5 * np.sum(mu**2) + p_weight * (delta**2)

        # Constraint 1: CLF (Standard form C(x) >= 0)
        # Rearranged: -LgV*mu + delta - (LfV + gamma*V) <= 0 ??
        # Let's stick to scipy 'ineq' (fun(x) >= 0)
        # LfV + LgV*mu - delta <= -gamma*V
        # => -LfV - LgV*mu + delta - gamma*V >= 0 -- WRONG SIGN
        # Correct: (-LgV)mu + delta + (-LfV - gamma*V) >= 0  ... Wait
        # Equation: LgV*mu - delta <= Bound
        # Ineq: Bound - (LgV*mu - delta) >= 0
        clf_bound = -self.gamma_clf * V_val - LfV
        def cons_clf(x):
            mu = x[:3]
            delta = x[3]
            return clf_bound - (LgV @ mu - delta)

        # Constraint 2: CBF
        # h_ddot = drift + grad @ (a_des + mu)
        # Ineq: h_ddot + k_1 * h_dot + k_2 * h >= 0
        k1 = self.cbf.K[1]
        k2 = self.cbf.K[0]
        cbf_rhs = -h_ddot_drift - (h_ddot_grad @ a_des) - k1*h_dot - k2*h_val
        # grad @ mu >= cbf_rhs
        # grad @ mu - cbf_rhs >= 0
        def cons_cbf(x):
            mu = x[:3]
            return (h_ddot_grad @ mu) - cbf_rhs

        cons = [
            {'type': 'ineq', 'fun': cons_clf},
            {'type': 'ineq', 'fun': cons_cbf}
        ]
        
        x0 = np.zeros(4)
        res = minimize(cost_func, x0, constraints=cons, method='SLSQP')

        if res.success:
            mu = res.x[:3]
        else:
            mu = -10.0*e - 5.0*de # Fallback
            # self.get_logger().warn("QP Failed")

        # 6. Compute Torque (Eq 9)
        # Gamma = M * J_pinv * (x_des_ddot + mu - dJ_dq) + nle
        J_pinv = np.linalg.pinv(J_linear)
        x_ddot_command = a_des + mu
        q_ddot_command = J_pinv @ (x_ddot_command - dJ_dq)
        tau_total = (M @ q_ddot_command) + nle

        # Publish
        tau_output = np.clip(tau_total[self.joint_indices_v], -50.0, 50.0)
        msg = Float64MultiArray(data=tau_output.tolist())
        self.torque_pub.publish(msg)

    def stop_robot(self):
        self.torque_pub.publish(Float64MultiArray(data=[0.0]*7))

def main():
    rclpy.init()
    node = RESCLF_CBF_Controller()
    
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()

    # VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Trajectory Plot
    ln_target, = ax1.plot([], [], 'b--', label='Target')
    ln_actual, = ax1.plot([], [], 'r-', label='Robot')
    
    # Draw Safe Set Boundary (approx rectangle for n=4)
    # Just simple box visualization
    rect = plt.Rectangle((-0.3, -0.3), 0.6, 0.6, linewidth=2, edgecolor='g', facecolor='none', label='Safe Set')
    ax1.add_patch(rect)
    
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title('Operational Space (XY)')
    ax1.legend()
    ax1.grid(True)

    # 2. Barrier Function Value
    ln_h, = ax2.plot([], [], 'g-', label='h(x)')
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlim(0, 300) # frames
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title('Barrier Function h(x) >= 0')
    ax2.grid(True)

    def update(frame):
        ln_target.set_data(node.target_x, node.target_y)
        ln_actual.set_data(node.actual_x, node.actual_y)
        
        y_data = list(node.h_values)
        x_data = range(len(y_data))
        ln_h.set_data(x_data, y_data)
        if len(x_data) > 0: ax2.set_xlim(max(0, len(x_data)-300), len(x_data)+10)
        
        return ln_target, ln_actual, ln_h,

    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import pinocchio as pin
# import numpy as np
# import math
# import threading
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.linalg import solve_continuous_are
# from scipy.optimize import minimize

# # --- CONFIGURATION ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# TARGET_JOINTS = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# class RESCLF_OperationalSpace(Node):
#     def __init__(self):
#         super().__init__('resclf_operational_space')

#         self.urdf_path = URDF_PATH
#         self.ee_frame_name = 'endeffector'
        
#         # --- TRAJECTORY TUNING ---
#         self.trajectory_period = 7.5 
#         self.center_z = 0.72
#         self.center_pos = np.array([0.0, 0.0, self.center_z]) 
#         self.ellipse_a = 0.150  
#         self.ellipse_b = 0.270  

#         # --- RESCLF SETUP (Matrices F, G, P) ---
#         # State dimension for Position control (x, y, z) is 3.
#         # Total error state eta has dimension 6 (3 pos error + 3 vel error).
#         self.dim = 3 
        
#         # Define F and G according to Eq (12)
#         # F = [0  I]
#         #     [0  0]
#         self.F_mat = np.zeros((2*self.dim, 2*self.dim))
#         self.F_mat[:self.dim, self.dim:] = np.eye(self.dim)
        
#         # G = [0]
#         #     [I]
#         self.G_mat = np.zeros((2*self.dim, self.dim))
#         self.G_mat[self.dim:, :] = np.eye(self.dim)

#         # Solve Algebraic Riccati Equation (ARE) - Eq (13)
#         # F.T*P + P*F - P*G*G.T*P + Q = 0
#         # We choose Q and R (where R=Identity implies the G*G.T term)
#         self.Q_are = np.eye(2*self.dim) * 1000.0  # Tunable weight on error convergence
#         self.R_are = np.eye(self.dim)   *0.1         # Weight on control effort
        
#         # Solve for P
#         self.P = solve_continuous_are(self.F_mat, self.G_mat, self.Q_are, self.R_are)
        
#         # Calculate Gamma for RESCLF constraint - Eq (15)
#         # gamma = min_eig(Q) / max_eig(P)
#         min_eig_Q = np.min(np.linalg.eigvals(self.Q_are).real)
#         max_eig_P = np.max(np.linalg.eigvals(self.P).real)
#         # self.gamma_clf = min_eig_Q / max_eig_P
#         calculated_gamma = min_eig_Q / max_eig_P
#         self.gamma_clf = 1.8 * calculated_gamma  # Slightly more aggressive than theoretical minimum

        
#         # --- ROBOT SETUP ---
#         self.model = pin.buildModelFromUrdf(self.urdf_path)
#         self.data = self.model.createData()
#         self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

#         self.joint_indices_q = []
#         self.joint_indices_v = []
#         for name in TARGET_JOINTS:
#             if self.model.existJointName(name):
#                 joint_id = self.model.getJointId(name)
#                 self.joint_indices_q.append(self.model.joints[joint_id].idx_q)
#                 self.joint_indices_v.append(self.model.joints[joint_id].idx_v)

#         self.q = pin.neutral(self.model) 
#         self.dq = np.zeros(self.model.nv) 
#         self.received_first_state = False
#         self.start_approach_pos = None

#         # --- PLOTTING ---
#         self.actual_x, self.actual_y = [], []
#         self.target_x, self.target_y = [], []

#         # --- ROS COMMUNICATION ---
#         self.dt = 0.002 
#         self.joint_state_sub = self.create_subscription(
#             JointState, '/joint_states', self.joint_state_callback, 10)
#         self.torque_pub = self.create_publisher(
#             Float64MultiArray, '/effort_arm_controller/commands', 10)

#         self.start_time = None
#         self.control_timer = self.create_timer(self.dt, self.control_loop)

#     def joint_state_callback(self, msg):
#         msg_map = {name: i for i, name in enumerate(msg.name)}
#         try:
#             for i, joint_name in enumerate(TARGET_JOINTS):
#                 if joint_name in msg_map:
#                     msg_idx = msg_map[joint_name]
#                     pin_q_idx = self.joint_indices_q[i]
#                     pin_v_idx = self.joint_indices_v[i]
#                     self.q[pin_q_idx] = msg.position[msg_idx]
#                     self.dq[pin_v_idx] = msg.velocity[msg_idx]
#             self.received_first_state = True
#         except IndexError:
#             pass

#     def get_desired_state(self, t_rel):
#         omega = 2 * math.pi / self.trajectory_period
#         angle = omega * t_rel
#         sin_a = math.sin(angle)
#         cos_a = math.cos(angle)

#         # Position
#         tx = self.center_pos[0] + self.ellipse_a * cos_a
#         ty = self.center_pos[1] + self.ellipse_b * sin_a
#         tz = self.center_z
#         p_des = np.array([tx, ty, tz])
        
#         # Velocity
#         vx = -self.ellipse_a * omega * sin_a
#         vy =  self.ellipse_b * omega * cos_a
#         vz = 0.0
#         v_des = np.array([vx, vy, vz])

#         # Acceleration (x_des_ddot)
#         ax = -self.ellipse_a * (omega**2) * cos_a
#         ay = -self.ellipse_b * (omega**2) * sin_a
#         az = 0.0
#         a_des = np.array([ax, ay, az])

#         return p_des, v_des, a_des

#     def control_loop(self):
#         if not self.received_first_state:
#             return

#         # 1. DYNAMICS & KINEMATICS UPDATE (Eq 1 & 2)
#         pin.computeAllTerms(self.model, self.data, self.q, self.dq)
#         pin.updateFramePlacements(self.model, self.data)
        
#         M = self.data.M      # Mass Matrix
#         nle = self.data.nle  # C(q,dq)dq + G(q)
        
#         # Operational Space State (x, v)
#         ee_pose = self.data.oMf[self.ee_frame_id]
#         p_curr = ee_pose.translation
        
#         J_full = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
#         J_linear = J_full[:3, :] # We are controlling XYZ only
        
#         v_curr_spatial = J_full @ self.dq
#         v_curr = v_curr_spatial[:3]

#         # Jacobian Drift (dJ * dq) - Needed for Eq (3) & (9)
#         drift_accel = pin.getFrameClassicalAcceleration(
#             self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
#         )
#         dJ_dq = drift_accel.linear

#         # 2. TRAJECTORY GENERATION
#         if self.start_time is None:
#             self.start_time = self.get_clock().now().nanoseconds / 1e9
        
#         elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        
#         # Simple Logic: Move to start (5s) then circle
#         if elapsed < 5.0:
#             # Move to start of ellipse
#             start_pt = self.center_pos + np.array([self.ellipse_a, 0, 0])
#             if self.start_approach_pos is None: self.start_approach_pos = p_curr
            
#             ratio = elapsed / 5.0
#             sm_ratio = (1 - math.cos(ratio * math.pi)) / 2
#             p_des = (1 - sm_ratio)*self.start_approach_pos + sm_ratio*start_pt
#             v_des, a_des = np.zeros(3), np.zeros(3)
#         else:
#             p_des, v_des, a_des = self.get_desired_state(elapsed - 5.0)

#         # Plotting storage
#         self.actual_x.append(p_curr[0])
#         self.actual_y.append(p_curr[1])
#         self.target_x.append(p_des[0])
#         self.target_y.append(p_des[1])

#         # 3. RESCLF FORMULATION
        
#         # A. Error Vector (Eq 11 Context)
#         # Note: Theory uses e = x - x_des. 
#         e = p_curr - p_des
#         de = v_curr - v_des
#         eta = np.hstack([e, de]) # 6x1 vector

#         # B. Lyapunov Function & Lie Derivatives (Eq 14)
#         V_val = eta.T @ self.P @ eta
        
#         # Gradient of V w.r.t eta: (eta.T * P + eta.T * P.T) -> 2 * eta.T * P (since P symmetric)
#         dV_deta = 2 * eta.T @ self.P
        
#         # Lie Derivatives
#         LfV = dV_deta @ (self.F_mat @ eta)
#         LgV = dV_deta @ self.G_mat # Shape (3,)

#         # C. QP Solver for auxiliary input mu
#         # Constraint (Eq 15): LfV + LgV*mu <= -gamma * V
#         # Rearranged: -LgV*mu >= LfV + gamma*V
#         # Standard form (C*x >= b): ( -LgV ) * mu >= ( LfV + gamma*V )
        
#         def cost_func(mu):
#             return 0.5 * np.sum(mu**2) # Min Norm

#         # Constraint: LfV + LgV @ mu + gamma * V <= 0
#         # => upper_bound - LgV @ mu >= 0
#         upper_bound = -self.gamma_clf * V_val - LfV

#         def constraint_clf(mu):
#             return upper_bound - (LgV @ mu)

#         cons = {'type': 'ineq', 'fun': constraint_clf}
#         mu0 = np.zeros(3)
        
#         res = minimize(cost_func, mu0, constraints=cons, method='SLSQP')
        
#         if res.success:
#             mu = res.x
#         else:
#             # Fallback (Robustness)
#             mu = -10.0*e - 5.0*de 

#         # 4. CONTROL LAW (Eq 9)
#         # Gamma = M * J_dag * (x_des_ddot + mu - dJ_dq) + C*dq + G
        
#         # Pseudo-Inverse of Jacobian (Linear part only for XYZ control)
#         J_pinv = np.linalg.pinv(J_linear)
        
#         # The desired acceleration in operational space
#         # x_ddot = x_des_ddot + mu
#         x_ddot_command = a_des + mu
        
#         # Transform to Joint Space
#         # J_dag * (x_ddot - dJ_dq)
#         q_ddot_command = J_pinv @ (x_ddot_command - dJ_dq)
        
#         # Inverse Dynamics: Tau = M * q_ddot + nle
#         # Note: In theory Eq (9), they multiply M * J_dag * (...). 
#         # This is equivalent to calculating q_ddot first then multiplying M.
#         tau_total = (M @ q_ddot_command) + nle

#         # 5. SEND TORQUES
#         tau_output = []
#         for i in range(len(TARGET_JOINTS)):
#             idx = self.joint_indices_v[i]
#             tau_output.append(tau_total[idx])

#         # Safety Clip
#         tau_output = np.clip(tau_output, -50.0, 50.0)

#         msg = Float64MultiArray()
#         msg.data = tau_output.tolist()
#         self.torque_pub.publish(msg)

#     def stop_robot(self):
#         msg = Float64MultiArray()
#         msg.data = [0.0] * 7
#         self.torque_pub.publish(msg)

# # --- MAIN ---
# def main(args=None):
#     rclpy.init(args=args)
#     node = RESCLF_OperationalSpace()

#     spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     spin_thread.start()

#     # Setup Plot
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ln_target, = ax.plot([], [], 'b--', linewidth=2, label='Target')
#     ln_actual, = ax.plot([], [], 'r-', linewidth=2, label='Actual (RESCLF)')
    
#     ax.set_xlim(-0.5, 0.5)
#     ax.set_ylim(-0.5, 0.5)
#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_title('RESCLF Operational Space Tracking')
#     ax.legend()
#     ax.grid(True)
#     ax.set_aspect('equal')

#     def init_plot():
#         ln_target.set_data([], [])
#         ln_actual.set_data([], [])
#         return ln_target, ln_actual

#     def update_plot(frame):
#         tx = list(node.target_x)
#         ty = list(node.target_y)
#         ax_dat = list(node.actual_x)
#         ay_dat = list(node.actual_y)

#         ln_target.set_data(tx, ty)
#         ln_actual.set_data(ax_dat, ay_dat)
#         return ln_target, ln_actual

#     ani = FuncAnimation(fig, update_plot, init_func=init_plot, blit=True, interval=50)

#     try:
#         plt.show()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
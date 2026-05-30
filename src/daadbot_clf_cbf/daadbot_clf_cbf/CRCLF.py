import pinocchio as pin
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# ==========================================
# 1. ROBOT SYSTEM
# ==========================================
class RobotSystem:
    def __init__(self, urdf_path, controlled_joints):
        full_model = pin.buildModelFromUrdf(urdf_path)
        joints_to_lock = []
        for joint_name in full_model.names:
            if joint_name not in controlled_joints and joint_name != "universe":
                joints_to_lock.append(full_model.getJointId(joint_name))
        
        q_ref = pin.neutral(full_model) 
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

    def get_dynamics(self, q, dq):
        pin.computeAllTerms(self.model, self.data, q, dq)
        return self.data.M, self.data.nle

# ==========================================
# 2. ROBUST RESCLF CONTROLLER
# ==========================================
class RESCLF_Controller:
    def __init__(self, num_joints, gamma=1.0):
        self.n = num_joints
        self.gamma = gamma 
        
        # Linear System
        self.A = np.zeros((2 * self.n, 2 * self.n))
        self.A[:self.n, self.n:] = np.eye(self.n)
        self.B = np.zeros((2 * self.n, self.n))
        self.B[self.n:, :] = np.eye(self.n)
        
        # Tuning
        Kp = 4.0 * np.eye(self.n) 
        Kd = 3.0 * np.eye(self.n)
        K = np.hstack([Kp, Kd])
        
        A_cl = self.A - self.B @ K
        Q = np.eye(2 * self.n)
        self.P = solve_continuous_lyapunov(A_cl.T, -Q)

    def V_func(self, z):
        return z.T @ self.P @ z

    def get_control(self, robot, q, dq, q_des):
        error = q - q_des
        z = np.hstack([error, dq])

        dV_dz = 2 * z.T @ self.P
        LfV = dV_dz @ (self.A @ z)
        LgV = dV_dz @ self.B
        V_val = self.V_func(z)
        
        # Robust QP with Slack
        p_slack = 1.0e6
        upper_bound = -self.gamma * V_val - LfV
        
        def cost(x):
            u = x[:self.n]
            delta = x[self.n]
            return 0.5 * np.sum(u**2) + p_slack * (delta**2)
        
        def constraint(x):
            u = x[:self.n]
            delta = x[self.n]
            return upper_bound - (LgV @ u - delta)
            
        x0 = np.zeros(self.n + 1)
        res = minimize(cost, x0, constraints={'type': 'ineq', 'fun': constraint}, method='SLSQP')
        
        u_aux = res.x[:self.n] if res.success else np.zeros(self.n)
        M, nle = robot.get_dynamics(q, dq)
        tau = M @ u_aux + nle
        
        return tau, V_val

# ==========================================
# 3. MAIN (With Debug Printing)
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node('resclf_visualizer')
    pub = node.create_publisher(JointState, 'joint_states', 10)
    
    # --- SETUP PATHS ---
    from ament_index_python.packages import get_package_share_directory
    import os

    URDF_PATH = os.path.join(
        get_package_share_directory("daadbot_desc"),
        "urdf",
        "urdf_inverted_torque",
        "daadbot.urdf"
    )
 
    
    # Must match your URDF names exactly!
    target_joints = [
        "joint_1", "joint_2", "joint_3", 
        "joint_4", "joint_5", "joint_6", "joint_7"
    ]
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    robot = RobotSystem(urdf_path, target_joints)
    ctrl = RESCLF_Controller(num_joints=7, gamma=3.0)
    
    # --- INITIALIZATION ---
    q = np.random.uniform(-0.5, 0.5, 7)
    dq = np.zeros(7)
    
    # Define your specific target angles here (in radians)
    q_des = np.array([0.5, -0.3, 1.0, -1.5, 0.2, 1.0, 0.0])
    
    dt = 0.002 # 500Hz
    iter_count = 0
    
    print("Publishing to /joint_states... Check RViz!")
    print(f"Target Config: {np.array2string(q_des, precision=3, separator=',')}")

    try:
        while rclpy.ok():
            # 1. Physics & Control
            tau, V = ctrl.get_control(robot, q, dq, q_des)
            M, nle = robot.get_dynamics(q, dq)
            ddq = np.linalg.solve(M, tau - nle)
            
            dq += ddq * dt
            q += dq * dt
            
            # 2. Visualization
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = target_joints
            msg.position = q.tolist()
            pub.publish(msg)
            
            # 3. DEBUG PRINTING (Every 50 steps / 0.1 seconds)
            if iter_count % 50 == 0:
                err_norm = np.linalg.norm(q - q_des)
                
                print(f"\n--- Iter {iter_count} ---")
                print(f"Lyapunov V(x): {V:.6f}")
                print(f"Error Norm:    {err_norm:.6f}")
                
                # Print nicely formatted arrays
                # We use np.array2string to prevent giant brackets
                print(f"Target:  {np.array2string(q_des, precision=3, suppress_small=True)}")
                print(f"Current: {np.array2string(q, precision=3, suppress_small=True)}")
                
                # Stop printing if converged to avoid spamming console
                if err_norm < 1e-4:
                    print(">>> CONVERGED <<<")
                    # Optional: uncomment break to stop script
                    # break 

            iter_count += 1
            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
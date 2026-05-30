import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Controller:
    """
    Conformally Robust Rapidly Exponentially Stabilizing CLF (CR-RES-CLF)
    
    Standard Constraint:
        LfV + LgV μ ≤ -γ V
        
    Robust Constraint (Paper Eq 4 / Def 5):
        LfV + LgV μ + ||∂V/∂x|| * q ≤ -γ V
    
    Rearranged for QP (Ax ≤ b):
        LgV μ ≤ -γ V - LfV - (||∂V/∂x|| * q)
    """
    def __init__(self, dim_task=3, kp=3.0, kv=3.0):
        self.dim = dim_task
        
        # 1. System Matrices for Error Dynamics
        zero = np.zeros((dim_task, dim_task))
        eye  = np.eye(dim_task)
        
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # 2. Optimal Control Weights (LQR)
        q_pos = 3000.0
        q_vel = 1500.0
        
        self.Q_mat = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        R_mat = np.eye(dim_task) * 0.00001
        
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, R_mat)
        
        # 3. Decay Rate
        eig_Q = np.min(np.linalg.eigvals(self.Q_mat).real)
        eig_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 3.9 * (eig_Q / eig_P)
        
        self.kp = kp
        self.kv = kv

    def get_nominal_acceleration(self, x, dx, x_des, dx_des):
        e = x - x_des
        de = dx - dx_des
        return -self.kp * e - self.kv * de

    def get_lyapunov_constraints(self, x, dx, x_des, dx_des, q_quantile=0.0):
        """
        Computes Conformally Robust Lie Derivatives.
        
        Args:
            q_quantile (float): The conformal uncertainty bound (q_1-delta).
                                If 0.0, reduces to standard CLF.
        """
        e = x - x_des
        de = dx - dx_des
        eta = np.hstack((e, de)).reshape(-1, 1)
        # print("quantile inside clf:", q_quantile)

        # A. Standard Terms
        V = (eta.T @ self.P @ eta)[0, 0]
        LfV = (eta.T @ (self.P @ self.F + self.F.T @ self.P) @ eta)[0, 0]
        LgV = 2 * eta.T @ self.P @ self.G

        # B. Robustness Term Calculation
        # Gradient of V w.r.t state eta: ∇V = 2 * P * eta
        grad_V = 2 * (self.P @ eta)
        
        # The uncertainty enters at the acceleration level (the bottom half of eta's dynamics).
        # We need the norm of the gradient components corresponding to acceleration.
        # Since eta = [e, de], the "actuated" rows are the last 'dim' rows.
        grad_V_actuated = grad_V[self.dim:, 0] # Extract bottom half
        
        # Robust Term = ||∇V_act|| * q
        robustness_cost = np.linalg.norm(grad_V_actuated) * q_quantile
        
        # We return the cost separately so the QP setup can subtract it from 'b'
        # b_clf = -gamma*V - LfV - robustness_cost
        print("Robustness cost in CLF:", robustness_cost)
        print("Gradient V in CLF:", grad_V_actuated)
        
        return LfV, LgV, V, self.gamma, robustness_cost
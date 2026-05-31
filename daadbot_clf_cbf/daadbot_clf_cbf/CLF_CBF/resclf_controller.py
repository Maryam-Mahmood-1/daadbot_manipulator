import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Controller:
    """
    Rapidly Exponentially Stabilizing Control Lyapunov Function (RES-CLF)
    
    Verified Math:
    • Error State:       η = [e, ė]ᵀ
    • Lyapunov Function: V(η) = ηᵀ P η
    • Stability:         V̇(η) + γ V(η) ≤ 0
    • QP Constraint:     LfV + LgV μ ≤ -γ V
    """
    def __init__(self, dim_task=3, kp=0.0, kv=0.0):
        self.dim = dim_task
        
        # ---------------------------------------------------------
        # 1. System Matrices for Error Dynamics
        #    η̇ = F η + G μ
        #    F = [[0, I], [0, 0]],  G = [[0], [I]]
        # ---------------------------------------------------------
        zero = np.zeros((dim_task, dim_task))
        eye  = np.eye(dim_task)
        
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # ---------------------------------------------------------
        # 2. Solve Algebraic Riccati Equation (ARE) for P
        #    FᵀP + PF - P G R⁻¹ Gᵀ P + Q = 0
        #    (Optimal solution for LQR -> V(η) is a valid CLF)
        # ---------------------------------------------------------
        # self.Q_mat = np.eye(2 * dim_task) * 10.0
        q_pos = 3000.0
        
        # 2. Velocity Weight (LOWER THIS): 10.0
        #    Was 1000.0. Dropping it prevents the controller from reacting to noise.
        q_vel = 1500.0
        
        # Build Diagonal Matrix
        self.Q_mat = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        R_mat = np.eye(dim_task)*0.00001
        
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, R_mat)
        
        # ---------------------------------------------------------
        # 3. Compute Decay Rate γ
        #    γ = min(eig(Q)) / max(eig(P))
        #    (Ensures exponential convergence rate)
        # ---------------------------------------------------------
        eig_Q = np.min(np.linalg.eigvals(self.Q_mat).real)
        eig_P = np.max(np.linalg.eigvals(self.P).real)
        # self.gamma = eig_Q / eig_P
        self.gamma = 3.9 * (eig_Q / eig_P)
        print(f"[RES-CLF] Decay Rate γ: {self.gamma:.3f}")
        
        # PD Gains for Nominal Control
        self.kp = kp
        self.kv = kv

    def get_nominal_acceleration(self, x, dx, x_des, dx_des):
        """
        Calculates Nominal Control (u_nom):
        u_nom = -Kp e - Kv ė
        """
        e = x - x_des
        de = dx - dx_des
        return -self.kp * e - self.kv * de

    def get_lyapunov_constraints(self, x, dx, x_des, dx_des):
        """
        Computes Lie Derivatives for QP constraints.
        
        Constraint Equation:
        LfV + LgV μ ≤ -γ V
        
        Returns:
            LfV (scalar), LgV (1x3 vector), V (scalar), gamma
        """
        e = x - x_des
        de = dx - dx_des
        eta = np.hstack((e, de)).reshape(-1, 1) # State vector η = [e; ė]

        # ---------------------------------------------------------
        # Lyapunov Function Value
        # V(η) = ηᵀ P η
        # ---------------------------------------------------------
        V = (eta.T @ self.P @ eta)[0, 0]

        # ---------------------------------------------------------
        # Lie Derivative along System Dynamics (Uncontrolled)
        # LfV = ∇V ⋅ F η
        #     = (2 P η)ᵀ (F η)
        #     = ηᵀ (P F + Fᵀ P) η
        # ---------------------------------------------------------
        LfV = (eta.T @ (self.P @ self.F + self.F.T @ self.P) @ eta)[0, 0]

        # ---------------------------------------------------------
        # Lie Derivative along Control Input
        # LgV = ∇V ⋅ G
        #     = (2 P η)ᵀ G
        #     = 2 ηᵀ P G
        # ---------------------------------------------------------
        LgV = 2 * eta.T @ self.P @ self.G
        
        return LfV, LgV, V, self.gamma
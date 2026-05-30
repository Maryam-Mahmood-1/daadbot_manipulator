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
    def __init__(self, dim_task=2, kp=10.0, kv=10.0):
        self.dim = dim_task
        
        # ---------------------------------------------------------
        # 1. System Matrices for Error Dynamics
        #    F (System) and G (Input) scale automatically with dim_task
        # ---------------------------------------------------------
        zero = np.zeros((dim_task, dim_task))
        eye  = np.eye(dim_task)
        
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # ---------------------------------------------------------
        # 2. Solve Algebraic Riccati Equation (ARE)
        # ---------------------------------------------------------
        q_pos = 1000.0
        q_vel = 500.0
        
        # --- FIX: DYNAMIC Q MATRIX CONSTRUCTION ---
        # This creates a list of length 2*dim_task automatically.
        # For dim_task=2, it makes [3000, 3000, 1500, 1500] (4 elements)
        q_diagonal = [q_pos] * dim_task + [q_vel] * dim_task
        
        self.Q_mat = np.diag(q_diagonal)
        R_mat = np.eye(dim_task) * 0.00001
        
        # Now F and Q are guaranteed to have the same shape (2*dim x 2*dim)
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, R_mat)
        
        # ---------------------------------------------------------
        # 3. Compute Decay Rate γ
        # ---------------------------------------------------------
        eig_Q = np.min(np.linalg.eigvals(self.Q_mat).real)
        eig_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 1.8 * (eig_Q / eig_P)
        print(f"[RES-CLF] Decay Rate γ: {self.gamma:.3f}")
        
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
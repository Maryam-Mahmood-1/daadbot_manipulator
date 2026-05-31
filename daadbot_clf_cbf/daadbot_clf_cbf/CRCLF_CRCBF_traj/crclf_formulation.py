import numpy as np
from scipy.linalg import solve_continuous_are

class CRCLF_Formulation:
    """
    Module 2: CR-CLF Stability Mathematics (Robust).
    Implements Eq (11) through (15) with Conformal Robustness term.
    """
    def __init__(self, dim=3, q_cp=0.0):
        self.dim = dim
        self.q_cp = q_cp  # Conformal Quantile
        
        # Matrices F and G (Eq 12)
        self.F = np.zeros((2*dim, 2*dim))
        self.F[:dim, dim:] = np.eye(dim)
        self.G = np.zeros((2*dim, dim))
        self.G[dim:, :] = np.eye(dim)
        
        # --- OPTIMIZED TUNING: Aggressive (From Normal Code) ---
        # Matches the high-performance tracking of your monolithic script.
        # Q = 1000 (High Accuracy), R = 0.1 (Cheap Control Effort)
        q_pos = 100.0
        
        # 2. Velocity Weight (LOWER THIS): 10.0
        #    Was 1000.0. Dropping it prevents the controller from reacting to noise.
        q_vel = 30.0
        
        # Build Diagonal Matrix
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        # self.Q = np.eye(2*dim) * 100.0 
        self.R = np.eye(dim) * 0.1
        
        # Solve ARE
        self.P = solve_continuous_are(self.F, self.G, self.Q, self.R)
        
        # Calculate Gamma
        min_Q = np.min(np.linalg.eigvals(self.Q).real)
        max_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 1.8 * (min_Q / max_P)

    def get_qp_constraints(self, e, de):
        """
        Calculates Lie Derivatives, V, and the Robustness Term.
        Returns LfV, LgV, V_val, gamma, robust_term
        """
        eta = np.hstack([e, de]) # Error state vector
        
        # Lyapunov Value
        V_val = eta.T @ self.P @ eta
        
        # Gradient dV/d_eta = 2 * eta.T * P
        dV_deta = 2 * eta.T @ self.P
        
        # Lie Derivatives
        LfV = dV_deta @ (self.F @ eta)
        LgV = dV_deta @ self.G  # Shape (dim,)
        
        # --- ROBUSTNESS LOGIC (Preserved) ---
        # The worst-case V increase due to uncertainty: ||LgV|| * q
        robust_term = np.linalg.norm(LgV) * self.q_cp
        
        return LfV, LgV, V_val, self.gamma, robust_term




# import numpy as np
# from scipy.linalg import solve_continuous_are

# class CRCLF_Formulation:
#     """
#     Module 2: CR-CLF Stability Mathematics (Robust).
#     Implements Eq (11) through (15) with Conformal Robustness term.
#     """
#     def __init__(self, dim=3, q_cp=0.0):
#         self.dim = dim
#         self.q_cp = q_cp  # <--- NEW: Store the quantile value
        
#         # Matrices F and G (Eq 12)
#         self.F = np.zeros((2*dim, 2*dim))
#         self.F[:dim, dim:] = np.eye(dim)
#         self.G = np.zeros((2*dim, dim))
#         self.G[dim:, :] = np.eye(dim)
        
#         # Weights for ARE (Eq 13)
#         self.Q = np.eye(2*dim) * 300.0 
#         self.R = np.eye(dim) * 20.0
        
#         # Solve ARE: F.T*P + P*F - P*G*R_inv*G.T*P + Q = 0
#         self.P = solve_continuous_are(self.F, self.G, self.Q, self.R)
        
#         # Calculate Gamma for Stability Condition (Eq 15)
#         min_Q = np.min(np.linalg.eigvals(self.Q).real)
#         max_P = np.max(np.linalg.eigvals(self.P).real)
#         self.gamma = 1.8 * (min_Q / max_P)

#     def get_qp_constraints(self, e, de):
#         """
#         Calculates Lie Derivatives, V, and the Robustness Term.
#         Returns LfV, LgV, V_val, gamma, robust_term
#         """
#         eta = np.hstack([e, de]) # Error state vector
        
#         # Lyapunov Value
#         V_val = eta.T @ self.P @ eta
        
#         # Gradient dV/d_eta = 2 * eta.T * P
#         dV_deta = 2 * eta.T @ self.P
        
#         # Lie Derivatives
#         LfV = dV_deta @ (self.F @ eta)
#         LgV = dV_deta @ self.G  # Shape (dim,), e.g., [x_sens, y_sens, z_sens]
        
#         # --- NEW: Robustness Term ---
#         # The paper's term is || dV/dx || * q. 
#         # Since our error is in acceleration (matched with G), we use ||LgV|| * q.
#         # This represents the worst-case destruction of V caused by the noise.
#         robust_term = np.linalg.norm(LgV) * self.q_cp
        
#         return LfV, LgV, V_val, self.gamma, robust_term
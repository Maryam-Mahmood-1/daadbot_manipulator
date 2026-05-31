import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Formulation:
    """
    Module 2: RESCLF Stability Mathematics.
    """
    def __init__(self, dim=3):
        self.dim = dim
        
        # Matrices F and G (Identical to monolithic)
        self.F = np.zeros((2*dim, 2*dim))
        self.F[:dim, dim:] = np.eye(dim)
        self.G = np.zeros((2*dim, dim))
        self.G[dim:, :] = np.eye(dim)
        
        # --- UPDATE: Match Monolithic Aggressiveness ---
        # Old: Q=300, R=1.0 (Too lazy)
        # New: Q=1000, R=0.1 (High accuracy, cheap torque)
        self.Q = np.eye(2*dim) * 1000.0 
        self.R = np.eye(dim) * 0.1
        
        # Solve ARE
        self.P = solve_continuous_are(self.F, self.G, self.Q, self.R)
        
        # Gamma Calculation (Identical)
        min_Q = np.min(np.linalg.eigvals(self.Q).real)
        max_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 1.8 * (min_Q / max_P)

    def get_qp_constraints(self, e, de):
        """ Identical Math to Monolithic """
        eta = np.hstack([e, de]) 
        V_val = eta.T @ self.P @ eta
        dV_deta = 2 * eta.T @ self.P
        LfV = dV_deta @ (self.F @ eta)
        LgV = dV_deta @ self.G
        
        return LfV, LgV, V_val, self.gamma
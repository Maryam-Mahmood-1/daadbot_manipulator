import numpy as np

class CRCBF_Formulation:
    """
    Module 4: CR-CBF (Conformally Robust Safety).
    Implements Super-Ellipsoid Safe Set with Uncertainty Margins.
    """
    def __init__(self, center, lengths, power_n=4, q_cp=0.0):
        self.xc = np.array(center)
        self.n = int(power_n)
        self.q_cp = q_cp  # Conformal Quantile
        
        # Matrix A setup
        self.dims = np.array(lengths)
        self.A_diag = 1.0 / (self.dims ** (2 * self.n))
        self.A = np.diag(self.A_diag)

        # --- OPTIMIZED TUNING: Pole Placement ---
        # Steal this from the monolithic code.
        # Places poles at -10, -10 for critical damping.
        p0 = 30.0
        p1 = 30.0
        
        self.k1 = p0 + p1  # 20.0 (Damping)
        self.k0 = p0 * p1  # 100.0 (Stiffness)

    def get_h(self, x):
        """ Barrier Value h(x) """
        diff = x - self.xc
        abs_diff_n = np.abs(diff) ** self.n
        term = abs_diff_n.T @ self.A @ abs_diff_n
        return 1.0 - term

    def get_h_dot(self, x, dx):
        """ Time Derivative h_dot(x) """
        diff = x - self.xc
        sign_diff = np.sign(diff)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        X_tilde = sign_diff * self.n * pow_n_1 * dx 
        
        term_abs_n = np.abs(diff) ** self.n
        h_dot = -2 * term_abs_n.T @ self.A @ X_tilde
        return h_dot, X_tilde

    def get_constraints(self, x, dx, a_des):
        """
        Returns Linear Constraints for QP: L_cbf * mu >= b_cbf
        """
        h_val = self.get_h(x)
        h_dot, X_tilde = self.get_h_dot(x, dx)
        
        # --- Calculate h_ddot terms ---
        diff = x - self.xc
        sign_diff = np.sign(diff)
        term_abs_n = np.abs(diff) ** self.n
        
        # Drift Terms
        pow_n_2 = np.abs(diff) ** (self.n - 2)
        dX_tilde_drift = self.n * (self.n - 1) * pow_n_2 * (dx * dx)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        dX_tilde_acc_coeff = self.n * sign_diff * pow_n_1
        
        term1 = -2 * X_tilde.T @ self.A @ X_tilde
        h_ddot_drift = term1 - 2 * term_abs_n.T @ self.A @ dX_tilde_drift
        
        # Gradient w.r.t Acceleration
        grad_x_ddot = -2 * term_abs_n.T @ self.A @ np.diag(dX_tilde_acc_coeff)
        
        # --- Robustness ---
        norm_grad = np.linalg.norm(grad_x_ddot)
        robust_margin = norm_grad * self.q_cp
        
        # --- Formulate QP Constraint ---
        # Same math as monolithic, just cleaner packaging
        L_cbf = grad_x_ddot
        
        boundary_val = (-h_ddot_drift 
                        - (grad_x_ddot @ a_des) 
                        - self.k1*h_dot 
                        - self.k0*h_val 
                        + robust_margin) 
        
        return L_cbf, boundary_val
import numpy as np

class CRCBF_Formulation:
    """
    Module 4: CR-CBF (Conformally Robust Safety).
    Implements Super-Ellipsoid Safe Set with Uncertainty Margins.
    """
    def __init__(self, center, lengths, power_n=4, q_cp=0.0):
        self.xc = np.array(center)
        self.n = int(power_n)
        self.q_cp = q_cp  # Conformal Quantile (Acceleration Error)
        
        # Matrix A setup
        self.dims = np.array(lengths)
        self.A_diag = 1.0 / (self.dims ** (2 * self.n))
        self.A = np.diag(self.A_diag)

        # --- OPTIMIZED TUNING: Pole Placement (From Normal Code) ---
        # Places poles at -30, -30 for stiff, critical damping.
        # This matches your working monolithic/normal implementation.
        p0 = 12.0
        p1 = 10.0
        
        self.k1 = p0 + p1  # 22.0
        self.k0 = p0 * p1  # 120.0
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
        
        # Chain rule vector term
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        X_tilde = sign_diff * self.n * pow_n_1 * dx 
        
        term_abs_n = np.abs(diff) ** self.n
        
        # h_dot = -2 * [|x-xc|^n]^T * A * X_tilde
        h_dot = -2 * term_abs_n.T @ self.A @ X_tilde
        return h_dot, X_tilde

    def get_constraints(self, x, dx, a_des):
        """
        Returns Linear Constraints for QP: L_cbf * mu >= b_cbf
        Robust Constraint: h_ddot + k1*h_dot + k0*h - ||grad_h||*q >= 0
        """
        h_val = self.get_h(x)
        h_dot, X_tilde = self.get_h_dot(x, dx)
        
        # --- Calculate h_ddot terms ---
        diff = x - self.xc
        sign_diff = np.sign(diff)
        term_abs_n = np.abs(diff) ** self.n
        
        # 1. Drift Terms
        pow_n_2 = np.abs(diff) ** (self.n - 2)
        dX_tilde_drift = self.n * (self.n - 1) * pow_n_2 * (dx * dx)
        pow_n_1 = np.abs(diff) ** (self.n - 1)
        dX_tilde_acc_coeff = self.n * sign_diff * pow_n_1
        
        term1 = -2 * X_tilde.T @ self.A @ X_tilde
        h_ddot_drift = term1 - 2 * term_abs_n.T @ self.A @ dX_tilde_drift
        
        # 2. Gradient w.r.t Acceleration
        grad_x_ddot = -2 * term_abs_n.T @ self.A @ np.diag(dX_tilde_acc_coeff)
        
        # --- ROBUSTNESS LOGIC (Preserved) ---
        # The robot must satisfy a stricter boundary to account for uncertainty q_cp.
        norm_grad = np.linalg.norm(grad_x_ddot)
        robust_margin = norm_grad * self.q_cp
        
        # --- Formulate QP Constraint ---
        L_cbf = grad_x_ddot
        
        # We ADD robust_margin to the RHS, raising the safety threshold
        boundary_val = (-h_ddot_drift 
                        - (grad_x_ddot @ a_des) 
                        - self.k1*h_dot 
                        - self.k0*h_val 
                        + robust_margin) 
        
        return L_cbf, boundary_val




# import numpy as np

# class CRCBF_Formulation:
#     """
#     Module 4: CR-CBF (Conformally Robust Safety).
#     Implements Super-Ellipsoid Safe Set with Uncertainty Margins.
#     """
#     def __init__(self, center, lengths, power_n=4, q_cp=0.0):
#         self.xc = np.array(center)
#         self.n = int(power_n)
#         self.q_cp = q_cp  # <--- NEW: Conformal Quantile (Acceleration Error)
        
#         # Matrix A setup
#         self.dims = np.array(lengths)
#         self.A_diag = 1.0 / (self.dims ** (2 * self.n))
#         self.A = np.diag(self.A_diag)

#         # Gains (Same as before)
#         self.k1 = 20.0       
#         self.k0 = 90.0       

#     def get_h(self, x):
#         """ Barrier Value h(x) """
#         diff = x - self.xc
#         abs_diff_n = np.abs(diff) ** self.n
#         term = abs_diff_n.T @ self.A @ abs_diff_n
#         return 1.0 - term

#     def get_h_dot(self, x, dx):
#         """ Time Derivative h_dot(x) """
#         diff = x - self.xc
#         sign_diff = np.sign(diff)
        
#         # Chain rule vector term
#         pow_n_1 = np.abs(diff) ** (self.n - 1)
#         X_tilde = sign_diff * self.n * pow_n_1 * dx 
        
#         term_abs_n = np.abs(diff) ** self.n
        
#         # h_dot = -2 * [|x-xc|^n]^T * A * X_tilde
#         h_dot = -2 * term_abs_n.T @ self.A @ X_tilde
#         return h_dot, X_tilde

#     def get_constraints(self, x, dx, a_des):
#         """
#         Returns Linear Constraints for QP: A_cbf * mu >= b_cbf
#         Robust Constraint: h_ddot_nom + k1*h_dot + k0*h - ||grad_h||*q >= 0
#         """
#         h_val = self.get_h(x)
#         h_dot, X_tilde = self.get_h_dot(x, dx)
        
#         # --- Calculate h_ddot terms ---
#         diff = x - self.xc
#         sign_diff = np.sign(diff)
#         term_abs_n = np.abs(diff) ** self.n
        
#         # 1. Drift Terms
#         pow_n_2 = np.abs(diff) ** (self.n - 2)
#         dX_tilde_drift = self.n * (self.n - 1) * pow_n_2 * (dx * dx)
#         pow_n_1 = np.abs(diff) ** (self.n - 1)
#         dX_tilde_acc_coeff = self.n * sign_diff * pow_n_1
        
#         term1 = -2 * X_tilde.T @ self.A @ X_tilde
#         h_ddot_drift = term1 - 2 * term_abs_n.T @ self.A @ dX_tilde_drift
        
#         # 2. Gradient w.r.t Acceleration (This is effectively grad_h w.r.t Position)
#         # This vector tells us how acceleration affects h directly
#         grad_x_ddot = -2 * term_abs_n.T @ self.A @ np.diag(dX_tilde_acc_coeff)
        
#         # --- NEW: Robustness Term ---
#         # We calculate the norm of the gradient that multiplies the acceleration/error
#         # This quantifies "How vulnerable is h to acceleration noise right now?"
#         norm_grad = np.linalg.norm(grad_x_ddot)
#         robust_margin = norm_grad * self.q_cp
        
#         # --- Formulate QP Constraint ---
#         # h_ddot_nom >= -k1*h_dot - k0*h + robust_margin
#         # (grad @ mu) + drift + (grad @ a_des) >= -k1*h_dot - k0*h + robust_margin
        
#         L_cbf = grad_x_ddot
        
#         # Move everything else to the RHS
#         # We ADD robust_margin to the required bound. 
#         # The robot must satisfy a higher (harder) threshold to ensure safety.
#         boundary_val = (-h_ddot_drift 
#                         - (grad_x_ddot @ a_des) 
#                         - self.k1*h_dot 
#                         - self.k0*h_val 
#                         + robust_margin) # <--- The CR Term
        
#         return L_cbf, boundary_val
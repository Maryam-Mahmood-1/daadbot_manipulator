import numpy as np

class CBF_SuperEllipsoid:
    """
    Conformally Robust Super-Ellipsoid CBF (CR-CBF)
    
    Standard Constraint:
        A_cbf * μ ≤ b_cbf
        
    Robust Constraint (Paper Eq 5 / Def 6):
        Reduces the available safety budget 'b' by the uncertainty margin.
    """
    def __init__(self, center, lengths, power_n=4, k_pos=20.0, k_vel=10.0):
        self.center = np.array(center)
        self.radii = np.array(lengths)
        self.power_n = power_n 
        self.kp = k_pos
        self.kv = k_vel

    def get_h_value(self, x):
        w = (x - self.center) / self.radii
        power_term = 2 * self.power_n 
        contributions = w ** power_term
        return 1.0 - np.sum(contributions)

    def get_constraints(self, x, dx, u_ref_total, q_quantile=0.0):
        """
        Calculates A_cbf * mu <= b_cbf with Robustness Margin.
        """
        # 1. Gradients & H-Val
        w = (x - self.center) / self.radii
        power_term = 2 * self.power_n
        
        h = 1.0 - np.sum(w ** power_term)
        grad_h = -power_term * (w ** (power_term - 1)) / self.radii
        h_dot = np.dot(grad_h, dx)
        
        # 2. Second Derivative Terms
        w_dot = dx / self.radii
        d_grad_dt = -power_term * (power_term - 1) * (w ** (power_term - 2)) * w_dot / self.radii
        drift_term = np.dot(d_grad_dt, dx)

        # 3. Robustness Margin
        # Uncertainty is on acceleration: h_ddot = drift + grad_h * (u + noise)
        # We need to ensure safety even if noise is worst-case aligned against grad_h.
        # Penalty = ||grad_h|| * q
        norm_grad_h = np.linalg.norm(grad_h)
        robustness_penalty = norm_grad_h * q_quantile

        # 4. Formulate Constraint
        # A * mu <= b
        barrier_term = self.kv * h_dot + self.kp * h
        
        # Subtract penalty from budget 'b'
        b_val = drift_term + np.dot(grad_h, u_ref_total) + barrier_term - robustness_penalty
        
        A_cbf = -grad_h.reshape(1, 3)
        b_cbf = np.array([[b_val]])
        
        return A_cbf, b_cbf
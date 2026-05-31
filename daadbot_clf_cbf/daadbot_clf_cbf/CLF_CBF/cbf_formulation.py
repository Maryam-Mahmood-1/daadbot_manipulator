import numpy as np

class CBF_SuperEllipsoid:
    """
    Implements the 'Virtual Cage' Super-Ellipsoid Safety Constraint.
    
    Mathematical Definition (from Eq. 24 & 19):
    -------------------------------------------
    1. Safe Set: h(x) ≥ 0
       h(x) = 1 - [|x - x_c|^n]^T A [|x - x_c|^n]
       (Simplified: h(x) = 1 - Σ ((x_i - c_i)/r_i)^(2n) )
       
    2. Constraint (ECBF):
       L_f^2 h + L_g L_f h Γ + K η ≥ 0
    """
    def __init__(self, center, lengths, power_n=4, k_pos=20.0, k_vel=10.0):
        """
        Args:
            center (list): x_c in Eq. 24 (e.g., [0.0, 0.0, 0.72])
            lengths (list): Radii corresponding to A matrix (e.g. [0.3, 0.24, 0.4])
            power_n (int): n in Eq. 24 (Higher = sharper corners, must be >= 2)
            k_pos, k_vel: Gains for K matrix in Eq. 19
        """
        self.center = np.array(center)
        self.radii = np.array(lengths)
        self.power_n = power_n # Note: The math uses '2n' as total power in summation form
        self.kp = k_pos
        self.kv = k_vel

    def get_h_value(self, x):
        """
        Helper to debug which axis is violating the constraint.
        """
        # Calculate normalized vector w = (x - c) / r
        w = (x - self.center) / self.radii
        
        # Calculate individual contributions
        # We use the same power term as in get_constraints
        # If your __init__ has power_n=2, the exponent is 4.
        power_term = 2 * self.power_n 
        
        contributions = w ** power_term
        h_val = 1.0 - np.sum(contributions)
        
        # --- PRINT DEBUG INFO ---
        # This will tell you: "Axis 0 is 0.1, Axis 2 is 1.5 (VIOLATION)"
        if h_val < 0.1:
            print(f"\n[CBF DEBUG] h(x): {h_val:.3f}")
            print(f"  X-contribution: {contributions[0]:.3f} (Pos: {x[0]:.2f}, Lim: {self.radii[0]})")
            print(f"  Y-contribution: {contributions[1]:.3f} (Pos: {x[1]:.2f}, Lim: {self.radii[1]})")
            print(f"  Z-contribution: {contributions[2]:.3f} (Pos: {x[2]:.2f}, Lim: {self.radii[2]})")
            
        return h_val

    def get_constraints(self, x, dx, u_ref_total):
        """
        Calculates A_cbf * mu <= b_cbf
        
        CRITICAL UPDATE:
        u_ref_total must be the FULL nominal input (a_des + u_pd).
        Otherwise, the PD controller fights the safety constraint.
        """
        # 1. Normalized State
        w = (x - self.center) / self.radii
        power_term = 2 * self.power_n
        
        # 2. Barrier States
        h = 1.0 - np.sum(w ** power_term)
        
        # Gradient ∇h
        grad_h = -power_term * (w ** (power_term - 1)) / self.radii
        
        # h_dot
        h_dot = np.dot(grad_h, dx)
        
        # 3. Second Derivative Drift (L_f^2 h)
        w_dot = dx / self.radii
        d_grad_dt = -power_term * (power_term - 1) * (w ** (power_term - 2)) * w_dot / self.radii
        drift_term = np.dot(d_grad_dt, dx)

        # 4. Formulate Constraint: A * mu <= b
        # Condition: ∇h * (u_ref + mu) >= -drift - K_v h_dot - K_p h
        # Rearranged: -∇h * mu <= drift + ∇h * u_ref + K_v h_dot + K_p h
        
        barrier_term = self.kv * h_dot + self.kp * h
        
        # The 'b' vector represents the "Available Safety Budget"
        # We add (grad_h @ u_ref_total) to account for the nominal controller's intent
        b_val = drift_term + np.dot(grad_h, u_ref_total) + barrier_term
        
        A_cbf = -grad_h.reshape(1, 3)
        b_cbf = np.array([[b_val]])
        
        return A_cbf, b_cbf
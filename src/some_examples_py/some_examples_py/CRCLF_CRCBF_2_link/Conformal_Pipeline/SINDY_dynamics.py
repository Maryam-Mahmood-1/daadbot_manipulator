import numpy as np

class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        """ Extracts a_hat and b_hat such that ddq = a_hat + b_hat * tau """
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        
        c2_sq, s2_sq = c2**2, s2**2
        dq0_sq, dq1_sq = dq[0]**2, dq[1]**2
        dq_cross = dq[0] * dq[1]
        
        # Passive Dynamics H(x)
        H_x = np.array([
            1.0, q[0], q[1], dq[0], dq[1], 
            s1, c1, s2, c2, s12, c12,
            dq0_sq, dq1_sq, dq_cross,
            dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
            dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
            np.sign(dq[0]), np.sign(dq[1])
        ])
        
        # Control interactions G(x)
        G_basis = np.array([1.0, c2, s2, c12, c2_sq, s2_sq])
        
        # The state-space Xi has 4 columns: [dq0, dq1, ddq0, ddq1]
        # We only need the acceleration columns (indices 2 and 3)
        a_hat = H_x @ self.Xi[:22, 2:4]
        
        b_hat = np.zeros((2, 2))
        b_hat[:, 0] = G_basis @ self.Xi[22:28, 2:4]
        b_hat[:, 1] = G_basis @ self.Xi[28:34, 2:4]
        
        return a_hat, b_hat
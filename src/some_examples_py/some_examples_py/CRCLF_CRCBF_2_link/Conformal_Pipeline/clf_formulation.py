

import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Controller:
    def __init__(self, dim_task=2):
        self.dim = dim_task
        zero, eye = np.zeros((dim_task, dim_task)), np.eye(dim_task)
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # Tuned LQR Weights
        q_pos, q_vel = 300.0, 210.0 
        self.Q_mat = np.diag([q_pos]*dim_task + [q_vel]*dim_task)
        R_mat = np.eye(dim_task) * 0.01 
        
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, R_mat)
        
        # Calculate optimal LQR gain K 
        self.K = np.linalg.inv(R_mat) @ self.G.T @ self.P
        
        eig_Q, eig_P = np.min(np.linalg.eigvals(self.Q_mat).real), np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 0.5 * (eig_Q / eig_P) 

    def get_nominal_acceleration(self, x, dx, x_des, dx_des):
        eta = np.hstack((x - x_des, dx - dx_des))
        print(f"State Error (eta): {eta}")
        print(f"Optimal LQR Gain (K): {self.K}")
        return -((self.K)/15) @ eta
        # return -self.K @ eta 

    # [FIXED] u_nom is now correctly added to the signature
    def get_lyapunov_constraints(self, x, dx, x_des, dx_des, u_nom, q_quantile=0.0, J=None):
        eta = np.hstack((x - x_des, dx - dx_des)).reshape(-1, 1)

        V = (eta.T @ self.P @ eta)[0, 0]
        LfV_open = (eta.T @ (self.P @ self.F + self.F.T @ self.P) @ eta)[0, 0]
        LgV = 2 * eta.T @ self.P @ self.G

        # Incorporate nominal control into closed-loop drift
        LfV_closed = LfV_open + (LgV @ u_nom.reshape(-1, 1))[0, 0]

        grad_V = 2 * (self.P @ eta)
        grad_V_actuated = grad_V[self.dim:, 0] 
        
        if J is not None:
            robustness_cost = np.linalg.norm(grad_V_actuated @ J) * q_quantile
        else:
            robustness_cost = np.linalg.norm(grad_V_actuated) * q_quantile
        print(f"Gamma: {self.gamma}, LfV_closed: {LfV_closed}, Robustness Cost: {robustness_cost}")
            
        return LfV_closed, LgV, V, self.gamma, robustness_cost
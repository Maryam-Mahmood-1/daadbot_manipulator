import numpy as np
from scipy.linalg import solve_continuous_are

class RESCLF_Controller:
    def __init__(self, dim_task=2):
        self.dim = dim_task
        zero, eye = np.zeros((dim_task, dim_task)), np.eye(dim_task)
        self.F = np.block([[zero, eye], [zero, zero]])
        self.G = np.block([[zero], [eye]])

        # --- PID GAINS ---
        # Adjust these to change the responsiveness of the nominal controller
        self.kp = 7.5
        self.kd = 7.5
        self.ki = 15.0
        self.error_integral = np.zeros(dim_task)
        
        # --- LYAPUNOV STABILITY BACKBONE ---
        # We keep P and gamma to define the 'Safety/Stability' boundary
        q_pos, q_vel = 300.0, 210.0 
        self.Q_mat = np.diag([q_pos]*dim_task + [q_vel]*dim_task)
        self.R_mat = np.eye(dim_task) * 0.1 
        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, self.R_mat)
        
        eig_Q = np.min(np.linalg.eigvals(self.Q_mat).real)
        eig_P = np.max(np.linalg.eigvals(self.P).real)
        self.gamma = 0.5 * (eig_Q / eig_P) 

    def get_nominal_acceleration(self, x, dx, x_des, dx_des, dt=0.01):
        # 1. Calculate Errors
        error_pos = x_des - x
        error_vel = dx_des - dx
        
        # 2. Update Integral with anti-windup (clipping)
        self.error_integral += error_pos * dt
        self.error_integral = np.clip(self.error_integral, -5.0, 5.0)
        
        # 3. PID Law: u = Kp*e + Kd*edot + Ki*e_int
        u_pid = (self.kp * error_pos) + (self.kd * error_vel) + (self.ki * self.error_integral)
        
        print(f"PID Effort: {u_pid}")
        return u_pid

    def get_lyapunov_constraints(self, x, dx, x_des, dx_des, u_nom, q_quantile=0.0, J=None):
        # We still use the state-error vector eta for the Lyapunov Energy V
        eta = np.hstack((x - x_des, dx - dx_des)).reshape(-1, 1)

        V = (eta.T @ self.P @ eta)[0, 0]
        LfV_open = (eta.T @ (self.P @ self.F + self.F.T @ self.P) @ eta)[0, 0]
        LgV = 2 * eta.T @ self.P @ self.G

        # Incorporate the PID nominal control into the closed-loop drift
        LfV_closed = LfV_open + (LgV @ u_nom.reshape(-1, 1))[0, 0]

        grad_V = 2 * (self.P @ eta)
        grad_V_actuated = grad_V[self.dim:, 0] 
        
        if J is not None:
            robustness_cost = np.linalg.norm(grad_V_actuated @ J) * q_quantile
        else:
            robustness_cost = np.linalg.norm(grad_V_actuated) * q_quantile
            
        return LfV_closed, LgV, V, self.gamma, robustness_cost
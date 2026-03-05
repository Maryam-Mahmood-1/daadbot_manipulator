#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import threading
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from ament_index_python.packages import get_package_share_directory

# --- INTEGRATED SINDy PREDICTOR ---
class SINDyPredictor:
    def __init__(self, xi_path, q_val_path):
        self.Xi = np.load(xi_path)
        with open(q_val_path, "r") as f:
            self.q_quantile = float(f.read())
            
    def get_dynamics(self, q, dq):
        s1, c1, s2, c2 = np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        dq0_sq, dq1_sq, dq_cross = dq[0]**2, dq[1]**2, dq[0]*dq[1]
        
        H_x = np.array([
            1.0, dq[0], dq[1], s1, c1, s2, c2, s12, c12,
            dq0_sq, dq1_sq, dq_cross,
            dq0_sq * s2, dq1_sq * s2, dq_cross * s2, 
            dq0_sq * c2, dq1_sq * c2, dq_cross * c2,
            np.sign(dq[0]), np.sign(dq[1])
        ])
        
        G_basis = np.array([1.0, c2, s2, c12, c2**2, s2**2, c2**3, c2**4])
        a_hat = (H_x @ self.Xi[:20, 2:4]).flatten()
        b_hat = np.zeros((2, 2))
        b_hat[:, 0] = G_basis @ self.Xi[20:28, 2:4]
        b_hat[:, 1] = G_basis @ self.Xi[28:36, 2:4]
        return a_hat, b_hat

# --- MODULAR IMPORTS ---
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.qp_solver import solve_optimization

class StabilizationComparison(Node):
    def __init__(self):
        super().__init__('stab_comparison_node')
        ws_path = "/home/maryammahmood/xdaadbot_ws/"
        self.sindy = SINDyPredictor(os.path.join(ws_path, "sindy_Xi_state_space2.npy"), 
                                    os.path.join(ws_path, "q_quantile_state_space2.txt"))
        
        # Load your iso-energy states
        self.df = pd.read_csv("iso_energy_states.csv").head(15)
        urdf = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
        self.model = pin.buildModelFromUrdf(urdf)
        self.ee_id = self.model.getFrameId("endEffector")
        
        self.num_robots = 15
        self.target_v_init = 15.0
        self.target_x = np.array([0.8, 0.2])
        self.clf = RESCLF_Controller(dim_task=2)
        
        # Double states for side-by-side comparison
        self.q_reg = self.df[['q1', 'q2']].values.copy()
        self.v_reg = self.df[['dq1', 'dq2']].values.copy()
        self.q_rob = self.df[['q1', 'q2']].values.copy()
        self.v_rob = self.df[['dq1', 'dq2']].values.copy()
        
        self.log = {'t': [], 'V_reg': [[] for _ in range(15)], 'V_rob': [[] for _ in range(15)]}
        self.lock = threading.Lock()
        self.start_t = time.time()
        
        self.timer = self.create_timer(0.01, self.control_loop)

    def control_loop(self):
        t_now = time.time() - self.start_t
        dt_sim = 0.01 
        
        with self.lock:
            self.log['t'].append(t_now)
            for i in range(15):
                # 1. Regular Robot Step
                v_val_reg, ddq_reg = self.compute_dynamics_step(self.q_reg[i], self.v_reg[i], robust=False)
                self.v_reg[i] += ddq_reg * dt_sim
                self.q_reg[i] += self.v_reg[i] * dt_sim
                self.log['V_reg'][i].append(v_val_reg)
                
                # 2. Robust Robot Step
                v_val_rob, ddq_rob = self.compute_dynamics_step(self.q_rob[i], self.v_rob[i], robust=True)
                self.v_rob[i] += ddq_rob * dt_sim
                self.q_rob[i] += self.v_rob[i] * dt_sim
                self.log['V_rob'][i].append(v_val_rob)

    def compute_dynamics_step(self, q, dq, robust=False):
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q, dq)
        pin.updateFramePlacements(self.model, data)
        x = data.oMf[self.ee_id].translation[:2]
        J = pin.computeFrameJacobian(self.model, data, q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:2, :]
        dx = J @ dq
        
        a_hat, b_hat = self.sindy.get_dynamics(q, dq)
        u_nom = self.clf.get_nominal_acceleration(x, dx, self.target_x, np.zeros(2))
        
        LfV, LgV, V, gamma, robust_term = self.clf.get_lyapunov_constraints(
            x, dx, self.target_x, np.zeros(2), u_nom, 
            q_quantile=(self.sindy.q_quantile if robust else 0.0), J=J
        )
        
        mu, feasible = solve_optimization(LfV, LgV, V, gamma, robust_clf_term=(robust_term if robust else 0.0))
        
        if not feasible:
            # Emergency damping if QP fails due to high energy/divergence
            return V, -5.0 * dq 
            
        dj_dq = pin.getFrameAcceleration(self.model, data, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[:2]
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(2))
        ddq_des = J_pinv @ (u_nom + mu - dj_dq)
        
        # Use learned B matrix for inversion
        tau = np.linalg.pinv(b_hat) @ (ddq_des - a_hat)
        # Apply Tau back to SINDy model for simulation
        ddq_actual = a_hat + b_hat @ tau
        
        return V, ddq_actual

def main():
    rclpy.init()
    node = StabilizationComparison()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(r"Lyapunov Energy Comparison at $V_0 = 15.0$")
    
    t_bound = np.linspace(0, 5, 100)
    ax.plot(t_bound, 15.0 * np.exp(-2.0 * t_bound), 'k:', lw=4, label="Theoretical Bound")
    
    lines_reg = [ax.plot([], [], 'r-', alpha=0.3, lw=1)[0] for _ in range(15)]
    lines_rob = [ax.plot([], [], 'b-', alpha=0.3, lw=1)[0] for _ in range(15)]
    
    ax.set_ylim(-0.5, 20); ax.grid(True, alpha=0.3)
    ax.legend(['Theoretical Bound', 'Regular CLF (SINDy)', 'CR-CLF (Robust SINDy)'])

    def update(frame):
        with node.lock:
            t = node.log['t']
            if not t: return lines_reg + lines_rob
            for i in range(15):
                lines_reg[i].set_data(t, node.log['V_reg'][i])
                lines_rob[i].set_data(t, node.log['V_rob'][i])
            ax.set_xlim(0, max(t) + 0.1)
        return lines_reg + lines_rob

    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.show()

if __name__ == '__main__': main()
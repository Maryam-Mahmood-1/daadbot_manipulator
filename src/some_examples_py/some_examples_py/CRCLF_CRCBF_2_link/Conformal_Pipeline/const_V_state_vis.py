import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import solve_continuous_are



class RESCLF_Controller:

    def __init__(self, dim_task=2):

        self.dim = dim_task

        zero, eye = np.zeros((dim_task, dim_task)), np.eye(dim_task)

        self.F = np.block([[zero, eye], [zero, zero]])

        self.G = np.block([[zero], [eye]])

        q_pos, q_vel = 300.0, 210.0 

        self.Q_mat = np.diag([q_pos]*dim_task + [q_vel]*dim_task)

        R_mat = np.eye(dim_task) * 0.01 

        self.P = solve_continuous_are(self.F, self.G, self.Q_mat, R_mat)



def get_v_contour_points(target_v=150, num_samples=15):

    ctrl = RESCLF_Controller(dim_task=2)

    P = ctrl.P

    target_pos = np.array([0.8, 0.2])

    

    samples = []

    attempts = 0

    max_attempts = 200000 

    

    print(f"Sampling {num_samples} points where V = {target_v}...")

    

    while len(samples) < num_samples and attempts < max_attempts:

        attempts += 1

        

        # --- MODIFIED RANGES ---

        # x sampled from [0.3, 1.2]

        x_val = np.random.uniform(0.3, 1.2)

        # y sampled from [-0.2, 1.2]

        y_val = np.random.uniform(-0.2, 1.2)

        

        x_rand = np.array([x_val, y_val])

        # Velocity remains in [-0.5, 0.5]

        v_rand = np.random.uniform(-0.65, 0.65, 2)

        

        # Calculate error vector eta

        eta = np.hstack((x_rand - target_pos, v_rand))

        

        # Calculate V based on your P matrix

        V = (eta.T @ P @ eta)

        

        # Tolerance check (1% error allowed)

        if np.abs(V - target_v) < (0.01 * target_v):

            samples.append({'pos': x_rand, 'vel': v_rand, 'V': V})

            

    return samples



# --- EXECUTION AND VISUALIZATION ---

points = get_v_contour_points(target_v=150, num_samples=15)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))



# Plot Position Samples

target_pos = [0.8, 0.2]

ax1.plot(target_pos[0], target_pos[1], 'bx', markersize=12, label='Target')

ax1.scatter([p['pos'][0] for p in points], [p['pos'][1] for p in points], c='red', label='V=150')

ax1.set_title("Position Space (x: 0.3-1.2, y: -0.2-1.2)")

ax1.set_xlim(-1.8, 1.8); ax1.set_ylim(-1.8, 1.8)

ax1.grid(True); ax1.legend()



# Plot Velocity Samples

ax2.scatter([p['vel'][0] for p in points], [p['vel'][1] for p in points], c='green')

ax2.set_title("Velocity Space (vx, vy: -1.0-1.0)")

ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.8, 1.8)

ax2.grid(True)



plt.tight_layout()

plt.show()



# Final output for your simulation configs

print("\n--- Initial States (x0) Found ---")

for i, p in enumerate(points):

    print(f"P{i+1}: pos=[{p['pos'][0]:.4f}, {p['pos'][1]:.4f}], vel=[{p['vel'][0]:.4f}, {p['vel'][1]:.4f}]")


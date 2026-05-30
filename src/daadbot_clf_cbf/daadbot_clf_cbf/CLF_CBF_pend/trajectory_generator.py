import numpy as np
import math

class TrajectoryGenerator:
    def __init__(self):
        # --- PATH PARAMETERS ---
        # The path is in the YZ plane at a fixed X
        self.fixed_x = -0.546
        self.center_z = 0.758
        self.center_y = -0.01  # Midpoint of 0.332 and -0.352
        
        self.radius = 0.356
        self.period = 10.0      # Time for one full back-and-forth loop
        self.omega = 2 * np.pi / self.period
        
        # --- APPROACH PHASE ---
        self.approach_duration = 4.0
        self.start_pos = None  
        
        # Target start point of the circle (where t_orbit = 0)
        # Starting at the first translation point provided
        self.orbit_start_pos = np.array([self.fixed_x, 0.332, self.center_z])

    def get_ref(self, t, current_actual_pos=None):
        """
        Computes desired trajectory state: pd, vd, ad.
        """
        
        # =========================================================
        # PHASE 1: Smooth Approach (0 < t < approach_duration)
        # Moves from robot's current position to the start of the loop
        # =========================================================
        if t < self.approach_duration:
            if self.start_pos is None:
                if current_actual_pos is None:
                    # Return zeros if we don't know where we are yet
                    return np.zeros(3), np.zeros(3), np.zeros(3)
                self.start_pos = current_actual_pos

            tau = t / self.approach_duration
            # Smooth S-curve (Cosine Interpolation)
            s = (1.0 - math.cos(tau * math.pi)) / 2.0
            ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
            dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)

            vector_diff = self.orbit_start_pos - self.start_pos
            
            pd = self.start_pos + (vector_diff * s)
            vd = vector_diff * ds
            ad = vector_diff * dds
            
            return pd, vd, ad

        # =========================================================
        # PHASE 2: Circular Loop (t > approach_duration)
        # Swings in YZ plane: y = R*cos(θ), z = R*sin(θ)
        # =========================================================
        else:
            t_orbit = t - self.approach_duration
            
            # θ starts at an offset to align with y = 0.332
            # 0.332 = center_y + radius * cos(theta_offset)
            # cos(theta_offset) = (0.332 - (-0.01)) / 0.356 ≈ 0.96
            theta_offset = math.acos((0.332 - self.center_y) / self.radius)
            
            theta = theta_offset + self.omega * t_orbit

            # Position
            pd = np.array([
                self.fixed_x,
                self.center_y + self.radius * np.cos(theta),
                self.center_z + self.radius * np.sin(theta)
            ])

            # Velocity
            vd = np.array([
                0.0,
                -self.radius * self.omega * np.sin(theta),
                self.radius * self.omega * np.cos(theta)
            ])

            # Acceleration
            ad = np.array([
                0.0,
                -self.radius * (self.omega**2) * np.cos(theta),
                -self.radius * (self.omega**2) * np.sin(theta)
            ])

            return pd, vd, ad
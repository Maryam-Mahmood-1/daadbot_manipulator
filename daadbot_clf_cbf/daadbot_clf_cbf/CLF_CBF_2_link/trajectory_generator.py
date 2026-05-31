import numpy as np
import math

class TrajectoryGenerator:
    def __init__(self):
        # --- ELLIPSE PARAMETERS ---
        self.center_pos = np.array([0.0, 0.0, 0.0])
        
        # Target Ellipse: 1.6m x 0.9m
        self.ellipse_a = 1.6 
        self.ellipse_b = 0.9
        
        self.period = 12.0     
        self.omega = 2 * np.pi / self.period
        
        # --- APPROACH PHASE (0 to 5 seconds) ---
        self.approach_duration = 5.0
        self.start_pos = None  # Will capture robot's actual start pos
        
        # The goal is to reach the ellipse start point [1.6, 0, 0] at t=5s
        self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])

    def get_ref(self, t, current_actual_pos=None):
        """
        Computes desired trajectory state: p_d, v_d, a_d.
        """
        
        # =========================================================
        # PHASE 1: Smooth Approach (0 < t < 5)
        # Moves from Robot_Start (approx 1.75) -> Ellipse_Start (1.6)
        # =========================================================
        if t < self.approach_duration:
            # Capture the robot's position at the very first timestep
            if self.start_pos is None:
                if current_actual_pos is None:
                    return np.zeros(3), np.zeros(3), np.zeros(3)
                self.start_pos = current_actual_pos

            # Normalized Time: τ goes from 0.0 to 1.0
            tau = t / self.approach_duration
            
            # Cosine Interpolation for smooth velocity
            # s(0) = 0, s(1) = 1
            s = (1.0 - math.cos(tau * math.pi)) / 2.0
            
            # Derivatives (Chain Rule)
            ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
            dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)

            # Interpolate
            vector_diff = self.orbit_start_pos - self.start_pos
            
            pd = self.start_pos + (vector_diff * s)
            vd = vector_diff * ds
            ad = vector_diff * dds
            
            return pd, vd, ad

        # =========================================================
        # PHASE 2: Elliptical Orbit (t > 5)
        # =========================================================
        else:
            t_orbit = t - self.approach_duration
            
            # Position
            x_des = self.center_pos.copy()
            x_des[0] += self.ellipse_a * np.cos(self.omega * t_orbit)
            x_des[1] += self.ellipse_b * np.sin(self.omega * t_orbit)

            # Velocity
            dx_des = np.zeros(3)
            dx_des[0] = -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit)
            dx_des[1] =  self.ellipse_b * self.omega * np.cos(self.omega * t_orbit)

            # Acceleration
            ddx_des = np.zeros(3)
            ddx_des[0] = -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit)
            ddx_des[1] = -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit)

            return x_des, dx_des, ddx_des
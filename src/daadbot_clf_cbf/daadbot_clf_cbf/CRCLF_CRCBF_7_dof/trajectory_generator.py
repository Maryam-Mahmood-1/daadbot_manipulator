import numpy as np
import math

class TrajectoryGenerator:
    def __init__(self, approach_duration=5.0):
        # --- ELLIPSE PARAMETERS (Updated for 7-DOF Workspace) ---
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.36
        
        self.period = 12.0     
        self.omega = 2 * np.pi / self.period
        
        # --- APPROACH PHASE PARAMETERS ---
        self.approach_duration = approach_duration  
        self.start_pos = None         
        
        # Target start point on the ellipse (at t_orbit = 0)
        self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])
        # Target velocity at the exact moment the orbit begins
        self.orbit_start_vel = np.array([0.0, self.ellipse_b * self.omega, 0.0])

    def get_ref(self, t, current_actual_pos=None, current_actual_vel=None):
        """
        Computes desired trajectory state: p_d, v_d, a_d.
        """
        
        # =========================================================
        # PHASE 1: Smooth Approach (Cosine Interpolation)
        # =========================================================
        if t < self.approach_duration:
            # 1. Capture starting position p_start at t=0
            if self.start_pos is None:
                # Synchronization: Wait for valid data from the robot
                if current_actual_pos is None or np.all(current_actual_pos == 0):
                    return np.zeros(3), np.zeros(3), np.zeros(3)
                self.start_pos = current_actual_pos

            # Normalized Time: tau = t / T_approach
            tau = t / self.approach_duration
            
            # Scalar Function s(tau) using Cosine Profile
            # This moves s from 0.0 to 1.0 smoothly
            s = (1.0 - math.cos(tau * math.pi)) / 2.0
            
            # s_dot (first derivative w.r.t time)
            ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
            
            # s_ddot (second derivative w.r.t time)
            dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)

            # Vector Difference between start and orbit entry point
            vector_diff = self.orbit_start_pos - self.start_pos
            
            pd = self.start_pos + (vector_diff * s)
            vd = vector_diff * ds
            ad = vector_diff * dds
            
            return pd, vd, ad

        # =========================================================
        # PHASE 2: Elliptical Orbit
        # =========================================================
        else:
            t_orbit = t - self.approach_duration
            
            # Position p_d(t)
            x_des = self.center_pos.copy()
            x_des[0] += self.ellipse_a * np.cos(self.omega * t_orbit)
            x_des[1] += self.ellipse_b * np.sin(self.omega * t_orbit)

            # Velocity v_d(t)
            dx_des = np.array([
                -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit),
                 self.ellipse_b * self.omega * np.cos(self.omega * t_orbit),
                 0.0
            ])

            # Acceleration a_d(t)
            ddx_des = np.array([
                -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit),
                -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit),
                 0.0
            ])

            return x_des, dx_des, ddx_des
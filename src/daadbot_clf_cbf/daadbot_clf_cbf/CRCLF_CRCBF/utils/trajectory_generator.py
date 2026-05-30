import numpy as np
import math

class TrajectoryGenerator:
    def __init__(self):
        # --- ELLIPSE PARAMETERS ---
        self.center_pos = np.array([0.0, 0.0, 0.72])
        self.ellipse_a = 0.15 
        self.ellipse_b = 0.36
        
        self.period = 12.0     
        # ω = 2π / T
        self.omega = 2 * np.pi / self.period
        
        # --- APPROACH PHASE PARAMETERS ---
        self.approach_duration = 5.0  # T_approach
        self.start_pos = None         # p_start
        
        # Target start point on ellipse (at angle = 0)
        # p_orbit = c + [a, 0, 0]ᵀ
        self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])

    def get_ref(self, t, current_actual_pos=None):
        """
        Computes desired trajectory state: p_d, v_d, a_d.
        """
        
        # =========================================================
        # PHASE 1: Smooth Approach (Cosine Interpolation)
        # Interpolates from p_start to p_orbit using scalar s(t) ∈ [0,1]
        # =========================================================
        if t < self.approach_duration:
            # 1. Capture starting position p_start at t=0
            if self.start_pos is None:
                if current_actual_pos is None:
                    return np.zeros(3), np.zeros(3), np.zeros(3)
                self.start_pos = current_actual_pos

            # -----------------------------------------------------
            # Normalized Time: τ = t / T_approach
            # -----------------------------------------------------
            tau = t / self.approach_duration
            
            # -----------------------------------------------------
            # Scalar Function s(τ) and derivatives
            # Using Cosine Profile for smooth acceleration
            #
            # s(τ) = (1 - cos(π τ)) / 2
            # ṡ(t) = (π / 2T) sin(π τ)
            # s̈(t) = (π² / 2T²) cos(π τ)
            # -----------------------------------------------------
            
            # s(τ)
            s = (1.0 - math.cos(tau * math.pi)) / 2.0
            
            # ṡ(t) (Chain rule applied)
            ds = (math.pi / (2.0 * self.approach_duration)) * math.sin(tau * math.pi)
            
            # s̈(t)
            dds = ((math.pi**2) / (2.0 * self.approach_duration**2)) * math.cos(tau * math.pi)

            # -----------------------------------------------------
            # Vector Trajectory Generation
            # p_d(t) = p_start + s(t) * (p_orbit - p_start)
            # v_d(t) = ṡ(t) * (p_orbit - p_start)
            # a_d(t) = s̈(t) * (p_orbit - p_start)
            # -----------------------------------------------------
            vector_diff = self.orbit_start_pos - self.start_pos
            
            pd = self.start_pos + (vector_diff * s)
            vd = vector_diff * ds
            ad = vector_diff * dds
            
            return pd, vd, ad

        # =========================================================
        # PHASE 2: Elliptical Orbit
        # Standard parametric equations for ellipse in XY plane
        # =========================================================
        else:
            # Shift time: t_orbit = t - T_approach
            t_orbit = t - self.approach_duration
            
            # -----------------------------------------------------
            # Position p_d(t)
            # x_d = c_x + a cos(ωt)
            # y_d = c_y + b sin(ωt)
            # -----------------------------------------------------
            x_des = self.center_pos.copy()
            x_des[0] += self.ellipse_a * np.cos(self.omega * t_orbit)
            x_des[1] += self.ellipse_b * np.sin(self.omega * t_orbit)

            # -----------------------------------------------------
            # Velocity v_d(t) = ṗ_d(t)
            # ẋ_d = -a ω sin(ωt)
            # ẏ_d =  b ω cos(ωt)
            # -----------------------------------------------------
            dx_des = np.zeros(3)
            dx_des[0] = -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit)
            dx_des[1] =  self.ellipse_b * self.omega * np.cos(self.omega * t_orbit)

            # -----------------------------------------------------
            # Acceleration a_d(t) = p̈_d(t)
            # ẍ_d = -a ω² cos(ωt)
            # ÿ_d = -b ω² sin(ωt)
            # -----------------------------------------------------
            ddx_des = np.zeros(3)
            ddx_des[0] = -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit)
            ddx_des[1] = -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit)

            return x_des, dx_des, ddx_des
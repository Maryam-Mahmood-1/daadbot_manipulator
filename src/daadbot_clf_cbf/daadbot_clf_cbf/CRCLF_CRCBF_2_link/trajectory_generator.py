import numpy as np

class TrajectoryGenerator:
    def __init__(self, approach_duration=2.0):
        # --- ELLIPSE PARAMETERS ---
        self.center_pos = np.array([0.0, 0.0, 0.0])
        self.ellipse_a = 1.6 
        self.ellipse_b = 0.9
        
        self.period = 12.0     
        self.omega = 2 * np.pi / self.period
        
        # --- CUSTOMIZABLE APPROACH PHASE ---
        self.approach_duration = approach_duration  # Set your desired time here
        self.start_pos = None  
        self.coeffs = None 
        
        # Target state at the end of approach (t_orbit = 0)
        # Matches the position, velocity, and acceleration of the ellipse exactly
        self.orbit_start_pos = self.center_pos + np.array([self.ellipse_a, 0.0, 0.0])
        self.orbit_start_vel = np.array([0.0, self.ellipse_b * self.omega, 0.0])
        self.orbit_start_acc = np.array([-self.ellipse_a * (self.omega**2), 0.0, 0.0])

    def _compute_quintic_coeffs(self, p0, v0, a0, pf, vf, af, T):
        """
        Solves for coefficients [c0, c1, c2, c3, c4, c5] matching 
        boundary conditions at t=0 and t=T.
        """
        b = np.array([
            pf - (p0 + v0*T + 0.5*a0*T**2),
            vf - (v0 + a0*T),
            af - a0
        ])
        
        A = np.array([
            [T**3,   T**4,    T**5],
            [3*T**2, 4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3]
        ])
        
        x = np.linalg.solve(A, b) 
        return np.array([p0, v0, 0.5*a0, x[0], x[1], x[2]])

    def get_ref(self, t, current_actual_pos=None, current_actual_vel=None):
        # PHASE 1: Customizable Smooth Approach (e.g., 1.75 -> 1.6)
        if t < self.approach_duration:
            if self.start_pos is None:
                # Synchronization Gate: Wait for valid Gazebo data
                if current_actual_pos is None or np.all(current_actual_pos == 0):
                    return np.zeros(3), np.zeros(3), np.zeros(3)
                
                self.start_pos = current_actual_pos
                # Match current robot velocity at t=0 to prevent 'jerking'
                v_start = current_actual_vel if current_actual_vel is not None else np.zeros(3)
                
                self.coeffs = []
                for i in range(3):
                    c = self._compute_quintic_coeffs(
                        p0=self.start_pos[i], v0=v_start[i], a0=0.0,
                        pf=self.orbit_start_pos[i], 
                        vf=self.orbit_start_vel[i], 
                        af=self.orbit_start_acc[i],
                        T=self.approach_duration
                    )
                    self.coeffs.append(c)
                self.coeffs = np.array(self.coeffs)

            # Evaluate Polynomial for Phase 1
            tt = np.array([1.0, t, t**2, t**3, t**4, t**5])
            vt = np.array([0.0, 1.0, 2*t, 3*t**2, 4*t**3, 5*t**4])
            at = np.array([0.0, 0.0, 2.0, 6*t, 12*t**2, 20*t**3])
            
            return self.coeffs @ tt, self.coeffs @ vt, self.coeffs @ at

        # PHASE 2: Standard Elliptical Orbit
        else:
            t_orbit = t - self.approach_duration
            
            # Position
            x_des = self.center_pos + np.array([
                self.ellipse_a * np.cos(self.omega * t_orbit),
                self.ellipse_b * np.sin(self.omega * t_orbit),
                0.0
            ])

            # Velocity
            dx_des = np.array([
                -self.ellipse_a * self.omega * np.sin(self.omega * t_orbit),
                 self.ellipse_b * self.omega * np.cos(self.omega * t_orbit),
                 0.0
            ])

            # Acceleration
            ddx_des = np.array([
                -self.ellipse_a * (self.omega**2) * np.cos(self.omega * t_orbit),
                -self.ellipse_b * (self.omega**2) * np.sin(self.omega * t_orbit),
                 0.0
            ])

            return x_des, dx_des, ddx_des
import pinocchio as pin
import numpy as np

class RobotDynamics:
    """
    Rigid Body Dynamics & Kinematics Wrapper
    Supports Single EE or Midpoint of Two EEs (e.g., for grippers).
    [NEW] Includes parameteric noise injection for robustness testing.
    """
    def __init__(self, urdf_path, end_effector_names, controlled_joints, noise_level=0.0):
        """
        end_effector_names: String (single frame) or List of Strings (two frames for midpoint).
        controlled_joints: List of ALL joint names to be included in the model.
        noise_level: Percentage of noise to add (e.g., 0.10 = 10% error)
        """
        # 1. Load the FULL model from URDF
        model_full = pin.buildModelFromUrdf(urdf_path)
        
        # Store noise parameter
        self.noise_percentage = noise_level
        
        # 2. Lock joints not in the controlled list
        joints_to_lock_ids = []
        for joint_name in model_full.names:
            if joint_name == "universe": continue
            
            if joint_name not in controlled_joints:
                if model_full.existJointName(joint_name):
                    joints_to_lock_ids.append(model_full.getJointId(joint_name))
        
        # 3. Build Reduced Model
        q_ref = pin.neutral(model_full)
        self.model = pin.buildReducedModel(model_full, joints_to_lock_ids, q_ref)
        self.data = self.model.createData()
        
        # 4. Get End Effector Frame IDs
        if isinstance(end_effector_names, str):
            self.ee_ids = [self.model.getFrameId(end_effector_names)]
        elif isinstance(end_effector_names, list):
            self.ee_ids = []
            for name in end_effector_names:
                if self.model.existFrame(name):
                    self.ee_ids.append(self.model.getFrameId(name))
                else:
                    raise ValueError(f"Frame '{name}' not found in URDF.")
        else:
            raise ValueError("end_effector_names must be a string or a list of strings.")

    def compute_dynamics(self, q, dq, use_joint1=True):
        """
        Computes dynamics matrices for the control law.
        """
        # Update model state
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        # ---------------------------------------------------------
        # 1. Mass Matrix M(q)
        # ---------------------------------------------------------
        M = self.data.M

        # ---------------------------------------------------------
        # 2. Nonlinear Effects n(q, q̇)
        # ---------------------------------------------------------
        nle = pin.nonLinearEffects(self.model, self.data, q, dq)
        
        # ---------------------------------------------------------
        # 3. Compute Kinematics (Single or Midpoint)
        # ---------------------------------------------------------
        if len(self.ee_ids) == 1:
            # --- Single Frame Mode ---
            fid = self.ee_ids[0]
            J, dJ, x, dx = self._get_kinematics_for_frame(fid)
        
        elif len(self.ee_ids) == 2:
            # --- Midpoint Mode ---
            fid1 = self.ee_ids[0]
            fid2 = self.ee_ids[1]
            
            J1, dJ1, x1, dx1 = self._get_kinematics_for_frame(fid1)
            J2, dJ2, x2, dx2 = self._get_kinematics_for_frame(fid2)
            
            x = (x1 + x2) / 2.0
            dx = (dx1 + dx2) / 2.0
            J = (J1 + J2) / 2.0
            dJ = (dJ1 + dJ2) / 2.0
            
        else:
            raise NotImplementedError("Only 1 frame or 2 frames (midpoint) supported.")

        # ---------------------------------------------------------
        # Dynamic Switch Logic (Joint 1 Lock)
        # ---------------------------------------------------------
        if not use_joint1:
            J[:, 0] = 0.0 
            dJ[:, 0] = 0.0

        # ---------------------------------------------------------
        # [NEW] Inject Noise (Robustness Test)
        # ---------------------------------------------------------
        if self.noise_percentage > 0.0:
            M = self.add_noise(M)
            nle = self.add_noise(nle)

        return M, nle, J, dJ, x, dx

    def _get_kinematics_for_frame(self, fid):
        """Helper to extract J, dJ, x, dx for a specific frame ID"""
        x = self.data.oMf[fid].translation
        v_frame = pin.getFrameVelocity(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dx = v_frame.linear
        
        J_full = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, :] 
        
        dJ_full = pin.getFrameJacobianTimeVariation(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ = dJ_full[:3, :]
        
        return J, dJ, x, dx

    def add_noise(self, matrix):
        """
        Adds Gaussian noise proportional to the magnitude of each element.
        Noise ~ N(0, (val * percentage)^2)
        """
        # Calculate standard deviation for each element
        std_dev = np.abs(matrix) * self.noise_percentage
        # Generate noise
        noise = np.random.normal(0, std_dev + 1e-6) # 1e-6 avoids 0 std dev
        return matrix + noise

    def get_pseudo_inverse(self, J):
        epsilon = 1e-6
        return J.T @ np.linalg.inv(J @ J.T + epsilon * np.eye(J.shape[0]))

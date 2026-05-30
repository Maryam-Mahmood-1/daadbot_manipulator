import pinocchio as pin
import numpy as np

class RobotDynamics:
    """
    Module 1: System Dynamics & Operational Space Kinematics.
    Implements Eq (1), (2), and (3) from the paper.
    """
    def __init__(self, urdf_path, ee_frame_name, target_joints):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        # Map joint names to Pinocchio internal indices
        self.q_indices = []
        self.v_indices = []
        for name in target_joints:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                self.q_indices.append(self.model.joints[jid].idx_q)
                self.v_indices.append(self.model.joints[jid].idx_v)
        
        self.q = pin.neutral(self.model)
        self.dq = np.zeros(self.model.nv)

    def update_state_from_ros(self, msg, msg_map, target_joint_names):
        """ Updates internal q and dq from ROS JointState message. """
        for i, (q_idx, v_idx) in enumerate(zip(self.q_indices, self.v_indices)):
            name = target_joint_names[i]
            if name in msg_map:
                idx = msg_map[name]
                self.q[q_idx] = msg.position[idx]
                self.dq[v_idx] = msg.velocity[idx]

    def compute(self):
        """ 
        Computes matrices for Eq (3): 
        Returns M, nle (C+G), position (x), velocity (x_dot), Jacobian (J), Drift (dJ*dq)
        """
        # 1. Dynamics (Mass Matrix & NLE)
        # computeAllTerms is good, but expensive. 
        # It computes M and nle correctly.
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)
        
        M = self.data.M
        nle = self.data.nle
        
        # 2. Kinematics & Drift
        # To get strictly J_dot*q_dot, we propagate kinematics with ddq=0.
        # This updates position, velocity, AND acceleration (data.a) of frames.
        pin.forwardKinematics(self.model, self.data, self.q, self.dq, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)
        
        # Operational Space State
        ee_pose = self.data.oMf[self.ee_frame_id]
        p = ee_pose.translation
        
        # Jacobian (Eq 2)
        # We use LOCAL_WORLD_ALIGNED to get world-oriented Jacobian
        J_full = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_linear = J_full[:3, :] 
        
        v = (J_full @ self.dq)[:3]
        
        # Drift Acceleration dJ * dq (Eq 3)
        # Since we ran forwardKinematics with ddq=0, the resulting frame accel is purely drift.
        drift = pin.getFrameClassicalAcceleration(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ_dq = drift.linear
        
        return M, nle, p, v, J_linear, dJ_dq
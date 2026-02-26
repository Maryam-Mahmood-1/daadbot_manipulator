import numpy as np
import pickle
import pinocchio as pin

class LearnedRobotDynamics:
    def __init__(self, pkl_path, urdf_path, ee_names):
        # Load the saved Scikit-Learn pipelines
        with open(pkl_path, "rb") as f:
            saved_data = pickle.load(f)
        
        self.m_model = saved_data["M_model"]
        self.c_model = saved_data["C_model"]
        # Pull the quantile calculated during calibration
        self.q_quantile = saved_data.get("q_quantile", 1.0) 

        # Pinocchio setup for Kinematics
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(ee_names[0])

    def compute_dynamics(self, q, dq):
        # 1. Feature Engineering (Must match training script)
        feat_M = np.array([[1.0, np.cos(q[1])]])
        feat_C = np.array([[dq[0]*np.sin(q[1]), dq[1]*np.sin(q[1]), 
                            dq[0]*np.cos(q[1]), dq[1]*np.cos(q[1])]])

        # 2. Predict Matrices
        M_learned = self.m_model.predict(feat_M).reshape(2, 2)
        C_learned = self.c_model.predict(feat_C).reshape(2, 2)
        nle_learned = C_learned @ dq

        # 3. Kinematics
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

        x = self.data.oMf[self.ee_id].translation
        J = pin.getFrameJacobian(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, self.ee_id, pin.LOCAL_WORLD_ALIGNED)

        return M_learned, nle_learned, J, dJ, x, dq
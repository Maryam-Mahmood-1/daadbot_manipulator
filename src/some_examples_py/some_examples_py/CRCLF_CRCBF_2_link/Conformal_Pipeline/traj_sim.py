#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
import os
import csv
import sys
from ament_index_python.packages import get_package_share_directory

# Modular Import from your workspace
from some_examples_py.CRCLF_CRCBF_2_link.Conformal_Pipeline.clf_formulation import RESCLF_Controller

class LargeScaleStateGenerator(Node):
    def __init__(self, num_to_generate):
        super().__init__('large_state_generator')
        
        # 1. Setup Robot Model
        urdf = os.path.join(get_package_share_directory("daadbot_desc"), "urdf", "2_link_urdf", "2link_robot.urdf")
        self.model = pin.buildModelFromUrdf(urdf)
        self.data = self.model.create_data() if hasattr(self.model, 'create_data') else self.model.createData()
        self.ee_id = self.model.getFrameId("endEffector")
        
        # 2. Setup Controller
        self.clf = RESCLF_Controller(dim_task=2)
        self.target_x = np.array([0.8, 0.2])
        self.target_V = 15.0
        
        # 3. Execution
        self.output_file = "iso_energy_states.csv"
        self.run_generation(num_to_generate)

    def run_generation(self, num_needed):
        found_states = []
        attempts = 0
        self.get_logger().info(f"Generating {num_needed} states. This may take a moment...")

        while len(found_states) < num_needed:
            attempts += 1
            
            # A. Sample Joint Positions
            q = np.random.uniform(-np.pi, np.pi, 2)
            
            # B. Task Space Check [-0.95, 0.95]
            pin.forwardKinematics(self.model, self.data, q, np.zeros(2))
            pin.updateFramePlacements(self.model, self.data)
            x_ee = self.data.oMf[self.ee_id].translation[:2]
            
            if not (np.all(x_ee >= -0.95) and np.all(x_ee <= 0.95)):
                continue

            # C. Energy Feasibility Check
            _, _, V_pos, _, _ = self.clf.get_lyapunov_constraints(
                x_ee, np.zeros(2), self.target_x, np.zeros(2), np.zeros(2)
            )
            
            if V_pos < self.target_V:
                # D. Sample and Scale Velocity
                v_dir = np.random.uniform(-1.0, 1.0, 2)
                v_dir /= (np.linalg.norm(v_dir) + 1e-6)
                
                # V_total = V_pos + V_vel. Calculate scale 's' to hit exactly 2.0
                s = np.sqrt(max(0, (self.target_V - V_pos) * 5.0)) 
                dq = s * v_dir
                
                # E. Final Joint Velocity Limit Check [-0.9, 0.9]
                if np.all(np.abs(dq) <= 0.9):
                    found_states.append([*q, *dq, *x_ee])

        self.save_to_csv(found_states)
        self.get_logger().info(f"Successfully saved {num_needed} states to {self.output_file}")

    def save_to_csv(self, data):
        header = ['q1', 'q2', 'dq1', 'dq2', 'x', 'y']
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

def main():
    rclpy.init()
    # Accept count from terminal: ros2 run ... 50
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    node = LargeScaleStateGenerator(count)
    rclpy.shutdown()

if __name__ == '__main__':
    main()






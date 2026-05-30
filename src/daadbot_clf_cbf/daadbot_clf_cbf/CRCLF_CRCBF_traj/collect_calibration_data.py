#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np
import os
import csv
import math
import datetime

# --- CONFIGURATION ---
URDF_CLEAN = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot.urdf"
URDF_NOISY = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_noisy.urdf"
SAVE_PATH = os.path.expanduser("~/xdaadbot_ws/calibration_data_sum_sines.csv")

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]
EE_FRAME_NAME = 'endeffector' 

# --- TRAJECTORY SETTINGS ---
N_TRAJECTORIES = 150        
TRAJ_DURATION = 8.0         # Longer duration for sum of sines
WARMUP_DURATION = 1.0       # Time to fade in trajectory
CONTROL_FREQ = 300.0        # 300 Hz (Target DT = 0.0033)

# --- SAFETY LIMITS (The Fix) ---
SAFETY_VEL_LIMIT = 1.5 
SAFETY_ACC_LIMIT = 5.0      # Max allowed acceleration per joint

# Controller Gains (CTC/PID)
KP = 100.0 
KD = 10.0

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_cp')
        
        # 1. Load Pinocchio Models
        self.model_true = pin.buildModelFromUrdf(URDF_CLEAN)
        self.data_true = self.model_true.createData()
        
        self.model_pred = pin.buildModelFromUrdf(URDF_NOISY)
        self.data_pred = self.model_pred.createData()
        
        self.ee_frame_id_true = self.model_true.getFrameId(EE_FRAME_NAME)
        self.ee_frame_id_pred = self.model_pred.getFrameId(EE_FRAME_NAME)

        # 2. Map Target Joints
        self.joint_indices_q = [] 
        self.joint_indices_v = [] 
        self.effort_limits = []   
        
        full_limits = self.model_true.effortLimit
        
        for name in TARGET_JOINTS:
            if self.model_true.existJointName(name):
                j_id = self.model_true.getJointId(name)
                idx_q = self.model_true.joints[j_id].idx_q
                idx_v = self.model_true.joints[j_id].idx_v
                self.joint_indices_q.append(idx_q)
                self.joint_indices_v.append(idx_v)
                self.effort_limits.append(full_limits[idx_v])
            else:
                self.get_logger().error(f"Joint {name} not found!")

        self.effort_limits = np.array(self.effort_limits)
        
        # 3. Initialize FULL State Vectors
        self.q_full = pin.neutral(self.model_true)
        self.v_full = np.zeros(self.model_true.nv)
        
        # 4. Initialize CSV
        self.csv_headers = ['traj_id', 'time', 'dt_actual']
        # Joint Columns
        for i in range(len(TARGET_JOINTS)): self.csv_headers.append(f'q_{i}')
        for i in range(len(TARGET_JOINTS)): self.csv_headers.append(f'v_{i}')
        for i in range(len(TARGET_JOINTS)): self.csv_headers.append(f'acc_{i}')
        # Cartesian Columns
        for axis in ['x', 'y', 'z']: self.csv_headers.append(f'pos_{axis}')
        for axis in ['x', 'y', 'z']: self.csv_headers.append(f'vel_{axis}')
        for axis in ['x', 'y', 'z']: self.csv_headers.append(f'acc_{axis}')
        # Metric
        self.csv_headers.append('error_norm')

        with open(SAVE_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
        print(f"CSV initialized at {SAVE_PATH}")
        
        # 5. ROS Setup
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        self.timer = self.create_timer(1.0/CONTROL_FREQ, self.control_loop)
        
        # 6. Trajectory State
        self.trajectory_count = 0
        self.start_time = None
        self.last_loop_time = None
        self.received_first_state = False
        self.is_collecting = False
        
        # Sum of Sines Params
        self.num_sines = 3
        self.sine_params = []
        self.q_start_snapshot = None

        print("Waiting for joint states...")

    def cb_joints(self, msg):
        try:
            msg_map = {name: i for i, name in enumerate(msg.name)}
            for i, name in enumerate(TARGET_JOINTS):
                if name in msg_map:
                    msg_idx = msg_map[name]
                    pin_idx_q = self.joint_indices_q[i]
                    pin_idx_v = self.joint_indices_v[i]
                    self.q_full[pin_idx_q] = msg.position[msg_idx]
                    self.v_full[pin_idx_v] = msg.velocity[msg_idx]
            self.received_first_state = True
        except Exception:
            pass 

    def reset_trajectory_params(self):
        """
        Generates Sum-of-Sines parameters with 1/w^2 scaling 
        to prevent high accelerations.
        """
        self.sine_params = []
        
        # Snapshot current state for smooth transition
        # extracting just the 7 target joints for simplicity in logic
        current_q_7 = [self.q_full[idx] for idx in self.joint_indices_q]
        self.q_start_snapshot = np.array(current_q_7)

        for i in range(len(TARGET_JOINTS)):
            # Limits (Approximated for safety, you can use model limits)
            lim_low, lim_high = -2.0, 2.0 
            
            # Center: biased towards current position to avoid large jumps
            center = current_q_7[i]
            
            # Generate Frequencies (0.5 to 2.5 Hz)
            freqs = np.random.uniform(0.5, 2.5, self.num_sines)
            phases = np.random.uniform(0, 2*np.pi, self.num_sines)
            
            # --- THE FIX: Scale Amplitude by Frequency ---
            # Max accel contribution per sine wave ~= MaxAccel / num_sines
            accel_budget = SAFETY_ACC_LIMIT / self.num_sines
            
            # A <= a_max / w^2
            max_amps = accel_budget / (freqs**2)
            
            # Randomize amplitude within this safe limit
            amps = np.random.uniform(0.05, max_amps)
            
            self.sine_params.append({
                'A': amps,
                'w': freqs,
                'p': phases,
                'c': center
            })

    def get_full_kinematics(self, model, data, q_full, v_full, tau_full, frame_id):
        # 1. Forward Dynamics (Computes Joint Accel ddq)
        ddq = pin.aba(model, data, q_full, v_full, tau_full)
        # 2. Forward Kinematics
        pin.forwardKinematics(model, data, q_full, v_full, ddq)
        pin.updateFramePlacements(model, data)
        # 3. Get Cartesian Quantities
        pos = data.oMf[frame_id].translation
        vel = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        acc = pin.getFrameAcceleration(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        return pos, vel, acc, ddq

    def control_loop(self):
        if not self.received_first_state: return
        
        now = self.get_clock().now()
        
        # --- Start New Trajectory ---
        if not self.is_collecting:
            if self.trajectory_count < N_TRAJECTORIES:
                print(f"--- Starting Trajectory {self.trajectory_count + 1}/{N_TRAJECTORIES} ---")
                self.reset_trajectory_params()
                self.start_time = now
                self.last_loop_time = now
                self.is_collecting = True
            else:
                self.save_and_exit()
                return

        # Time Calculation
        t_total = (now - self.start_time).nanoseconds / 1e9
        dt_nano = (now - self.last_loop_time).nanoseconds
        dt_actual = dt_nano / 1e9
        self.last_loop_time = now

        if t_total > TRAJ_DURATION:
            self.is_collecting = False
            self.trajectory_count += 1
            return

        # --- 1. Compute Desired State (Sum of Sines) ---
        q_des_7 = np.zeros(len(TARGET_JOINTS))
        v_des_7 = np.zeros(len(TARGET_JOINTS))
        a_des_7 = np.zeros(len(TARGET_JOINTS))

        # Smooth Fade-in
        fade = 1.0
        if t_total < WARMUP_DURATION:
            ratio = t_total / WARMUP_DURATION
            fade = 0.5 * (1.0 - math.cos(ratio * math.pi))

        for i in range(len(TARGET_JOINTS)):
            P = self.sine_params[i]
            val_q = P['c']
            val_v = 0.0
            val_a = 0.0
            
            for k in range(self.num_sines):
                arg = P['w'][k] * t_total + P['p'][k]
                val_q += P['A'][k] * math.sin(arg)
                val_v += P['A'][k] * P['w'][k] * math.cos(arg)
                val_a += -P['A'][k] * (P['w'][k]**2) * math.sin(arg)
            
            if fade >= 1.0:
                q_des_7[i] = val_q
                v_des_7[i] = val_v
                a_des_7[i] = val_a
            else:
                q_des_7[i] = (1.0 - fade) * self.q_start_snapshot[i] + fade * val_q
                v_des_7[i] = fade * val_v
                a_des_7[i] = fade * val_a

        # --- 2. Compute Control Torque (Computed Torque Control) ---
        # Current 7-DOF state
        q_curr_7 = np.array([self.q_full[i] for i in self.joint_indices_q])
        v_curr_7 = np.array([self.v_full[i] for i in self.joint_indices_v])

        # PID Error Term
        ddq_cmd_7 = a_des_7 + KP * (q_des_7 - q_curr_7) + KD * (v_des_7 - v_curr_7)

        # Full Inverse Dynamics (using 'clean' model for control)
        # We assume 0 torque for non-target joints
        ddq_cmd_full = np.zeros(self.model_true.nv)
        for i, idx in enumerate(self.joint_indices_v):
            ddq_cmd_full[idx] = ddq_cmd_7[i]

        tau_full = pin.rnea(self.model_true, self.data_true, self.q_full, self.v_full, ddq_cmd_full)
        
        # Clip to Limits
        tau_full = np.clip(tau_full, -self.model_true.effortLimit, self.model_true.effortLimit)
        
        # Extract 7 torques for publishing
        tau_7_out = np.array([tau_full[i] for i in self.joint_indices_v])

        # --- 3. Compute Metrics & Save ---
        
        # True Physics (Clean Model)
        pos_true, vel_true, acc_true, ddq_true = self.get_full_kinematics(
            self.model_true, self.data_true, 
            self.q_full, self.v_full, tau_full, self.ee_frame_id_true
        )
        
        # Predicted Physics (Noisy Model) - This is where the error comes from
        _, _, acc_pred, _ = self.get_full_kinematics(
            self.model_pred, self.data_pred, 
            self.q_full, self.v_full, tau_full, self.ee_frame_id_pred
        )
        
        error_norm = np.linalg.norm(acc_true - acc_pred)

        row_data = {
            'traj_id': self.trajectory_count,
            'time': t_total,
            'dt_actual': dt_actual,
            'error_norm': error_norm,
            'pos_x': pos_true[0], 'pos_y': pos_true[1], 'pos_z': pos_true[2],
            'vel_x': vel_true[0], 'vel_y': vel_true[1], 'vel_z': vel_true[2],
            'acc_x': acc_true[0], 'acc_y': acc_true[1], 'acc_z': acc_true[2]
        }
        
        for i in range(len(TARGET_JOINTS)):
            row_data[f'q_{i}'] = q_curr_7[i]
            row_data[f'v_{i}'] = v_curr_7[i]
            # Log the acceleration that RESULTED from the torque
            row_data[f'acc_{i}'] = ddq_true[self.joint_indices_v[i]]

        with open(SAVE_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(row_data)

        # --- 4. Publish ---
        msg = Float64MultiArray()
        msg.data = tau_7_out.tolist()
        self.pub.publish(msg)

    def save_and_exit(self):
        print("Collection Complete.")
        stop_msg = Float64MultiArray(data=[0.0]*len(TARGET_JOINTS))
        self.pub.publish(stop_msg)
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
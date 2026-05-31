#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import csv
import datetime

# --- 1. CONDA + ROS FIX ---
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin

# --- 2. CONFIGURATION ---
from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

COMMAND_TOPIC = '/effort_arm_controller/commands' 
SAVE_DIR = "/home/maryammahmood/xdaadbot_ws/"

# --- SAFETY LIMITS ---
# We use slightly conservative limits because Sum of Sines can add up quickly
TRAJ_MAX_VEL = 1.8  
TRAJ_MAX_ACC = 3.5  

SAFETY_VEL_LIMIT = 2.5 
SAFETY_ACC_LIMIT = 6.0 

# TIME SETTINGS
TARGET_DT = 0.002 # 500Hz
TOTAL_EPISODES = 300        # <--- INCREASED for better generalization
EPISODE_DURATION = 8.0     
WARMUP_DURATION = 1.0     

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = f"{SAVE_DIR}robot_data_sum_sines_{TIMESTAMP}.csv"

CONTROLLER_JOINT_ORDER = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        # Load Pinocchio Model
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        
        # Internal State
        self.q_current = np.zeros(self.nq)
        self.v_current = np.zeros(self.nv)
        self.v_prev = np.zeros(self.nv) 
        self.state_received = False
        
        self.prev_q = None
        self.prev_v = None
        self.prev_tau = None
        
        self.last_loop_time = None
        self.episode_start_time = None
        
        # Filter for recording Acceleration
        self.alpha = 0.05 
        self.filtered_accel = np.zeros(self.nv)

        self.global_max_vel = 0.0
        self.global_max_acc = 0.0

        # Map indices
        self.save_indices = []
        for name in CONTROLLER_JOINT_ORDER:
            if self.model.existJointName(name):
                idx = self.model.getJointId(name) - 1
                self.save_indices.append(idx)
        self.save_indices = np.array(self.save_indices, dtype=int)

        # ROS Setup
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
        # CSV Setup
        self.csv_file = open(DATA_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ["episode_id", "dt_actual"] + \
                 [f"q_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"v_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"tau_{name}" for name in CONTROLLER_JOINT_ORDER] + \
                 [f"acc_{name}" for name in CONTROLLER_JOINT_ORDER]
        self.csv_writer.writerow(header)
        
        self.timer = self.create_timer(TARGET_DT, self.control_loop)
        
        # Gains
        self.kp = 100.0 
        self.kd = 10.0

        self.current_episode = 0
        self.is_collecting = False
        
        # --- NEW: Multi-Sine Parameters ---
        # Each joint gets 3 sine waves: q = Sum(A_k * sin(w_k * t + p_k))
        self.num_sines = 3
        self.sine_params = [] # List of (Amps, Freqs, Phases, Center) per joint
        
        self.q_start_snapshot = np.zeros(self.nq)

    def joint_state_callback(self, msg):
        msg_map = {name: i for i, name in enumerate(msg.name)}
        for i, name in enumerate(CONTROLLER_JOINT_ORDER):
            if name in msg_map:
                ros_idx = msg_map[name]
                pin_idx = self.save_indices[i]
                self.q_current[pin_idx] = msg.position[ros_idx]
                self.v_current[pin_idx] = msg.velocity[ros_idx]

        if not self.state_received:
            self.state_received = True
            self.start_new_episode()

    def reset_trajectory_params(self):
        """Generates SUM OF SINES parameters for chaotic motion"""
        self.sine_params = []
        
        for i in range(self.nq):
            lim_low = self.model.lowerPositionLimit[i]
            lim_high = self.model.upperPositionLimit[i]
            
            # Safety clamp for limits
            if lim_high > 10: lim_high = 3.14
            if lim_low < -10: lim_low = -3.14
            
            # Center of motion (Random spot in range)
            center = np.random.uniform(lim_low * 0.5, lim_high * 0.5)
            
            # Generate 3 random components
            # High Freq (jitter), Med Freq, Low Freq (base motion)
            freqs = np.random.uniform(0.5, 3.0, self.num_sines) 
            phases = np.random.uniform(0, 2*np.pi, self.num_sines)
            
            # Amplitudes must be small enough that sum doesn't exceed limits
            # Total Amp roughly 0.5 rad
            total_allowed_amp = min(abs(lim_high-center), abs(center-lim_low)) * 0.8
            amps = np.random.uniform(0.05, total_allowed_amp/self.num_sines, self.num_sines)
            
            self.sine_params.append({
                'A': amps,
                'w': freqs,
                'p': phases,
                'c': center
            })

        self.filtered_accel = np.zeros(self.nv)
        if hasattr(self, 'q_current'):
            self.q_start_snapshot = self.q_current.copy()

    def start_new_episode(self):
        if self.current_episode >= TOTAL_EPISODES:
            self.close_file()
            raise SystemExit
            
        self.current_episode += 1
        self.reset_trajectory_params()
        
        now = self.get_clock().now()
        self.episode_start_time = now
        self.last_loop_time = now
        
        self.is_collecting = True
        self.get_logger().info(f"Start Ep {self.current_episode}/{TOTAL_EPISODES} (Sum-of-Sines)")

    def control_loop(self):
        if not self.state_received or not self.is_collecting:
            return

        current_ros_time = self.get_clock().now()
        t_total = (current_ros_time - self.episode_start_time).nanoseconds / 1e9
        
        if t_total > EPISODE_DURATION:
            self.is_collecting = False
            self.start_new_episode()
            return

        dt_nano = (current_ros_time - self.last_loop_time).nanoseconds
        dt_actual = dt_nano / 1e9 
        if dt_actual <= 0.0001: return

        # 1. LOGGING (Same as before)
        raw_accel = (self.v_current - self.v_prev) / dt_actual
        self.filtered_accel = (self.alpha * raw_accel) + ((1.0 - self.alpha) * self.filtered_accel)
        
        if t_total > WARMUP_DURATION and self.prev_q is not None:
            q_s = self.prev_q[self.save_indices]
            v_s = self.prev_v[self.save_indices]
            tau_s = self.prev_tau[self.save_indices]
            acc_s = self.filtered_accel[self.save_indices]
            
            row = np.concatenate([[self.current_episode], [dt_actual], q_s, v_s, tau_s, acc_s])
            self.csv_writer.writerow(row)

        self.v_prev = self.v_current.copy()
        self.last_loop_time = current_ros_time 

        # 2. TRAJECTORY GENERATION (Sum of Sines)
        q_des = np.zeros(self.nq)
        v_des = np.zeros(self.nv)
        a_des = np.zeros(self.nv)
        
        # Smooth start
        fade = 1.0
        if t_total < WARMUP_DURATION:
            ratio = t_total / WARMUP_DURATION
            fade = 0.5 * (1.0 - math.cos(ratio * math.pi))

        for i in range(self.nq):
            # Retrieve params
            P = self.sine_params[i]
            
            # q = Center + Sum( A * sin(wt + p) )
            # v = Sum( A * w * cos(...) )
            # a = Sum( -A * w^2 * sin(...) )
            
            val_q = P['c']
            val_v = 0.0
            val_a = 0.0
            
            for k in range(self.num_sines):
                arg = P['w'][k] * t_total + P['p'][k]
                val_q += P['A'][k] * math.sin(arg)
                val_v += P['A'][k] * P['w'][k] * math.cos(arg)
                val_a += -P['A'][k] * (P['w'][k]**2) * math.sin(arg)
            
            # Blend
            if fade >= 1.0:
                q_des[i] = val_q
                v_des[i] = val_v
                a_des[i] = val_a
            else:
                q_des[i] = (1.0 - fade) * self.q_start_snapshot[i] + fade * val_q
                v_des[i] = fade * val_v 
                a_des[i] = fade * val_a

        # 3. CONTROL (Inverse Dynamics)
        # Clip desireds for safety
        v_des = np.clip(v_des, -SAFETY_VEL_LIMIT, SAFETY_VEL_LIMIT)
        a_des = np.clip(a_des, -SAFETY_ACC_LIMIT, SAFETY_ACC_LIMIT)
        
        # PD + Feedforward
        kp_vec = np.full(self.nv, self.kp)
        kd_vec = np.full(self.nv, self.kd)
        
        ddq_cmd = a_des + kp_vec * (q_des - self.q_current) + kd_vec * (v_des - self.v_current)
        tau_ideal = pin.rnea(self.model, self.data, self.q_current, self.v_current, ddq_cmd)
        
        tau_final = np.clip(tau_ideal, -self.model.effortLimit, self.model.effortLimit)
        
        self.prev_q = self.q_current.copy()
        self.prev_v = self.v_current.copy()
        self.prev_tau = tau_final.copy()

        ordered_tau = tau_final[self.save_indices]
        msg = Float64MultiArray()
        msg.data = ordered_tau.tolist()
        self.torque_pub.publish(msg)

    def close_file(self):
        if hasattr(self, 'csv_file'): self.csv_file.close()
        print("\n" + "="*40)
        print(f"DONE. {self.current_episode} Episodes Collected.")
        print("="*40 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit): pass
    finally:
        node.close_file()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()













# #!/usr/bin/env python3
# import sys
# import os
# import math
# import numpy as np
# import csv
# import datetime

# # --- 1. CONDA + ROS FIX ---
# sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
# sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import pinocchio as pin

# # --- 2. CONFIGURATION ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# COMMAND_TOPIC = '/effort_arm_controller/commands' 
# SAVE_DIR = "/home/maryammahmood/xdaadbot_ws/"

# # --- SAFETY LIMITS ---
# # Limits for the generated sine waves
# TRAJ_MAX_VEL = 1.5  
# TRAJ_MAX_ACC = 2.5  

# # STRICT limits for the controller command (prevents jumps)
# SAFETY_VEL_LIMIT = 2.0 
# SAFETY_ACC_LIMIT = 5.0  # Keeps acceleration small (user requested <10)

# # TIME SETTINGS
# TARGET_DT = 0.001 
# TOTAL_EPISODES = 100
# EPISODE_DURATION = 6.0    
# WARMUP_DURATION = 1.5     

# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# DATA_FILE = f"{SAVE_DIR}robot_data_dynamic_{TIMESTAMP}.csv"

# CONTROLLER_JOINT_ORDER = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# class DataCollectorNode(Node):
#     def __init__(self):
#         super().__init__('data_collector_node')
        
#         # Load Pinocchio Model
#         self.model = pin.buildModelFromUrdf(URDF_PATH)
#         self.data = self.model.createData()
#         self.nq = self.model.nq
#         self.nv = self.model.nv
        
#         # Internal State
#         self.q_current = np.zeros(self.nq)
#         self.v_current = np.zeros(self.nv)
#         self.v_prev = np.zeros(self.nv) 
#         self.state_received = False
        
#         self.prev_q = None
#         self.prev_v = None
#         self.prev_tau = None
        
#         self.last_loop_time = None
#         self.episode_start_time = None
        
#         # Filter for recording Acceleration data (smooths out noise)
#         self.alpha = 0.1 
#         self.filtered_accel = np.zeros(self.nv)

#         self.global_max_vel = 0.0
#         self.global_max_acc = 0.0

#         # Map indices
#         self.save_indices = []
#         self.get_logger().info("Mapping joints...")
#         for name in CONTROLLER_JOINT_ORDER:
#             if self.model.existJointName(name):
#                 idx = self.model.getJointId(name) - 1
#                 self.save_indices.append(idx)
#         self.save_indices = np.array(self.save_indices, dtype=int)

#         # ROS Setup
#         self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
#         self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
#         # CSV Setup
#         self.csv_file = open(DATA_FILE, 'w', newline='')
#         self.csv_writer = csv.writer(self.csv_file)
#         header = ["episode_id", "dt_actual"] + \
#                  [f"q_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"v_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"tau_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"acc_{name}" for name in CONTROLLER_JOINT_ORDER]
#         self.csv_writer.writerow(header)
        
#         self.timer = self.create_timer(TARGET_DT, self.control_loop)
        
#         # Controller Gains
#         self.kp = 120.0 
#         self.kd = 2 * np.sqrt(self.kp)

#         self.current_episode = 0
#         self.is_collecting = False
#         self.q_center = np.zeros(self.nq)
#         self.amp = np.zeros(self.nq)
#         self.phase_offset = np.zeros(self.nq)
        
#         # Initialize placeholders
#         self.q_start_snapshot = np.zeros(self.nq)

#     def reset_trajectory_params(self):
#         """Generates random sine waves that respect velocity/accel limits"""
#         min_safe_omega = 3.0 
        
#         for i in range(self.nq):
#             lim_low = self.model.lowerPositionLimit[i]
#             lim_high = self.model.upperPositionLimit[i]
            
#             # Sanity check for infinite/undefined limits
#             if lim_high > 100 or lim_low < -100: 
#                 lim_low, lim_high = -np.pi, np.pi

#             max_possible_amp = (lim_high - lim_low) / 2.0 * 0.9
#             a = np.random.uniform(0.1, max_possible_amp)
            
#             safe_min = lim_low + a
#             safe_max = lim_high - a
            
#             if safe_min >= safe_max:
#                 c = (lim_low + lim_high) / 2.0
#                 a = (lim_high - lim_low) / 2.0 * 0.9 
#             else:
#                 c = np.random.uniform(safe_min, safe_max)
            
#             self.q_center[i] = c
#             self.amp[i] = a
#             self.phase_offset[i] = np.random.uniform(0, math.pi)

#             # Calculate max Omega (speed of sine wave) for this amplitude
#             w_limit_v = TRAJ_MAX_VEL / (a + 1e-6)
#             w_limit_a = math.sqrt(TRAJ_MAX_ACC / (a + 1e-6))
            
#             joint_safe_omega = min(w_limit_v, w_limit_a)
#             if joint_safe_omega < min_safe_omega:
#                 min_safe_omega = joint_safe_omega

#         if min_safe_omega < 0.3:
#              min_safe_omega = 0.3 
             
#         self.omega = np.random.uniform(0.3, min_safe_omega)
#         self.filtered_accel = np.zeros(self.nv)
        
#         # Take a snapshot of where the robot is RIGHT NOW so we can ramp from it
#         if hasattr(self, 'q_current'):
#             self.q_start_snapshot = self.q_current.copy()

#     def joint_state_callback(self, msg):
#         # Update internal state
#         for i, name in enumerate(msg.name):
#             if self.model.existJointName(name):
#                 idx = self.model.getJointId(name) - 1
#                 if 0 <= idx < self.nq:
#                     self.q_current[idx] = msg.position[i]
#                     self.v_current[idx] = msg.velocity[i]

#         # Start first episode only after receiving valid data
#         if not self.state_received:
#             self.state_received = True
#             self.start_new_episode()

#     def start_new_episode(self):
#         if self.current_episode >= TOTAL_EPISODES:
#             self.close_file()
#             raise SystemExit
            
#         self.current_episode += 1
#         self.reset_trajectory_params()
#         self.prev_q = None 
#         self.prev_v = None 
#         self.prev_tau = None
#         self.v_prev = self.v_current.copy()
        
#         now = self.get_clock().now()
#         self.episode_start_time = now
#         self.last_loop_time = now
        
#         self.is_collecting = True
#         self.get_logger().info(f"Start Ep {self.current_episode} (Omega: {self.omega:.2f})")

#     def control_loop(self):
#         if not self.state_received or not self.is_collecting:
#             return

#         current_ros_time = self.get_clock().now()
#         t_total = (current_ros_time - self.episode_start_time).nanoseconds / 1e9
        
#         if t_total > EPISODE_DURATION:
#             self.is_collecting = False
#             self.start_new_episode()
#             return

#         dt_nano = (current_ros_time - self.last_loop_time).nanoseconds
#         dt_actual = dt_nano / 1e9 
        
#         if dt_actual <= 0.0005:
#             return

#         # 1. CALC FILTERED ACCELERATION (For CSV recording only)
#         raw_accel = (self.v_current - self.v_prev) / dt_actual
#         self.filtered_accel = (self.alpha * raw_accel) + ((1.0 - self.alpha) * self.filtered_accel)
        
#         # 2. SAVE DATA
#         if t_total > WARMUP_DURATION and self.prev_q is not None:
#             q_saved   = self.prev_q[self.save_indices]
#             v_saved   = self.prev_v[self.save_indices]
#             tau_saved = self.prev_tau[self.save_indices]
#             acc_saved = self.filtered_accel[self.save_indices]
            
#             curr_max_v = np.max(np.abs(v_saved))
#             curr_max_a = np.max(np.abs(acc_saved))
#             if curr_max_v > self.global_max_vel: self.global_max_vel = curr_max_v
#             if curr_max_a > self.global_max_acc: self.global_max_acc = curr_max_a

#             row = np.concatenate([
#                 [self.current_episode], 
#                 [dt_actual], 
#                 q_saved, v_saved, tau_saved, acc_saved
#             ])
#             self.csv_writer.writerow(row)

#         self.v_prev = self.v_current.copy()
#         self.last_loop_time = current_ros_time 

#         # 3. TRAJECTORY GENERATION (With Smooth Ramp)
#         q_des = np.zeros(self.nq)
#         v_des = np.zeros(self.nv)
#         a_des = np.zeros(self.nv)
        
#         # Ramping Logic (Cosine Fade)
#         if t_total < WARMUP_DURATION:
#             ratio = t_total / WARMUP_DURATION
#             fade = 0.5 * (1.0 - math.cos(ratio * math.pi)) 
#         else:
#             fade = 1.0

#         for i in range(self.nq):
#             p = self.phase_offset[i]
#             c = self.q_center[i]
#             a = self.amp[i]
            
#             # The Sinusoidal Goal
#             target_q = c + a * math.sin(self.omega * t_total + p)
#             target_v = a * self.omega * math.cos(self.omega * t_total + p)
#             target_a = -a * (self.omega**2) * math.sin(self.omega * t_total + p)
            
#             # Blend current position with Goal
#             if fade >= 1.0:
#                 q_des[i] = target_q
#                 v_des[i] = target_v
#                 a_des[i] = target_a
#             else:
#                 q_des[i] = (1.0 - fade) * self.q_start_snapshot[i] + fade * target_q
#                 v_des[i] = fade * target_v 
#                 a_des[i] = fade * target_a 

#         # 4. CONTROLLER & SAFETY CLAMPS
        
#         # A. Clip Desired Velocity
#         v_des = np.clip(v_des, -SAFETY_VEL_LIMIT, SAFETY_VEL_LIMIT)
        
#         # B. Calculate Desired Acceleration (Computed Torque)
#         error_q = pin.difference(self.model, self.q_current, q_des)
#         error_v = v_des - self.v_current
#         u_ideal = a_des + self.kp * error_q + self.kd * error_v
        
#         # C. Clip Desired Acceleration (Prevents 100s or 1000s rad/s^2)
#         u_clamped = np.clip(u_ideal, -SAFETY_ACC_LIMIT, SAFETY_ACC_LIMIT)
        
#         # D. Calculate Ideal Torque
#         tau_ideal = pin.rnea(self.model, self.data, self.q_current, self.v_current, u_clamped)
        
#         # E. Clip Torque to URDF Limits [NEW ADDITION]
#         tau_final = np.clip(tau_ideal, -self.model.effortLimit, self.model.effortLimit)
        
#         # Update History & Publish
#         self.prev_q = self.q_current.copy()
#         self.prev_v = self.v_current.copy()
#         self.prev_tau = tau_final.copy() # Save the CLIPPED torque

#         ordered_tau = tau_final[self.save_indices]
#         msg = Float64MultiArray()
#         msg.data = ordered_tau.tolist()
#         self.torque_pub.publish(msg)

#     def close_file(self):
#         if hasattr(self, 'csv_file') and not self.csv_file.closed:
#             self.csv_file.flush()
#             self.csv_file.close()
        
#         print("\n" + "="*40)
#         print("DATA COLLECTION COMPLETE")
#         print("="*40)
#         print(f"Total Episodes: {self.current_episode}")
#         print(f"Max Velocity:     {self.global_max_vel:.4f} rad/s")
#         print(f"Max Acceleration: {self.global_max_acc:.4f} rad/s^2")
#         print("="*40 + "\n")

# def main(args=None):
#     rclpy.init(args=args)
#     node = DataCollectorNode()
#     try:
#         rclpy.spin(node)
#     except (KeyboardInterrupt, SystemExit):
#         pass
#     finally:
#         node.close_file()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()















    # #!/usr/bin/env python3
# import sys
# import os
# import time
# import math
# import numpy as np
# import csv
# import datetime

# # --- 1. CONDA + ROS FIX ---
# sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
# sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import pinocchio as pin

# # --- 2. CONFIGURATION ---
# URDF_PATH = "/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/urdf_inverted_torque/daadbot_2.urdf"
# COMMAND_TOPIC = '/effort_arm_controller/commands' 

# # Output Path
# SAVE_DIR = "/home/maryammahmood/xdaadbot_ws/"
# TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# DATA_FILE = f"{SAVE_DIR}robot_data_filtered_{TIMESTAMP}.csv"

# # --- 3. FILTER SETTINGS ---
# # ONLY these joints will be saved to the CSV.
# CONTROLLER_JOINT_ORDER = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# # Parameters
# TOTAL_EPISODES = 100
# EPISODE_DURATION = 5.0 # seconds
# DT = 0.001 

# class DataCollectorNode(Node):
#     def __init__(self):
#         super().__init__('data_collector_node')
        
#         # A. Setup Pinocchio
#         self.model = pin.buildModelFromUrdf(URDF_PATH)
#         self.data = self.model.createData()
#         self.nq = self.model.nq
#         self.nv = self.model.nv
        
#         # B. Internal Robot State (Full Size)
#         self.q_current = np.zeros(self.nq)
#         self.v_current = np.zeros(self.nv)
#         self.v_prev = np.zeros(self.nv) 
#         self.state_received = False
        
#         # Alignment Buffers
#         self.prev_q = None
#         self.prev_v = None
#         self.prev_tau = None
        
#         # C. Create Filter Mappings
#         self.save_indices = []
        
#         self.get_logger().info("Mapping joints for filtering...")
#         for name in CONTROLLER_JOINT_ORDER:
#             if self.model.existJointName(name):
#                 # Pinocchio joint IDs start at 1 (universe=0). 
#                 # For standard revolute joints, the q/v index is (id - 1).
#                 idx = self.model.getJointId(name) - 1
#                 self.save_indices.append(idx)
#                 self.get_logger().info(f"  Mapped '{name}' to Index {idx}")
#             else:
#                 self.get_logger().error(f"  CRITICAL: Joint '{name}' not found in URDF!")

#         # Convert to numpy array for easy indexing later
#         self.save_indices = np.array(self.save_indices, dtype=int)

#         # D. ROS Interface
#         self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
#         self.torque_pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        
#         # E. REAL-TIME FILE SETUP
#         self.get_logger().info(f"Opening file: {DATA_FILE}")
#         self.csv_file = open(DATA_FILE, 'w', newline='')
#         self.csv_writer = csv.writer(self.csv_file)
        
#         # Dynamic Header based on joint names
#         header = ["episode_id"] + \
#                  [f"q_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"v_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"tau_{name}" for name in CONTROLLER_JOINT_ORDER] + \
#                  [f"acc_{name}" for name in CONTROLLER_JOINT_ORDER]
#         self.csv_writer.writerow(header)
#         self.csv_file.flush()
        
#         # F. Control Loop
#         self.timer = self.create_timer(DT, self.control_loop)
        
#         # Gains
#         self.kp = 100.0 
#         self.kd = 2 * np.sqrt(self.kp)

#         # G. Episode Management
#         self.current_episode = 0
#         self.episode_start_time = 0.0
#         self.is_collecting = False
#         self.reset_trajectory_params()
        
#     def reset_trajectory_params(self):
#         self.omega = np.random.uniform(0.5, 2.0)
#         self.amp = np.random.uniform(0.1, 0.6)
#         # Random phase for ALL joints (full model), we just control the active ones
#         self.phase_offset = np.random.uniform(0, math.pi, self.nq)

#     def joint_state_callback(self, msg):
#         if not self.state_received:
#             self.get_logger().info("Connected to Gazebo!")
#             self.state_received = True
#             self.start_new_episode()

#         for i, name in enumerate(msg.name):
#             if self.model.existJointName(name):
#                 j_id = self.model.getJointId(name)
#                 idx = j_id - 1
#                 if 0 <= idx < self.nq:
#                     self.q_current[idx] = msg.position[i]
#                     self.v_current[idx] = msg.velocity[i]

#     def start_new_episode(self):
#         if self.current_episode >= TOTAL_EPISODES:
#             self.close_file()
#             self.get_logger().info("Collection Complete. File closed.")
#             raise SystemExit
            
#         self.current_episode += 1
#         self.reset_trajectory_params()
        
#         # Reset buffers
#         self.prev_q = None 
#         self.prev_v = None
#         self.prev_tau = None
#         self.v_prev = self.v_current.copy()
        
#         self.episode_start_time = time.time()
#         self.is_collecting = True
#         self.get_logger().info(f"Starting Episode {self.current_episode}/{TOTAL_EPISODES}")

#     def control_loop(self):
#         if not self.state_received or not self.is_collecting:
#             return

#         now = time.time()
#         t = now - self.episode_start_time
        
#         if t > EPISODE_DURATION:
#             self.is_collecting = False
#             self.start_new_episode()
#             return

#         # --- 1. OBSERVE RESULT (Full State Accel) ---
#         accel_full = (self.v_current - self.v_prev) / DT
        
#         # --- 2. WRITE FILTERED DATA TO DISK ---
#         if self.prev_q is not None:
#             # EXTRACT ONLY THE 7 JOINTS
#             q_saved   = self.prev_q[self.save_indices]
#             v_saved   = self.prev_v[self.save_indices]
#             tau_saved = self.prev_tau[self.save_indices]
#             acc_saved = accel_full[self.save_indices]
            
#             # Combine into one row
#             row = np.concatenate([
#                 [self.current_episode], 
#                 q_saved, 
#                 v_saved, 
#                 tau_saved, 
#                 acc_saved
#             ])
#             self.csv_writer.writerow(row)

#         self.v_prev = self.v_current.copy()

#         # --- 3. GENERATE TRAJECTORY ---
#         q_des = np.zeros(self.nq)
#         v_des = np.zeros(self.nv)
#         a_des = np.zeros(self.nv)
        
#         # Only generating excitation for the joints we care about? 
#         # Actually it's safer to generate for all, and just use the ones we need.
#         for i in range(self.nq):
#             p = self.phase_offset[i]
#             q_des[i] = self.amp * math.sin(self.omega * t + p)
#             v_des[i] = self.amp * self.omega * math.cos(self.omega * t + p)
#             a_des[i] = -self.amp * (self.omega**2) * math.sin(self.omega * t + p)

#         # --- 4. COMPUTE TORQUE (Full Model) ---
#         error_q = pin.difference(self.model, self.q_current, q_des)
#         error_v = v_des - self.v_current
#         u = a_des + self.kp * error_q + self.kd * error_v
        
#         tau_current = pin.rnea(self.model, self.data, self.q_current, self.v_current, u)
        
#         # --- 5. STORE & PUBLISH ---
#         self.prev_q = self.q_current.copy()
#         self.prev_v = self.v_current.copy()
#         self.prev_tau = tau_current.copy()

#         # Send filtered torque to Gazebo (using the same indices, convenient!)
#         ordered_tau = tau_current[self.save_indices]
#         msg = Float64MultiArray()
#         msg.data = ordered_tau.tolist()
#         self.torque_pub.publish(msg)

#     def close_file(self):
#         if hasattr(self, 'csv_file') and not self.csv_file.closed:
#             self.csv_file.flush()
#             self.csv_file.close()
#             self.get_logger().info("CSV File Closed Safely.")

# def main(args=None):
#     rclpy.init(args=args)
#     node = DataCollectorNode()
#     try:
#         rclpy.spin(node)
#     except (KeyboardInterrupt, SystemExit):
#         node.get_logger().info("Stopping...")
#     finally:
#         node.close_file()
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()
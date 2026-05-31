#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import csv
import datetime
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONDA + ROS FIX ---
# (Ensures compatible Python paths for ROS 2 Humble)
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

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
SAFETY_VEL_LIMIT = 2.0 
SAFETY_ACC_LIMIT = 5.0  
TORQUE_LIMIT = 50.0

# TIME SETTINGS
TARGET_DT = 0.002 # 500Hz Control Loop

# SETUP DIRECTORY
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_FILE = f"{SAVE_DIR}robot_data_ellipse_{TIMESTAMP}.csv"

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class EllipseDataCollector(Node):
    def __init__(self):
        super().__init__('ellipse_data_collector')

        self.urdf_path = URDF_PATH
        self.ee_frame_name = 'endeffector'
        
        # --- DATA LOGGING SETUP ---
        self.csv_file = open(DATA_FILE, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header Format: Matches the template exactly
        header = ["episode_id", "dt_actual"] + \
                 [f"q_{name}" for name in TARGET_JOINTS] + \
                 [f"v_{name}" for name in TARGET_JOINTS] + \
                 [f"tau_{name}" for name in TARGET_JOINTS] + \
                 [f"acc_{name}" for name in TARGET_JOINTS]
        self.csv_writer.writerow(header)
        self.get_logger().info(f"Data logging started: {DATA_FILE}")

        # --- ELLIPSE PARAMETERS ---
        # Slower speed: ~7.5 seconds per loop
        self.trajectory_period = 7.5 
        self.center_z = 0.72
        self.center_pos = np.array([0.0, 0.0, self.center_z]) 
        self.ellipse_a = 0.150  # Radius X
        self.ellipse_b = 0.270  # Radius Y

        # --- TUNING GAINS (Task Space PD) ---
        self.Kp_task = np.array([800.0, 800.0, 800.0, 50.0, 50.0, 50.0]) 
        self.Kd_task = np.array([40.0, 40.0, 40.0, 2.0, 2.0, 2.0])       
        self.Ki_task = np.array([100.0, 100.0, 100.0, 0.5, 0.5, 0.5])
        self.sum_error = np.zeros(6) 

        # --- PINOCCHIO SETUP ---
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Joint Mapping
        self.joint_indices_q = []
        self.joint_indices_v = []
        for name in TARGET_JOINTS:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.joint_indices_q.append(self.model.joints[joint_id].idx_q)
                self.joint_indices_v.append(self.model.joints[joint_id].idx_v)
            else:
                self.get_logger().error(f"Joint {name} not found in URDF!")

        # State Variables
        self.q = pin.neutral(self.model) 
        self.dq = np.zeros(self.model.nv) 
        self.dq_prev = np.zeros(self.model.nv) # For acceleration calc
        self.received_first_state = False
        
        # Acceleration Filtering (Alpha Filter)
        self.alpha = 0.1
        self.filtered_accel = np.zeros(self.model.nv)
        
        # Plotting Data
        self.actual_x, self.actual_y = [], []
        self.target_x, self.target_y = [], []

        # Start Logic
        self.start_time = None
        self.last_loop_time = None
        self.start_approach_pos = None
        
        # ROS Communication
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(
            Float64MultiArray, COMMAND_TOPIC, 10)

        self.control_timer = self.create_timer(TARGET_DT, self.control_loop)

    def joint_state_callback(self, msg):
        msg_map = {name: i for i, name in enumerate(msg.name)}
        try:
            for i, joint_name in enumerate(TARGET_JOINTS):
                if joint_name in msg_map:
                    msg_idx = msg_map[joint_name]
                    pin_q_idx = self.joint_indices_q[i]
                    pin_v_idx = self.joint_indices_v[i]
                    self.q[pin_q_idx] = msg.position[msg_idx]
                    self.dq[pin_v_idx] = msg.velocity[msg_idx]
            self.received_first_state = True
        except IndexError:
            pass

    def get_desired_state(self, t_rel):
        """Standard Ellipse Math"""
        omega = 2 * math.pi / self.trajectory_period
        angle = omega * t_rel
        
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)

        target_x = self.center_pos[0] + self.ellipse_a * cos_angle
        target_y = self.center_pos[1] + self.ellipse_b * sin_angle
        target_z = self.center_z
        p_des = np.array([target_x, target_y, target_z])
        
        v_x = -self.ellipse_a * omega * sin_angle
        v_y =  self.ellipse_b * omega * cos_angle
        v_z = 0.0
        v_des = np.array([v_x, v_y, v_z])

        a_x = -self.ellipse_a * (omega**2) * cos_angle
        a_y = -self.ellipse_b * (omega**2) * sin_angle
        a_z = 0.0
        a_des = np.array([a_x, a_y, a_z])

        return p_des, v_des, a_des

    def control_loop(self):
        if not self.received_first_state:
            return
        
        current_ros_time = self.get_clock().now()
        
        # --- TIMING ---
        if self.start_time is None:
            self.start_time = current_ros_time.nanoseconds / 1e9
            self.last_loop_time = current_ros_time
            return # Skip first loop to establish delta time

        curr_time_sec = current_ros_time.nanoseconds / 1e9
        elapsed = curr_time_sec - self.start_time
        
        dt_nano = (current_ros_time - self.last_loop_time).nanoseconds
        dt_actual = dt_nano / 1e9
        self.last_loop_time = current_ros_time
        
        if dt_actual <= 0.0001: 
            return # Avoid division by zero

        # --- 1. COMPUTE FILTERED ACCELERATION (Like Template) ---
        # Calculate raw acceleration via finite difference
        raw_accel = (self.dq - self.dq_prev) / dt_actual
        # Apply Low Pass Filter
        self.filtered_accel = (self.alpha * raw_accel) + ((1.0 - self.alpha) * self.filtered_accel)
        self.dq_prev = self.dq.copy()

        # --- DYNAMICS UPDATE ---
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        
        ee_pose = self.data.oMf[self.ee_frame_id]
        p_curr = ee_pose.translation
        
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        v_curr_spatial = J @ self.dq
        v_curr = v_curr_spatial[:3]
        w_curr = v_curr_spatial[3:] 

        # --- TRAJECTORY PHASES (Smooth Approach) ---
        HOLD_TIME = 1.0         
        APPROACH_DURATION = 5.0 
        START_TRAJ_TIME = HOLD_TIME + APPROACH_DURATION
        
        p_des = p_curr
        v_des = np.zeros(3)
        a_des = np.zeros(3)

        if elapsed < HOLD_TIME:
             if self.start_approach_pos is None:
                 self.start_approach_pos = p_curr 
             p_des = self.start_approach_pos
             self.sum_error = np.zeros(6) 

        elif elapsed < START_TRAJ_TIME:
            # Smooth interpolation to start of ellipse
            ellipse_start_pos = self.center_pos + np.array([self.ellipse_a, 0, 0])
            t_move = elapsed - HOLD_TIME
            ratio = t_move / APPROACH_DURATION
            smooth_ratio = (1 - math.cos(ratio * math.pi)) / 2
            
            p_des = (1 - smooth_ratio) * self.start_approach_pos + smooth_ratio * ellipse_start_pos
            v_des = np.zeros(3) 
            a_des = np.zeros(3)

        else:
            # Execute Ellipse
            traj_time = elapsed - START_TRAJ_TIME
            p_des, v_des, a_des = self.get_desired_state(traj_time)

        # Plotting storage
        self.actual_x.append(p_curr[0])
        self.actual_y.append(p_curr[1])
        self.target_x.append(p_des[0])
        self.target_y.append(p_des[1])

        # --- CONTROL LAW ---
        pos_error = p_des - p_curr
        vel_error = v_des - v_curr
        rot_error = np.zeros(3)
        ang_vel_error = -w_curr
        
        current_error_vector = np.concatenate([pos_error, rot_error])
        self.sum_error += current_error_vector * dt_actual
        self.sum_error = np.clip(self.sum_error, -2.0, 2.0)

        F_linear = (self.Kp_task[:3] * pos_error) + \
                   (self.Kd_task[:3] * vel_error) + \
                   (self.Ki_task[:3] * self.sum_error[:3]) + \
                   (1.0 * a_des) 
                   
        F_angular = (self.Kp_task[3:] * rot_error) + \
                    (self.Kd_task[3:] * ang_vel_error) + \
                    (self.Ki_task[3:] * self.sum_error[3:])
                    
        F_task = np.concatenate([F_linear, F_angular])

        tau_task = J.T @ F_task
        tau_gravity = self.data.g
        
        tau_total_full = tau_task + tau_gravity

        # --- SAFETY CLIPPING ---
        tau_output_list = []
        for i in range(len(TARGET_JOINTS)):
            idx = self.joint_indices_v[i] 
            tau_output_list.append(tau_total_full[idx])

        # Clip Torque (Hardware Protection)
        tau_output_clipped = np.clip(tau_output_list, -TORQUE_LIMIT, TORQUE_LIMIT)

        # --- 2. CSV LOGGING (Like Template) ---
        # Extract data strictly for the TARGET JOINTS
        log_q = []
        log_dq = []
        log_acc = []
        log_tau = tau_output_clipped.tolist()

        for i in range(len(TARGET_JOINTS)):
            idx = self.joint_indices_v[i]
            log_q.append(self.q[idx])
            log_dq.append(self.dq[idx])
            log_acc.append(self.filtered_accel[idx]) # Use the filtered value

        # Write Row: [episode_id, dt, q..., v..., tau..., acc...]
        # We treat this as "Episode 1"
        row_data = [1, dt_actual] + log_q + log_dq + log_tau + log_acc
        self.csv_writer.writerow(row_data)

        # --- PUBLISH ---
        msg = Float64MultiArray()
        msg.data = tau_output_clipped.tolist() 
        self.torque_pub.publish(msg)

    def stop_robot(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 7
        self.torque_pub.publish(msg)
        self.close_file()

    def close_file(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Data saved to {DATA_FILE}")
            
            # Print Summary (Like Template)
            print("\n" + "="*40)
            print("DATA COLLECTION COMPLETE")
            print("="*40)
            print(f"Data File: {DATA_FILE}")
            print("="*40 + "\n")

# --- MAIN ---
def main(args=None):
    rclpy.init(args=args)
    node = EllipseDataCollector()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ln_target, = ax.plot([], [], 'b--', linewidth=2, label='Target Ellipse')
    ln_actual, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Ellipse Trajectory Tracking')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    def init_plot():
        ln_target.set_data([], [])
        ln_actual.set_data([], [])
        return ln_target, ln_actual

    def update_plot(frame):
        tx = list(node.target_x)
        ty = list(node.target_y)
        ax_dat = list(node.actual_x)
        ay_dat = list(node.actual_y)

        ln_target.set_data(tx, ty)
        ln_actual.set_data(ax_dat, ay_dat)
        return ln_target, ln_actual

    ani = FuncAnimation(fig, update_plot, init_func=init_plot, blit=True, interval=30)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
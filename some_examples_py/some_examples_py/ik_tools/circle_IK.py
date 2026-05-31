import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION ---
from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

TARGET_JOINTS = [
    'joint_1', 'joint_2', 'joint_3', 'joint_4', 
    'joint_5', 'joint_6', 'joint_7'
]

class CircleTorqueController(Node):
    def __init__(self):
        super().__init__('circle_torque_controller')

        self.urdf_path = URDF_PATH
        self.ee_frame_name = 'endeffector'
        
        # --- TUNING: SPEED & SIZE ---
        # UPDATED: Slower speed (10 seconds per circle instead of 5)
        self.circle_period = 7.0 
        self.circle_z = 0.72
        self.circle_radius = 0.285
        self.center_pos = np.array([0.0, 0.0, self.circle_z]) 

        # --- TUNING GAINS (PID) ---
        self.Kp_task = np.array([800.0, 800.0, 800.0, 50.0, 50.0, 50.0]) 
        self.Kd_task = np.array([40.0, 40.0, 40.0, 2.0, 2.0, 2.0])       
        self.Ki_task = np.array([100.0, 100.0, 100.0, 0.5, 0.5, 0.5])

        self.sum_error = np.zeros(6) 
        self.dt = 0.002 
        # ---------------------

        # Load Pinocchio
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

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

        self.q = pin.neutral(self.model) 
        self.dq = np.zeros(self.model.nv) 
        self.received_first_state = False

        # State storage for smoothing
        self.start_approach_pos = None

        # Plotting Data
        self.actual_x, self.actual_y = [], []
        self.target_x, self.target_y = [], []

        # ROS
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/effort_arm_controller/commands', 10)

        self.start_time = None
        self.control_timer = self.create_timer(self.dt, self.control_loop)

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
        omega = 2 * math.pi / self.circle_period
        angle = omega * t_rel
        
        # Position
        target_x = self.center_pos[0] + self.circle_radius * math.cos(angle)
        target_y = self.center_pos[1] + self.circle_radius * math.sin(angle)
        target_z = self.circle_z
        p_des = np.array([target_x, target_y, target_z])
        
        # Velocity
        v_x = -self.circle_radius * omega * math.sin(angle)
        v_y =  self.circle_radius * omega * math.cos(angle)
        v_z = 0.0
        v_des = np.array([v_x, v_y, v_z])

        # Acceleration
        a_x = -self.circle_radius * (omega**2) * math.cos(angle)
        a_y = -self.circle_radius * (omega**2) * math.sin(angle)
        a_z = 0.0
        a_des = np.array([a_x, a_y, a_z])

        return p_des, v_des, a_des

    def control_loop(self):
        if not self.received_first_state:
            return

        # --- DYNAMICS ---
        pin.computeAllTerms(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)
        
        ee_pose = self.data.oMf[self.ee_frame_id]
        p_curr = ee_pose.translation
        
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        v_curr_spatial = J @ self.dq
        v_curr = v_curr_spatial[:3]
        w_curr = v_curr_spatial[3:] 

        # --- TRAJECTORY GENERATION ---
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        curr_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = curr_time - self.start_time
        
        # Timings
        HOLD_TIME = 1.0       # Hold still for 1s
        APPROACH_DURATION = 5.0 # Take 5s to move to start (Very slow and safe)
        START_CIRCLE_TIME = HOLD_TIME + APPROACH_DURATION
        
        # Default Targets
        p_des = p_curr
        v_des = np.zeros(3)
        a_des = np.zeros(3)

        if elapsed < HOLD_TIME:
             # Phase 0: Just Hold Current Position
             if self.start_approach_pos is None:
                 self.start_approach_pos = p_curr # Capture where we are
             
             p_des = self.start_approach_pos
             self.sum_error = np.zeros(6) # Don't accumulate error yet

        elif elapsed < START_CIRCLE_TIME:
            # Phase 1: Smoothly Move to Circle Start
            # We want to go from [start_approach_pos] to [circle_start_pos]
            circle_start_pos = self.center_pos + np.array([self.circle_radius, 0, 0])
            
            # Calculate how far along we are (0.0 to 1.0)
            t_move = elapsed - HOLD_TIME
            ratio = t_move / APPROACH_DURATION
            
            # Use Cosine Interpolation for smoothness (Ease-In, Ease-Out)
            # This makes velocity 0 at start and 0 at end -> NO OVERSHOOT
            smooth_ratio = (1 - math.cos(ratio * math.pi)) / 2
            
            p_des = (1 - smooth_ratio) * self.start_approach_pos + smooth_ratio * circle_start_pos
            v_des = np.zeros(3) # Keeping velocity target 0 for stability during approach
            a_des = np.zeros(3)

        else:
            # Phase 2: Execute Circle
            circle_time = elapsed - START_CIRCLE_TIME
            p_des, v_des, a_des = self.get_desired_state(circle_time)

        # Store for plotting
        self.actual_x.append(p_curr[0])
        self.actual_y.append(p_curr[1])
        self.target_x.append(p_des[0])
        self.target_y.append(p_des[1])

        # --- PID CONTROL LAW ---
        pos_error = p_des - p_curr
        vel_error = v_des - v_curr
        rot_error = np.zeros(3)
        ang_vel_error = -w_curr
        
        current_error_vector = np.concatenate([pos_error, rot_error])
        self.sum_error += current_error_vector * self.dt
        
        integral_limit = 2.0 
        self.sum_error = np.clip(self.sum_error, -integral_limit, integral_limit)

        F_linear = (self.Kp_task[:3] * pos_error) + \
                   (self.Kd_task[:3] * vel_error) + \
                   (self.Ki_task[:3] * self.sum_error[:3]) + \
                   (1.0 * a_des) 
                   
        F_angular = (self.Kp_task[3:] * rot_error) + \
                    (self.Kd_task[3:] * ang_vel_error) + \
                    (self.Ki_task[3:] * self.sum_error[3:])
                    
        F_task = np.concatenate([F_linear, F_angular])

        # Torque Mapping
        tau_task = J.T @ F_task
        tau_gravity = self.data.g
        
        tau_total_full = tau_task + tau_gravity

        tau_output = []
        for i in range(len(TARGET_JOINTS)):
            idx = self.joint_indices_v[i] 
            tau_output.append(tau_total_full[idx])

        tau_output = np.clip(tau_output, -50.0, 50.0)

        msg = Float64MultiArray()
        msg.data = tau_output.tolist() 
        self.torque_pub.publish(msg)

    def stop_robot(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 7
        self.torque_pub.publish(msg)

# --- MAIN ---
def main(args=None):
    rclpy.init(args=args)
    node = CircleTorqueController()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ln_target, = ax.plot([], [], 'b--', linewidth=2, label='Target')
    ln_actual, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
    
    ax.set_xlim(-0.450, 0.450)
    ax.set_ylim(-0.450, 0.450)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Slow & Smooth Trajectory')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    def init_plot():
        ln_target.set_data([], [])
        ln_actual.set_data([], [])
        return ln_target, ln_actual

    def update_plot(frame):
        # tx = list(node.target_x)[-500:]
        # ty = list(node.target_y)[-500:]
        # ax_dat = list(node.actual_x)[-500:]
        # ay_dat = list(node.actual_y)[-500:]
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
"""
Code for y-z plane movement of the end-effector in a circular trajectory
using torque control with Pinocchio and ROS2.
"""
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray
# import pinocchio as pin
# import numpy as np
# import math
# import threading
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- CONFIGURATION ---
# from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(
    get_package_share_directory("daadbot_desc"),
    "urdf",
    "urdf_inverted_torque",
    "daadbot.urdf"
)

# TARGET_JOINTS = [
#     'joint_1', 'joint_2', 'joint_3', 'joint_4', 
#     'joint_5', 'joint_6', 'joint_7'
# ]

# class CircleTorqueControllerYZ(Node):
#     def __init__(self):
#         super().__init__('circle_torque_controller_yz')

#         self.urdf_path = URDF_PATH
#         self.ee_frame_name = 'endeffector'
        
#         # --- CIRCLE PARAMETERS (YZ PLANE) ---
#         self.circle_period = 10.0 
#         self.circle_z_center = 0.75  # Center height
#         self.circle_radius = 0.24
        
#         # Center is at X=-0.36, Y=0, Z=0.6
#         self.center_pos = np.array([-0.36, 0.0, self.circle_z_center]) 

#         # --- TUNING GAINS (PID) ---
#         self.Kp_task = np.array([800.0, 800.0, 800.0, 50.0, 50.0, 50.0]) 
#         self.Kd_task = np.array([40.0, 40.0, 40.0, 2.0, 2.0, 2.0])       
#         self.Ki_task = np.array([100.0, 100.0, 100.0, 0.5, 0.5, 0.5])

#         self.sum_error = np.zeros(6) 
#         self.dt = 0.002 
#         # ---------------------

#         # Load Pinocchio
#         self.model = pin.buildModelFromUrdf(self.urdf_path)
#         self.data = self.model.createData()
#         self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)

#         # Joint Mapping
#         self.joint_indices_q = []
#         self.joint_indices_v = []
#         for name in TARGET_JOINTS:
#             if self.model.existJointName(name):
#                 joint_id = self.model.getJointId(name)
#                 self.joint_indices_q.append(self.model.joints[joint_id].idx_q)
#                 self.joint_indices_v.append(self.model.joints[joint_id].idx_v)
#             else:
#                 self.get_logger().error(f"Joint {name} not found in URDF!")

#         self.q = pin.neutral(self.model) 
#         self.dq = np.zeros(self.model.nv) 
#         self.received_first_state = False
#         self.start_approach_pos = None

#         # Plotting Data (Storing Y and Z now)
#         self.actual_y, self.actual_z = [], []
#         self.target_y, self.target_z = [], []

#         # ROS
#         self.joint_state_sub = self.create_subscription(
#             JointState, '/joint_states', self.joint_state_callback, 10)
#         self.torque_pub = self.create_publisher(
#             Float64MultiArray, '/effort_arm_controller/commands', 10)

#         self.start_time = None
#         self.control_timer = self.create_timer(self.dt, self.control_loop)

#     def joint_state_callback(self, msg):
#         msg_map = {name: i for i, name in enumerate(msg.name)}
#         try:
#             for i, joint_name in enumerate(TARGET_JOINTS):
#                 if joint_name in msg_map:
#                     msg_idx = msg_map[joint_name]
#                     pin_q_idx = self.joint_indices_q[i]
#                     pin_v_idx = self.joint_indices_v[i]
#                     self.q[pin_q_idx] = msg.position[msg_idx]
#                     self.dq[pin_v_idx] = msg.velocity[msg_idx]
#             self.received_first_state = True
#         except IndexError:
#             pass

#     def get_desired_state(self, t_rel):
#         omega = 2 * math.pi / self.circle_period
#         angle = omega * t_rel
        
#         # --- POSITION (YZ Plane) ---
#         # X is fixed at Center X (0.0)
#         target_x = self.center_pos[0]
        
#         # Y = r * cos(angle)
#         target_y = self.center_pos[1] + self.circle_radius * math.cos(angle)
        
#         # Z = Center_Z + r * sin(angle)
#         target_z = self.center_pos[2] + self.circle_radius * math.sin(angle)
        
#         p_des = np.array([target_x, target_y, target_z])
        
#         # --- VELOCITY ---
#         v_x = 0.0
#         v_y = -self.circle_radius * omega * math.sin(angle)
#         v_z =  self.circle_radius * omega * math.cos(angle)
#         v_des = np.array([v_x, v_y, v_z])

#         # --- ACCELERATION (Centripetal) ---
#         a_x = 0.0
#         a_y = -self.circle_radius * (omega**2) * math.cos(angle)
#         a_z = -self.circle_radius * (omega**2) * math.sin(angle)
#         a_des = np.array([a_x, a_y, a_z])

#         return p_des, v_des, a_des

#     def control_loop(self):
#         if not self.received_first_state:
#             return

#         # Dynamics Update
#         pin.computeAllTerms(self.model, self.data, self.q, self.dq)
#         pin.updateFramePlacements(self.model, self.data)
        
#         ee_pose = self.data.oMf[self.ee_frame_id]
#         p_curr = ee_pose.translation
        
#         J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
#         v_curr_spatial = J @ self.dq
#         v_curr = v_curr_spatial[:3]
#         w_curr = v_curr_spatial[3:] 

#         # Trajectory Logic
#         if self.start_time is None:
#             self.start_time = self.get_clock().now().nanoseconds / 1e9
        
#         curr_time = self.get_clock().now().nanoseconds / 1e9
#         elapsed = curr_time - self.start_time
        
#         HOLD_TIME = 1.0       
#         APPROACH_DURATION = 5.0 
#         START_CIRCLE_TIME = HOLD_TIME + APPROACH_DURATION
        
#         p_des = p_curr
#         v_des = np.zeros(3)
#         a_des = np.zeros(3)

#         if elapsed < HOLD_TIME:
#              # Phase 0: Hold
#              if self.start_approach_pos is None:
#                  self.start_approach_pos = p_curr 
             
#              p_des = self.start_approach_pos
#              self.sum_error = np.zeros(6) 

#         elif elapsed < START_CIRCLE_TIME:
#             # Phase 1: Smooth Approach to Start Point on YZ Circle
#             # Start Point: (0, 0 + radius, center_z)
#             # This is Y = 0.24, Z = 0.72
#             circle_start_pos = self.center_pos + np.array([0, self.circle_radius, 0])
            
#             t_move = elapsed - HOLD_TIME
#             ratio = t_move / APPROACH_DURATION
#             smooth_ratio = (1 - math.cos(ratio * math.pi)) / 2
            
#             p_des = (1 - smooth_ratio) * self.start_approach_pos + smooth_ratio * circle_start_pos
#             v_des = np.zeros(3)
#             a_des = np.zeros(3)

#         else:
#             # Phase 2: Circle in YZ
#             circle_time = elapsed - START_CIRCLE_TIME
#             p_des, v_des, a_des = self.get_desired_state(circle_time)

#         # Store for plotting (YZ Plane)
#         self.actual_y.append(p_curr[1])
#         self.actual_z.append(p_curr[2])
#         self.target_y.append(p_des[1])
#         self.target_z.append(p_des[2])

#         # Control Law
#         pos_error = p_des - p_curr
#         vel_error = v_des - v_curr
#         rot_error = np.zeros(3)
#         ang_vel_error = -w_curr
        
#         current_error_vector = np.concatenate([pos_error, rot_error])
#         self.sum_error += current_error_vector * self.dt
        
#         integral_limit = 2.0 
#         self.sum_error = np.clip(self.sum_error, -integral_limit, integral_limit)

#         F_linear = (self.Kp_task[:3] * pos_error) + \
#                    (self.Kd_task[:3] * vel_error) + \
#                    (self.Ki_task[:3] * self.sum_error[:3]) + \
#                    (1.0 * a_des) 
                   
#         F_angular = (self.Kp_task[3:] * rot_error) + \
#                     (self.Kd_task[3:] * ang_vel_error) + \
#                     (self.Ki_task[3:] * self.sum_error[3:])
                    
#         F_task = np.concatenate([F_linear, F_angular])

#         # Torque Mapping
#         tau_task = J.T @ F_task
#         tau_gravity = self.data.g
        
#         tau_total_full = tau_task + tau_gravity

#         tau_output = []
#         for i in range(len(TARGET_JOINTS)):
#             idx = self.joint_indices_v[i] 
#             tau_output.append(tau_total_full[idx])

#         tau_output = np.clip(tau_output, -50.0, 50.0)

#         msg = Float64MultiArray()
#         msg.data = tau_output.tolist() 
#         self.torque_pub.publish(msg)

#     def stop_robot(self):
#         msg = Float64MultiArray()
#         msg.data = [0.0] * 7
#         self.torque_pub.publish(msg)

# # --- MAIN ---
# def main(args=None):
#     rclpy.init(args=args)
#     node = CircleTorqueControllerYZ()

#     spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     spin_thread.start()

#     # Setup Plot for YZ Plane
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ln_target, = ax.plot([], [], 'b--', linewidth=2, label='Target')
#     ln_actual, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
    
#     # Adjust limits for YZ view
#     # Y goes from -0.3 to 0.3
#     # Z goes from 0.4 to 1.0 (since center is 0.72)
#     ax.set_xlim(-0.35, 0.35)
#     ax.set_ylim(0.40, 1.10)
#     ax.set_xlabel('Y [m]')
#     ax.set_ylabel('Z [m]')
#     ax.set_title('YZ Plane Trajectory (Side View)')
#     ax.legend()
#     ax.grid(True)
#     ax.set_aspect('equal')

#     def init_plot():
#         ln_target.set_data([], [])
#         ln_actual.set_data([], [])
#         return ln_target, ln_actual

#     def update_plot(frame):
#         # Plot Y vs Z
#         ty = list(node.target_y)
#         tz = list(node.target_z)
#         ay = list(node.actual_y)
#         az = list(node.actual_z)

#         ln_target.set_data(ty, tz)
#         ln_actual.set_data(ay, az)
#         return ln_target, ln_actual

#     ani = FuncAnimation(fig, update_plot, init_func=init_plot, blit=True, interval=30)

#     try:
#         plt.show()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
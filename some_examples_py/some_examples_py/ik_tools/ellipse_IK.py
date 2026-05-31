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

class EllipseTorqueController(Node):
    def __init__(self):
        super().__init__('ellipse_torque_controller')

        self.urdf_path = URDF_PATH
        self.ee_frame_name = 'endeffector'
        
        # --- TUNING: SPEED & SIZE ---
        # Slower speed: 10 seconds to complete one full ellipse
        self.trajectory_period = 7.5 
        self.center_z = 0.72
        self.center_pos = np.array([0.0, 0.0, self.center_z]) 

        # --- ELLIPSE PARAMETERS (Tunable) ---
        self.ellipse_a = 0.150  # Semi-major axis (Radius in X)
        self.ellipse_b = 0.270  # Semi-minor axis (Radius in Y)

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
        """
        Computes the desired Position, Velocity, and Acceleration 
        for an Ellipse trajectory in the XY plane.
        """
        omega = 2 * math.pi / self.trajectory_period
        angle = omega * t_rel
        
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)

        # --- Position ---
        # x = cx + a * cos(t)
        # y = cy + b * sin(t)
        target_x = self.center_pos[0] + self.ellipse_a * cos_angle
        target_y = self.center_pos[1] + self.ellipse_b * sin_angle
        target_z = self.center_z
        p_des = np.array([target_x, target_y, target_z])
        
        # --- Velocity ---
        # dx = -a * w * sin(t)
        # dy =  b * w * cos(t)
        v_x = -self.ellipse_a * omega * sin_angle
        v_y =  self.ellipse_b * omega * cos_angle
        v_z = 0.0
        v_des = np.array([v_x, v_y, v_z])

        # --- Acceleration ---
        # ddx = -a * w^2 * cos(t)
        # ddy = -b * w^2 * sin(t)
        a_x = -self.ellipse_a * (omega**2) * cos_angle
        a_y = -self.ellipse_b * (omega**2) * sin_angle
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
        HOLD_TIME = 1.0         # Hold still for 1s
        APPROACH_DURATION = 5.0 # Take 5s to move to start (Slow & Safe)
        START_TRAJ_TIME = HOLD_TIME + APPROACH_DURATION
        
        # Default Targets
        p_des = p_curr
        v_des = np.zeros(3)
        a_des = np.zeros(3)

        if elapsed < HOLD_TIME:
             # Phase 0: Just Hold Current Position
             if self.start_approach_pos is None:
                 self.start_approach_pos = p_curr 
             
             p_des = self.start_approach_pos
             self.sum_error = np.zeros(6) 

        elif elapsed < START_TRAJ_TIME:
            # Phase 1: Smoothly Move to Ellipse Start
            # The ellipse starts at angle=0, which is:
            # x = center_x + a, y = center_y
            ellipse_start_pos = self.center_pos + np.array([self.ellipse_a, 0, 0])
            
            # Calculate how far along we are (0.0 to 1.0)
            t_move = elapsed - HOLD_TIME
            ratio = t_move / APPROACH_DURATION
            
            # Cosine Interpolation (Ease-In, Ease-Out)
            smooth_ratio = (1 - math.cos(ratio * math.pi)) / 2
            
            p_des = (1 - smooth_ratio) * self.start_approach_pos + smooth_ratio * ellipse_start_pos
            v_des = np.zeros(3) 
            a_des = np.zeros(3)

        else:
            # Phase 2: Execute Ellipse
            traj_time = elapsed - START_TRAJ_TIME
            p_des, v_des, a_des = self.get_desired_state(traj_time)

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
    node = EllipseTorqueController()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ln_target, = ax.plot([], [], 'b--', linewidth=2, label='Target Ellipse')
    ln_actual, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
    
    # Adjusted limits for potentially larger ellipses
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
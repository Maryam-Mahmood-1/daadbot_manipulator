#!/usr/bin/env python3
import sys
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
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

EE_FRAME_NAME = 'endeffector'

# Target setup
TARGET_POS = np.array([0.195, 0.0, 0.563]) 
CIRCLE_RADIUS = 0.195                      
MOVE_TIME = 5.0                            
CIRCLE_PERIOD = 10.0                        

class CircularMotionNode(Node):
    def __init__(self):
        super().__init__('circular_motion_node')
        
        # 1. Setup Model
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(EE_FRAME_NAME)
        
        # 2. State Headers
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)
        self.q_des_integ = None 
        self.start_pos = None
        self.start_time = None
        self.initialized = False

        # 3. Data Buffers for Plotting (Shared Resources)
        # We use lists; Python lists are thread-safe for append/read
        self.vis_des_x = []
        self.vis_des_y = []
        self.vis_des_z = []
        
        self.vis_act_x = []
        self.vis_act_y = []
        self.vis_act_z = []

        # 4. ROS Setup
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        self.joint_indices = [self.model.getJointId(n)-1 for n in self.joint_names]
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.state_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/effort_arm_controller/commands', 10)
        
        # Control Loop at 1000 Hz
        self.timer = self.create_timer(0.001, self.control_loop) 
        print("Control Node Initialized. Waiting for robot state...")

    def state_cb(self, msg):
        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                idx = self.model.getJointId(name) - 1
                self.q[idx] = msg.position[i]
                self.v[idx] = msg.velocity[i]

        if not self.initialized:
            self.initialized = True
            self.q_des_integ = self.q.copy()
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            self.start_pos = self.data.oMf[self.ee_id].translation.copy()
            self.start_time = self.get_clock().now()
            print(f"Phase 1: Moving to start point {TARGET_POS}...")

    def control_loop(self):
        if not self.initialized: return

        # 1. Time Calculation
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        
        p_des = np.zeros(3)
        v_des = np.zeros(3)

        # PHASE 1: Smooth Move
        if t < MOVE_TIME:
            tau = t / MOVE_TIME
            s = 10*(tau**3) - 15*(tau**4) + 6*(tau**5)
            ds = (30*(tau**2) - 60*(tau**3) + 30*(tau**4)) / MOVE_TIME
            p_des = self.start_pos + s * (TARGET_POS - self.start_pos)
            v_des = ds * (TARGET_POS - self.start_pos)

        # PHASE 2: Circle
        else:
            t_circle = t - MOVE_TIME
            omega = 2 * np.pi / CIRCLE_PERIOD
            p_des[0] = CIRCLE_RADIUS * np.cos(omega * t_circle)
            p_des[1] = CIRCLE_RADIUS * np.sin(omega * t_circle)
            p_des[2] = TARGET_POS[2] 
            v_des[0] = -CIRCLE_RADIUS * omega * np.sin(omega * t_circle)
            v_des[1] =  CIRCLE_RADIUS * omega * np.cos(omega * t_circle)
            v_des[2] = 0.0

        # 2. Kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        p_curr = self.data.oMf[self.ee_id].translation
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # 3. Store Data for Plotting (Only keep last 2000 points to save RAM/Speed)
        # Note: We append here, the Visualization thread reads.
        if len(self.vis_des_x) > 2000:
            self.vis_des_x.pop(0); self.vis_des_y.pop(0); self.vis_des_z.pop(0)
            self.vis_act_x.pop(0); self.vis_act_y.pop(0); self.vis_act_z.pop(0)
            
        self.vis_des_x.append(p_des[0]); self.vis_des_y.append(p_des[1]); self.vis_des_z.append(p_des[2])
        self.vis_act_x.append(p_curr[0]); self.vis_act_y.append(p_curr[1]); self.vis_act_z.append(p_curr[2])

        # 4. Control Law
        err_pos = p_des - p_curr
        v_cmd_spatial = np.concatenate([v_des + 5.0 * err_pos, np.zeros(3)]) 
        J_dls = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dq_des = J_dls @ v_cmd_spatial

        self.q_des_integ += dq_des * 0.001 
        err_q = pin.difference(self.model, self.q, self.q_des_integ)
        err_v = dq_des - self.v
        kp = 100.0
        kd = 20.0
        ddq_des = (dq_des - self.v) / 0.001 
        tau = pin.rnea(self.model, self.data, self.q, self.v, ddq_des + kp*err_q + kd*err_v)

        msg = Float64MultiArray()
        msg.data = tau[self.joint_indices].tolist()
        self.pub.publish(msg)

# --- THREADING WRAPPER ---
def run_ros_node(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    node = CircularMotionNode()
    
    # Start ROS node in a separate thread so it doesn't block plotting
    ros_thread = threading.Thread(target=run_ros_node, args=(node,), daemon=True)
    ros_thread.start()

    # --- SETUP REALTIME PLOT ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title("Real-Time Trajectory Tracking")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    # Initialize Lines
    line_des, = ax.plot([], [], [], 'g--', label='Desired')
    line_act, = ax.plot([], [], [], 'r-', label='Actual')
    
    # Point at current head
    head_point, = ax.plot([], [], [], 'ro', markersize=5)

    ax.legend()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.0)

    def update_plot(frame):
        # Read data from the node safely
        if len(node.vis_des_x) > 1:
            # Update data
            line_des.set_data(node.vis_des_x, node.vis_des_y)
            line_des.set_3d_properties(node.vis_des_z)
            
            line_act.set_data(node.vis_act_x, node.vis_act_y)
            line_act.set_3d_properties(node.vis_act_z)
            
            # Update head point
            head_point.set_data([node.vis_act_x[-1]], [node.vis_act_y[-1]])
            head_point.set_3d_properties([node.vis_act_z[-1]])

        return line_des, line_act, head_point

    # Animate at 50ms interval (20Hz) - Fast enough for eyes, slow enough for CPU
    ani = FuncAnimation(fig, update_plot, interval=50, blit=False)
    
    try:
        plt.show() # This blocks the main thread
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
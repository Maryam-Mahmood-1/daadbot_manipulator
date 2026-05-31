#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math

class TargetAngleSetter(Node):
    def __init__(self):
        super().__init__('target_angle_setter')
        self.current_positions = [0.0] * 7

        # Subscribe to actual joint positions
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # Publisher for degrees
        self.target_pub = self.create_publisher(
            Float64MultiArray,
            '/target_joint_angles_deg',
            10
        )

        # Publish at fixed rate
        self.timer = self.create_timer(0.01, self.publish_target)  # 10 Hz

        self.get_logger().info("Publishing current joint angles in degrees.")

    def joint_callback(self, msg: JointState):
        if len(msg.position) >= 7:
            # radians → degrees
            self.current_positions = [
                pos * 180.0 / math.pi for pos in msg.position[:7]
            ]

    def publish_target(self):
        msg = Float64MultiArray()
        msg.data = self.current_positions
        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TargetAngleSetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

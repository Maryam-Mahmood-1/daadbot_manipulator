import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
import math


class TargetToControllerBridge(Node):
    def __init__(self):
        super().__init__("target_to_controller_bridge")

        # --- Arm bridge ---
        self.sub = self.create_subscription(
            Float64MultiArray,
            "/target_joint_angles_deg",
            self.target_callback,
            10
        )
        self.pub = self.create_publisher(
            Float64MultiArray,
            "/position_forw_arm_controller/commands",
            10
        )

        # --- Gripper bridge ---
        self.gripper_sub = self.create_subscription(
            String,
            "/gripper_command",
            self.gripper_callback,
            10
        )
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            "/position_forw_gripper_controller/commands",
            10
        )

        # Publish default gripper position = closed (0 rad)
        default_msg = Float64MultiArray()
        default_msg.data = [0.0]
        self.gripper_pub.publish(default_msg)
        self.get_logger().info("Initialized gripper at 0.0 rad (closed)")

    # --- Arm ---
    def target_callback(self, msg: Float64MultiArray):
        rad_msg = Float64MultiArray()
        rad_msg.data = [math.radians(a) for a in msg.data]
        self.get_logger().info(f"Publishing to arm controller: {rad_msg.data}")
        self.pub.publish(rad_msg)

    # --- Gripper ---
    def gripper_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        rad_msg = Float64MultiArray()

        if cmd == "c":  # close
            rad_msg.data = [0.0]
            self.get_logger().info("Gripper CLOSE → 0.0 rad")
        elif cmd == "o":  # open
            rad_msg.data = [1.0]
            self.get_logger().info("Gripper OPEN → 1.5 rad")
        else:
            self.get_logger().warn(f"Unknown gripper command: '{cmd}'")
            return

        self.gripper_pub.publish(rad_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetToControllerBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

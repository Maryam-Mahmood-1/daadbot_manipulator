import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Assuming the Ignition Gazebo force-torque sensor message is compatible with this placeholder:
# You need to confirm the actual ROS 2 message type after bridging
from geometry_msgs.msg import WrenchStamped  # Often force-torque sensors publish this

class TorqueZPublisher(Node):
    def __init__(self):
        super().__init__('torquez_publisher')

        # Joint names as strings
        self.joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.torque_z_values = [0.0] * 7

        # Subscribers for each joint's torque sensor topic
        self.subscribers = []
        for i in range(1, 8):
            topic = f"/world/empty/model/daadbot/joint/joint_{i}/sensor/tcp_fts_sensor_joint_{i}/forcetorque"
            sub = self.create_subscription(
                WrenchStamped,
                topic,
                lambda msg, idx=i-1: self.callback(msg, idx),
                10)
            self.subscribers.append(sub)

        # Publisher for joint torque z
        self.pub = self.create_publisher(JointState, '/daadbot/joint_torque_z', 10)

        # Timer to publish at 50 Hz or so
        self.timer = self.create_timer(0.02, self.publish_torque_z)

    def callback(self, msg: WrenchStamped, idx: int):
        # Extract torque.z
        self.torque_z_values[idx] = msg.wrench.torque.z

    def publish_torque_z(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.effort = self.torque_z_values
        self.pub.publish(joint_state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TorqueZPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

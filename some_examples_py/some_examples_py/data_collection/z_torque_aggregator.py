import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from daadbot_msgs.msg import ZTorques

class ZTorqueAggregator(Node):
    def __init__(self):
        super().__init__('z_torque_aggregator')

        self.z_torques = [0.0] * 7
        self.subscribers = []

        # Subscribe to each joint's forcetorque topic
        for i in range(7):
            topic = f"/world/empty/model/daadbot/joint/joint_{i+1}/sensor/tcp_fts_sensor_joint_{i+1}/forcetorque"
            sub = self.create_subscription(
                WrenchStamped,
                topic,
                self.make_callback(i),
                10
            )
            self.subscribers.append(sub)

        self.publisher = self.create_publisher(ZTorques, '/daadbot/z_torques', 10)
        self.timer = self.create_timer(0.02, self.publish_z_torques)  # 50 Hz

    def make_callback(self, index):
        def callback(msg):
            self.z_torques[index] = msg.wrench.torque.z
        return callback

    def publish_z_torques(self):
        msg = ZTorques()
        msg.z_torques = self.z_torques
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ZTorqueAggregator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

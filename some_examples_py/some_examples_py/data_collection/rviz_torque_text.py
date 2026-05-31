import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from daadbot_msgs.msg import ZTorques

class ZTorqueTextMarkers(Node):
    def __init__(self):
        super().__init__('z_torque_text_markers')

        self.joint_frames = [
            ("joint_1", (-0.75, 0.75, 1.4)),
            ("joint_2", (-0.75, 0.75, 1.2)),
            ("joint_3", (-0.75, 0.75, 1.0)),
            ("joint_4", (-0.75, 0.75, 0.8)),
            ("joint_5", (-0.75, 0.75, 0.6)),
            ("joint_6", (-0.75, 0.75, 0.4)),
            ("joint_7", (-0.75, 0.75, 0.2))
        ]

        self.subscription = self.create_subscription(
            ZTorques,
            '/daadbot/z_torques',
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/z_torque_text_array', 10)


        self.latest_msg = None

        # Timer: publish markers at 5 Hz (every 0.2s)
        self.timer = self.create_timer(0.2, self.publish_markers)

    def callback(self, msg):
        self.latest_msg = msg

    def publish_markers(self):
        if self.latest_msg is None:
            return

        marker_array = MarkerArray()

        for i, (joint_name, position) in enumerate(self.joint_frames):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "z_torque_text"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2] + 0.3
            marker.pose.orientation.w = 1.0

            z_torque = self.latest_msg.z_torques[i] if i < len(self.latest_msg.z_torques) else 0.0
            marker.text = f"{joint_name}: {z_torque:.2f}"

            marker.scale.z = 0.1  # Text height
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ZTorqueTextMarkers()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

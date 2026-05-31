#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker


class TFVisualizer(Node):
    def __init__(self):
        super().__init__('tf_visualizer')

        # Frames of interest
        self.parent_frame = 'camera_link'
        self.child_frame = 'endeffector'

        # TF buffer/listener
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'tf_pose', 10)
        self.marker_pub = self.create_publisher(Marker, 'tf_arrow', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.publish_tf)

    def publish_tf(self):
        try:
            trans = self.buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time()
            )

            # --- PoseStamped for orientation (RViz triad) ---
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.parent_frame
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            self.pose_pub.publish(pose)

            # --- Marker Arrow for translation vector (world -> EE) ---
            marker = Marker()
            marker.header = pose.header
            marker.ns = "tf_visualizer"
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # shaft diameter
            marker.scale.y = 0.05  # head diameter
            marker.scale.z = 0.1   # head length
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Start at parent (world origin), end at EE position
            start = Point(x=0.0, y=0.0, z=0.0)
            end = Point(
                x=trans.transform.translation.x,
                y=trans.transform.translation.y,
                z=trans.transform.translation.z
            )
            marker.points = [start, end]

            self.marker_pub.publish(marker)

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")


def main():
    rclpy.init()
    node = TFVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

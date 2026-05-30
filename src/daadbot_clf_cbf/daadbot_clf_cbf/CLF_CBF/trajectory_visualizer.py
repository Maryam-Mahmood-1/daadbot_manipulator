import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker) # Publish once per second
        
        # Ellipse Parameters (Must match your TrajectoryGenerator)
        self.center_z = 0.72
        self.a = 0.15
        self.b = 0.36
        
        self.get_logger().info("Trajectory Visualizer Node Started")

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "world" # Or "base_link" if your robot is fixed
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # ID and Type
        marker.ns = "reference_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Scale (Line Width)
        marker.scale.x = 0.005 # 5mm thick line
        
        # Color (Red, fully opaque)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Geometry: Draw the Ellipse
        # We create 100 points to make a smooth loop
        steps = 100
        for i in range(steps + 1):
            theta = 2.0 * math.pi * i / steps
            
            # Parametric Ellipse Equations
            x = self.a * math.cos(theta)
            y = self.b * math.sin(theta)
            z = self.center_z
            
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            marker.points.append(p)
            
        self.publisher_.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
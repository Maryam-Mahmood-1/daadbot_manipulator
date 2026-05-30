import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')
        self.publisher_ = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker) # Publish 1 Hz
        
        # --- ELLIPSE PARAMETERS (Matched to your TrajectoryGenerator) ---
        self.center_x = 0.0
        self.center_y = 0.0
        self.center_z = 0.0  # Planar robot is at z=0
        
        self.a = 1.6  # Radius on X
        self.b = 0.9  # Radius on Y
        
        self.get_logger().info("Trajectory Visualizer Node Started (1.6m x 0.9m)")

    def publish_marker(self):
        marker = Marker()
        # "world" is usually the fixed frame for Gazebo/Pinocchio. 
        # If your marker doesn't appear, try changing this to "base" or "base_link"
        marker.header.frame_id = "world" 
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # ID and Type
        marker.ns = "reference_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Scale (Line Width)
        marker.scale.x = 0.01 # 1cm thick line for better visibility
        
        # Color (Green, fully opaque)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Geometry: Draw the Ellipse
        steps = 200 # More steps for a smoother large ellipse
        for i in range(steps + 1):
            theta = 2.0 * math.pi * i / steps
            
            # Parametric Ellipse Equations (x = cx + a*cos(t), y = cy + b*sin(t))
            x = self.center_x + self.a * math.cos(theta)
            y = self.center_y + self.b * math.sin(theta)
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
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class UnifiedSafetyVisualizer(Node):
    def __init__(self):
        super().__init__('unified_safety_visualizer')
        # Publisher for markers
        self.pub = self.create_publisher(Marker, '/safety_marker', 10)
        
        # Timer to keep publishing (Rviz needs constant updates or persistent markers)
        self.timer = self.create_timer(0.5, self.publish_all_zones)
        self.get_logger().info("Unified Visualizer Started: Green Safe Zone + Red Danger Zones")

    def publish_all_zones(self):
        """
        Publishes 3 markers:
        1. Green Safe Zone (Center)
        2. Red Unsafe Zone (Left)
        3. Red Unsafe Zone (Right)
        """
        
        # === 1. GREEN SAFE ZONE (Center) ===
        # Position: Center (0,0,0.72)
        # Dimensions: 0.6m x 0.48m x 0.01m
        self.publish_box(
            marker_id=0,
            ns="safe_zone",
            center_x=0.0,
            center_y=0.0,
            center_z=0.72,
            scale_x=2.4,
            scale_y=0.48,
            scale_z=0.01,
            r=0.0, g=1.0, b=0.0, a=0.27 # Green, Semi-Transparent
        )

        # === 2. LEFT UNSAFE ZONE (Negative Y) ===
        # Covering Y from -1.0 to -0.24
        self.publish_box(
            marker_id=1,
            ns="unsafe_zone",
            center_x=0.0,
            center_y=-0.62, 
            center_z=0.75,
            scale_x=2.4,
            scale_y=0.76,
            scale_z=1.5,
            r=1.0, g=0.0, b=0.0, a=0.27 # Red, Semi-Transparent
        )

        # === 3. RIGHT UNSAFE ZONE (Positive Y) ===
        # Covering Y from 0.24 to 1.0
        self.publish_box(
            marker_id=2,
            ns="unsafe_zone",
            center_x=0.0,
            center_y=0.62,
            center_z=0.75,
            scale_x=2.4,
            scale_y=0.76,
            scale_z=1.5,
            r=1.0, g=0.0, b=0.0, a=0.27 # Red, Semi-Transparent
        )

    def publish_box(self, marker_id, ns, center_x, center_y, center_z, scale_x, scale_y, scale_z, r, g, b, a):
        """
        Generic helper to publish any box marker.
        """
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z
        
        # Orientation (Standard Flat)
        marker.pose.orientation.w = 1.0
        
        # Scale (Dimensions)
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        
        # Color & Transparency
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a  # Alpha: 0.0=Invisible, 1.0=Solid
        
        # Lifetime (0 = forever until overwritten)
        marker.lifetime.sec = 0
        
        self.pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedSafetyVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
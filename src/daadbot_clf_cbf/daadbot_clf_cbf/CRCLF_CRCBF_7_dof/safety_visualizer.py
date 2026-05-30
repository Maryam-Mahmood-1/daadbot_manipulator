"""
Unified Safety Visualizer (XY Plane)
Configured for limits: X = [-1.2, 1.2], Y = [-1.2, 1.2]
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class UnifiedSafetyVisualizer(Node):
    def __init__(self):
        super().__init__('unified_safety_visualizer')
        self.pub = self.create_publisher(Marker, '/safety_marker', 10)
        self.timer = self.create_timer(0.5, self.publish_all_zones)
        self.get_logger().info("Unified Visualizer Started: XY Plane Zones (Limits: +/- 1.2m)")

    def publish_all_zones(self):
        """
        Publishes 5 markers in the XY plane:
        1. Green Safe Zone (Center)
        2. Red Unsafe Zone (Left)
        3. Red Unsafe Zone (Right)
        4. Red Unsafe Zone (Front)
        5. Red Unsafe Zone (Back)
        """
        
        # Base Z height for the XY plane drawing (just above the ground to prevent Z-fighting)
        Z_PLANE = 0.05 
        THICKNESS = 0.01 # Flat in Z
        
        # Boundary constraints based on lengths=[1.2, 1.2, 3.0]
        SAFE_X = 2.4 # From -1.2 to 1.2
        SAFE_Y = 2.4 # From -1.2 to 1.2

        # Outer limit of the world you want to color red (Arbitrary 3.0m in each direction)
        WORLD_LIMIT = 3.0 

        # === 1. GREEN SAFE ZONE (Center) ===
        self.publish_box(
            marker_id=0,
            ns="safe_zone",
            center_x=0.0,
            center_y=0.0,
            center_z=Z_PLANE,
            scale_x=SAFE_X,
            scale_y=SAFE_Y,
            scale_z=THICKNESS,
            r=0.0, g=1.0, b=0.0, a=0.3 # Green
        )

        # === 2. RED UNSAFE ZONE (Left / Negative Y) ===
        # Width = (World_Limit - Safe_Y/2)
        left_width = WORLD_LIMIT - (SAFE_Y / 2.0)
        self.publish_box(
            marker_id=1, ns="unsafe_zone",
            center_x=0.0, center_y=-(SAFE_Y/2.0 + left_width/2.0), center_z=Z_PLANE,
            scale_x=SAFE_X, scale_y=left_width, scale_z=THICKNESS,
            r=1.0, g=0.0, b=0.0, a=0.3 # Red
        )

        # === 3. RED UNSAFE ZONE (Right / Positive Y) ===
        right_width = WORLD_LIMIT - (SAFE_Y / 2.0)
        self.publish_box(
            marker_id=2, ns="unsafe_zone",
            center_x=0.0, center_y=(SAFE_Y/2.0 + right_width/2.0), center_z=Z_PLANE,
            scale_x=SAFE_X, scale_y=right_width, scale_z=THICKNESS,
            r=1.0, g=0.0, b=0.0, a=0.3 # Red
        )

        # === 4. RED UNSAFE ZONE (Back / Negative X) ===
        # This covers the entire Y span (2 * World_Limit)
        back_length = WORLD_LIMIT - (SAFE_X / 2.0)
        self.publish_box(
            marker_id=3, ns="unsafe_zone",
            center_x=-(SAFE_X/2.0 + back_length/2.0), center_y=0.0, center_z=Z_PLANE,
            scale_x=back_length, scale_y=(2*WORLD_LIMIT), scale_z=THICKNESS,
            r=1.0, g=0.0, b=0.0, a=0.3 # Red
        )

        # === 5. RED UNSAFE ZONE (Front / Positive X) ===
        front_length = WORLD_LIMIT - (SAFE_X / 2.0)
        self.publish_box(
            marker_id=4, ns="unsafe_zone",
            center_x=(SAFE_X/2.0 + front_length/2.0), center_y=0.0, center_z=Z_PLANE,
            scale_x=front_length, scale_y=(2*WORLD_LIMIT), scale_z=THICKNESS,
            r=1.0, g=0.0, b=0.0, a=0.3 # Red
        )


    def publish_box(self, marker_id, ns, center_x, center_y, center_z, scale_x, scale_y, scale_z, r, g, b, a):
        marker = Marker()
        marker.header.frame_id = "world" # Ensure this matches your Gazebo fixed frame
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        
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
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import mediapipe as mp

class MediaPipeNode(Node):

    def __init__(self):
        super().__init__('mediapipe_node')

        # ROS 2 Publisher
        self.publisher_ = self.create_publisher(String, 'pose_status', 10)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Open a video capture (0 = webcam)
        self.cap = cv2.VideoCapture(0)

        # ROS 2 Timer (process every 0.1s)
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read frame from camera")
            return

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Check if a pose is detected
        if results.pose_landmarks:
            msg = String()
            msg.data = "Pose detected!"
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Show output frame
        cv2.imshow('MediaPipe Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Shutting down...")
            self.cleanup()

    def cleanup(self):
        """ Releases resources and destroys the node """
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = MediaPipeNode()
    rclpy.spin(node)
    node.cleanup()

if __name__ == '__main__':
    main()

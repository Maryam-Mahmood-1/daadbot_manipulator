import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from daadbot_msgs.action import Fibonacci
import time

class SimpleActionServer(Node):
    def __init__(self):
        super().__init__('my_action_server')
        self._action_server = ActionServer(self, Fibonacci, 'fibonacci', self.execute_callback)
        self.get_logger().info('Starting...')
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Received Goal Request with order %d' % goal_handle.request.order)
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.partial_sequence.append(feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i - 1])
            self.get_logger().info('Publishing feedback: %d' % feedback_msg.partial_sequence[-1])
            self.get_logger().info('Publishing feedback: {0}'.format(feedback_msg.partial_sequence))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result

def main():
    rclpy.init()
    my_action_server = SimpleActionServer()
    rclpy.spin(my_action_server)
    my_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
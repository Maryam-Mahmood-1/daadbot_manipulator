#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

class DirectJointAngleNode : public rclcpp::Node
{
public:
    DirectJointAngleNode()
        : Node("direct_joint_angle_node")
    {
        publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/position_arm_controller/joint_trajectory", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 10 Hz update rate
            std::bind(&DirectJointAngleNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        // Example: replace these with your vision-provided joint angles
        std::vector<double> target_positions = {0.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.0};

        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = {
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        };

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = target_positions;
        point.time_from_start = rclcpp::Duration::from_seconds(1.5); // short execution time

        traj_msg.points.push_back(point);

        publisher_->publish(traj_msg);
    }

    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DirectJointAngleNode>());
    rclcpp::shutdown();
    return 0;
}

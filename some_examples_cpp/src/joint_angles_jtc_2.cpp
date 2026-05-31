#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

class SequentialJointAngleNode : public rclcpp::Node
{
public:
    SequentialJointAngleNode()
        : Node("sequential_joint_angle_node"), current_point_(0)
    {
        publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/position_arm_controller/joint_trajectory", 10);

        joint_trajectories_ = {
            {0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
            {0.1,  0.2,  0.1,  0.0,  0.1, -0.1,  0.0},
            {0.2,  0.3,  0.2,  0.1,  0.2, -0.2,  0.1},
            {0.3,  0.4,  0.3,  0.2,  0.3, -0.3,  0.2},
            {0.4,  0.5,  0.4,  0.3,  0.4, -0.4,  0.3},
            {0.5,  0.6,  0.5,  0.4,  0.5, -0.5,  0.4},
            {0.6,  0.7,  0.6,  0.5,  0.6, -0.6,  0.5},
            {0.7,  0.8,  0.7,  0.6,  0.7, -0.7,  0.6},
            {0.8,  0.9,  0.8,  0.7,  0.8, -0.8,  0.7},
            {0.9,  1.0,  0.9,  0.8,  0.9, -0.9,  0.8}
        };

        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&SequentialJointAngleNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        if (current_point_ >= joint_trajectories_.size())
        {
            RCLCPP_INFO(this->get_logger(), "All points sent, stopping timer.");
            timer_->cancel();
            return;
        }

        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = {
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        };

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = joint_trajectories_[current_point_];
        point.time_from_start = rclcpp::Duration::from_seconds(1.5);

        traj_msg.points.push_back(point);
        publisher_->publish(traj_msg);

        RCLCPP_INFO(this->get_logger(), "Published point %zu", current_point_);
        current_point_++;
    }

    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::vector<double>> joint_trajectories_;
    size_t current_point_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SequentialJointAngleNode>());
    rclcpp::shutdown();
    return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <cmath>

class SequentialJointAngleNode : public rclcpp::Node
{
public:
    SequentialJointAngleNode()
        : Node("sequential_joint_angle_node"), current_angle_rad_(0.0)
    {
        // Declare and get target angle parameter (degrees)
        this->declare_parameter<double>("target_angle_deg", 70.0); // default: 30°
        double target_angle_deg;
        this->get_parameter("target_angle_deg", target_angle_deg);

        // Convert degrees to radians
        target_angle_rad_ = target_angle_deg * M_PI / 180.0;
        increment_rad_ = 0.3 * M_PI / 180.0; // 0.3° in radians

        publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/position_arm_controller/joint_trajectory", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 0.01 seconds
            std::bind(&SequentialJointAngleNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
        // Stop if target reached (within increment size)
        if (std::fabs(target_angle_rad_ - current_angle_rad_) < increment_rad_)
        {
            current_angle_rad_ = target_angle_rad_;
            publish_angle(current_angle_rad_);
            RCLCPP_INFO(this->get_logger(), "Target reached: %.3f rad", current_angle_rad_);
            timer_->cancel();
            return;
        }

        // Increment towards target
        if (current_angle_rad_ < target_angle_rad_)
            current_angle_rad_ += increment_rad_;
        else
            current_angle_rad_ -= increment_rad_;

        publish_angle(current_angle_rad_);
    }

    void publish_angle(double angle_rad)
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = {
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        };

        trajectory_msgs::msg::JointTrajectoryPoint point;
        // Keep all joints fixed except joint_2
        point.positions = {0.0, angle_rad, 0.0, 0.0, 0.0, 0.0, 0.0};
        point.time_from_start = rclcpp::Duration::from_seconds(0.01);

        traj_msg.points.push_back(point);
        publisher_->publish(traj_msg);

        RCLCPP_INFO(this->get_logger(), "Published joint_2: %.3f rad (%.2f°)",
                    angle_rad, angle_rad * 180.0 / M_PI);
    }

    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    double current_angle_rad_;
    double target_angle_rad_;
    double increment_rad_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SequentialJointAngleNode>());
    rclcpp::shutdown();
    return 0;
}

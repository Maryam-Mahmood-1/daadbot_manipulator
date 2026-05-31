#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <cmath>
#include <vector>

class MultiJointTrajectoryNode : public rclcpp::Node
{
public:
    MultiJointTrajectoryNode()
        : Node("multi_joint_trajectory_node"),
          current_angles_rad_(7, 0.0), target_angles_rad_(7, 0.0),
          increment_rad_(0.3 * M_PI / 180.0) // 0.3° in radians
    {
        // Subscriber for target angles (degrees)
        target_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/target_joint_angles_deg", 10,
            std::bind(&MultiJointTrajectoryNode::target_callback, this, std::placeholders::_1));

        // Publisher to JointTrajectoryController
        traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/position_arm_controller/joint_trajectory", 10);

        // Timer for sending incremental steps
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 10ms
            std::bind(&MultiJointTrajectoryNode::timer_callback, this));
    }

private:
    void target_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 7)
        {
            RCLCPP_ERROR(this->get_logger(), "Target array must have exactly 7 elements.");
            return;
        }

        // Convert degrees to radians
        for (size_t i = 0; i < 7; i++)
            target_angles_rad_[i] = msg->data[i] * M_PI / 180.0;

        RCLCPP_INFO(this->get_logger(), "Received new target angles (deg):");
        for (double val : msg->data)
            RCLCPP_INFO(this->get_logger(), "%.2f", val);
    }

    void timer_callback()
    {
        bool all_reached = true;

        for (size_t i = 0; i < 7; i++)
        {
            if (std::fabs(target_angles_rad_[i] - current_angles_rad_[i]) >= increment_rad_)
            {
                all_reached = false;
                if (current_angles_rad_[i] < target_angles_rad_[i])
                    current_angles_rad_[i] += increment_rad_;
                else
                    current_angles_rad_[i] -= increment_rad_;
            }
            else
            {
                current_angles_rad_[i] = target_angles_rad_[i];
            }
        }

        publish_angles(current_angles_rad_);

        if (all_reached)
        {
            RCLCPP_INFO_ONCE(this->get_logger(), "Target reached for all joints.");
        }
    }

    void publish_angles(const std::vector<double> &angles_rad)
    {
        trajectory_msgs::msg::JointTrajectory traj_msg;
        traj_msg.joint_names = {
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7"
        };

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = angles_rad;
        point.time_from_start = rclcpp::Duration::from_seconds(0.01);

        traj_msg.points.push_back(point);
        traj_pub_->publish(traj_msg);
    }

    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<double> current_angles_rad_;
    std::vector<double> target_angles_rad_;
    double increment_rad_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MultiJointTrajectoryNode>());
    rclcpp::shutdown();
    return 0;
}

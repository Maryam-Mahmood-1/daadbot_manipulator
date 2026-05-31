#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

class ArmPoseSubscriber : public rclcpp::Node
{
public:
  ArmPoseSubscriber() : Node("arm_pose_subscriber_node")
  {
    RCLCPP_INFO(get_logger(), "Initializing MoveIt and subscribing to joint states...");
    
    // Schedule initialization after node setup
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&ArmPoseSubscriber::initializeMoveIt, this));
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::TimerBase::SharedPtr timer_; // Timer to delay initialization

  void initializeMoveIt()
  {
    // Stop the timer to prevent repeated execution
    timer_->cancel();
    
    // Now it's safe to use shared_from_this()
    arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "slider_joint_states", 10,
        std::bind(&ArmPoseSubscriber::jointStateCallback, this, std::placeholders::_1));
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->position.size() < 7)
    {
      RCLCPP_ERROR(get_logger(), "Received joint state message with insufficient joint positions.");
      return;
    }

    std::vector<double> joint_positions(msg->position.begin(), msg->position.begin() + 7);
    arm_move_group_->setJointValueTarget(joint_positions);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Plan successful! Executing...");
      arm_move_group_->move();
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Planning failed.");
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmPoseSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
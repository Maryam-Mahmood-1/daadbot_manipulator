#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include "daadbot_msgs/action/daadbot_task_server.hpp"
#include <memory>

using namespace std::placeholders;

namespace daadbot_posegen
{
class ArmTaskServer : public rclcpp::Node
{
public:
  explicit ArmTaskServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("multi_task_server", options)
  {
    RCLCPP_INFO(get_logger(), "Starting the Server");

    action_server_ = rclcpp_action::create_server<daadbot_msgs::action::DaadbotTaskServer>(
        this, "multi_task_server",
        std::bind(&ArmTaskServer::goalCallback, this, _1, _2),
        std::bind(&ArmTaskServer::cancelCallback, this, _1),
        std::bind(&ArmTaskServer::acceptedCallback, this, _1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/slider_joint_states", 10, std::bind(&ArmTaskServer::jointStateCallback, this, _1));
  }

private:
  rclcpp_action::Server<daadbot_msgs::action::DaadbotTaskServer>::SharedPtr action_server_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_;
  std::vector<double> input_joint_positions_;

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    input_joint_positions_.clear();
    input_joint_positions_.resize(7, 0.0);  // Ensure exactly 7 joint values

    for (size_t i = 0; i < msg->name.size(); ++i)
    {
        if (msg->name[i] == "joint_1") input_joint_positions_[0] = msg->position[i];
        else if (msg->name[i] == "joint_2") input_joint_positions_[1] = msg->position[i];
        else if (msg->name[i] == "joint_3") input_joint_positions_[2] = msg->position[i];
        else if (msg->name[i] == "joint_4") input_joint_positions_[3] = msg->position[i];
        else if (msg->name[i] == "joint_5") input_joint_positions_[4] = msg->position[i];
        else if (msg->name[i] == "joint_6") input_joint_positions_[5] = msg->position[i];
        else if (msg->name[i] == "joint_7") input_joint_positions_[6] = msg->position[i];
    }

    // Debugging log
    RCLCPP_INFO(get_logger(), "Received joint states: [%f, %f, %f, %f, %f, %f, %f]",
                input_joint_positions_[0], input_joint_positions_[1], input_joint_positions_[2], 
                input_joint_positions_[3], input_joint_positions_[4], input_joint_positions_[5], 
                input_joint_positions_[6]);
  }

  rclcpp_action::GoalResponse goalCallback(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const daadbot_msgs::action::DaadbotTaskServer::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received goal request");
    (void)uuid;
    (void)goal;  // Suppress unused parameter warning
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse cancelCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    if (arm_move_group_) arm_move_group_->stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    std::thread{ std::bind(&ArmTaskServer::execute, this, goal_handle) }.detach();
  }

  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "Executing goal");
    auto result = std::make_shared<daadbot_msgs::action::DaadbotTaskServer::Result>();

    if (!arm_move_group_)
    {
      arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    }

    if (input_joint_positions_.empty())
    {
      RCLCPP_ERROR(get_logger(), "No joint states received!");
      return;
    }

    arm_move_group_->setStartState(*arm_move_group_->getCurrentState());

    RCLCPP_INFO(get_logger(), "Setting joint target: [%f, %f, %f, %f, %f, %f, %f]",
                input_joint_positions_[0], input_joint_positions_[1], input_joint_positions_[2], 
                input_joint_positions_[3], input_joint_positions_[4], input_joint_positions_[5], 
                input_joint_positions_[6]);

    bool arm_within_bounds = arm_move_group_->setJointValueTarget(input_joint_positions_);
    if (!arm_within_bounds)
    {
      RCLCPP_WARN(get_logger(), "Target joint positions out of limits, clamping to limits");
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan arm_plan;
    bool arm_plan_success = (arm_move_group_->plan(arm_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (arm_plan_success)
    {
      RCLCPP_INFO(get_logger(), "Planner SUCCEEDED, executing movement");
      arm_move_group_->move();
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Planner failed!");
      return;
    }

    result->success = true;
    goal_handle->succeed(result);
    RCLCPP_INFO(get_logger(), "Goal succeeded");
  }
};
}  // namespace daadbot_posegen

RCLCPP_COMPONENTS_REGISTER_NODE(daadbot_posegen::ArmTaskServer)
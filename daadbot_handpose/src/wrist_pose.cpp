#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include "daadbot_msgs/action/daadbot_task_server.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

#include <memory>

using namespace std::placeholders;

namespace daadbot_handpose
{
class HandPose : public rclcpp::Node
{
public:
  explicit HandPose(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("multi_task_server", options)
  {
    RCLCPP_INFO(get_logger(), "Starting the Server");
    action_server_ = rclcpp_action::create_server<daadbot_msgs::action::DaadbotTaskServer>(
        this, "multi_task_server",
        std::bind(&HandPose::goalCallback, this, _1, _2),
        std::bind(&HandPose::cancelCallback, this, _1),
        std::bind(&HandPose::acceptedCallback, this, _1));
  }

private:
  rclcpp_action::Server<daadbot_msgs::action::DaadbotTaskServer>::SharedPtr action_server_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_, gripper_move_group_;
  geometry_msgs::msg::Pose target_pose_;
  std::vector<double> gripper_joint_goal_;

  rclcpp_action::GoalResponse goalCallback(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const daadbot_msgs::action::DaadbotTaskServer::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received goal request with task id %d", goal->task_id);
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse cancelCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    if (arm_move_group_)
    {
      arm_move_group_->stop();
    }
    if (gripper_move_group_)
    {
      gripper_move_group_->stop();
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    // Execute in a new thread to avoid blocking the executor
    std::thread{ std::bind(&HandPose::execute, this, _1), goal_handle }.detach();
  }

  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<daadbot_msgs::action::DaadbotTaskServer>> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "Executing goal");
    auto result = std::make_shared<daadbot_msgs::action::DaadbotTaskServer::Result>();

    // Initialize MoveIt 2 Interface if not already
    if (!arm_move_group_)
    {
      RCLCPP_INFO(get_logger(), "Initializing MoveGroupInterface for arm...");
      arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    }
    if (!gripper_move_group_)
    {
      RCLCPP_INFO(get_logger(), "Initializing MoveGroupInterface for gripper...");
      gripper_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "gripper");
    }

    // Get the current pose of the arm
    geometry_msgs::msg::Pose current_pose = arm_move_group_->getCurrentPose().pose;

    // Print the current pose
    RCLCPP_INFO(get_logger(), "Current Pose: [x: %f, y: %f, z: %f, q_w: %f, q_x: %f, q_y: %f, q_z: %f]", 
                current_pose.position.x, current_pose.position.y, current_pose.position.z, 
                current_pose.orientation.w, current_pose.orientation.x, 
                current_pose.orientation.y, current_pose.orientation.z);

    // Define target pose based on task_id
    if (goal_handle->get_goal()->task_id == 0)
    {
      target_pose_.position.x = 0.35;
      target_pose_.position.y = 0.0;
      target_pose_.position.z = 0.6;
      target_pose_.orientation.w = 0.015;
      target_pose_.orientation.x = 0.1;
      target_pose_.orientation.y = 0.01;
      target_pose_.orientation.z = 0.0;
      gripper_joint_goal_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Invalid Task Number: %d", goal_handle->get_goal()->task_id);
      return;
    }

    // Set the target pose for MoveIt
    arm_move_group_->setPoseTarget(target_pose_);
    gripper_move_group_->setJointValueTarget(gripper_joint_goal_);

    // Plan
    moveit::planning_interface::MoveGroupInterface::Plan arm_plan;
    moveit::planning_interface::MoveGroupInterface::Plan gripper_plan;

    auto arm_plan_status = arm_move_group_->plan(arm_plan);
    auto gripper_plan_status = gripper_move_group_->plan(gripper_plan);

    RCLCPP_INFO(get_logger(), "Arm Plan Status Code: %d", arm_plan_status.val);
    RCLCPP_INFO(get_logger(), "Gripper Plan Status Code: %d", gripper_plan_status.val);

    if (arm_plan_status == moveit::core::MoveItErrorCode::SUCCESS &&
        gripper_plan_status == moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Planning SUCCEEDED, executing movement");
      arm_move_group_->move();
      gripper_move_group_->move();
    }

    result->success = true;
    goal_handle->succeed(result);
    RCLCPP_INFO(get_logger(), "Goal succeeded");
  }
};
}  // namespace daadbot_handpose

RCLCPP_COMPONENTS_REGISTER_NODE(daadbot_handpose::HandPose)

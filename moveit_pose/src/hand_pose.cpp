#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

class HandPose : public rclcpp::Node
{
public:
  HandPose() : Node("hand_pose_node")
  {
    RCLCPP_INFO(get_logger(), "Initializing MoveIt for direct execution...");

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&HandPose::initializeMoveIt, this));
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_;
  rclcpp::TimerBase::SharedPtr timer_;

  void initializeMoveIt()
  {
    timer_->cancel();

    arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");

    // 🟢 Set the pose reference frame to camera_link
    arm_move_group_->setPoseReferenceFrame("camera_link");

    // 🟢 Tune planning parameters
    arm_move_group_->setPlannerId("RRTConnectkConfigDefault");
    arm_move_group_->setPlanningTime(3.0);             // default is 5.0
    arm_move_group_->setNumPlanningAttempts(3);        // default is higher
    arm_move_group_->setMaxVelocityScalingFactor(0.8); // faster but safe
    arm_move_group_->setMaxAccelerationScalingFactor(0.8);

    RCLCPP_INFO(get_logger(), "Planning frame: %s", arm_move_group_->getPlanningFrame().c_str());

    moveArm();
  }

  void moveArm()
  {
    geometry_msgs::msg::Pose target_pose;
    // target_pose.position.x = -0.421;
    // target_pose.position.y = -0.079;
    // target_pose.position.z = -0.047;

    target_pose.position.x = -0.5861;
    target_pose.position.y = -0.025;
    target_pose.position.z = 0.187;

    target_pose.orientation.x = 0.707;
    target_pose.orientation.y = -0.707;
    target_pose.orientation.z = -0.000;
    target_pose.orientation.w = 0.000;

    arm_move_group_->setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_INFO(get_logger(), "Plan successful! Executing...");
      arm_move_group_->execute(plan);  // 🟢 only execute the already computed plan
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
  auto node = std::make_shared<HandPose>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}






// #include <rclcpp/rclcpp.hpp>
// #include <moveit/move_group_interface/move_group_interface.h>
// #include <geometry_msgs/msg/pose.hpp>

// class HandPose : public rclcpp::Node
// {
// public:
//   HandPose() : Node("hand_pose_node")
//   {
//     RCLCPP_INFO(get_logger(), "Initializing MoveIt for direct execution...");

//     timer_ = this->create_wall_timer(
//         std::chrono::milliseconds(500),
//         std::bind(&HandPose::initializeMoveIt, this));
//   }

// private:
//   std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_;
//   rclcpp::TimerBase::SharedPtr timer_;

//   void initializeMoveIt()
//   {
//     timer_->cancel();

//     arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");

//     // 🟢 Tune planning parameters
//     arm_move_group_->setPlannerId("RRTConnectkConfigDefault");
//     arm_move_group_->setPlanningTime(3.0);             // default is 5.0
//     arm_move_group_->setNumPlanningAttempts(3);        // default is higher
//     arm_move_group_->setMaxVelocityScalingFactor(0.8); // faster but safe
//     arm_move_group_->setMaxAccelerationScalingFactor(0.8);

//     RCLCPP_INFO(get_logger(), "Planning frame: %s", arm_move_group_->getPlanningFrame().c_str());

//     moveArm();
//   }

//   void moveArm()
//   {
//     geometry_msgs::msg::Pose target_pose;
//     target_pose.position.x = 0.745;
//     target_pose.position.y = 0.246;
//     target_pose.position.z = 1.683;

//     target_pose.orientation.x = -0.041;
//     target_pose.orientation.y = 0.466;
//     target_pose.orientation.z = 0.855;
//     target_pose.orientation.w = 0.224;

//     arm_move_group_->setPoseTarget(target_pose);

//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     if (arm_move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//     {
//       RCLCPP_INFO(get_logger(), "Plan successful! Executing...");
//       arm_move_group_->execute(plan);  // 🟢 only execute the already computed plan
//     }
//     else
//     {
//       RCLCPP_ERROR(get_logger(), "Planning failed.");
//     }
//   }
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<HandPose>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }

/*
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

class HandPose : public rclcpp::Node
{
public:
  HandPose() : Node("hand_pose_node")
  {
    RCLCPP_INFO(get_logger(), "Initializing MoveIt for direct execution...");

    // Schedule initialization after node setup
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&HandPose::initializeMoveIt, this));
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_, gripper_move_group_;
  rclcpp::TimerBase::SharedPtr timer_;  // Timer to delay initialization

  void initializeMoveIt()
  {
    // Stop the timer to prevent repeated execution
    timer_->cancel();

    // Now it's safe to use shared_from_this()
    arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "arm");
    gripper_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "gripper");

    RCLCPP_INFO(get_logger(), "Planning frame: %s", arm_move_group_->getPlanningFrame().c_str());

    moveArm();
  }

  void moveArm()
  {
    geometry_msgs::msg::Pose target_pose;
    // target_pose.position.x = 0.487;
    // target_pose.position.y = -0.005;
    // target_pose.position.z = 0.604;
    // target_pose.orientation.w = 1.0;
    // target_pose.orientation.x = 0.0;
    // target_pose.orientation.y = 0.0;
    // target_pose.orientation.z = 0.025;

    target_pose.position.x = 0.745;
    target_pose.position.y = 0.246;
    target_pose.position.z = 1.683;
    
    target_pose.orientation.x = -0.041;
    target_pose.orientation.y = 0.466;
    target_pose.orientation.z = 0.855;
    target_pose.orientation.w = 0.224;

    arm_move_group_->setPoseTarget(target_pose);

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
  auto node = std::make_shared<HandPose>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

*/
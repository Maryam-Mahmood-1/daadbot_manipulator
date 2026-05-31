#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>

#include <random>
#include <cmath>

class IKWorkspaceVisualizer : public rclcpp::Node
{
public:
  IKWorkspaceVisualizer()
      : Node("ik_workspace_visualizer"), iteration_(0)
  {
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "ik_workspace_markers", 10);

    RCLCPP_INFO(get_logger(), "IK Workspace Visualizer node created, waiting for delayed initialization...");

    // Delay initialization slightly to ensure robot_description is available
    init_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&IKWorkspaceVisualizer::init_model, this));
  }

private:
  // --- MoveIt state ---
  moveit::core::RobotModelPtr kinematic_model_;
  moveit::core::RobotStatePtr kinematic_state_;
  const moveit::core::JointModelGroup* joint_model_group_;

  // --- ROS 2 ---
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::TimerBase::SharedPtr main_timer_;

  int iteration_;

  // ============================================================
  void init_model()
  {
    init_timer_->cancel();

    auto node_ptr = this->shared_from_this();
    robot_model_loader::RobotModelLoader loader(node_ptr, "robot_description");
    kinematic_model_ = loader.getModel();

    if (!kinematic_model_)
    {
      RCLCPP_ERROR(get_logger(), "❌ Failed to load robot model from parameter 'robot_description'.");
      return;
    }

    kinematic_state_ = std::make_shared<moveit::core::RobotState>(kinematic_model_);
    joint_model_group_ = kinematic_model_->getJointModelGroup("arm");

    if (!joint_model_group_)
    {
      RCLCPP_ERROR(get_logger(), "❌ JointModelGroup 'arm' not found. Check your MoveIt SRDF.");
      return;
    }

    RCLCPP_INFO(get_logger(), "✅ Robot model and kinematic state initialized.");

    main_timer_ = this->create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&IKWorkspaceVisualizer::timer_callback, this));

    RCLCPP_INFO(get_logger(), "🔁 Continuous sampling started.");
  }

  // ============================================================
  void timer_callback()
  {
    iteration_++;
    RCLCPP_INFO(get_logger(), "🔁 Generating new sample set #%d", iteration_);
    sample_and_publish(300, 0.9, 0.9);
  }

  // ============================================================
  void sample_and_publish(int samples, double range_xy, double range_z)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<double> dist_xy(-range_xy, range_xy);
    std::uniform_real_distribution<double> dist_z(-range_z, range_z);

    int id = 0;

    for (int i = 0; i < samples; ++i)
    {
      // --- Random pose generation ---
      Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
      target.translation() = Eigen::Vector3d(dist_xy(rng), dist_xy(rng), dist_z(rng));

      Eigen::AngleAxisd rot_x(M_PI * ((double)rand() / RAND_MAX), Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd rot_y(M_PI * ((double)rand() / RAND_MAX), Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd rot_z(M_PI * ((double)rand() / RAND_MAX), Eigen::Vector3d::UnitZ());
      target.linear() = (rot_z * rot_y * rot_x).toRotationMatrix();

      // --- Check IK ---
      bool found_ik = kinematic_state_->setFromIK(joint_model_group_, target, 0.1);
      if (!found_ik)
        continue; // Skip unreachable poses entirely

      // --- Draw a point (sphere) for reachable pose ---
      visualization_msgs::msg::Marker sphere;
      sphere.header.frame_id = "base_link";
      sphere.header.stamp = this->now();
      sphere.ns = "reachable_points";
      sphere.id = id++;
      sphere.type = visualization_msgs::msg::Marker::SPHERE;
      sphere.action = visualization_msgs::msg::Marker::ADD;
      sphere.pose = tf2::toMsg(target);
      sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.025;
      sphere.color.r = 0.3178;
      sphere.color.g = sphere.color.b = 0.3882;
      sphere.color.a = 1.0;
      marker_array.markers.push_back(sphere);

      // --- Draw orientation axes ---
      Eigen::Vector3d origin = target.translation();
      Eigen::Matrix3d rot = target.rotation();
      add_axis_marker(marker_array, id++, origin, rot.col(0), "x_axis", 1.0, 0.0, 0.0);
      add_axis_marker(marker_array, id++, origin, rot.col(1), "y_axis", 0.0, 1.0, 0.0);
      add_axis_marker(marker_array, id++, origin, rot.col(2), "z_axis", 0.0, 0.0, 1.0);
    }

    marker_pub_->publish(marker_array);
    RCLCPP_INFO(get_logger(), "📡 Published %zu reachable poses with orientation axes", marker_array.markers.size());
  }

  // ============================================================
  void add_axis_marker(visualization_msgs::msg::MarkerArray &array, int id,
                       const Eigen::Vector3d &origin, const Eigen::Vector3d &axis,
                       const std::string &ns, double r, double g, double b)
  {
    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = "base_link";
    arrow.header.stamp = this->now();
    arrow.ns = ns;
    arrow.id = id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p0, p1;
    p0.x = origin.x();
    p0.y = origin.y();
    p0.z = origin.z();
    p1.x = origin.x() + 0.05 * axis.x();
    p1.y = origin.y() + 0.05 * axis.y();
    p1.z = origin.z() + 0.05 * axis.z();

    arrow.points.push_back(p0);
    arrow.points.push_back(p1);

    arrow.scale.x = 0.005;
    arrow.scale.y = 0.01;
    arrow.scale.z = 0.0;

    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;
    arrow.color.a = 1.0;

    array.markers.push_back(arrow);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IKWorkspaceVisualizer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}










// // ik_fk_workspace_sampler_loop.cpp
// // Continuously publishes FK-sampled workspace points every few seconds

// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/point_cloud2.hpp>
// #include <sensor_msgs/point_cloud2_iterator.hpp>

// #include <moveit/robot_model_loader/robot_model_loader.hpp>
// #include <moveit/robot_state/robot_state.hpp>

// #include <random>
// #include <vector>
// #include <string>

// class FKWorkspaceSampler : public rclcpp::Node
// {
// public:
//   FKWorkspaceSampler()
//   : Node("fk_workspace_sampler")
//   {
//     this->declare_parameter<std::string>("group_name", "arm");
//     this->declare_parameter<std::string>("ee_link", "endeffector");
//     this->declare_parameter<std::string>("frame_id", "base_link");
//     this->declare_parameter<int>("n_samples", 50000);
//     this->declare_parameter<int>("seed", 0);

//     group_name_ = this->get_parameter("group_name").as_string();
//     ee_link_ = this->get_parameter("ee_link").as_string();
//     frame_id_ = this->get_parameter("frame_id").as_string();
//     n_samples_ = this->get_parameter("n_samples").as_int();
//     int seed = this->get_parameter("seed").as_int();

//     pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("workspace_cloud", 10);

//     // Run every few seconds (continuous publishing)
//     timer_ = this->create_wall_timer(std::chrono::seconds(5),
//                                      std::bind(&FKWorkspaceSampler::publishWorkspace, this));

//     if (seed == 0) {
//       std::random_device rd;
//       rng_.seed(rd());
//     } else {
//       rng_.seed((unsigned)seed);
//     }

//     RCLCPP_INFO(this->get_logger(), "Continuous FK workspace sampler started.");
//   }

// private:
//   void publishWorkspace()
//   {
//     if (!kmodel_) {
//       robot_model_loader::RobotModelLoader loader(shared_from_this(), "robot_description");
//       kmodel_ = loader.getModel();
//       if (!kmodel_) {
//         RCLCPP_ERROR(this->get_logger(), "Failed to load robot model from 'robot_description'");
//         return;
//       }

//       joint_model_group_ = kmodel_->getJointModelGroup(group_name_);
//       if (!joint_model_group_) {
//         RCLCPP_ERROR(this->get_logger(), "Could not find joint model group '%s'", group_name_.c_str());
//         return;
//       }

//       kinematic_state_ = std::make_shared<moveit::core::RobotState>(kmodel_);
//       kinematic_state_->setToDefaultValues();

//       const std::vector<const moveit::core::JointModel*>& joints =
//         joint_model_group_->getActiveJointModels();

//       distributions_.clear();
//       variable_names_.clear();
//       for (const moveit::core::JointModel* jm : joints) {
//         const auto& vars = jm->getVariableNames();
//         for (const std::string &var : vars) {
//           variable_names_.push_back(var);
//           const auto& b = jm->getVariableBounds(var);
//           double lo = b.min_position_;
//           double hi = b.max_position_;
//           if (!(lo < hi)) { lo = -3.14159; hi = 3.14159; }
//           distributions_.emplace_back(lo, hi);
//         }
//       }
//     }

//     // Prepare PointCloud2
//     sensor_msgs::msg::PointCloud2 cloud;
//     cloud.header.frame_id = frame_id_;
//     cloud.header.stamp = this->now();
//     cloud.height = 1;
//     cloud.width = n_samples_;
//     cloud.is_bigendian = false;
//     cloud.is_dense = false;

//     sensor_msgs::PointCloud2Modifier modifier(cloud);
//     modifier.setPointCloud2FieldsByString(1, "xyz");
//     modifier.resize(n_samples_);

//     sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
//     sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
//     sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

//     std::vector<double> joint_positions(variable_names_.size(), 0.0);

//     for (int i = 0; i < n_samples_; ++i) {
//       for (size_t j = 0; j < distributions_.size(); ++j)
//         joint_positions[j] = distributions_[j](rng_);

//       kinematic_state_->setVariablePositions(variable_names_, joint_positions);
//       kinematic_state_->update();

//       const Eigen::Isometry3d& pose = kinematic_state_->getGlobalLinkTransform(ee_link_);
//       *iter_x = static_cast<float>(pose.translation().x());
//       *iter_y = static_cast<float>(pose.translation().y());
//       *iter_z = static_cast<float>(pose.translation().z());
//       ++iter_x; ++iter_y; ++iter_z;
//     }

//     pub_->publish(cloud);
//     RCLCPP_INFO(this->get_logger(), "Published %d workspace samples", n_samples_);
//   }

//   rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
//   rclcpp::TimerBase::SharedPtr timer_;

//   moveit::core::RobotModelPtr kmodel_;
//   const moveit::core::JointModelGroup* joint_model_group_{nullptr};
//   moveit::core::RobotStatePtr kinematic_state_;

//   std::string group_name_, ee_link_, frame_id_;
//   int n_samples_;
//   std::mt19937 rng_;
//   std::vector<std::uniform_real_distribution<double>> distributions_;
//   std::vector<std::string> variable_names_;
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<FKWorkspaceSampler>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }




// #include <rclcpp/rclcpp.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>
// #include <moveit/robot_model_loader/robot_model_loader.hpp>
// #include <moveit/robot_state/robot_state.hpp>

// class WorkspaceVisualizer : public rclcpp::Node
// {
// public:
//   WorkspaceVisualizer()
//   : Node("workspace_visualizer")
//   {
//     // Publisher (standard QoS, publishes repeatedly)
//     marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
//       "workspace_markers", 10);

//     // Timer publishes workspace every 2 seconds
//     timer_ = this->create_wall_timer(
//       std::chrono::seconds(2),
//       std::bind(&WorkspaceVisualizer::publish_workspace, this));
//   }

// private:
//   void publish_workspace()
//   {
//     // Load robot model from parameter "robot_description"
//     robot_model_loader::RobotModelLoader loader(
//       shared_from_this(), "robot_description");
//     auto kinematic_model = loader.getModel();
//     if (!kinematic_model) {
//       RCLCPP_ERROR(this->get_logger(), "Failed to load robot model from 'robot_description'");
//       return;
//     }

//     auto kinematic_state = std::make_shared<moveit::core::RobotState>(kinematic_model);
//     kinematic_state->setToDefaultValues();

//     visualization_msgs::msg::MarkerArray markers;
//     int id = 0;

//     // Example: a cube of sample points
//     for (double x = -0.5; x <= 0.5; x += 0.1) {
//       for (double y = -0.5; y <= 0.5; y += 0.1) {
//         for (double z = 0.0; z <= 1.0; z += 0.1) {
//           visualization_msgs::msg::Marker m;
//           m.header.frame_id = "base_link";   // ✅ change to your base link if different
//           m.header.stamp = this->now();
//           m.ns = "workspace";
//           m.id = id++;
//           m.type = visualization_msgs::msg::Marker::SPHERE;
//           m.action = visualization_msgs::msg::Marker::ADD;
//           m.pose.position.x = x;
//           m.pose.position.y = y;
//           m.pose.position.z = z;
//           m.scale.x = m.scale.y = m.scale.z = 0.02;
//           m.color.a = 1.0;
//           m.color.g = 1.0;  // green
//           markers.markers.push_back(m);
//         }
//       }
//     }

//     marker_pub_->publish(markers);
//     RCLCPP_INFO(this->get_logger(), "Published %zu workspace markers!", markers.markers.size());
//   }

//   rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
//   rclcpp::TimerBase::SharedPtr timer_;
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<WorkspaceVisualizer>();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }

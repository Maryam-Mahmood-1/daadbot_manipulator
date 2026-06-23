// reachability_map.cpp
// Capability/reachability-map visualizer for an arm group.
//
// Samples the joint space within the URDF joint limits (uniform random, so it
// respects the limits exactly), runs forward kinematics, and bins the
// end-effector positions into a voxel grid. Each voxel's REACHABILITY INDEX is
// the number of distinct end-effector orientations reached there (normalized) —
// the classic capability-map metric. Voxels are drawn as a CUBE_LIST coloured
// with a jet colormap (red = low reachability shell, blue/green = high), with
// optional alpha-by-reachability for a translucent-shell look.
//
// Optional: filter out self/table-colliding configurations using the MoveIt
// planning scene + SRDF allowed-collision matrix (param check_collision).
//
// Publishes visualization_msgs/MarkerArray on "reachability_markers"
// (frame: base_frame). Computed once, re-published so RViz always catches it.

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <random_numbers/random_numbers.h>

#include <Eigen/Geometry>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

namespace
{
// pack an (ix,iy,iz) voxel index into one 64-bit key
inline int64_t voxelKey(int ix, int iy, int iz)
{
  auto enc = [](int v) -> int64_t { return static_cast<int64_t>(v + (1 << 20)); };
  return (enc(ix) << 42) ^ (enc(iy) << 21) ^ enc(iz);
}

// jet colormap: t in [0,1] -> RGB in [0,1]
void jet(double t, float& r, float& g, float& b)
{
  t = std::clamp(t, 0.0, 1.0);
  r = static_cast<float>(std::clamp(1.5 - std::fabs(4.0 * t - 3.0), 0.0, 1.0));
  g = static_cast<float>(std::clamp(1.5 - std::fabs(4.0 * t - 2.0), 0.0, 1.0));
  b = static_cast<float>(std::clamp(1.5 - std::fabs(4.0 * t - 1.0), 0.0, 1.0));
}
}  // namespace

class ReachabilityMap : public rclcpp::Node
{
public:
  ReachabilityMap() : Node("reachability_map")
  {
    group_      = declare_parameter<std::string>("group", "arm");
    ee_link_    = declare_parameter<std::string>("ee_link", "endeffector");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    n_samples_  = declare_parameter<int>("n_samples", 300000);
    voxel_      = declare_parameter<double>("voxel_size", 0.05);
    n_theta_    = declare_parameter<int>("orient_theta_bins", 8);
    n_phi_      = declare_parameter<int>("orient_phi_bins", 12);
    check_collision_ = declare_parameter<bool>("check_collision", false);
    alpha_min_  = declare_parameter<double>("alpha_min", 0.35);
    alpha_max_  = declare_parameter<double>("alpha_max", 1.0);
    alpha_by_reach_ = declare_parameter<bool>("alpha_by_reachability", true);
    seed_       = declare_parameter<int>("seed", 0);

    pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("reachability_markers", 1);
    init_timer_ = create_wall_timer(std::chrono::milliseconds(600),
                                    std::bind(&ReachabilityMap::build, this));
  }

private:
  void build()
  {
    init_timer_->cancel();

    robot_model_loader::RobotModelLoader loader(shared_from_this(), "robot_description");
    auto model = loader.getModel();
    if (!model) { RCLCPP_ERROR(get_logger(), "no robot_description"); return; }
    const auto* jmg = model->getJointModelGroup(group_);
    if (!jmg) { RCLCPP_ERROR(get_logger(), "group '%s' not found", group_.c_str()); return; }

    moveit::core::RobotState state(model);
    state.setToDefaultValues();
    planning_scene::PlanningScene scene(model);  // for optional collision filtering

    random_numbers::RandomNumberGenerator rng =
        seed_ ? random_numbers::RandomNumberGenerator(static_cast<uint32_t>(seed_))
              : random_numbers::RandomNumberGenerator();

    struct Cell { int ix = 0, iy = 0, iz = 0; std::unordered_set<int> orient; uint32_t count = 0; };
    std::unordered_map<int64_t, Cell> grid;

    int collided = 0;
    for (int i = 0; i < n_samples_; ++i)
    {
      state.setToRandomPositions(jmg, rng);
      state.update();

      if (check_collision_)
      {
        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;
        scene.checkSelfCollision(req, res, state, scene.getAllowedCollisionMatrix());
        if (res.collision) { ++collided; continue; }
      }

      // EE pose expressed in base_frame
      const Eigen::Isometry3d Tb = state.getGlobalLinkTransform(base_frame_);
      const Eigen::Isometry3d Te = state.getGlobalLinkTransform(ee_link_);
      const Eigen::Isometry3d rel = Tb.inverse() * Te;
      const Eigen::Vector3d p = rel.translation();

      int ix = static_cast<int>(std::floor(p.x() / voxel_));
      int iy = static_cast<int>(std::floor(p.y() / voxel_));
      int iz = static_cast<int>(std::floor(p.z() / voxel_));
      Cell& c = grid[voxelKey(ix, iy, iz)];
      c.ix = ix; c.iy = iy; c.iz = iz;
      ++c.count;

      // orientation bin from the EE approach axis (z column)
      const Eigen::Vector3d a = rel.rotation().col(2).normalized();
      const double theta = std::acos(std::clamp(a.z(), -1.0, 1.0));      // [0,pi]
      const double phi   = std::atan2(a.y(), a.x()) + M_PI;              // [0,2pi)
      int bt = std::min(n_theta_ - 1, static_cast<int>(theta / (M_PI / n_theta_)));
      int bp = std::min(n_phi_ - 1, static_cast<int>(phi / (2.0 * M_PI / n_phi_)));
      c.orient.insert(bt * n_phi_ + bp);
    }

    size_t max_orient = 1;
    for (auto& kv : grid) max_orient = std::max(max_orient, kv.second.orient.size());

    visualization_msgs::msg::Marker m;
    m.header.frame_id = base_frame_;
    m.header.stamp = now();
    m.ns = "reachability";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::CUBE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = voxel_;

    for (auto& kv : grid)
    {
      const Cell& c = kv.second;
      double reach = static_cast<double>(c.orient.size()) / static_cast<double>(max_orient);

      geometry_msgs::msg::Point pt;
      pt.x = (c.ix + 0.5) * voxel_;
      pt.y = (c.iy + 0.5) * voxel_;
      pt.z = (c.iz + 0.5) * voxel_;
      m.points.push_back(pt);

      std_msgs::msg::ColorRGBA col;
      jet(reach, col.r, col.g, col.b);
      col.a = static_cast<float>(alpha_by_reach_ ? (alpha_min_ + (alpha_max_ - alpha_min_) * reach)
                                                 : alpha_max_);
      m.colors.push_back(col);
    }

    marker_array_.markers.clear();
    marker_array_.markers.push_back(m);

    RCLCPP_INFO(get_logger(),
                "reachability map: %zu voxels from %d samples (collision-skipped %d). "
                "republishing.",
                grid.size(), n_samples_, collided);

    // republish so RViz catches it even if started late
    pub_timer_ = create_wall_timer(std::chrono::seconds(2),
                                   [this]() { pub_->publish(marker_array_); });
    pub_->publish(marker_array_);
  }

  std::string group_, ee_link_, base_frame_;
  int n_samples_, n_theta_, n_phi_, seed_;
  double voxel_, alpha_min_, alpha_max_;
  bool check_collision_, alpha_by_reach_;

  visualization_msgs::msg::MarkerArray marker_array_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr init_timer_, pub_timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ReachabilityMap>());
  rclcpp::shutdown();
  return 0;
}

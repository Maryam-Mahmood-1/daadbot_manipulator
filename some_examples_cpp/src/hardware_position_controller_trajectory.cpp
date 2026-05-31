#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

class A4TrajectoryNode : public rclcpp::Node {
public:
  A4TrajectoryNode()
      : Node("a4_trajectory_node"),
        joint_names_({"joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"}),
        motor_map_({{1, 0x141}, {2, 0x142}, {3, 0x144}, {4, 0x145}, {5, 0x146}, {6, 0x147}, {7, 0x148}}),
        current_angles_deg_(7, 0.0), target_angles_deg_(7, 0.0), initial_angles_deg_(7, 0.0) {

    // ---- Setup CAN ----
    const char *can_interface = "can0";
    struct ifreq ifr {};
    struct sockaddr_can addr {};

    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
      RCLCPP_FATAL(this->get_logger(), "Failed to create CAN socket");
      rclcpp::shutdown();
      return;
    }

    std::strcpy(ifr.ifr_name, can_interface);
    ioctl(socket_fd_, SIOCGIFINDEX, &ifr);
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
      RCLCPP_FATAL(this->get_logger(), "Failed to bind CAN socket");
      rclcpp::shutdown();
      return;
    }

    // ---- Read initial positions with 0x92 ----
    for (auto &[jidx, mid] : motor_map_) {
      if (mid == 0x141) continue; // skip joint_1
      float init_angle = 0.0f;
      if (!readMultiTurnAngle(mid, init_angle)) {
        RCLCPP_WARN(this->get_logger(), "Failed to read motor 0x%X", mid);
        continue;
      }
      initial_angles_deg_[jidx-1] = init_angle;
      current_angles_deg_[jidx-1] = init_angle;
      target_angles_deg_[jidx-1]  = init_angle;
      RCLCPP_INFO(this->get_logger(), "Motor 0x%X init angle = %.2f deg", mid, init_angle);
    }

    // ---- Subscriber for JointTrajectory ----
    traj_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "/position_arm_controller/joint_trajectory", 10,
      std::bind(&A4TrajectoryNode::traj_callback, this, std::placeholders::_1));

    // ---- Control loop timer ----
    timer_ = this->create_wall_timer(10ms, std::bind(&A4TrajectoryNode::control_loop, this));
  }

  ~A4TrajectoryNode() {
    close(socket_fd_);
  }

private:
  int socket_fd_;
  std::vector<std::string> joint_names_;
  std::map<int,uint32_t> motor_map_;   // joint idx → CAN ID
  std::vector<double> current_angles_deg_;
  std::vector<double> target_angles_deg_;
  std::vector<double> initial_angles_deg_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // --- Receive JointTrajectory goals ---
  void traj_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
    if (msg->points.empty()) return;
    auto point = msg->points[0]; // only first point for now
    if (point.positions.size() != 7) {
      RCLCPP_ERROR(this->get_logger(), "Expected 7 joints, got %zu", point.positions.size());
      return;
    }

    for (size_t i=0; i<7; i++) {
      double rel_deg = point.positions[i] * 180.0 / M_PI; // rad → deg
      target_angles_deg_[i] = initial_angles_deg_[i] + rel_deg;  // relative to initial
    }
    RCLCPP_INFO(this->get_logger(), "New relative trajectory target received.");
  }

  // --- Control loop: step toward targets, send A4 command ---
  void control_loop() {
    const double step = 0.3; // deg per tick
    for (auto &[jidx, mid] : motor_map_) {
      if (mid == 0x141) continue; // skip joint_1

      double error = target_angles_deg_[jidx-1] - current_angles_deg_[jidx-1];
      if (std::fabs(error) > step) {
        current_angles_deg_[jidx-1] += (error > 0 ? step : -step);
      } else {
        current_angles_deg_[jidx-1] = target_angles_deg_[jidx-1];
      }

      // Send CAN A4 command
      int32_t angle_cdeg = static_cast<int32_t>(current_angles_deg_[jidx-1] * 100.0); // centi-deg
      sendA4PositionCommand(mid, angle_cdeg, 30); // fixed max speed
    }
  }

  // --- CAN helpers ---
  bool readMultiTurnAngle(uint32_t motor_id, float &angle_deg_out) {
    struct can_frame cmd = {};
    cmd.can_id = motor_id;
    cmd.can_dlc = 8;
    cmd.data[0] = 0x92;
    write(socket_fd_, &cmd, sizeof(cmd));

    struct can_frame reply;
    struct timeval timeout = {0, 10000}; // 10ms timeout
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket_fd_, &read_set);

    int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
    if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
      ssize_t bytes = read(socket_fd_, &reply, sizeof(reply));
      if (bytes > 0 && reply.data[0] == 0x92 && reply.can_id == (motor_id + 0x100)) {
        int32_t angle_raw;
        std::memcpy(&angle_raw, &reply.data[4], sizeof(angle_raw));
        angle_deg_out = angle_raw * 0.01f;
        return true;
      }
    }
    return false;
  }

  void sendA4PositionCommand(uint32_t motor_id, int32_t angleControl, uint16_t maxSpeed) {
    struct can_frame frame {};
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    frame.data[0] = 0xA4;
    frame.data[1] = 0x00;
    frame.data[2] = static_cast<uint8_t>(maxSpeed);
    frame.data[3] = static_cast<uint8_t>(maxSpeed >> 8);
    frame.data[4] = static_cast<uint8_t>(angleControl);
    frame.data[5] = static_cast<uint8_t>(angleControl >> 8);
    frame.data[6] = static_cast<uint8_t>(angleControl >> 16);
    frame.data[7] = static_cast<uint8_t>(angleControl >> 24);
    write(socket_fd_, &frame, sizeof(frame));
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<A4TrajectoryNode>());
  rclcpp::shutdown();
  return 0;
}

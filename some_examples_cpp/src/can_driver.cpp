#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <map>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class A4CanDriverNode : public rclcpp::Node {
public:
  A4CanDriverNode()
      : Node("a4_can_driver_node"),
        motor_ids_({0x142, 0x144, 0x145, 0x146, 0x147, 0x148}),
        initial_angles_(), target_angles_() {

    // --- Open CAN socket ---
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

    // --- Read initial absolute angles once ---
    for (auto id : motor_ids_) {
      float init_angle = 0.0f;
      if (!readMultiTurnAngle(id, init_angle)) {
        RCLCPP_FATAL(this->get_logger(), "Failed to read angle for motor ID: 0x%X", id);
        rclcpp::shutdown();
        return;
      }

      initial_angles_[id] = init_angle;
      target_angles_[id] = static_cast<int32_t>(init_angle * 100.0); // centi-deg

      RCLCPP_INFO(this->get_logger(), "Motor 0x%X initial angle: %.2f deg", id, init_angle);
    }

    // --- Subscribe to target angles topic ---
    sub_target_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/target_joint_angles_deg", 10,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
          if (msg->data.size() != 7) { // expect 7 joints total
            RCLCPP_WARN(this->get_logger(), "Received %zu angles but expected 7",
                        msg->data.size());
            return;
          }

          // Map joints 2–7 (indices 1..6) to motor_ids_
          for (size_t i = 0; i < motor_ids_.size(); i++) {
            uint32_t id = motor_ids_[i];
            double rel_angle = msg->data[i + 1]; // skip joint_1
            double abs_target = 0.0;
            if (id == 0x142) {
                // Only motor 0x142 uses scaled relative angle
                abs_target = initial_angles_[id] + (rel_angle * 10.0);
            } else {
                abs_target = initial_angles_[id] + rel_angle;
            }
            target_angles_[id] = static_cast<int32_t>(abs_target * 100.0); // centi-deg
            RCLCPP_INFO(this->get_logger(),
                "Motor 0x%X → Init: %.2f deg | RelCmd: %.2f deg | Target: %.2f deg (%.d cdeg)",
                id, initial_angles_[id], rel_angle, abs_target, target_angles_[id]);
          }
        });

    // --- Publisher for joint states ---
    pub_joint_states_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    // --- Timer for control loop every 14ms ---
    timer_ = this->create_wall_timer(14ms, std::bind(&A4CanDriverNode::control_loop, this));
  }

  ~A4CanDriverNode() override {
    close(socket_fd_);
  }

private:
  int socket_fd_;
  std::vector<uint32_t> motor_ids_;
  std::map<uint32_t, float> initial_angles_;  // startup reference
  std::map<uint32_t, int32_t> target_angles_; // centi-deg
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_target_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_states_;
  rclcpp::TimerBase::SharedPtr timer_;

  void control_loop() {
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.name.resize(7);
    js.position.resize(7);
    js.velocity.resize(7);
    js.effort.resize(7);

    // joint_1 dummy (not controlled here)
    js.name[0] = "joint_1";
    js.position[0] = 0.0;
    js.velocity[0] = 0.0;
    js.effort[0] = 0.0;

    // joints 2–7 → motors
    for (size_t i = 0; i < motor_ids_.size(); i++) {
      uint32_t id = motor_ids_[i];
      int32_t target_cdeg = target_angles_[id];
      const uint16_t max_speed_limit =
        (id == 0x142) ? 300 : (id == 0x148 ? 75 : 50);

      int8_t temp;
      int16_t iq, speed, angle_deg;

      if (sendA4PositionCommand(id, target_cdeg, max_speed_limit, temp, iq, speed, angle_deg)) {
        js.name[i + 1] = "joint_" + std::to_string(i + 2);

        // --- Relative angle calculation ---
        double rel_angle_deg;
        if (id == 0x142) {
            rel_angle_deg = (static_cast<double>(angle_deg) - initial_angles_[id]) / 10.0;
        } else {
            rel_angle_deg = static_cast<double>(angle_deg) - initial_angles_[id];
        }

        // publish relative position in radians
        js.position[i + 1] = rel_angle_deg * M_PI / 180.0;

        // velocity (rad/s) and effort
        js.velocity[i + 1] = static_cast<double>(speed) * M_PI / 180.0;
        js.effort[i + 1]   = static_cast<double>(iq) * 0.01;
        }

    }

    pub_joint_states_->publish(js);
  }

  bool readMultiTurnAngle(uint32_t motor_id, float &angle_deg_out) {
    struct can_frame cmd = {};
    cmd.can_id = motor_id;
    cmd.can_dlc = 8;
    cmd.data[0] = 0x92;
    write(socket_fd_, &cmd, sizeof(cmd));

    struct can_frame reply;
    struct timeval timeout = {0, 20000}; // 20ms timeout
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

  bool sendA4PositionCommand(uint32_t motor_id, int32_t angle_cdeg, uint16_t max_speed,
                             int8_t &temp, int16_t &iq, int16_t &speed, int16_t &angle_deg) {
    struct can_frame frame {};
    frame.can_id = motor_id;
    frame.can_dlc = 8;

    frame.data[0] = 0xA4;
    frame.data[1] = 0x00;
    frame.data[2] = static_cast<uint8_t>(max_speed);
    frame.data[3] = static_cast<uint8_t>(max_speed >> 8);
    frame.data[4] = static_cast<uint8_t>(angle_cdeg);
    frame.data[5] = static_cast<uint8_t>(angle_cdeg >> 8);
    frame.data[6] = static_cast<uint8_t>(angle_cdeg >> 16);
    frame.data[7] = static_cast<uint8_t>(angle_cdeg >> 24);

    write(socket_fd_, &frame, sizeof(frame));

    // ---- Read reply ----
    struct can_frame reply;
    struct timeval timeout = {0, 20000}; // 20ms
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket_fd_, &read_set);

    int ret = select(socket_fd_ + 1, &read_set, nullptr, nullptr, &timeout);
    if (ret > 0 && FD_ISSET(socket_fd_, &read_set)) {
      ssize_t bytes = read(socket_fd_, &reply, sizeof(reply));
      if (bytes > 0 && reply.data[0] == 0xA4 && reply.can_id == (motor_id + 0x100)) {
        temp = static_cast<int8_t>(reply.data[1]);
        iq = static_cast<int16_t>(reply.data[2] | (reply.data[3] << 8));
        speed = static_cast<int16_t>(reply.data[4] | (reply.data[5] << 8));
        angle_deg = static_cast<int16_t>(reply.data[6] | (reply.data[7] << 8));
        return true;
      }
    }
    return false;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<A4CanDriverNode>());
  rclcpp::shutdown();
  return 0;
}

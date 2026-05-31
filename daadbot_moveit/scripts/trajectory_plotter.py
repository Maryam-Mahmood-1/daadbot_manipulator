#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PipelineState
import matplotlib.pyplot as plt
import time  # Import the time module


class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')
        self.subscription = self.create_subscription(
            PipelineState,
            '/pipeline_state',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("TrajectoryPlotter node started. Waiting for /pipeline_state...")

    def listener_callback(self, msg):
        self.get_logger().info("Received message on /pipeline_state")
        
        # Check if the pipeline stage is 'DisplayMotionPath'
        if msg.pipeline_stage != 'DisplayMotionPath':
            self.get_logger().info(f"Skipping stage: {msg.pipeline_stage}")
            return

        trajectory = msg.response.trajectory.joint_trajectory
        joint_names = trajectory.joint_names
        self.get_logger().info(f"Joint Names: {joint_names}")

        points = trajectory.points

        # Check if trajectory has valid points
        if not points:
            self.get_logger().warn("Trajectory contains no points!")
            rclpy.shutdown()
            return

        # Prepare time, position, velocity, and acceleration data for plotting
        time_list = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        num_joints = len(joint_names)

        # Initialize lists to store positions, velocities, and accelerations for each joint
        positions = [[] for _ in range(num_joints)]
        velocities = [[] for _ in range(num_joints)]
        accelerations = [[] for _ in range(num_joints)]

        # Process each point in the trajectory
        for idx, p in enumerate(points):
            if len(p.positions) < num_joints or len(p.velocities) < num_joints or len(p.accelerations) < num_joints:
                self.get_logger().warn(f"Point {idx} has missing joint data. Skipping.")
                continue
            for i in range(num_joints):
                positions[i].append(p.positions[i])
                velocities[i].append(p.velocities[i])
                accelerations[i].append(p.accelerations[i])

        # Plot the trajectory for each joint
        for i in range(num_joints):
            plt.figure(figsize=(10, 5))
            #plt.plot(time_list, positions[i], label='Position')
            plt.plot(time_list, velocities[i], label='Velocity')
            plt.plot(time_list, accelerations[i], label='Acceleration')
            plt.xlabel('Time (s)')
            plt.ylabel(joint_names[i])
            plt.title(f'Trajectory for {joint_names[i]}')
            plt.legend()
            plt.grid(True)
            # Optionally save each plot
            # plt.savefig(f"trajectory_{joint_names[i]}.png")

        plt.tight_layout()
        plt.show()  # Show the plots

        self.get_logger().info("Plotting complete. Shutting down node.")
        rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    plotter = TrajectoryPlotter()
    rclpy.spin(plotter)


if __name__ == '__main__':
    main()

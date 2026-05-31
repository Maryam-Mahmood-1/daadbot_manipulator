#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from daadbot_msgs.action import DaadbotTaskServer
import time
import numpy as np
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

class MultiTaskServer(Node):
    def __init__(self):
        super().__init__('multi_task_server')
        self.get_logger().info('Starting...')
        self._action_server = ActionServer(self, DaadbotTaskServer, 'multi_task_server', self.execute_callback)

        self.daadbot = MoveItPy(node_name="moveit_py")
        self.daadbot_arm = self.daadbot.get_planning_component("arm")
        self.daadbot_gripper = self.daadbot.get_planning_component("gripper")
        
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Received Goal Request with Task ID %d' % goal_handle.request.task_id)
        
        arm_state = RobotState(self.daadbot.get_robot_model())
        gripper_state = RobotState(self.daadbot.get_robot_model())

        arm_joint_goal = []
        gripper_joint_goal = []

        if goal_handle.request.task_id == 0:
            arm_joint_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            gripper_joint_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        elif goal_handle.request.task_id == 1:
            arm_joint_goal = np.array([0.0, 0.543, 0.0, -0.209, 0.0, -1.239, 0.0])
            gripper_joint_goal = np.array([0.5, 0.5, -0.5, 0.5, 0.5, -0.5])
        elif goal_handle.request.task_id == 2:
            arm_joint_goal = np.array([0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0])
            gripper_joint_goal = np.array([0.5, 0.5, -0.5, 0.5, 0.5, -0.5])
        elif goal_handle.request.task_id == 3:
            arm_joint_goal = np.array([0.0, -0.4363, 0.0, 3.333, 0.0, -2.3 , 0.0])
            gripper_joint_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            self.get_logger().info('Invalid Task ID')
            return
        
        arm_state.set_joint_group_positions("arm", arm_joint_goal)
        gripper_state.set_joint_group_positions("gripper", gripper_joint_goal)

        self.daadbot_arm.set_start_state_to_current_state()
        self.daadbot_gripper.set_start_state_to_current_state()

        self.daadbot_arm.set_gaol_state(robot_state = arm_state)
        self.daadbot_gripper.set_gaol_state(robot_state = gripper_state)

        arm_plan_success = self.daadbot_arm.plan()
        gripper_plan_success = self.daadbot_gripper.plan() 

        if arm_plan_success and gripper_plan_success:
            self.daadbot_arm.execute(arm_plan_success.trajectory, controllers=[])
            self.daadbot_gripper.execute(gripper_plan_success.trajectory, controllers=[])
        else:
            self.get_logger().info('Planning failed for either arm or hand')
        
        goal_handle.succeed()
        result = DaadbotTaskServer.Result()
        result.success = True
        return result

def main():
    rclpy.init()
    multi_task_server = MultiTaskServer()
    rclpy.spin(multi_task_server)
    multi_task_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
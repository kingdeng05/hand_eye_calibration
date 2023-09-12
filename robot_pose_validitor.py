#!/usr/bin/env python3
"""
Brief:  This library acts as the main kinetic interface to the ROS Moveit! control library.
Input:  Robot enable from `kinetic_parameters/robot_0_input_overall_enable` ros parameter
Input: 	Pose goal from `kinetic_parameters/robot_0_input_end_effector_pose_request` ros parameter
Output: Robot movement and publishing the pose of the end effector pose with respect
        to the base_link frame
Author: Sander
"""

import sys
import rospy
import moveit_commander
import geometry_msgs.msg

class MoveGroup(object):
    def __init__(self):
        super(MoveGroup, self).__init__()
        rospy.init_node('kinetic_robot_control')

        moveit_commander.roscpp_initialize(sys.argv)

        robot_type = 0

        if robot_type > 0:
            group_name = "manipulator"
        else:
            group_name = "planning"

        move_group = moveit_commander.MoveGroupCommander(group_name)
        user_end_effector_link = 'link_6'
        user_base_link = 'base_link'
        move_group.set_end_effector_link(user_end_effector_link)
        move_group.set_pose_reference_frame(user_base_link)
        move_group.set_planner_id("RRTConnect")

        self.move_group = move_group

    def is_pose_valid(self, param_pose_goal):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = param_pose_goal[0]
        pose_goal.position.y = param_pose_goal[1]
        pose_goal.position.z = param_pose_goal[2]
        pose_goal.orientation.x = param_pose_goal[3]
        pose_goal.orientation.y = param_pose_goal[4]
        pose_goal.orientation.z = param_pose_goal[5]
        pose_goal.orientation.w = param_pose_goal[6]
        self.move_group.set_pose_target(pose_goal)
        self.move_group.set_planning_time(0.5)  # Max sensible time for valid plan
        plan_valid = self.move_group.plan(pose_goal)[0] 
        return plan_valid


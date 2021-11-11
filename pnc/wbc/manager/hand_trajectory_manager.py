import numpy as np
from util import util
from util import interpolation

from config.draco_manipulation_config import HandState

class HandTrajectoryManager(object):
    def __init__(self, pos_task, ori_task, side, robot):
        self._pos_task = pos_task
        self._ori_task = ori_task
        self._robot = robot
        self._side = side

        assert self._pos_task.target_id == self._ori_task.target_id


    def update_desired(self, hand_iso):
        hand_pos_des = hand_iso[0:3,3]
        hand_vel_des = np.zeros(3)
        hand_acc_des = np.zeros(3)

        hand_ori_des = util.rot_to_quat(hand_iso[0:3,0:3])
        hand_ang_vel_des = np.zeros(3)
        hand_ang_acc_des = np.zeros(3)

        self._pos_task.update_desired(hand_pos_des,hand_vel_des,hand_acc_des)
        self._ori_task.update_desired(hand_ori_des,hand_ang_vel_des,hand_ang_acc_des)

        # if self._side == HandState.LEFT:
            # print("========Left===============")
            # print("hand_ori_act:")
            # print(hand_iso[0:3,0:3])
            # print("hand_ori_after_transform:") 
            # print(util.quat_to_rot(hand_ori_des))
        # else:
            # print("========Right===============")
            # print("hand_ori_act:")
            # print(hand_iso[0:3,0:3])
            # print("hand_ori_after_transform:") 
            # print(util.quat_to_rot(hand_ori_des))
    def use_current():
        hand_iso = self._robot.get_link_iso(self._pos_task.target_id)
        hand_vel = self._robot.get_link_vel(self._pos_task.target_id)

        hand_pos_des = hand_iso[0:3,3]
        hand_vel_des = hand_vel[3:6]
        hand_acc_des = np.zeros(3)

        hand_ori_des = util.rot_to_quat(hand_iso[0:3,0:3])
        hand_ang_vel_des = hand_vel[0:3]
        hand_ang_acc_des = np.zeros(3)

        self._pos_task.update_desired(hand_pos_des,hand_vel_des,hand_acc_des)
        self._ori_task.update_desired(hand_ori_des,hand_ang_vel_des,hand_ang_acc_des)

    # def initialize_trajectory():

    # def update_trajectory():


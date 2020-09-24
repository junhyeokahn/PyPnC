import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from util import util


class FloatingBaseTrajectoryManager(object):
    def __init__(self, com_task, base_ori_task, robot):
        self._com_task = com_task
        self._base_ori_task = base_ori_task
        self._robot = robot

        self._start_time = 0.
        self._duration = 0.
        self._ini_com_pos = self._target_com_pos = np.zeros(3)
        self._ini_base_quat = self._target_base_quat = np.zeros(4)

    def initialize_floating_base_trajectory(self, start_time, duration,
                                            target_com_pos, target_base_quat):
        self._start_time = start_time
        self._duration = duration

        self._ini_com_pos = self._robot.get_com_pos()
        self._target_com_pos = target_com_pos

        self._ini_base_quat = util.rot_to_quat(
            self._robot.get_link_iso(self._base_ori_task.target_id)[0:3, 0:3])
        self._target_base_quat = target_base_quat
        self._quat_error = (R.from_quat(self._target_base_quat) *
                            R.from_quat(self._ini_base_quat).inv()).as_quat()
        self._exp_error = util.quat_to_exp(self._quat_error)

        ## TODO : Check this with dart function
        # import dartpy as dart
        # _quat_error = dart.math.Quaternion()
        # _quat_error.set_wxyz(self._quat_error[3], self._quat_error[0],
        # self._quat_error[1], self._quat_error[2])
        # _exp_error = dart.math.quatToExp(_quat_error)
        # print(self._exp_error, _exp_error)
        # __import__('ipdb').set_trace()
        ##

    def update_floating_base_desired(self, current_time):
        com_pos_des = com_vel_des = com_acc_des = np.zeros(3)
        for i in range(3):
            com_pos_des[i] = util.smooth_changing(
                self._ini_com_pos[i], self._target_com_pos[i], self._duration,
                current_time - self._start_time)
            com_vel_des[i] = util.smooth_changing_vel(
                self._ini_com_pos[i], self._target_com_pos[i], self._duration,
                current_time - self._start_time)
            com_acc_des[i] = util.smooth_changing_acc(
                self._ini_com_pos[i], self._target_com_pos[i], self._duration,
                current_time - self._start_time)

        self._com_task.update_desired(com_pos_des, com_vel_des, com_acc_des)

        scaled_t = util.smooth_changing(0, 1, self._duration,
                                        current_time - self._start_time)
        scaled_tdot = util.smooth_changing_vel(0, 1, self._duration,
                                               current_time - self._start_time)
        scaled_tddot = util.smooth_changing_acc(
            0, 1, self._duration, current_time - self._start_time)

        exp_inc = self._exp_error * scaled_t
        quat_inc = util.exp_to_quat(exp_inc)
        ## TODO : Check this with dart function
        # import dartpy as dart
        # quat_inc_ = dart.math.expToQuat(exp_inc)
        # print(quat_inc, quat_inc_)
        # __import__('ipdb').set_trace()
        ##
        base_quat_des = (R.from_quat(quat_inc) *
                         R.from_quat(self._ini_base_quat)).as_quat()
        base_angvel_des = self._exp_error * scaled_tdot
        base_angacc_des = self._exp_error * scaled_tddot

        self._base_ori_task.update_desired(base_quat_des, base_angvel_des,
                                           base_angacc_des)

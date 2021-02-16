import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from util import util
from pnc.wbc.task import Task
from pnc.data_saver import DataSaver


class BasicTask(Task):
    def __init__(self, robot, task_type, dim, target_id=None, data_save=False):
        super(BasicTask, self).__init__(robot, dim)

        self._target_id = target_id
        self._task_type = task_type
        self._b_data_save = data_save

        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def target_id(self):
        return self._target_id

    def update_cmd(self):

        if self._task_type == "JOINT":
            pos = self._robot.joint_positions
            pos_err = self._pos_des - pos
            vel_act = self._robot.joint_velocities
            if self._b_data_save:
                self._data_saver.add('joint_pos_des', self._pos_des.copy())
                self._data_saver.add('joint_vel_des', self._vel_des.copy())
                self._data_saver.add('joint_pos', pos.copy())
                self._data_saver.add('joint_vel', vel_act.copy())
                self._data_saver.add('w_joint', self._w_hierarchy)
        elif self._task_type == "SELECTED_JOINT":
            pos = self._robot.joint_positions[self._robot.get_joint_idx(
                self._target_id)]
            pos_err = self._pos_des - pos
            vel_act = self._robot.joint_velocities[self._robot.get_joint_idx(
                self._target_id)]
            if self._b_data_save:
                self._data_saver.add('joint_pos_des', self._pos_des.copy())
                self._data_saver.add('joint_vel_des', self._vel_des.copy())
                self._data_saver.add('joint_pos', pos.copy())
                self._data_saver.add('joint_vel', vel_act.copy())
                self._data_saver.add('w_joint', self._w_hierarchy)
        elif self._task_type == "LINK_XYZ":
            pos = self._robot.get_link_iso(self._target_id)[0:3, 3]
            pos_err = self._pos_des - pos
            vel_act = self._robot.get_link_vel(self._target_id)[3:6]
            if self._b_data_save:
                self._data_saver.add(self._target_id + '_pos_des',
                                     self._pos_des.copy())
                self._data_saver.add(self._target_id + '_vel_des',
                                     self._vel_des.copy())
                self._data_saver.add(self._target_id + '_pos', pos.copy())
                self._data_saver.add(self._target_id + '_vel', vel_act.copy())
                self._data_saver.add('w_' + self._target_id, self._w_hierarchy)
        elif self._task_type == "LINK_ORI":
            quat_des = R.from_quat(self._pos_des)
            quat_act = R.from_matrix(
                self._robot.get_link_iso(self._target_id)[0:3, 0:3])
            quat_err = R.from_matrix(
                np.dot(quat_des.as_matrix(),
                       quat_act.as_matrix().transpose())).as_quat()
            pos_err = util.quat_to_exp(quat_err)
            vel_act = self._robot.get_link_vel(self._target_id)[0:3]
            if self._b_data_save:
                self._data_saver.add(self._target_id + '_quat_des',
                                     quat_des.as_quat())
                self._data_saver.add(self._target_id + '_ang_vel_des',
                                     self._vel_des.copy())
                self._data_saver.add(self._target_id + '_quat',
                                     quat_act.as_quat())
                self._data_saver.add(self._target_id + '_ang_vel',
                                     vel_act.copy())
                self._data_saver.add('w_' + self._target_id + "_ori",
                                     self._w_hierarchy)
        elif self._task_type == "COM":
            pos = self._robot.get_com_pos()
            pos_err = self._pos_des - pos
            vel_act = self._robot.get_com_lin_vel()
            if self._b_data_save:
                self._data_saver.add(self._target_id + '_pos_des',
                                     self._pos_des.copy())
                self._data_saver.add(self._target_id + '_vel_des',
                                     self._vel_des.copy())
                self._data_saver.add(self._target_id + '_pos', pos.copy())
                self._data_saver.add(self._target_id + '_vel', vel_act.copy())
                self._data_saver.add('w_' + self._target_id, self._w_hierarchy)
        else:
            raise ValueError

        for i in range(self._dim):
            self._op_cmd[i] = self._acc_des[i] + self._kp[i] * pos_err[
                i] + self._kd[i] * (self._vel_des[i] - vel_act[i])

    def update_jacobian(self):
        if self._task_type == "JOINT":
            self._jacobian[:, self._robot.n_floating:self._robot.n_floating +
                           self._robot.n_a] = np.eye(self._dim)
            self._jacobian_dot_q_dot = np.zeros(self._dim)
        elif self._task_type == "SELECTED_JOINT":
            for i, jid in enumerate(self._robot.get_q_dot_idx(
                    self._target_id)):
                self._jacobian[i, jid] = 1
            self._jacobian_dot_q_dot = np.zeros(self._dim)
        elif self._task_type == "LINK_XYZ":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[3:6, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[3:6]
        elif self._task_type == "LINK_ORI":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[0:3, :]
            self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
                self._target_id)[0:3]
        elif self._task_type == "COM":
            self._jacobian = self._robot.get_com_lin_jacobian()
            self._jacobian_dot_q_dot = np.dot(
                self._robot.get_com_lin_jacobian_dot(),
                self._robot.get_q_dot())
        else:
            raise ValueError

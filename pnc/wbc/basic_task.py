import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from util import util


class BasicTask(Task):
    def __init__(self, robot, task_type, dim, target_id=None):
        super(BasicTask, self).__init__(robot, dim)

        self._target_id = target_id
        self._task_type = task_type

        if task_type is "JOINT":
            pass
        elif task_type is "SELECTED_JOINT":
            pass
        elif task_type is "LINK_XYZ":
            pass
        elif task_type is "LINK_ORI":
            pass
        elif task_type is "COM":
            pass
        else:
            raise ValueError

    def update_cmd(self):

        if task_type is "JOINT":
            pos_err = self._pos_des - self._robot.get_q()[-self._dim:]
            vel_act = self._robot.get_q_dot()[-self._dim:]
        elif task_type is "SELECTED_JOINT":
            pos_err = self._pos_des - self._robot.get_q()[
                self._robot.get_q_idx(self._target_id)]
            vel_act = self._robot.get_q_dot()[self._robot.get_q_idx(
                self._target_id)]
        elif task_type is "LINK_XYZ":
            pos_err = self._pos_des - self._robot.get_link_iso(
                self._target_id)[0:3, 3]
            vel_act = self._robot.get_link_vel(self._target_id)[3:6]
        elif task_type is "LINK_ORI":
            quat_des = R.from_quat(self._pos_des)
            quat_act = R.from_matrix(
                self._robot.get_link_iso(self._target_id)[0:3, 0:3])
            quat_err = (quat_des * quat_act.inv()).as_quat()

            pos_err = util.quat_to_exp(quat_err)
            vel_act = self._robot.get_link_vel(self._target_id)[0:3]
        elif task_type is "COM":
            pos_err = self._pos_des - self._robot.get_com_pos()
            vel_act = self._robot.get_com_lin_vel()
        else:
            raise ValueError

        for i in range(self._dim):
            self._op_cmd[i] = self._acc_des[i] + self._kp[i] * pos_err[
                i] + self._kd[i] * (self._vel_des[i] - vel_act[i])

    def update_jacobian(self, arg1):
        if task_type is "JOINT":
            self._jacobian[:, self._robot.n_virtual:] = np.eye(self._dim)
            self._jacobian_dot_q_dot = np.zeros(self._dim)
        elif task_type is "SELECTED_JOINT":
            for jid in self._robot.get_q_idx(self._target_id):
                self._jacobian[jid, jid + self._robot.n_virtual] = 1
            self._jacobian_dot_q_dot = np.zeros(self._dim)
        elif task_type is "LINK_XYZ":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[3:6, :]
            self._jacobian_dot_q_dot = np.dot(
                self._robot.get_link_jacobian_dot(self._target_id)[3:6, :],
                self._robot.get_q_dot())
        elif task_type is "LINK_ORI":
            self._jacobian = self._robot.get_link_jacobian(
                self._target_id)[0:3, :]
            self._jacobian_dot_q_dot = np.dot(
                self._robot.get_link_jacobian_dot(self._target_id)[0:3, :],
                self._robot.get_q_dot())
        elif task_type is "COM":
            self._jacobian = self._robot.get_com_lin_jacobian()
            self._jacobian_dot_q_dot = np.dot(
                self._robot.get_com_lin_jacobian_dot(),
                self._robot.get_q_dot())
        else:
            raise ValueError

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

from pnc.robot_system.robot_system import RobotSystem
from util import util as util


class PyBulletRobotsystem(RobotSystem):
    """
    PyBullet considers floating base with 7 positions and 6 velocities.
    q = [quat_x, quat_y, quat_z, x, y, z, joints, quat_w]
    qdot = [ang_vel, lin_vel, joints]
    Note that first 6 elements in qdot should be represented in the base frame.
    This is how PyBullet represents the floating base.
    You could double-check it by observing a jacobian matrix from PyBullet
    """
    def __init__(self, filepath, b_fixed_base, b_print_info=False):
        super(PyBulletRobotsystem, self).__init__(filepath, b_fixed_base,
                                                  b_print_info)

    def _config_robot(self, filepath):

        if self._b_use_fixed_base:
            self._robot_id = p.loadURDF(filepath, useFixedBase=True)
        else:
            self._robot_id = p.loadURDF(filepath)
        __import__('ipdb').set_trace()
        print(self._robot_id)

        jpos_lb, jpos_ub, jtrq_ub, jtrq_lb, jvel_ub, jvel_lb = [],[],[],[],[],[]

        self._link_id[(p.getBodyInfo(self._robot_id)[0]).decode("utf-8")] = -1
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self._joint_id[info[1].decode("utf-8")] = info[0]
                jpos_lb.append(info[8])
                jpos_ub.append(info[9])
                jtrq_ub.append(info[10])
                jtrq_lb.append(-info[10])
                jvel_ub.append(info[11])
                jvel_lb.append(-info[11])
            self._link_id[info[12].decode("utf-8")] = info[0]
            self._nq = max(self._nq, info[3])
            self._n_q_dot = max(self._n_q_dot, info[4])
        self._nq += 1
        self._n_q_dot += 1
        self._n_a = len(self._joint_id)

        self._joint_pos_limit = np.stack((jpos_lb, jpos_ub), axis=1)
        self._joint_vel_limit = np.stack((jvel_lb, jvel_ub), axis=1)
        self._joint_trq_limit = np.stack((jtrq_lb, jtrq_ub), axis=1)

    def get_q_idx(self, joint_id):
        if type(joint_id) is list:
            return [self._joint_id[id] for id in joint_id]
        else:
            self._joint_id[id]

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd,
                                joint_trq_cmd):
        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in self._joint_id.items():
            joint_idx = self._joint_id[k] - self._n_floating
            command["joint_pos"][k] = joint_pos_cmd[joint_idx]
            command["joint_vel"][k] = joint_vel_cmd[joint_idx]
            command["joint_trq"][k] = joint_trq_cmd[joint_idx]

        return command

    def update_system(self,
                      base_pos,
                      base_quat,
                      base_lin_vel,
                      base_ang_vel,
                      joint_pos,
                      joint_vel,
                      b_cent=False):
        assert len(joint_pos.keys()) == self._n_a

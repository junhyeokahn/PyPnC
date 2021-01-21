import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import numpy as np
import pinocchio as pin

from pnc.robot_system.robot_system import RobotSystem
from util import util as util


class PinocchioRobotSystem(RobotSystem):
    """
    Pinnochio considers floating base with 7 positions and 6 velocities with the
    order of [x, y, z, quat_x, quat_y, quat_z, quat_w, joints] and [].
    """
    def __init__(self,
                 urdf_file,
                 package_dir,
                 b_fixed_base,
                 b_print_info=False):
        supare(PinocchioRobotSystem, self).__init__(urdf_file, package_dir,
                                                    b_fixed_base, b_print_info)

    def _config_robot(self, urdf_file, package_dir):
        if self._b_fixed_base:
            # Fixed based robot
            self._model, self._collision_model, self._visual_model = pin.buildModelsFromUrdf(
                urdf_filename, package_dir)
            self._n_floating = 0
        else:
            # Floating based robot
            self._model, self._collision_model, self._visual_model = pin.buildModelsFromUrdf(
                urdf_filename, package_dir, pin.JointModelFreeFlyer())
            self._n_floating = 6

        self._n_q = self._model.nq
        self._n_q_dot = self._model.nv
        self._n_a = self._model.njoints

        # model.names : joints
        # model.frames : body, joint, body, joint, ...

        for j_id, j_name in enumerate(self._model.names):
            if j_name == 'root_joint' or j_name == 'universe':
                pass
            else:
                self._joint_id[j_name] = j_id

        for f_id, frame in enumerate(self._model.frames):
            if frame.name == 'root_joint' or frame.name == 'universe':
                pass
            else:
                if f_id % 2 == 0:
                    # Link
                    link_id = int(f_id / 2 - 1)
                    self._link_id[frame.name] = link_id
                else:
                    # Joint
                    joint_id = int((f_id - 1) / 2 - 1)
                    self._joint_id[frame.name] = joint_id

        assert len(self._joint_name == self._n_a)

        self._total_mass = sum([inertia.mass for inertia in model.inertias])

        self._joint_pos_limit = np.stack(
            [
                self._model.lowerPositionLimit(),
                self._model.upperPositionLimit()
            ],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]
        self._joint_vel_limit = np.stack(
            [-self._model.velocityLimit, self._model.velocityLimit],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]
        self._joint_trq_limit = np.stack(
            [-self._model.effortLimit, self._model.effortLimit],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]

    def get_q_idx(self, joint_id):
        if type(joint_id) is list:
            return [7 + self._joint_id[id] for id in joint_id]
        else:
            return 7 + self._joint_id[joint_id]

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd,
                                joint_trq_cmd):
        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in self._joint_id.items():
            command["joint_pos"][k] = joint_pos_cmd[v]
            command["joint_vel"][k] = joint_vel_cmd[v]
            command["joint_trq"][k] = joint_trq_cmd[v]

        return command

    def update_system(self,
                      base_com_pos,
                      base_com_quat,
                      base_com_lin_vel,
                      base_com_ang_vel,
                      base_joint_pos,
                      base_joint_quat,
                      base_joint_lin_vel,
                      base_joint_ang_vel,
                      joint_pos,
                      joint_vel,
                      b_cent=False):

        assert len(joint_pos.keys()) == self._n_a

        if not self._b_fixed_base:
            # Floating Based Robot
            q = np.zeros(self._n_q)
            q[0:3] = np.copy(base_pos)
            q[3:7] = np.copy(base_quat)
            q[7:7 + self._n_a] = np.copy(list(joint_pos.values()))

            rot_w_base = util.quat_to_rot(base_quat)
            qdot = np.zeros(self._n_q_dot)
            qdot[0:3] = np.dot(rot_w_base.transpose(), np.copy(base_lin_vel))
            qdot[3:6] = np.dot(rot_w_base.transpose(), np.copy(base_ang_vel))
            qdot[6:6 + self._n_a] = np.copy(list(joint_vel.values()))
        else:
            # Fixed Based Robot
            raise NotImplementedError

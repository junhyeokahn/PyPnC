import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import pprint

import pybullet as p
import dartpy as dart
import numpy as np

from pnc.robot_system.robot_system import RobotSystem
from util import util as util


class DartRobotSystem(RobotSystem):
    def __init__(self, n_virtual, filepath):
        super(DartRobotSystem, self).__init__(n_virtual, filepath)

    def config_robot(self, filepath):
        self._skel = dart.utils.DartLoader().parseSkeleton(filepath)

        self._n_q = self._n_q_dot = self._skel.getNumDofs()
        self._n_a = self._n_q_dot - self.n_virtual
        self._total_mass = self._skel.getMass()
        self._skel.getPositionLowerLimits()
        self._skel.getPositionUpperLimits()
        self._joint_pos_limit = np.stack([
            self._skel.getPositionLowerLimits(),
            self._skel.getPositionUpperLimits()
        ],
                                         axis=1)[self._n_virtual:, :]
        self._joint_vel_limit = np.stack([
            self._skel.getVelocityLowerLimits(),
            self._skel.getVelocityUpperLimits()
        ],
                                         axis=1)[self._n_virtual:, :]
        self._joint_trq_limit = np.stack([
            self._skel.getForceLowerLimits(),
            self._skel.getForceUpperLimits()
        ],
                                         axis=1)[self._n_virtual:, :]

        for i in range(self._skel.getNumJoints()):
            j = self._skel.getJoint(i)
            self._joint_id[j.getName()] = j

        for i in range(self._skel.getNumBodyNodes()):
            bn = self._skel.getBodyNode(i)
            self._link_id[bn.getName()] = bn

        # print("=" * 80)
        # print("DartRobotSystem")
        # print("nq: ", self._n_q, ", nv: ", self._n_q_dot, ", na: ", self._n_a)
        # print("+" * 80)
        # print("Joint Infos")
        # util.pretty_print(self._joint_id)
        # print("+" * 80)
        # print("Link Infos")
        # util.pretty_print(self._link_id)
        # print("=" * 80)

    def get_q_idx(self, joint_id):
        """
        Get joint index in generalized coordinate

        Parameters
        ----------
        joint_id (str or list of str)

        Returns
        -------
        joint_idx (int or list of int)
        """
        if type(joint_id) is list:
            return [
                self._joint_id[id].getIndexInSkeleton(0) for id in joint_id
            ]
        else:
            return self._joint_id[joint].getIndexInSkeleton(0)

    def update_system(self, base_pos, base_quat, base_lin_vel, base_ang_vel,
                      joint_pos, joint_vel):
        if base_pos is not None:
            base_iso = dart.math.Isometry3()
            base_iso.set_rotation(
                np.reshape(np.asarray(p.getMatrixFromQuaternion(base_quat)),
                           (3, 3)))
            base_iso.set_translation(base_pos)
            base_vel = np.concatenate([base_ang_vel, base_lin_vel])
            self._skel.getJoint(0).setSpatialMotion(
                base_iso, dart.dynamics.Frame.World(),
                np.reshape(base_vel, (6, 1)), dart.dynamics.Frame.World(),
                dart.dynamics.Frame.World(), np.zeros((6, 1)),
                dart.dynamics.Frame.World(), dart.dynamics.Frame.World())

        assert len(joint_pos.keys()) == self._n_a

        for (p_k, p_v), (v_k, v_v) in zip(joint_pos.items(),
                                          joint_vel.items()):
            # Assume the joints have 1 dof
            self._joint_id[p_k].setPosition(0, p_v)
            self._joint_id[v_k].setVelocity(0, v_v)

    def get_q(self):
        return self._skel.getPositions()

    def get_q_dot(self):
        return self._skel.getVelocities()

    def get_mass_matrix(self):
        return self._skel.getMassMatrix()

    def get_gravity(self):
        return self._skel.getGravityForces()

    def get_coriolis(self):
        return self._skel.getCoriolisForces()

    def get_com_pos(self):
        return self._skel.getCOM()

    def get_com_lin_vel(self):
        return self._skel.getCOMLinearVelocity()

    def get_com_lin_jacobian(self):
        return self._skel.getCOMLinearJacobian()

    def get_com_lin_jacobian_dot(self):
        return self._skel.getCOMLinearJacobianDeriv()

    def get_link_iso(self, link_id):
        link_iso = self._link_id[link_id].getTransform()
        ret = np.eye(4)
        ret[0:3, 0:3] = link_iso.get_rotation()
        ret[0:3, 3] = link_iso.get_translation()
        return ret

    def get_link_vel(self, link_id):
        return self._link_id[link_id].getSpatialVelocity()

    def get_link_jacobian(self, link_id):
        return self._link_id[link_id].getJacobian(dart.dynamics.Frame.World())

    def get_link_jacobian_dot(self, link_id):
        return self._link_id[link_id].getJacobianClassicDeriv()

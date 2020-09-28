import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import pprint
from collections import OrderedDict

import pybullet as p
import dartpy as dart
import numpy as np
from scipy.spatial.transform import Rotation as R

from pnc.robot_system.robot_system import RobotSystem
from util import util as util


class DartRobotSystem(RobotSystem):
    def __init__(self, filepath, floating_joint_list, fixed_joint_list):
        super(DartRobotSystem, self).__init__(filepath, floating_joint_list,
                                              fixed_joint_list)

    def config_robot(self, filepath, floating_joint_list, fixed_joint_list):
        self._skel = dart.utils.DartLoader().parseSkeleton(filepath)

        for i in range(self._skel.getNumJoints()):
            j = self._skel.getJoint(i)
            if j.getName() in floating_joint_list:
                self._n_virtual += j.getNumDofs()
                self._floating_id[j.getName()] = j
            elif j.getName() not in fixed_joint_list:
                self._joint_id[j.getName()] = j
            else:
                pass

        for i in range(self._skel.getNumBodyNodes()):
            bn = self._skel.getBodyNode(i)
            self._link_id[bn.getName()] = bn

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

        # print("=" * 80)
        # print("DartRobotSystem")
        # print("nq: ", self._n_q, ", nv: ", self._n_q_dot, ", na: ", self._n_a,
        # ", nvirtual: ", self._n_virtual)
        # print("+" * 80)
        # print("Joint Infos")
        # util.pretty_print([*self._joint_id.keys()])
        # print("+" * 80)
        # print("Floating Joint Infos")
        # util.pretty_print([*self._floating_id.keys()])
        # print("+" * 80)
        # print("Link Infos")
        # util.pretty_print([*self._link_id.keys()])
        # print("=" * 80)

    def get_q_idx(self, joint_id):
        if type(joint_id) is list:
            return [
                self._joint_id[id].getIndexInSkeleton(0) for id in joint_id
            ]
        else:
            return self._joint_id[joint].getIndexInSkeleton(0)

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd,
                                joint_trq_cmd):

        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in (self._joint_id).items():
            joint_idx = self._joint_id[k].getIndexInSkeleton(
                0) - self._n_virtual
            command["joint_pos"][k] = joint_pos_cmd[joint_idx]
            command["joint_vel"][k] = joint_vel_cmd[joint_idx]
            command["joint_trq"][k] = joint_trq_cmd[joint_idx]
        return command

    def update_system(self, base_pos, base_quat, base_lin_vel, base_ang_vel,
                      joint_pos, joint_vel):

        assert len(joint_pos.keys()) == self._n_a

        if len(self._floating_id) > 1:
            # Assume (x,y,z,rz,ry,rx)
            #TODO
            pass
        elif len(self._floating_id) == 1:
            # Assume base_iso is representing root com frame
            p_com_joint = self._skel.getRootBodyNode().getLocalCOM()
            base_iso = dart.math.Isometry3()
            base_iso.set_rotation(
                np.reshape(np.asarray(p.getMatrixFromQuaternion(base_quat)),
                           (3, 3)))
            base_iso.set_translation(base_pos - p_com_joint)
            base_vel = np.concatenate([base_ang_vel, base_lin_vel])
            self._skel.getRootJoint().setSpatialMotion(
                base_iso, dart.dynamics.Frame.World(),
                np.reshape(base_vel, (6, 1)), dart.dynamics.Frame.World(),
                dart.dynamics.Frame.World(), np.zeros((6, 1)),
                dart.dynamics.Frame.World(), dart.dynamics.Frame.World())
        else:
            pass

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
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM SE(3)
        """
        link_iso = self._link_id[link_id].getTransform()
        ret = np.eye(4)
        ret[0:3, 0:3] = link_iso.rotation()
        ret[0:3, 3] = self._link_id[link_id].getCOM()
        return ret

    def get_link_vel(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM Screw
        """
        return self._link_id[link_id].getCOMSpatialVelocity()

    def get_link_jacobian(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM Jacobian
        """
        return self._skel.getJacobian(self._link_id[link_id],
                                      self._link_id[link_id].getLocalCOM())

    def get_link_jacobian_dot(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM Jacobian Dot
        """
        return self._skel.getJacobianClassicDeriv(
            self._link_id[link_id], self._link_id[link_id].getLocalCOM())

    def debug_print_link_info(self):
        print("-" * 80)
        print("Controller")
        print("-" * 80)
        for (k, v) in self._link_id.items():
            print(k,
                  self.get_link_iso(k)[0:3, 3],
                  R.from_matrix(self.get_link_iso(k)[0:3, 0:3]).as_quat(),
                  self.get_link_vel(k)[3:6],
                  self.get_link_vel(k)[0:3])

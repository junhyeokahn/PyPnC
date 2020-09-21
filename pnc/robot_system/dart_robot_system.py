import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p
import dartpy as dart

from pnc.robot_system.robot_system import RobotSystem


class DartRobotSystem(RobotSystem):
    def __init__(self, filepath):
        super(DartRobotSystem, self).__init__(filepath)

    def config_robot(self, filepath):
        self._skel = dart.utils.DartLoader().parseSkeleton(filepath)
        __import__('ipdb').set_trace()
        n_dof = self._skel.getNumDofs()
        n_joint = self._skel.getNumJoints()
        n_bn = self._skel.getNumBodyNodes()

        # print("#" * 80)
        # print("DOF")
        # print("#" * 80)
        # for dof in self._skel.getDofs():
        # print(dof.getName())

        # __import__('ipdb').set_trace()
        # print("#" * 80)
        # print("JOINT")
        # print("#" * 80)
        # for joint in self._skel.getJoints():
        # print(joint.getName(), joint.getNumDofs(), joint.getType())

        # print("#" * 80)
        # print("BodyNode")
        # print("#" * 80)
        # for i in range(n_bn):
        # bn = self._skel.getBodyNode(i)
        # print(bn.getName(), bn.getMass())

    def update_system(self, base_pos, base_quat, base_lin_vel, base_ang_vel,
                      joint_pos, joint_vel):
        pass

    def get_q(self):
        pass

    def get_qdot(self):
        pass

    def get_mass_matrix(self):
        pass

    def get_gravity(self):
        pass

    def get_coriolis(self):
        pass

    def get_com_pos(self):
        pass

    def get_com_vel(self):
        pass

    def get_com_jac(self):
        pass

    def get_body_iso(self):
        pass

    def get_body_vel(self):
        pass

    def get_body_jac(self):
        pass

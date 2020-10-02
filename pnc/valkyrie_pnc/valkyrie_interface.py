import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np

from pnc.interface import Interface
from config.valkyrie_config import SimConfig, PnCConfig
from pnc.valkyrie_pnc.valkyrie_state_provider import ValkyrieStateProvider
from pnc.valkyrie_pnc.valkyrie_control_architecture import ValkyrieControlArchitecture
from pnc.data_saver import DataSaver


class ValkyrieInterface(Interface):
    def __init__(self):
        super(ValkyrieInterface, self).__init__()

        if PnCConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                cwd +
                "/../PnC/RobotModel/Robot/Valkyrie/ValkyrieSim_PyPnC.urdf",
                ['rootJoint'])
        else:
            raise ValueError
        self._control_architecture = ValkyrieControlArchitecture(self._robot)
        self._sp = ValkyrieStateProvider(self._robot)
        self._data_saver = DataSaver()
        self._data_saver.create('time', 1)

    def get_command(self, sensor_data):
        self._data_saver.add('time', self._running_time)

        if self._count == 0:
            self._sp.nominal_joint_pos = sensor_data["joint_pos"]

        # Update RobotSystem
        self._robot.update_system(sensor_data["base_pos"],
                                  sensor_data["base_quat"],
                                  sensor_data["base_lin_vel"],
                                  sensor_data["base_ang_vel"],
                                  sensor_data["joint_pos"],
                                  sensor_data["joint_vel"])
        # self._robot.debug_print_link_info()
        # Compute Cmd
        command = self._control_architecture.get_command()

        # print(self._running_time, " : ", self._robot.get_com_pos())

        # Increase time variables
        self._count += 1
        self._running_time += PnCConfig.CONTROLLER_DT
        self._sp.curr_time = self._running_time
        self._sp.state = self._control_architecture.state

        self._data_saver.advance()

        return command

    def debug_get_command2(self, q, qdot):
        k = [
            "leftHipYaw", "leftHipRoll", "leftHipPitch", "leftKneePitch",
            "leftAnklePitch", "leftAnkleRoll", "rightHipYaw", "rightHipRoll",
            "rightHipPitch", "rightKneePitch", "rightAnklePitch",
            "rightAnkleRoll", "torsoYaw", "torsoPitch", "torsoRoll",
            "leftShoulderPitch", "leftShoulderRoll", "leftShoulderYaw",
            "leftElbowPitch", "leftForearmYaw", "lowerNeckPitch", "neckYaw",
            "upperNeckPitch", "rightShoulderPitch", "rightShoulderRoll",
            "rightShoulderYaw", "rightElbowPitch", "rightForearmYaw"
        ]
        v = np.array([
            -8.17856e-07, 4.30013e-07, -0.599995, 1.19998, -0.599895,
            1.7189e-06, 1.48176e-06, -9.07747e-07, -0.599995, 1.19998,
            -0.599895, 1.11673e-05, 8.14396e-07, 1.70736e-06, -2.99131e-07,
            0.200008, -1.09997, 0.000353999, -0.39999, 1.49888, 3.43861e-06,
            -1.24655e-06, -3.03511e-08, 0.199983, 1.09996, 0.00036993,
            0.399943, 1.49889
        ])[-28:]

        nominal_jpos = OrderedDict()
        for i in range(28):
            nominal_jpos[k[i]] = v[i]
        self._sp.nominal_joint_pos = nominal_jpos

        self._robot.debug_update_system(q, qdot)

        command = self._control_architecture.get_command()

        return command

    def debug_get_command(self):
        q = np.array([
            0.499999, 0.499999, 1.024971, 0.523598, 0., 0., -0.000002,
            0.000001, -0.599981, 1.199952, -0.599704, 0.000005, 0.000004,
            -0.000002, -0.599981, 1.199951, -0.599704, 0.000027, 0.000002,
            0.000007, -0.000000, 0.200009, -1.099929, 0.000725, -0.399953,
            1.497563, 0.000007, -0.000003, 0.000003, 0.199960, 1.099927,
            0.000764, 0.399862, 1.497582
        ])

        qdot = np.array([
            -0.000673, -0.000341, -0.019141, 0.000049, -0.006214, -0.000134,
            -0.001497, 0.000957, 0.013401, -0.032492, 0.191210, 0.003190,
            0.002164, -0.001256, 0.013613, -0.032847, 0.191303, 0.015665,
            0.001002, 0.005408, 0.000010, 0.000552, 0.036206, 0.371331,
            0.036440, -1.312410, 0.003630, -0.001278, 0.003398, -0.023173,
            -0.037727, 0.393993, -0.081278, -1.310931
        ])

        k = [
            "leftHipYaw", "leftHipRoll", "leftHipPitch", "leftKneePitch",
            "leftAnklePitch", "leftAnkleRoll", "rightHipYaw", "rightHipRoll",
            "rightHipPitch", "rightKneePitch", "rightAnklePitch",
            "rightAnkleRoll", "torsoYaw", "torsoPitch", "torsoRoll",
            "leftShoulderPitch", "leftShoulderRoll", "leftShoulderYaw",
            "leftElbowPitch", "leftForearmYaw", "lowerNeckPitch", "neckYaw",
            "upperNeckPitch", "rightShoulderPitch", "rightShoulderRoll",
            "rightShoulderYaw", "rightElbowPitch", "rightForearmYaw"
        ]
        v = np.array([
            -8.17856e-07, 4.30013e-07, -0.599995, 1.19998, -0.599895,
            1.7189e-06, 1.48176e-06, -9.07747e-07, -0.599995, 1.19998,
            -0.599895, 1.11673e-05, 8.14396e-07, 1.70736e-06, -2.99131e-07,
            0.200008, -1.09997, 0.000353999, -0.39999, 1.49888, 3.43861e-06,
            -1.24655e-06, -3.03511e-08, 0.199983, 1.09996, 0.00036993,
            0.399943, 1.49889
        ])[-28:]

        nominal_jpos = OrderedDict()
        for i in range(28):
            nominal_jpos[k[i]] = v[i]
        self._sp.nominal_joint_pos = nominal_jpos

        self._robot.debug_update_system(q, qdot)

        ## Check Kinematics
        # print("com pos\n", self._robot.get_com_pos())
        # print("com vel\n", self._robot.get_com_lin_vel())
        # print("com jac\n", self._robot.get_com_lin_jacobian())
        # print("com jac dot\n", self._robot.get_com_lin_jacobian_dot())
        # print("right foot iso\n", self._robot.get_link_iso('rightCOP_Frame'))
        # print("right foot vel\n", self._robot.get_link_vel('rightCOP_Frame'))
        # print("right foot jac\n",
        # self._robot.get_link_jacobian('rightCOP_Frame'))
        # print("right foot jacdot\n",
        # self._robot.get_link_jacobian_dot('rightCOP_Frame'))
        # print("gravity\n", self._robot.get_gravity())
        # exit()

        command = self._control_architecture.get_command()
        exit()

        return command

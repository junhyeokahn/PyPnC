import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import numpy as np

from pnc.interface import Interface
from config.manipulator_config import ManipulatorConfig
from pnc.wbc.ihwbc.joint_integrator import JointIntegrator
from util import interpolation
from pnc.data_saver import DataSaver


class ManipulatorInterface(Interface):
    def __init__(self):
        super(ManipulatorInterface, self).__init__()

        if ManipulatorConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                cwd + "/robot_model/manipulator/three_link_manipulator.urdf",
                True, ManipulatorConfig.PRINT_ROBOT_INFO)
        elif ManipulatorConfig.DYN_LIB == "pinocchio":
            from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
            self._robot = PinocchioRobotSystem(
                cwd + "/robot_model/manipulator/three_link_manipulator.urdf",
                cwd + "/robot_model/manipulator", True,
                ManipulatorConfig.PRINT_ROBOT_INFO)
        else:
            raise ValueError("wrong dynamics library")
        self._joint_integrator = JointIntegrator(self._robot.n_a,
                                                 ManipulatorConfig.DT)
        self._joint_integrator.pos_cutoff_freq = 0.001  # hz
        self._joint_integrator.vel_cutoff_freq = 0.002  # hz
        self._joint_integrator.max_pos_err = 0.2  # rad
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit
        self._b_first_visit = True

        self._data_saver = DataSaver()

    def get_command(self, sensor_data):
        # Update Robot
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        if self._b_first_visit:
            self._joint_integrator.initialize_states(
                self._robot.joint_velocities, self._robot.joint_positions)
            self._ini_ee_pos = self._robot.get_link_iso('ee')[0:3, 3]
            self._b_first_visit = False

        # Operational Space Control
        jpos_cmd, jvel_cmd, jtrq_cmd = self._compute_osc_command()

        # Compute Cmd
        command = self._robot.create_cmd_ordered_dict(jpos_cmd, jvel_cmd,
                                                      jtrq_cmd)

        # Increase time variables
        self._count += 1
        self._running_time += ManipulatorConfig.DT

        self._data_saver.add('time', self._running_time)
        self._data_saver.advance()

        return command

    def _compute_osc_command(self):
        jtrq = np.zeros(self._robot.n_a)
        jac = self._robot.get_link_jacobian('ee')[3:6, :]
        pos = self._robot.get_link_iso('ee')[0:3, 3]
        vel = self._robot.get_link_vel('ee')[3:6]
        pos_des = np.zeros(3)
        vel_des = np.zeros(3)
        acc_des = np.zeros(3)

        for i in range(3):
            pos_des[i] = interpolation.smooth_changing(
                self._ini_ee_pos[i], ManipulatorConfig.DES_EE_POS[i], 3.,
                self._running_time)
            vel_des[i] = interpolation.smooth_changing_vel(
                self._ini_ee_pos[i], ManipulatorConfig.DES_EE_POS[i], 3.,
                self._running_time)
            acc_des[i] = interpolation.smooth_changing_acc(
                self._ini_ee_pos[i], ManipulatorConfig.DES_EE_POS[i], 3.,
                self._running_time)

        err = pos_des - pos
        err_d = vel_des - vel
        # xddot_des = acc_des + ManipulatorConfig.KP * err + ManipulatorConfig.KD * err_d
        xddot_des = ManipulatorConfig.KP * err + ManipulatorConfig.KD * err_d
        qddot_des = np.dot(np.linalg.pinv(jac, rcond=1e-3), xddot_des)
        # smoothing qddot
        s = interpolation.smooth_changing(0, 1, 0.5, self._running_time)
        qddot_des *= s

        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            qddot_des, self._robot.joint_velocities,
            self._robot.joint_positions)

        mass_matrix = self._robot.get_mass_matrix()
        c = self._robot.get_coriolis()
        g = self._robot.get_gravity()

        jtrq = np.dot(mass_matrix, qddot_des) + c + g

        self._data_saver.add('ee_pos_des', pos_des)
        self._data_saver.add('ee_pos_act', pos)
        self._data_saver.add('ee_vel_des', vel_des)
        self._data_saver.add('ee_vel_act', vel)
        self._data_saver.add('ee_acc_des', acc_des)
        self._data_saver.add('jpos_des', joint_pos_cmd)
        self._data_saver.add('jpos_act', self._robot.joint_positions)
        self._data_saver.add('jvel_des', joint_vel_cmd)
        self._data_saver.add('jvel_act', self._robot.joint_velocities)
        self._data_saver.add('qddot_des', qddot_des)

        return joint_pos_cmd, joint_vel_cmd, jtrq

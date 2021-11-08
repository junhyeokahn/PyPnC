import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import copy

import pybullet as p

from util import util
from config.laikago_config import PnCConfig
from pnc.interface import Interface
from pnc.laikago_pnc.laikago_interrupt_logic import LaikagoInterruptLogic
from pnc.laikago_pnc.laikago_state_provider import LaikagoStateProvider
from pnc.laikago_pnc.laikago_state_estimator import LaikagoStateEstimator
from pnc.laikago_pnc.laikago_control_architecture import LaikagoControlArchitecture
from pnc.data_saver import DataSaver
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem


class LaikagoInterface(Interface):
    def __init__(self):
        super(LaikagoInterface, self).__init__()

        self._robot = PinocchioRobotSystem(
            cwd + "/robot_model/laikago/laikago_toes.urdf",
            cwd + "/robot_model/laikago", False, True)

        self._sp = LaikagoStateProvider(self._robot)
        self._se = LaikagoStateEstimator(self._robot)
        self._control_architecture = LaikagoControlArchitecture(self._robot)
        self._interrupt_logic = LaikagoInterruptLogic(
            self._control_architecture)
        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()

    def get_command(self, sensor_data):
        if PnCConfig.SAVE_DATA:
            self._data_saver.add('time', self._running_time)
            self._data_saver.add('phase', self._control_architecture.state)

        # Update State Estimator
        if self._count == 0:
            print("=" * 80)
            print("Initialize")
            print("=" * 80)
            self._se.initialize(sensor_data)
        self._se.update(sensor_data)

        # print("total mass")
        # print(self._robot.total_mass)
        # print("com pos")
        # print(self._robot.get_com_pos())

        # Process Interrupt Logic
        self._interrupt_logic.process_interrupts()

        # Compute Cmd
        command = self._control_architecture.get_command()

        if PnCConfig.SAVE_DATA and (self._count % PnCConfig.SAVE_FREQ == 0):
            self._data_saver.advance()

        # Increase time variables
        self._count += 1
        self._running_time += PnCConfig.CONTROLLER_DT
        self._sp.curr_time = self._running_time
        self._sp.prev_state = self._control_architecture.prev_state
        self._sp.state = self._control_architecture.state

        return copy.deepcopy(command)

    @property
    def interrupt_logic(self):
        return self._interrupt_logic

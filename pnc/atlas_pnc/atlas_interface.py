import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p

from pnc.interface import Interface
from config.atlas_config import SimConfig, PnCConfig
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider
from pnc.atlas_pnc.atlas_control_architecture import AtlasControlArchitecture


class AtlasInterface(Interface):
    def __init__(self):
        super(AtlasInterface, self).__init__()

        if PnCConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
                ['rootJoint'], ["l_sole_fixed", "r_sole_fixed"])
        else:
            raise ValueError
        self._sp = AtlasStateProvider(self._robot)
        self._control_architecture = AtlasControlArchitecture(self._robot)

    def get_command(self, sensor_data):

        if self._count == 0:
            self._sp.nominal_joint_pos = sensor_data["joint_pos"]

        # Update RobotSystem
        self._robot.update_system(sensor_data["base_pos"],
                                  sensor_data["base_quat"],
                                  sensor_data["base_lin_vel"],
                                  sensor_data["base_ang_vel"],
                                  sensor_data["joint_pos"],
                                  sensor_data["joint_vel"])
        # Compute Cmd
        command = self._control_architecture.get_command()

        # Increase time variables
        self._count += 1
        self._running_time += PnCConfig.CONTROLLER_DT
        self._sp.curr_time = self._running_time
        self._sp.state = self._control_architecture.state

        return command

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p

from config.atlas_config import SimConfig, PnCConfig
from pnc.interface import Interface


class AtlasInterface(Interface):
    def __init__(self):
        if PnCConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf")
        else:
            raise ValueError
        super(AtlasInterface, self).__init__()

    def get_command(self, sensor_data):
        pass

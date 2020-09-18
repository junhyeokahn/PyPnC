import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p

from config.atlas_config import SimConfig
from pnc.interface import Interface
from pnc.robot_system import RobotSystem


class AtlasInterface(Interface):
    def __init__(self):
        self._robot = RobotSystem(0)
        super(AtlasInterface, self).__init__()

    def get_command(self, sensor_data):
        pass

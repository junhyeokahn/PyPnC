import numpy as np

from config.laikago_config import PnCConfig
from util import util
from pnc.laikago_pnc.laikago_state_provider import LaikagoStateProvider


class LaikagoStateEstimator(object):
    def __init__(self, robot):
        super(LaikagoStateEstimator, self).__init__()
        self._robot = robot
        self._sp = LaikagoStateProvider(self._robot)

    def initialize(self, sensor_data):
        self._sp.nominal_joint_pos = sensor_data["joint_pos"]

    def update(self, sensor_data):

        # Update Encoders
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

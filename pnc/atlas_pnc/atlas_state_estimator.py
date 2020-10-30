import numpy as np

from config.atlas_config import PnCConfig
from util import util
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class AtlasStateEstimator(object):
    def __init__(self, robot):
        super(AtlasStateEstimator, self).__init__()
        self._robot = robot
        self._sp = AtlasStateProvider(self._robot)

    def initialize(self, sensor_data):
        self._sp.nominal_joint_pos = sensor_data["joint_pos"]

    def update(Self, sensor_data):

        self._robot.update_system(sensor_data["base_pos"],
                                  sensor_data["base_quat"],
                                  sensor_data["base_lin_vel"],
                                  sensor_data["base_ang_vel"],
                                  sensor_data["joint_pos"],
                                  sensor_data["joint_vel"])

        self._update_dcm()

    def _update_dcm(self):
        com_pos = self.get_com_pos()
        com_vel = self.get_com_vel()
        dcm_omega = np.sqrt(9.81 / com_pos[2])
        self._sp._prev_dcm = np.copy(self._dcm)
        self._sp._dcm = com_pos + com_vel / dcm_omega
        alpha_dcm_vel = 0.1  # TODO : Get this from Hz
        self._sp._dcm_vel = alpha_dcm_vel * (
            self._dcm - self._prev_dcm) / PnCConfig.CONTROLLER_DT
        +(1.0 - alpha_dcm_vel) * self._dcm_vel

import numpy as np


class ReactionForceManager(object):
    def __init__(self, contact, rf_z_max, robot):
        self._contact = contact
        self._rf_z_max = rf_z_max
        self._robot = robot
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_zero(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration

    def update_ramp_to_zero(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        max_normal = -self._rf_z_max / duration * (
            t - self._start_time) + self._rf_z_max
        self._contact.rf_z_max = max_normal

    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        max_normal = self._rf_z_max / duration * (t - self._start_time)
        self._contact.rf_z_max = max_normal

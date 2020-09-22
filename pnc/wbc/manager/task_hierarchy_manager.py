import numpy as np


class TaskHierarchyManager(object):
    def __init__(self, task, w_max, w_min, robot):
        self._task = task
        self._w_max = w_max
        self._w_min = w_min
        self._robot = robot
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration

    def update_ramp_to_zero(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        w = (self._w_min -
             self._w_max) / duration * (t - self._start_time) + self._w_max
        self._task.w_hierarchy = w

    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        w = (self._w_max -
             self._w_min) / duration * (t - self._start_time) + self._w_min
        self._task.w_hierarchy = w

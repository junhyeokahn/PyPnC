from collections import OrderedDict

import numpy as np


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LaikagoStateProvider(metaclass=MetaSingleton):
    def __init__(self, robot):
        self._robot = robot
        self._nominal_joint_pos = OrderedDict()
        self._state = 0
        self._prev_state = 0
        self._curr_time = 0.

    @property
    def nominal_joint_pos(self):
        return self._nominal_joint_pos

    @property
    def state(self):
        return self._state

    @property
    def prev_state(self):
        return self._prev_state

    @prev_state.setter
    def prev_state(self, value):
        self._prev_state = value

    @property
    def curr_time(self):
        return self._curr_time

    @nominal_joint_pos.setter
    def nominal_joint_pos(self, val):
        assert self._robot.n_a == len(val.keys())
        self._nominal_joint_pos = val

    @state.setter
    def state(self, val):
        self._state = val

    @curr_time.setter
    def curr_time(self, val):
        self._curr_time = val

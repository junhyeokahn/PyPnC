import numpy as np


class AtlasStateProvider(object):
    __instance = None

    def __new__(cls, robot):
        if cls.__instance is None:
            cls.__instance = super(AtlasStateProvider, cls).__new__(cls, robot)
            self._robot = robot
            self._nominal_joint_pos = np.zeros(self._robot.na)
            self.curr_time = 0.
        return cls.__instance

    @property
    def nominal_joint_pos(self):
        return self._nominal_joint_pos

    @nominal_joint_pos.setter
    def nominal_joint_pos(self, val):
        assert self._robot.na == val.shape[0]
        self._nominal_joint_pos = val

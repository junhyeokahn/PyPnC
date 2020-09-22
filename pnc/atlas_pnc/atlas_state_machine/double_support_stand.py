import numpy as np

from pnc.atlas_pnc.atlas_control_architecture import AtlasStates
from pnc.state_machine import StateMachine


class DoubleSupportStand(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(DoubleSupportStand, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._end_time = 0.

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, val):
        self._end_time = val

    def first_visit(self):
        pass

    def one_step(self):
        pass

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        return AtlasStates.BALANCE

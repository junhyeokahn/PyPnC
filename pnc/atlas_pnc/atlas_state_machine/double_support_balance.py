import numpy as np

from pnc.state_machine import StateMachine
from pnc.atlas_pnc.atlas_control_architecture import AtlasStates
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class DoubleSupportBalance(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(DoubleSupportBalance, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = AtlasStateProvider()
        self._start_time = 0.
        self._b_state_switch_trigger = False

    @b_state_switch_trigger.setter
    def b_state_switch_trigger(self, val):
        self._b_state_switch_trigger = val

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Foot Task
        self._trajectory_managers["l_foot"].use_current()
        self._trajectory_managers["r_foot"].use_current()

    def first_visit(self):
        self._b_state_switch_trigger = False
        self._start_time = self._sp.curr_time
        pass

    def last_visit(self):
        pass

    def end_of_state(self):
        ## TODO
        return False

    def get_next_state(self):
        ## TODO
        pass

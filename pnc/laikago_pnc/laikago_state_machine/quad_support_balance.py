import numpy as np

from config.laikago_config import PushRecoveryState
from pnc.state_machine import StateMachine
from pnc.planner.locomotion.dcm_planner.footstep import Footstep
from pnc.laikago_pnc.laikago_state_provider import LaikagoStateProvider


class QuadSupportBalance(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(QuadSupportBalance, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = LaikagoStateProvider()
        self._start_time = 0.
        self._b_state_switch_trigger = False

    @property
    def b_state_switch_trigger(self):
        return self._b_state_switch_trigger

    @b_state_switch_trigger.setter
    def b_state_switch_trigger(self, val):
        self._b_state_switch_trigger = val

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Foot Task
        self._trajectory_managers["fl_foot"].use_current()
        self._trajectory_managers["fr_foot"].use_current()
        self._trajectory_managers["rl_foot"].use_current()
        self._trajectory_managers["rr_foot"].use_current()

    def first_visit(self):
        print("[PushRecoveryState] BALANCE")
        self._b_state_switch_trigger = False
        self._start_time = self._sp.curr_time

    def last_visit(self):
        pass

    def end_of_state(self):
        return False

    def get_next_state(self):
        raise NotImplementedError

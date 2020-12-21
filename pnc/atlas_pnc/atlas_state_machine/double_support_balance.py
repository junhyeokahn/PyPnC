import numpy as np

from config.atlas_config import WalkingState
from pnc.state_machine import StateMachine
from pnc.planner.locomotion.dcm_planner.footstep import Footstep
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

    @property
    def b_state_switch_trigger(self):
        return self._b_state_switch_trigger

    @b_state_switch_trigger.setter
    def b_state_switch_trigger(self, val):
        self._b_state_switch_trigger = val

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Foot Task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

    def first_visit(self):
        print("[WalkingState] BALANCE")
        self._b_state_switch_trigger = False
        self._start_time = self._sp.curr_time

    def last_visit(self):
        pass

    def end_of_state(self):
        if (self._b_state_switch_trigger) and (
                len(self._trajectory_managers["dcm"].footstep_list) > 0
        ) and not (self._trajectory_managers["dcm"].no_reaming_steps()):
            return True
        return False

    def get_next_state(self):
        b_valid_step, robot_side = self._trajectory_managers[
            "dcm"].next_step_side()
        if b_valid_step:
            if robot_side == Footstep.LEFT_SIDE:
                return WalkingState.LF_CONTACT_TRANS_START
            elif robot_side == Footstep.RIGHT_SIDE:
                return WalkingState.RF_CONTACT_TRANS_START
            else:
                raise ValueError("Wrong Footstep Side")

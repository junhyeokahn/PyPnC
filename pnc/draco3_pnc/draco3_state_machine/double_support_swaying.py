import numpy as np

from config.draco3_config import WalkingState
from pnc.state_machine import StateMachine
from pnc.planner.locomotion.dcm_planner.footstep import Footstep
from pnc.draco3_pnc.draco3_state_provider import Draco3StateProvider


class DoubleSupportSwaying(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(DoubleSupportSwaying, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = Draco3StateProvider()
        self._start_time = 0.
        self._amp = np.zeros(3)
        self._freq = np.zeros(3)

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value):
        self._amp = value

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        self._freq = value

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Foot Task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

        # Update Floating Base
        self._trajectory_managers[
            'floating_base'].update_floating_base_desired(self._sp._curr_time)

    def first_visit(self):
        print("[WalkingState] SWAYING")
        self._start_time = self._sp.curr_time

        self._trajectory_managers['floating_base'].b_swaying = True
        self._trajectory_managers[
            'floating_base'].initialize_floating_base_swaying_trajectory(
                self._sp.curr_time, self._amp, self._freq)

    def last_visit(self):
        pass

    def end_of_state(self):
        return False

    def get_next_state(self):
        pass

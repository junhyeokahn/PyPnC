import numpy as np

from pnc.state_machine import StateMachine
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class ContactTransitionStart(StateMachine):
    def __init__(self, id, tm, hm, fm, leg_side, robot):
        super(ContactTransitionStart, self).__init__()
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._leg_side = leg_side
        self._sp = AtlasStateProvider()
        self._start_time = 0.

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Task

    def first_visit(self):
        if self._leg_side == Footstep.RIGHT_SIDE:
            print("[WalkingState] RightLeg ContactTransitionStart")
        else:
            print("[WalkingState] LeftLeg ContactTransitionStart")
        self._start_time = self._sp.curr_time

        # Initialize Reaction Force Ramp to Max
        for fm in self._force_managers.values():
            fm.initialize_ramp_to_max(self._sp.curr_time, self._rf_z_max_time)

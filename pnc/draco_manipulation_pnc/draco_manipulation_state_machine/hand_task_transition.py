import numpy as np

from config.draco_manipulation_config import LocomanipulationState
from pnc.state_machine import StateMachine
from pnc.draco_manipulation_pnc.draco_manipulation_state_provider import DracoManipulationStateProvider


class HandTaskTransition(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super().__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = DracoManipulationStateProvider()
        self._start_time = 0.
        self._duration = 0.
        self._rhand_reaching_trigger = False
        self._lhand_reaching_trigger = False

    def first_visit(self):
        print("[LocomanipulationState] HandTaskTransition")

        # Initialize Hierarchy Ramp to Max
        self._hierarchy_managers["upper_body"].initialize_ramp_to_min(
            self._sp.curr_time, self._duration)
        self._hierarchy_managers["neck"].initialize_ramp_to_max(
            self._sp.curr_time, self._duration)
        self._hierarchy_managers["rhand_pos"].initialize_ramp_to_max(
            self._sp.curr_time, self._duration)
        self._hierarchy_managers["lhand_pos"].initialize_ramp_to_max(
            self._sp.curr_time, self._duration)
        self._hierarchy_managers["rhand_ori"].initialize_ramp_to_max(
            self._sp.curr_time, self._duration)
        self._hierarchy_managers["lhand_ori"].initialize_ramp_to_max(
            self._sp.curr_time, self._duration)

        # Set Desired neck and hand endeffector pose
        # self._trajectory_managers["neck"].use_nominal_upper_body_joint_pos()
        self._trajectory_managers["rhand"].update_desired(
            self._robot.get_link_iso("r_hand_contact"))
        self._trajectory_managers["lhand"].update_desired(
            self._robot.get_link_iso("l_hand_contact"))

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update task hieararchy weights
        self._hierarchy_managers["upper_body"].update_ramp_to_min(
            self._sp.curr_time)
        self._hierarchy_managers["neck"].update_ramp_to_max(self._sp.curr_time)
        self._hierarchy_managers["rhand_pos"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["lhand_pos"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["rhand_ori"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["lhand_ori"].update_ramp_to_max(
            self._sp.curr_time)

        # Update foot task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

    def last_visit(self):
        pass

    def end_of_state(self):
        if (self._rhand_reaching_trigger or self._lhand_reaching_trigger):
            return True
        return False

    def get_next_state(self):
        if self._rhand_reaching_trigger:
            return LocomanipulationState.RH_HANDREACH
        if self._lhand_reaching_trigger:
            return LocomanipulationState.LH_HANDREACH

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, val):
        self._duration = val

    @property
    def lhand_reaching_trigger(self):
        return self._lhand_reaching_trigger

    @lhand_reaching_trigger.setter
    def lhand_reaching_trigger(self, value):
        self._lhand_reaching_trigger = value

    @property
    def rhand_reaching_trigger(self):
        return self._rhand_reaching_trigger

    @rhand_reaching_trigger.setter
    def rhand_reaching_trigger(self, value):
        self._rhand_reaching_trigger = value

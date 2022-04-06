from pnc.state_machine import StateMachine
from pnc.draco_manipulation_pnc.draco_manipulation_state_provider import DracoManipulationStateProvider
from config.draco_manipulation_config import LocomanipulationState


class DoubleSupportHandReturn(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super().__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = DracoManipulationStateProvider()
        self._start_time = 0.
        self._trans_duration = 0.

    def first_visit(self):
        self._start_time = self._sp.curr_time

        if self._state_id == LocomanipulationState.LH_HANDRETURN:
            self._hierarchy_managers['lhand_pos'].initialize_ramp_to_min(
                self._start_time, self._trans_duration)
            self._hierarchy_managers['lhand_ori'].initialize_ramp_to_min(
                self._start_time, self._trans_duration)
        elif self._state_id == LocomanipulationState.RH_HANDRETURN:
            self._hierarchy_managers['rhand_pos'].initialize_ramp_to_min(
                self._start_time, self._trans_duration)
            self._hierarchy_managers['rhand_ori'].initialize_ramp_to_min(
                self._start_time, self._trans_duration)
        else:
            raise ValueError("Wrong LocomanipulationState: Handside")

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        if self._state_id == LocomanipulationState.LH_HANDRETURN:
            self._hierarchy_managers['lhand_pos'].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers['lhand_ori'].update_ramp_to_min(
                self._sp.curr_time)
        elif self._state_id == LocomanipulationState.RH_HANDRETURN:
            self._hierarchy_managers['rhand_pos'].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers['rhand_ori'].update_ramp_to_min(
                self._sp.curr_time)
        else:
            raise ValueError("Wrong LocomanipulationState: Handside")

        self._trajectory_managers['lfoot'].use_current()
        self._trajectory_managers['rfoot'].use_current()

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._trans_duration + 0.1:
            return True
        else:
            return False

    def get_next_state(self):
        return LocomanipulationState.BALANCE

    @property
    def trans_duration(self):
        return self._trans_duration

    @trans_duration.setter
    def trans_duration(self, value):
        self._trans_duration = value

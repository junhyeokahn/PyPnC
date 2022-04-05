import numpy as np

from pnc.state_machine import StateMachine
from pnc.draco_manipulation_pnc.draco_manipulation_state_provider import DracoManipulationStateProvider
from config.draco_manipulation_config import ManipulationConfig, LocomanipulationState
from util import util


class DoubleSupportHandReach(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super().__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._sp = DracoManipulationStateProvider()
        self._start_time = 0.
        self._moving_duration = 0.
        self._trans_duration = 0.
        self._rh_target_pos = np.zeros(3)
        self._rh_waypoint_pos = np.zeros(3)
        self._rh_target_quat = np.zeros(4)
        self._lh_target_pos = np.zeros(3)
        self._lh_waypoint_pos = np.zeros(3)
        self._lh_target_quat = np.zeros(4)

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Hierarchy
        if self._state_id == LocomanipulationState.LH_HANDREACH:
            self._hierarchy_managers["lhand_pos"].update_ramp_to_max(
                self._sp.curr_time)
            self._hierarchy_managers["lhand_ori"].update_ramp_to_max(
                self._sp.curr_time)
        if self._state_id == LocomanipulationState.RH_HANDREACH:
            self._hierarchy_managers["rhand_pos"].update_ramp_to_max(
                self._sp.curr_time)
            self._hierarchy_managers["rhand_ori"].update_ramp_to_max(
                self._sp.curr_time)

        # Update Foot Task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

        # Update Hand Task
        if self._state_id == LocomanipulationState.RH_HANDREACH:
            # self._trajectory_managers['rhand'].update_hand_trajectory(
            # self._sp.curr_time)
            self._trajectory_managers['rhand'].update_keypoint_hand_trajectory(
                self._sp.curr_time)
        elif self._state_id == LocomanipulationState.LH_HANDREACH:
            # self._trajectory_managers['lhand'].update_hand_trajectory(
            # self._sp.curr_time)
            self._trajectory_managers['lhand'].update_keypoint_hand_trajectory(
                self._sp.curr_time)
        else:
            raise ValueError("Wrong LocomanipulationState: HandSide")

    def first_visit(self):
        self._start_time = self._sp.curr_time

        if self._state_id == LocomanipulationState.RH_HANDREACH:
            # print("[LocomanipulationState] Right Hand Reaching")
            target_hand_iso = np.eye(4)
            target_hand_iso[0:3, 0:3] = util.quat_to_rot(self._rh_target_quat)
            target_hand_iso[0:3, 3] = self._rh_target_pos

            # self._trajectory_managers['rhand'].initialize_hand_trajectory(
            # self._start_time, self._moving_duration, target_hand_iso)

            self._trajectory_managers[
                'rhand'].initialize_keypoint_hand_trajectory(
                    self._start_time, self._moving_duration,
                    self._rh_waypoint_pos, target_hand_iso)

            self._hierarchy_managers["rhand_pos"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["rhand_ori"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
        elif self._state_id == LocomanipulationState.LH_HANDREACH:
            # print("[LocomanipulationState] Left Hand Reaching")
            target_hand_iso = np.eye(4)
            target_hand_iso[0:3, 0:3] = util.quat_to_rot(self._lh_target_quat)
            target_hand_iso[0:3, 3] = self._lh_target_pos

            # self._trajectory_managers['lhand'].initialize_hand_trajectory(
            # self._start_time, self._moving_duration, target_hand_iso)

            self._trajectory_managers[
                'lhand'].initialize_keypoint_hand_trajectory(
                    self._start_time, self._moving_duration,
                    self._lh_waypoint_pos, target_hand_iso)

            self._hierarchy_managers["lhand_pos"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["lhand_ori"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
        else:
            raise ValueError("Wrong LocomanipulationState: HandSide")

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._moving_duration + 0.1:
            return True
        else:
            return False

    def get_next_state(self):
        return LocomanipulationState.BALANCE

    @property
    def moving_duration(self):
        return self._moving_duration

    @moving_duration.setter
    def moving_duration(self, value):
        self._moving_duration = value

    @property
    def trans_duration(self):
        return self._trans_duration

    @trans_duration.setter
    def trans_duration(self, value):
        self._trans_duration = value

    @property
    def rh_target_pos(self):
        return self._rh_target_pos

    @rh_target_pos.setter
    def rh_target_pos(self, value):
        self._rh_target_pos = value

    @property
    def rh_target_quat(self):
        return self._rh_target_quat

    @rh_target_quat.setter
    def rh_target_quat(self, value):
        self._rh_target_quat = value

    @property
    def lh_target_pos(self):
        return self._lh_target_pos

    @lh_target_pos.setter
    def lh_target_pos(self, value):
        self._lh_target_pos = value

    @property
    def lh_target_quat(self):
        return self._lh_target_quat

    @lh_target_quat.setter
    def lh_target_quat(self, value):
        self._lh_target_quat = value

    @property
    def rh_waypoint_pos(self):
        return self._rh_waypoint_pos

    @rh_waypoint_pos.setter
    def rh_waypoint_pos(self, value):
        self._rh_waypoint_pos = value

    @property
    def lh_waypoint_pos(self):
        return self._lh_waypoint_pos

    @lh_waypoint_pos.setter
    def lh_waypoint_pos(self, value):
        self._lh_waypoint_pos = value

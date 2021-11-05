import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from config.laikago_config import PushRecoveryState
from pnc.state_machine import StateMachine
from pnc.laikago_pnc.laikago_state_provider import LaikagoStateProvider


class QuadSupportStand(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(QuadSupportStand, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._end_time = 0.
        self._rf_z_max_time = 0.
        self._start_time = 0.
        self._com_pos_des = np.zeros(3)
        self._base_quat_des = np.array([0., 0., 0., 1.])
        self._sp = LaikagoStateProvider()

    @property
    def end_time(self):
        return self._end_time

    @property
    def rf_z_max_time(self):
        return self.rf_z_max_time

    @property
    def com_pos_des(self):
        return self._com_pos_des

    @property
    def base_quat_des(self):
        return self._base_quat_des

    @end_time.setter
    def end_time(self, val):
        self._end_time = val

    @rf_z_max_time.setter
    def rf_z_max_time(self, val):
        self._rf_z_max_time = val

    @com_pos_des.setter
    def com_pos_des(self, val):
        self._com_pos_des = val

    @base_quat_des.setter
    def base_quat_des(self, val):
        self._base_quat_des = val

    def first_visit(self):
        print("[PushRecoveryState] STAND")
        self._start_time = self._sp.curr_time

        # Initialize CoM Trajectory
        self._trajectory_managers[
            "floating_base"].initialize_floating_base_interpolation_trajectory(
                self._sp.curr_time, self._end_time, self._com_pos_des,
                self._base_quat_des)

        # Initialize Reaction Force Ramp to Max
        for fm in self._force_managers.values():
            fm.initialize_ramp_to_max(self._sp.curr_time, self._rf_z_max_time)

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Floating Base Task
        self._trajectory_managers[
            "floating_base"].update_floating_base_desired(self._sp.curr_time)
        # Update Foot Task
        self._trajectory_managers["fl_foot"].use_current()
        self._trajectory_managers["fr_foot"].use_current()
        self._trajectory_managers["rl_foot"].use_current()
        self._trajectory_managers["rr_foot"].use_current()

        # Update Max Normal Reaction Force
        for fm in self._force_managers.values():
            fm.update_ramp_to_max(self._sp.curr_time)

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        return PushRecoveryState.BALANCE

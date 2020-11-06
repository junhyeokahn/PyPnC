import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from config.atlas_config import WalkingState
from pnc.state_machine import StateMachine
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class DoubleSupportStand(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super(DoubleSupportStand, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._end_time = 0.
        self._rf_z_max_time = 0.
        self._com_height_des = 0.
        self._start_time = 0.
        self._sp = AtlasStateProvider()

    @property
    def end_time(self):
        return self._end_time

    @property
    def rf_z_max_time(self):
        return self.rf_z_max_time

    @property
    def com_height_des(self):
        return self.com_height_des

    @end_time.setter
    def end_time(self, val):
        self._end_time = val

    @rf_z_max_time.setter
    def rf_z_max_time(self, val):
        self._rf_z_max_time = val

    @com_height_des.setter
    def com_height_des(self, val):
        self._com_height_des = val

    def first_visit(self):
        print("[WalkingState] STAND")
        self._start_time = self._sp.curr_time

        # Initialize CoM Trajectory
        lfoot_iso = self._robot.get_link_iso("l_sole")
        rfoot_iso = self._robot.get_link_iso("r_sole")
        com_pos_des = (lfoot_iso[0:3, 3] + rfoot_iso[0:3, 3]) / 2.0
        com_pos_des[2] = self._com_height_des
        base_quat_slerp = Slerp(
            [0, 1], R.from_matrix([lfoot_iso[0:3, 0:3], rfoot_iso[0:3, 0:3]]))
        base_quat_des = base_quat_slerp(0.5).as_quat()
        self._trajectory_managers[
            "floating_base"].initialize_floating_base_trajectory(
                self._sp.curr_time, self._end_time, com_pos_des, base_quat_des)

        # Initialize Reaction Force Ramp to Max
        for fm in self._force_managers.values():
            fm.initialize_ramp_to_max(self._sp.curr_time, self._rf_z_max_time)

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Floating Base Task
        self._trajectory_managers[
            "floating_base"].update_floating_base_desired(self._sp.curr_time)
        # Update Foot Task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

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
        return WalkingState.BALANCE

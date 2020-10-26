import numpy as np

from pnc.planner.locomotion.footstep import Footstep
from pnc.planner.locomotion.footstep import interpolate
from util import util


class DCMTransferType(object):
    INI = 0
    MID = 1


class DCMTrajectoryManager(object):
    def __init__(self, dcm_planner, com_task, base_ori_task, robot, lfoot_id,
                 rfoot_id):
        self._dcm_planner = dcm_planner
        self._com_task = com_task
        self._base_ori_task = base_ori_task
        self._robot = robot
        self._lfoot_id = lfoot_id
        self._rfoot_id = rfoot_id

        self._des_com_pos = np.zeros(3)
        self._des_com_vel = np.zeros(3)
        self._des_com_acc = np.zeros(3)

        self._des_quat = np.array([0., 0., 0., 1.])
        self._des_ang_vel = np.zeros(3)
        self._des_ang_acc = np.zeros(3)

        self._reset_step_idx()

        self._robot_side = Footstep.RIGHT_SIDE

        self._footstep_list = []
        self._footstep_preview_list = []

        self._lf_stance = Footstep()
        self._rf_stance = Footstep()
        self._mf_stance = Footstep()

        # Attributes
        self._nominal_com_height = 1.015
        self._t_additional_init_transfer = 0.
        self._t_contact_transition = 0.45
        self._t_swing = 1.
        self._percentage_settle = 0.99
        self._alpha_ds = 0.5
        self._nominal_footwidth = 0.27
        self._nominal_forward_step = 0.25
        self._nominal_backward_step = -0.25
        self._nominal_turn_randians = np.pi / 4.
        self._nominal_strafe_distance = 0.125

        self._set_temporal_params()

    def compute_ini_contact_transfer_time(self):
        return self._t_additional_init_transfer + self._t_ds + (
            1 - self._alpha_ds) * self._t_ds

    def compute_mid_step_contact_transfer_time(self):
        return self._t_ds

    def compute_final_contact_transfer_time(self):
        return self._t_ds + self._dcm_planner.compute_settling_time()

    def compute_swing_time(self):
        return self._t_ss

    def compute_rf_z_ramp_up_time(self):
        return self._alpha_ds * self._t_ds

    def compute_rf_z_ramp_down_time(self):
        return (1.0 - self._alpha_ds) * self._t_ds

    def _update_starting_stance(self):
        self._lf_stance.iso = self._robot.get_link_iso(self._lfoot_id)
        self._rf_stance.iso = self._robot.get_link_iso(self._rfoot_id)
        self._mf_stance.iso = interpolate(self._lf_stance, self._rf_stance,
                                          0.5)

    # TODO : Put initial angular velocity
    def initialize(self, t_start, transfer_type, quat_start, dcm_pos_start,
                   dcm_vel_start):
        self._update_starting_stance()
        self._update_footstep_preview()

        self._dcm_planner.robot_mass = self._robot.total_mass
        self._dcm_planner.z_vrp = self._nominal_com_height
        self._dcm_planner.t_start = t_start
        self._dcm_planner.ini_quat = quat_start

        if transfer_type == DCMTransferType.INI:
            self._dcm_planner.t_transfer = self._t_transfer_ini
        elif transfer_type == DCMTransferType.MID:
            self._dcm_planner.t_transfer = self._t_transfer_mid
        else:
            raise ValueError, "Wrong DCMTransferType"

        self._dcm_planner.initialize(self._footstep_preview_list,
                                     self._lf_stance, self._rf_stance,
                                     dcm_pos_start, dcm_vel_start)

    def update_floating_base_task_desired(self, curr_time):
        self._des_com_pos = self._dcm_planner.compute_reference_com_pos(
            curr_time)
        self._des_com_vel = self._dcm_planner.compute_reference_com_vel(
            curr_time)
        self._des_com_acc = np.zeros(3)  # TODO : Compute com_acc
        self._des_quat, self._des_ang_vel, self._des_ang_acc = self._dcm_planner.compute_reference_base_ori(
            curr_time)

        self._com_task.update_desired(self._des_com_pos, self._des_com_vel,
                                      self._des_com_acc)
        self._base_ori_task.update_desired(self._des_quat, self._des_ang_vel,
                                           self_des_ang_acc)

    def next_step_side(self):
        if len(self._footstep_list) > 0 and self._curr_footstep_idx < len(
                self._footstep_list):
            return True, self._footstep_list[self._curr_footstep_idx].side
        else:
            return False, None

    def no_reaming_steps(self):
        if self._curr_footstep_idx >= len(self._footstep_list):
            return True
        else:
            return False

    def walk_in_place(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_step_in_place(3, self._robot_side)
        self._alternate_leg()

    def walk_forward(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_walk_forward(3, self._nominal_forward_step)
        self._alternate_leg()

    def walk_backward(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_walk_forward(3, self._nominal_backward_step)
        self._alternate_leg()

    def strafe_left(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_strafe(2, self._nominal_strafe_distance)

    def strafe_right(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_strafe(2, -self._nominal_strafe_distance)

    def turn_left(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_turn(2, self._nominal_turn_randians)

    def turn_right(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_turn(2, -self._nominal_turn_randians)

    def _populate_step_in_place(self, num_step, robot_side_first):
        self._update_starting_stance()

        lf_stance = copy.deepcopy(self._lf_stance)
        rf_stance = copy.deepcopy(self._rf_stance)
        mf_stance = copy.deepcopy(self._mf_stance)
        robot_side = robot_side_first
        for i in range(num_step):
            if robot_side == Footstep.LEFT_SIDE:
                lf_stance.pos = mf_stance.pos + np.dot(
                    mf_stance.ori,
                    np.array([0., self._nominal_footwidth / 2., 0.]))
                lf_stance.ori = np.copy(mf_stance.ori)
                self._footstep_list.append(lf_stance)
                robot_side = Footstep.RIGHT_SIDE

            else:
                rf_stance.pos = mf_stance.pos + np.dot(
                    mf_stance.ori,
                    np.array([0., -self._nominal_footwidth / 2., 0.]))
                rf_stance_ori = np.copy(mf_stance.ori)
                self._footstep_list.append(rf_stance)
                robot_side = Footstep.LEFT_SIDE

    def _alternate_leg(self):
        if self._robot_side == Footstep.LEFT_SIDE:
            self._robot_side = Footstep.RIGHT_SIDE
        else:
            self._robot_side = Footstep.LEFT_SIDE

    def _reset_idx_and_clear_footstep_list(self):
        self._reset_step_idx()
        self._footstep_list = []

    def _update_footstep_preview(self, max_footsteps_to_preview=40):
        self._footstep_preview_list = []
        for i in range(max_footsteps_to_preview):
            if (i + self._current_footstep_idx) < len(self._footstep_list):
                self._footstep_preview_list.append(
                    self._footstep_list[i + self._curr_footstep_idx])
            else:
                break

    def _reset_step_idx(self):
        self._curr_footstep_idx = 0

    def _increment_step_idx(self):
        self._curr_footstep_idx += 1

    def _set_temporal_params(self):
        self._t_ds = self._t_contact_transition
        self._t_ss = self._t_swing
        self._t_transfer_ini = self._t_additional_init_transfer
        self._t_transfer_mid = (self._alpha_ds - 1.) * self._t_ds

        self._dcm_planner.t_transfer = self._t_transfer_ini
        self._dcm_planner.t_ds = self._t_ds
        self._dcm_planner.t_ss = self._t_ss
        self._dcm_planner.percentage_settle = self._percentage_settle
        self._dcm_planner.alpha_ds = self._alpha_ds

    @property
    def nominal_com_height(self):
        return self._nominal_com_height

    @nominal_com_height.setter
    def nominal_com_height(self, value):
        self._nominal_com_height = value

    @property
    def t_additional_init_transfer(self):
        return self._t_additional_init_transfer

    @t_additional_init_transfer.setter
    def t_additional_init_transfer(self, value):
        self._t_additional_init_transfer = value
        self._t_transfer_ini = self._t_additional_init_transfer
        self._dcm_planner.t_transfer = self._t_transfer_ini

    @property
    def t_contact_transition(self):
        return self._t_contact_transition

    @t_contact_transition.setter
    def t_contact_transition(self, value):
        self._t_contact_transition = value
        self._t_ds = self._t_contact_transition
        self._t_transfer_mid = (self._alpha_ds - 1.) * self._t_ds
        self._dcm_planner.t_ds = self._t_ds

    @property
    def t_swing(self):
        return self._t_swing

    @t_swing.setter
    def t_swing(self, value):
        self._t_swing = value
        self._t_ss = self._t_swing
        self._dcm_planner.t_ss = self._t_ss

    @property
    def percentage_settle(self):
        return self._percentage_settle

    @percentage_settle.setter
    def percentage_settle(self, value):
        self._percentage_settle = value
        self._dcm_planner.percentage_settle = self._percentage_settle

    @property
    def alpha_ds(self):
        return self._alpha_ds

    @alpha_ds.setter
    def alpha_ds(self, value):
        self._alpha_ds = value
        self._t_transfer_mid = (self._alpha_ds - 1.) * self._t_ds
        self._dcm_planner.alpha_ds = self._alpha_ds

    @property
    def nominal_footwidth(self):
        return self._nominal_footwidth

    @nominal_footwidth.setter
    def nominal_footwidth(self, value):
        self._nominal_footwidth = value

    @property
    def nominal_forward_step(self):
        return self._nominal_forward_step

    @nominal_forward_step.setter
    def nominal_forward_step(self, value):
        self._nominal_forward_step = value

    @property
    def nominal_backward_step(self):
        return self._nominal_backward_step

    @nominal_backward_step.setter
    def nominal_backward_step(self, value):
        self._nominal_backward_step = value

    @property
    def nominal_turn_radians(self):
        return self._nominal_turn_radians

    @nominal_turn_radians.setter
    def nominal_turn_radians(self, value):
        self._nominal_turn_radians = value

    @property
    def nominal_strafe_distance(self):
        return self._nominal_strafe_distance

    @nominal_strafe_distance.setter
    def nominal_strafe_distance(self, value):
        self._nominal_strafe_distance = value

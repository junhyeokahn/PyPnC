import numpy as np

from config.draco_manipulation_config import WalkingConfig, WBCConfig, LocomanipulationState, ManipulationConfig
from pnc.control_architecture import ControlArchitecture
from pnc.planner.locomotion.dcm_planner.dcm_planner import DCMPlanner
from pnc.planner.locomotion.dcm_planner.footstep import Footstep
from pnc.wbc.manager.dcm_trajectory_manager import DCMTrajectoryManager
from pnc.wbc.manager.task_hierarchy_manager import TaskHierarchyManager
from pnc.wbc.manager.floating_base_trajectory_manager import FloatingBaseTrajectoryManager
from pnc.wbc.manager.foot_trajectory_manager import FootTrajectoryManager
from pnc.wbc.manager.reaction_force_manager import ReactionForceManager
from pnc.wbc.manager.upper_body_trajectory_manager import UpperBodyTrajectoryManager
from pnc.wbc.manager.hand_trajectory_manager import HandTrajectoryManager
from pnc.draco_manipulation_pnc.draco_manipulation_tci_container import DracoManipulationTCIContainer
from pnc.draco_manipulation_pnc.draco_manipulation_controller import DracoManipulationController
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.double_support_stand import DoubleSupportStand
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.double_support_balance import DoubleSupportBalance
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.contact_transition_start import ContactTransitionStart
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.contact_transition_end import ContactTransitionEnd
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.single_support_swing import SingleSupportSwing
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.double_support_hand_reaching import DoubleSupportHandReach
from pnc.draco_manipulation_pnc.draco_manipulation_state_machine.double_support_hand_return import DoubleSupportHandReturn
from pnc.draco_manipulation_pnc.draco_manipulation_state_provider import DracoManipulationStateProvider


class DracoManipulationControlArchitecture(ControlArchitecture):
    def __init__(self, robot):
        super(DracoManipulationControlArchitecture, self).__init__(robot)

        # ======================================================================
        # Initialize TCIContainer
        # ======================================================================
        self._tci_container = DracoManipulationTCIContainer(robot)

        # ======================================================================
        # Initialize Controller
        # ======================================================================
        self._draco_manipulation_controller = DracoManipulationController(
            self._tci_container, robot)

        # ======================================================================
        # Initialize Planner
        # ======================================================================
        self._dcm_planner = DCMPlanner()

        # ======================================================================
        # Initialize Task Manager
        # ======================================================================
        self._rfoot_tm = FootTrajectoryManager(
            self._tci_container.rfoot_pos_task,
            self._tci_container.rfoot_ori_task, robot)
        self._rfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT

        self._lfoot_tm = FootTrajectoryManager(
            self._tci_container.lfoot_pos_task,
            self._tci_container.lfoot_ori_task, robot)
        self._lfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT

        self._upper_body_tm = UpperBodyTrajectoryManager(
            self._tci_container.upper_body_task, robot)

        self._lhand_tm = HandTrajectoryManager(
            self._tci_container.lhand_pos_task,
            self._tci_container.lhand_ori_task, robot)

        self._rhand_tm = HandTrajectoryManager(
            self._tci_container.rhand_pos_task,
            self._tci_container.rhand_ori_task, robot)

        self._floating_base_tm = FloatingBaseTrajectoryManager(
            self._tci_container.com_task, self._tci_container.torso_ori_task,
            robot)

        self._dcm_tm = DCMTrajectoryManager(self._dcm_planner,
                                            self._tci_container.com_task,
                                            self._tci_container.torso_ori_task,
                                            self._robot, "l_foot_contact",
                                            "r_foot_contact")
        self._dcm_tm.nominal_com_height = WalkingConfig.COM_HEIGHT
        self._dcm_tm.t_additional_init_transfer = WalkingConfig.T_ADDITIONAL_INI_TRANS
        self._dcm_tm.t_contact_transition = WalkingConfig.T_CONTACT_TRANS
        self._dcm_tm.t_swing = WalkingConfig.T_SWING
        self._dcm_tm.percentage_settle = WalkingConfig.PERCENTAGE_SETTLE
        self._dcm_tm.alpha_ds = WalkingConfig.ALPHA_DS
        self._dcm_tm.nominal_footwidth = WalkingConfig.NOMINAL_FOOTWIDTH
        self._dcm_tm.nominal_forward_step = WalkingConfig.NOMINAL_FORWARD_STEP
        self._dcm_tm.nominal_backward_step = WalkingConfig.NOMINAL_BACKWARD_STEP
        self._dcm_tm.nominal_turn_radians = WalkingConfig.NOMINAL_TURN_RADIANS
        self._dcm_tm.nominal_strafe_distance = WalkingConfig.NOMINAL_STRAFE_DISTANCE

        self._trajectory_managers = {
            "rfoot": self._rfoot_tm,
            "lfoot": self._lfoot_tm,
            "upper_body": self._upper_body_tm,
            "lhand": self._lhand_tm,
            "rhand": self._rhand_tm,
            "floating_base": self._floating_base_tm,
            "dcm": self._dcm_tm
        }

        # ======================================================================
        # Initialize Hierarchy Manager
        # ======================================================================
        self._rfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.rfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)

        self._lfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.lfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)

        self._rfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.rfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)

        self._lfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.lfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)

        self._lhand_pos_hm = TaskHierarchyManager(
            self._tci_container.lhand_pos_task, WBCConfig.W_HAND_POS_MAX,
            WBCConfig.W_HAND_POS_MIN)

        self._lhand_ori_hm = TaskHierarchyManager(
            self._tci_container.lhand_ori_task, WBCConfig.W_HAND_ORI_MAX,
            WBCConfig.W_HAND_ORI_MIN)

        self._rhand_pos_hm = TaskHierarchyManager(
            self._tci_container.rhand_pos_task, WBCConfig.W_HAND_POS_MAX,
            WBCConfig.W_HAND_POS_MIN)

        self._rhand_ori_hm = TaskHierarchyManager(
            self._tci_container.rhand_ori_task, WBCConfig.W_HAND_ORI_MAX,
            WBCConfig.W_HAND_ORI_MIN)

        self._hierarchy_managers = {
            "rfoot_pos": self._rfoot_pos_hm,
            "lfoot_pos": self._lfoot_pos_hm,
            "rfoot_ori": self._rfoot_ori_hm,
            "lfoot_ori": self._lfoot_ori_hm,
            "lhand_pos": self._lhand_pos_hm,
            "lhand_ori": self._lhand_ori_hm,
            "rhand_pos": self._rhand_pos_hm,
            "rhand_ori": self._rhand_ori_hm
        }

        # ======================================================================
        # Initialize Reaction Force Manager
        # ======================================================================
        self._rfoot_fm = ReactionForceManager(
            self._tci_container.rfoot_contact, WBCConfig.RF_Z_MAX)

        self._lfoot_fm = ReactionForceManager(
            self._tci_container.lfoot_contact, WBCConfig.RF_Z_MAX)

        self._reaction_force_managers = {
            "rfoot": self._rfoot_fm,
            "lfoot": self._lfoot_fm
        }

        # ======================================================================
        # Initialize State Machines
        # ======================================================================
        self._state_machine[LocomanipulationState.STAND] = DoubleSupportStand(
            LocomanipulationState.STAND, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, robot)
        self._state_machine[LocomanipulationState.
                            STAND].end_time = WalkingConfig.INIT_STAND_DUR
        self._state_machine[LocomanipulationState.
                            STAND].rf_z_max_time = WalkingConfig.RF_Z_MAX_TIME
        self._state_machine[LocomanipulationState.
                            STAND].com_height_des = WalkingConfig.COM_HEIGHT

        self._state_machine[
            LocomanipulationState.BALANCE] = DoubleSupportBalance(
                LocomanipulationState.BALANCE, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers, robot)

        self._state_machine[LocomanipulationState.
                            LF_CONTACT_TRANS_START] = ContactTransitionStart(
                                LocomanipulationState.LF_CONTACT_TRANS_START,
                                self._trajectory_managers,
                                self._hierarchy_managers,
                                self._reaction_force_managers,
                                Footstep.LEFT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.LF_CONTACT_TRANS_END] = ContactTransitionEnd(
                LocomanipulationState.LF_CONTACT_TRANS_END,
                self._trajectory_managers, self._hierarchy_managers,
                self._reaction_force_managers, Footstep.LEFT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.LF_SWING] = SingleSupportSwing(
                LocomanipulationState.LF_SWING, self._trajectory_managers,
                Footstep.LEFT_SIDE, self._robot)

        self._state_machine[LocomanipulationState.
                            RF_CONTACT_TRANS_START] = ContactTransitionStart(
                                LocomanipulationState.RF_CONTACT_TRANS_START,
                                self._trajectory_managers,
                                self._hierarchy_managers,
                                self._reaction_force_managers,
                                Footstep.RIGHT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.RF_CONTACT_TRANS_END] = ContactTransitionEnd(
                LocomanipulationState.RF_CONTACT_TRANS_END,
                self._trajectory_managers, self._hierarchy_managers,
                self._reaction_force_managers, Footstep.RIGHT_SIDE,
                self._robot)

        self._state_machine[
            LocomanipulationState.RF_SWING] = SingleSupportSwing(
                LocomanipulationState.RF_SWING, self._trajectory_managers,
                Footstep.RIGHT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.RH_HANDREACH] = DoubleSupportHandReach(
                LocomanipulationState.RH_HANDREACH, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            LocomanipulationState.
            RH_HANDREACH].moving_duration = ManipulationConfig.T_REACHING_DURATION
        self._state_machine[
            LocomanipulationState.
            RH_HANDREACH].rh_target_pos = ManipulationConfig.RH_TARGET_POS
        self._state_machine[
            LocomanipulationState.
            RH_HANDREACH].rh_target_quat = ManipulationConfig.RH_TARGET_QUAT
        self._state_machine[
            LocomanipulationState.
            RH_HANDREACH].trans_duration = ManipulationConfig.T_REACHING_TRANS_DURATION

        self._state_machine[
            LocomanipulationState.LH_HANDREACH] = DoubleSupportHandReach(
                LocomanipulationState.LH_HANDREACH, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            LocomanipulationState.
            LH_HANDREACH].moving_duration = ManipulationConfig.T_REACHING_DURATION
        self._state_machine[
            LocomanipulationState.
            LH_HANDREACH].lh_target_pos = ManipulationConfig.LH_TARGET_POS
        self._state_machine[
            LocomanipulationState.
            LH_HANDREACH].lh_target_quat = ManipulationConfig.LH_TARGET_QUAT
        self._state_machine[
            LocomanipulationState.
            LH_HANDREACH].trans_duration = ManipulationConfig.T_REACHING_TRANS_DURATION

        self._state_machine[
            LocomanipulationState.RH_HANDRETURN] = DoubleSupportHandReturn(
                LocomanipulationState.RH_HANDRETURN, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            LocomanipulationState.
            RH_HANDRETURN].trans_duration = ManipulationConfig.T_RETURNING_TRANS_DURATION

        self._state_machine[
            LocomanipulationState.LH_HANDRETURN] = DoubleSupportHandReturn(
                LocomanipulationState.LH_HANDRETURN, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            LocomanipulationState.
            LH_HANDRETURN].trans_duration = ManipulationConfig.T_RETURNING_TRANS_DURATION

        # Set Starting State
        self._state = LocomanipulationState.STAND
        self._prev_state = LocomanipulationState.STAND
        self._b_state_first_visit = True

        self._sp = DracoManipulationStateProvider()

    def get_command(self):
        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False

        # Update State Machine
        self._state_machine[self._state].one_step()

        # Update State Machine Independent Trajectories
        self._upper_body_tm.use_nominal_upper_body_joint_pos(
            self._sp.nominal_joint_pos)

        # Get Whole Body Control Commands
        command = self._draco_manipulation_controller.get_command()

        if self._state_machine[self._state].end_of_state():
            self._state_machine[self._state].last_visit()
            self._prev_state = self._state
            self._state = self._state_machine[self._state].get_next_state()
            self._b_state_first_visit = True

        return command

    @property
    def dcm_tm(self):
        return self._dcm_tm

    @property
    def state_machine(self):
        return self._state_machine

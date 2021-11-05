import numpy as np

from config.laikago_config import PushRecoveryConfig, WBCConfig, PushRecoveryState
from pnc.control_architecture import ControlArchitecture
from pnc.wbc.manager.task_hierarchy_manager import TaskHierarchyManager
from pnc.wbc.manager.floating_base_trajectory_manager import FloatingBaseTrajectoryManager
from pnc.wbc.manager.point_foot_trajectory_manager import PointFootTrajectoryManager
from pnc.wbc.manager.reaction_force_manager import ReactionForceManager
from pnc.laikago_pnc.laikago_tci_container import LaikagoTCIContainer
from pnc.laikago_pnc.laikago_controller import LaikagoController
from pnc.laikago_pnc.laikago_state_machine.quad_support_stand import QuadSupportStand
from pnc.laikago_pnc.laikago_state_machine.quad_support_balance import QuadSupportBalance
from pnc.laikago_pnc.laikago_state_provider import LaikagoStateProvider


class LaikagoControlArchitecture(ControlArchitecture):
    def __init__(self, robot):
        super(LaikagoControlArchitecture, self).__init__(robot)

        # Initialize Task Force Container
        self._tci_container = LaikagoTCIContainer(robot)

        # Initialize Controller
        self._laikago_controller = LaikagoController(self._tci_container,
                                                     robot)

        # Initialize Task Manager
        self._floating_base_tm = FloatingBaseTrajectoryManager(
            self._tci_container.com_task, self._tci_container.base_ori_task,
            robot)
        self._fl_foot_tm = PointFootTrajectoryManager(
            self._tci_container.fl_foot_task, robot)
        self._fr_foot_tm = PointFootTrajectoryManager(
            self._tci_container.fr_foot_task, robot)
        self._rr_foot_tm = PointFootTrajectoryManager(
            self._tci_container.rr_foot_task, robot)
        self._rl_foot_tm = PointFootTrajectoryManager(
            self._tci_container.rl_foot_task, robot)

        self._trajectory_managers = {
            "floating_base": self._floating_base_tm,
            "fl_foot": self._fl_foot_tm,
            "fr_foot": self._fr_foot_tm,
            "rr_foot": self._rr_foot_tm,
            "rl_foot": self._rl_foot_tm
        }

        # Initialize Hierarchy Manager
        self._fl_foot_hm = TaskHierarchyManager(
            self._tci_container.fl_foot_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._fr_foot_hm = TaskHierarchyManager(
            self._tci_container.fr_foot_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._rr_foot_hm = TaskHierarchyManager(
            self._tci_container.rr_foot_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._rl_foot_hm = TaskHierarchyManager(
            self._tci_container.rl_foot_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._hierarchy_managers = {
            "fl_foot": self._fl_foot_hm,
            "fr_foot": self._fr_foot_hm,
            "rr_foot": self._rr_foot_hm,
            "rl_foot": self._rl_foot_hm
        }

        # Initialize Reaction Force Manager
        self._fl_foot_fm = ReactionForceManager(
            self._tci_container.fl_foot_contact, WBCConfig.RF_Z_MAX)
        self._fr_foot_fm = ReactionForceManager(
            self._tci_container.fr_foot_contact, WBCConfig.RF_Z_MAX)
        self._rr_foot_fm = ReactionForceManager(
            self._tci_container.rr_foot_contact, WBCConfig.RF_Z_MAX)
        self._rl_foot_fm = ReactionForceManager(
            self._tci_container.rl_foot_contact, WBCConfig.RF_Z_MAX)
        self._reaction_force_managers = {
            "fl_foot": self._fl_foot_fm,
            "fr_foot": self._fr_foot_fm,
            "rr_foot": self._rr_foot_fm,
            "rl_foot": self._rl_foot_fm,
        }

        # Initialize State Machines
        self._state_machine[PushRecoveryState.STAND] = QuadSupportStand(
            PushRecoveryState.STAND, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, robot)
        self._state_machine[PushRecoveryState.
                            STAND].end_time = PushRecoveryConfig.INIT_STAND_DUR
        self._state_machine[
            PushRecoveryState.
            STAND].rf_z_max_time = PushRecoveryConfig.RF_Z_MAX_TIME
        self._state_machine[
            PushRecoveryState.
            STAND].com_pos_des = PushRecoveryConfig.INITIAL_COM_POS
        self._state_machine[
            PushRecoveryState.
            STAND].base_quat_des = PushRecoveryConfig.INITIAL_BASE_ORI

        self._state_machine[PushRecoveryState.BALANCE] = QuadSupportBalance(
            PushRecoveryState.BALANCE, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, robot)

        # Set Starting State
        self._state = PushRecoveryState.STAND
        self._prev_state = PushRecoveryState.STAND
        self._b_state_first_visit = True

        self._sp = LaikagoStateProvider()

    def get_command(self):
        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False

        # Update State Machine
        self._state_machine[self._state].one_step()
        # Get Whole Body Control Commands
        command = self._laikago_controller.get_command()

        if self._state_machine[self._state].end_of_state():
            self._state_machine[self._state].last_visit()
            self._prev_state = self._state
            self._state = self._state_machine[self._state].get_next_state()
            self._b_state_first_visit = True

        return command

    @property
    def state_machine(self):
        return self._state_machine

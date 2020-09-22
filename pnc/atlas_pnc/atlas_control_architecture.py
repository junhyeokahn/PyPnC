import numpy as np

from config.atlas_config import WalkingConfig, WBCConfig
from pnc.control_architecture import ControlArchitecture
from pnc.atlas_pnc.atlas_task_force_container import AtlasTaskForceContainer
from pnc.atlas_pnc.atlas_controller import AtlasController


class AtlasControlArchitecture(ControlArchitecture):
    def __init__(self, robot):
        super(AtlasControlArchitecture, self).__init__(robot)

        # Initialize Task Force Container
        self._taf_container = AtlasTaskForceContainer(robot)

        # Initialize Controller
        self._main_controller = AtlasController(self._taf_container, robot)

        # Initialize Planner
        # self._dcm_planner = DCMPlanner()
        # setting dcm parameters...

        # Initialize Task Manager
        self._rfoot_tm = FootTrajectoryManager(
            self._taf_container.rfoot_pos_task,
            self._taf_container.rfoot_ori_task, robot)
        self._rfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT
        self._lfoot_tm = FootTrajectoryManager(
            self._taf_container.lfoot_pos_task,
            self._taf_container.lfoot_ori_task, robot)
        self._lfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT
        self._upper_body_tm = UpperBodyTrajectoryManager(
            self._taf_container.upper_body_task, robot)
        self._trajectory_managers = {
            "rfoot": self._rfoot_tm,
            "lfoot": self._lfoot_tm,
            "upper_body": self._upper_body_tm
        }
        # self._dcm_tm = DCMTrajectoryManager(self._dcm_planner, self._taf_container.com_task, self._taf_container.pelvis_ori_task, "l_sole", "r_sole")

        # Initialize Reaction Force Manager
        self._rfoot_fm = ReactionForceManager(
            self._taf_container.rfoot_contact, WBCConfig.FR_Z_MAX, robot)
        self._lfoot_fm = ReactionForceManager(
            self._taf_container.lfoot_contact, WBCConfig.FR_Z_MAX, robot)
        self._reaction_force_managers = {
            "rfoot": self._rfoot_fm,
            "lfoot": self._lfoot_fm
        }

        # Initialize Hierarchy Manager
        self._rfoot_pos_hm = TaskHierarchyManager(
            self._taf_container.rfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCCongif.W_SWING_FOOT, robot)
        self._lfoot_pos_hm = TaskHierarchyManager(
            self._taf_container.lfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT, robot)
        self._rfoot_ori_hm = TaskHierarchyManager(
            self._taf_container.rfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT, robot)
        self._lfoot_ori_hm = TaskHierarchyManager(
            self._taf_container.lfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT, robot)
        self._hierarchy_managers = {
            "rfoot_pos": self._rfoot_pos_hm,
            "lfoot_pos": self._lfoot_pos_hm,
            "rfoot_ori": self._rfoot_ori_hm,
            "lfoot_ori": self._lfoot_ori_hm
        }

        # Initialize State Machines
        self._state_machine["STAND"] = DoubleSupportStand(
            0, self._trajectory_managers, self._hierarchy_managers,
            self._reaction_force_managers, robot)

        self._state_machine["BALANCE"] = DoubleSupportStand(
            1, self._trajectory_managers, self._hierarchy_managers,
            self._reaction_force_managers, robot)

        # Set Starting State
        self._state = "STAND"
        self._prev_state = "STAND"

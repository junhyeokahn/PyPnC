import numpy as np

from config.laikago_config import WBCConfig, PnCConfig
from pnc.wbc.tci_container import TCIContainer
from pnc.wbc.basic_task import BasicTask
from pnc.wbc.basic_contact import PointContact


class LaikagoTCIContainer(TCIContainer):
    def __init__(self, robot):
        super(LaikagoTCIContainer, self).__init__(robot)

        # ======================================================================
        # Initialize Task
        # ======================================================================
        # COM Task
        self._com_task = BasicTask(robot, "COM", 3, 'com', PnCConfig.SAVE_DATA)
        self._com_task.kp = WBCConfig.KP_COM
        self._com_task.kd = WBCConfig.KD_COM
        self._com_task.w_hierarchy = WBCConfig.W_COM
        # Base ori Task
        self._base_ori_task = BasicTask(robot, "LINK_ORI", 3, "chassis",
                                        PnCConfig.SAVE_DATA)
        self._base_ori_task.kp = WBCConfig.KP_BASE_ORI
        self._base_ori_task.kd = WBCConfig.KD_BASE_ORI
        self._base_ori_task.w_hierarchy = WBCConfig.W_BASE_ORI
        # FL foot Pos Task
        self._fl_foot_task = BasicTask(robot, "LINK_XYZ", 3, "toeFL",
                                       PnCConfig.SAVE_DATA)
        self._fl_foot_task.kp = WBCConfig.KP_FOOT_POS
        self._fl_foot_task.kd = WBCConfig.KD_FOOT_POS
        self._fl_foot_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # FR foot Pos Task
        self._fr_foot_task = BasicTask(robot, "LINK_XYZ", 3, "toeFR",
                                       PnCConfig.SAVE_DATA)
        self._fr_foot_task.kp = WBCConfig.KP_FOOT_POS
        self._fr_foot_task.kd = WBCConfig.KD_FOOT_POS
        self._fr_foot_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        # RR foot Pos Task
        self._rr_foot_task = BasicTask(robot, "LINK_XYZ", 3, "toeRR",
                                       PnCConfig.SAVE_DATA)
        self._rr_foot_task.kp = WBCConfig.KP_FOOT_POS
        self._rr_foot_task.kd = WBCConfig.KD_FOOT_POS
        self._rr_foot_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # RL foot Pos Task
        self._rl_foot_task = BasicTask(robot, "LINK_XYZ", 3, "toeRL",
                                       PnCConfig.SAVE_DATA)
        self._rl_foot_task.kp = WBCConfig.KP_FOOT_POS
        self._rl_foot_task.kd = WBCConfig.KD_FOOT_POS
        self._rl_foot_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        self._task_list = [
            self._com_task, self._base_ori_task, self._fl_foot_task,
            self._fr_foot_task, self._rr_foot_task, self._rl_foot_task
        ]

        # ======================================================================
        # Initialize Contact
        # ======================================================================
        # FL foot Contact
        self._fl_foot_contact = PointContact(robot, "toeFL", 0.4,
                                             PnCConfig.SAVE_DATA)
        self._fl_foot_contact.rf_z_max = 1e-3  # Initial rf_z_max
        # FR foot Contact
        self._fr_foot_contact = PointContact(robot, "toeFR", 0.4,
                                             PnCConfig.SAVE_DATA)
        self._fr_foot_contact.rf_z_max = 1e-3  # Initial rf_z_max
        # RR foot Contact
        self._rr_foot_contact = PointContact(robot, "toeRR", 0.4,
                                             PnCConfig.SAVE_DATA)
        self._rr_foot_contact.rf_z_max = 1e-3  # Initial rf_z_max
        # RL foot Contact
        self._rl_foot_contact = PointContact(robot, "toeRL", 0.4,
                                             PnCConfig.SAVE_DATA)
        self._rl_foot_contact.rf_z_max = 1e-3  # Initial rf_z_max

        self._contact_list = [
            self._fl_foot_contact, self._fr_foot_contact,
            self._rr_foot_contact, self._rl_foot_contact
        ]

    @property
    def com_task(self):
        return self._com_task

    @property
    def base_ori_task(self):
        return self._base_ori_task

    @property
    def fl_foot_task(self):
        return self._fl_foot_task

    @property
    def fr_foot_task(self):
        return self._fr_foot_task

    @property
    def rr_foot_task(self):
        return self._rr_foot_task

    @property
    def rl_foot_task(self):
        return self._rl_foot_task

    @property
    def fl_foot_contact(self):
        return self._fl_foot_contact

    @property
    def fr_foot_contact(self):
        return self._fr_foot_contact

    @property
    def rr_foot_contact(self):
        return self._rr_foot_contact

    @property
    def rl_foot_contact(self):
        return self._rl_foot_contact

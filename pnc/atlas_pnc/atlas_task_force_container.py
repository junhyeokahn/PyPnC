import numpy as np

from config.atlas_config import WBCConfig, PnCConfig
from pnc.task_force_container import TaskForceContainer
from pnc.wbc.basic_task import BasicTask
from pnc.wbc.basic_contact import SurfaceContact


class AtlasTaskForceContainer(TaskForceContainer):
    def __init__(self, robot):
        super(AtlasTaskForceContainer, self).__init__(robot)

        # ======================================================================
        # Initialize Task
        # ======================================================================
        # COM Task
        self._com_task = BasicTask(robot, "COM", 3, 'com', PnCConfig.SAVE_DATA)
        self._com_task.kp = WBCConfig.KP_COM
        self._com_task.kd = WBCConfig.KD_COM
        self._com_task.w_hierarchy = WBCConfig.W_COM
        # Pelvis Task
        self._pelvis_ori_task = BasicTask(robot, "LINK_ORI", 3, "pelvis_com",
                                          PnCConfig.SAVE_DATA)
        self._pelvis_ori_task.kp = WBCConfig.KP_PELVIS
        self._pelvis_ori_task.kd = WBCConfig.KD_PELVIS
        self._pelvis_ori_task.w_hierarchy = WBCConfig.W_PELVIS
        # Non leg related joints (18, so 18 + 6*2 = n_a)
        selected_joint = [
            "back_bkx", "back_bky", "back_bkz", "l_arm_elx", "l_arm_ely",
            "l_arm_shx", "l_arm_shz", "l_arm_wrx", "l_arm_wry", "l_arm_wry2",
            "neck_ry", "r_arm_elx", "r_arm_ely", "r_arm_shx", "r_arm_shz",
            "r_arm_wrx", "r_arm_wry", "r_arm_wry2"
        ]
        self._upper_body_task = BasicTask(robot, "SELECTED_JOINT",
                                          len(selected_joint), selected_joint,
                                          PnCConfig.SAVE_DATA)
        self._upper_body_task.kp = np.array([WBCConfig.KP_UPPER_BODY] *
                                            self._upper_body_task.dim)
        self._upper_body_task.kd = np.array([WBCConfig.KD_UPPER_BODY] *
                                            self._upper_body_task.dim)
        self._upper_body_task.w_hierarchy = WBCConfig.W_UPPER_BODY
        # Rfoot Pos Task
        self._rfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3, "r_sole",
                                         PnCConfig.SAVE_DATA)
        self._rfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._rfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._rfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Lfoot Pos Task
        self._lfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3, "l_sole",
                                         PnCConfig.SAVE_DATA)
        self._lfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._lfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._lfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Rfoot Ori Task
        self._rfoot_ori_task = BasicTask(robot, "LINK_ORI", 3, "r_sole",
                                         PnCConfig.SAVE_DATA)
        self._rfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._rfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._rfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Lfoot Ori Task
        self._lfoot_ori_task = BasicTask(robot, "LINK_ORI", 3, "l_sole",
                                         PnCConfig.SAVE_DATA)
        self._lfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._lfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._lfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        self._task_list = [
            self._com_task, self._pelvis_ori_task, self._upper_body_task,
            self._rfoot_pos_task, self._lfoot_pos_task, self._rfoot_ori_task,
            self._lfoot_ori_task
        ]

        # ======================================================================
        # Initialize Contact
        # ======================================================================
        # Rfoot Contact
        self._rfoot_contact = SurfaceContact(robot, "r_sole", 0.11, 0.065, 0.3,
                                             PnCConfig.SAVE_DATA)
        self._rfoot_contact.rf_z_max = 1e-3  # Initial rf_z_max
        # Lfoot Contact
        self._lfoot_contact = SurfaceContact(robot, "l_sole", 0.11, 0.065, 0.3,
                                             PnCConfig.SAVE_DATA)
        self._lfoot_contact.rf_z_max = 1e-3  # Initial rf_z_max

        self._contact_list = [self._rfoot_contact, self._lfoot_contact]

    @property
    def com_task(self):
        return self._com_task

    @property
    def pelvis_ori_task(self):
        return self._pelvis_ori_task

    @property
    def upper_body_task(self):
        return self._upper_body_task

    @property
    def rfoot_pos_task(self):
        return self._rfoot_pos_task

    @property
    def lfoot_pos_task(self):
        return self._lfoot_pos_task

    @property
    def rfoot_ori_task(self):
        return self._rfoot_ori_task

    @property
    def lfoot_ori_task(self):
        return self._lfoot_ori_task

    @property
    def rfoot_contact(self):
        return self._rfoot_contact

    @property
    def lfoot_contact(self):
        return self._lfoot_contact

    @property
    def task_list(self):
        return self._task_list

    @property
    def contact_list(self):
        return self._contact_list

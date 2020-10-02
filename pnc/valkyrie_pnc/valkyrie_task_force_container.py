import numpy as np

from config.valkyrie_config import WBCConfig, PnCConfig
from pnc.task_force_container import TaskForceContainer
from pnc.wbc.basic_task import BasicTask
from pnc.wbc.basic_contact import SurfaceContact


class ValkyrieTaskForceContainer(TaskForceContainer):
    def __init__(self, robot):
        super(ValkyrieTaskForceContainer, self).__init__(robot)

        # ======================================================================
        # Initialize Task
        # ======================================================================
        # COM Task
        self._com_task = BasicTask(robot, "COM", 3, "com", PnCConfig.SAVE_DATA)
        self._com_task.kp = WBCConfig.KP_COM
        self._com_task.kd = WBCConfig.KD_COM
        self._com_task.w_hierarchy = WBCConfig.W_COM
        # Pelvis Task
        self._pelvis_ori_task = BasicTask(robot, "LINK_ORI", 3, "pelvis",
                                          PnCConfig.SAVE_DATA)
        self._pelvis_ori_task.kp = WBCConfig.KP_PELVIS
        self._pelvis_ori_task.kd = WBCConfig.KD_PELVIS
        self._pelvis_ori_task.w_hierarchy = WBCConfig.W_PELVIS
        selected_joint = [
            'torsoYaw', 'torsoPitch', 'torsoRoll', 'leftShoulderPitch',
            'leftShoulderRoll', 'leftShoulderYaw', 'leftElbowPitch',
            'leftForearmYaw', 'lowerNeckPitch', 'neckYaw', 'upperNeckPitch',
            'rightShoulderPitch', 'rightShoulderRoll', 'rightShoulderYaw',
            'rightElbowPitch', 'rightForearmYaw'
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
        self._rfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3,
                                         "rightCOP_Frame", PnCConfig.SAVE_DATA)
        self._rfoot_pos_task.kp = WBCConfig.KP_FOOT
        self._rfoot_pos_task.kd = WBCConfig.KD_FOOT
        self._rfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Lfoot Pos Task
        self._lfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3, "leftCOP_Frame",
                                         PnCConfig.SAVE_DATA)
        self._lfoot_pos_task.kp = WBCConfig.KP_FOOT
        self._lfoot_pos_task.kd = WBCConfig.KD_FOOT
        self._lfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Rfoot Ori Task
        self._rfoot_ori_task = BasicTask(robot, "LINK_ORI", 3,
                                         "rightCOP_Frame", PnCConfig.SAVE_DATA)
        self._rfoot_ori_task.kp = WBCConfig.KP_FOOT
        self._rfoot_ori_task.kd = WBCConfig.KD_FOOT
        self._rfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT
        # Lfoot Ori Task
        self._lfoot_ori_task = BasicTask(robot, "LINK_ORI", 3, "leftCOP_Frame",
                                         PnCConfig.SAVE_DATA)
        self._lfoot_ori_task.kp = WBCConfig.KP_FOOT
        self._lfoot_ori_task.kd = WBCConfig.KD_FOOT
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
        self._rfoot_contact = SurfaceContact(robot, "rightCOP_Frame", 0.135,
                                             0.08, 0.9 / np.sqrt(2.0),
                                             PnCConfig.SAVE_DATA)
        self._rfoot_contact.rf_z_max = WBCConfig.RF_Z_MAX
        # Lfoot Contact
        self._lfoot_contact = SurfaceContact(robot, "leftCOP_Frame", 0.135,
                                             0.08, 0.9 / np.sqrt(2.0),
                                             PnCConfig.SAVE_DATA)
        self._lfoot_contact.rf_z_max = WBCConfig.RF_Z_MAX

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

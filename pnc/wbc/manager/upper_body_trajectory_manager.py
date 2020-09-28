import numpy as np

from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class UpperBodyTrajectoryManager(object):
    def __init__(self, upper_body_task, robot):
        self._upper_body_task = upper_body_task
        self._robot = robot
        self._sp = AtlasStateProvider(self._robot)

    def use_nominal_upper_body_joint_pos(self):
        joint_pos_des = np.array([
            self._sp.nominal_joint_pos[k]
            for k in self._upper_body_task.target_id
        ])
        joint_vel_des, joint_acc_des = np.zeros_like(
            joint_pos_des), np.zeros_like(joint_pos_des)

        self._upper_body_task.update_desired(joint_pos_des, joint_vel_des,
                                             joint_acc_des)

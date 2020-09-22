import numpy as np

from util import util


class FootTrajectoryManager(object):
    def __init__(self, pos_task, ori_task, robot):
        self._pos_task = pos_task
        self._ori_task = ori_task
        self._robot = robot
        self._swing_height = 0.05

        assert self._pos_task.link_id == self._ori_task.link_id
        self._link_id = self._pos_task_link_id

    @property
    def swing_height(self):
        return self._swing_height

    @swing_height.setter
    def swing_height(self, val):
        self._swing_height = val

    def use_current(self):
        foot_iso = self._robot.get_link_iso(self._link_id)
        foot_vel = self._robot.get_link_vel(self._link_id)

        foot_pos_des = foot_iso[0:3, 3]
        foot_lin_vel_des = foot_vel[3:6]
        self._pos_task.update_desired(foot_pos_des, foot_lin_vel_des,
                                      np.zeros(3))

        foot_rot_des = util.rot_to_quat(foot_iso[0:3, 0:3])
        fooot_ang_vel_des = foot_vel[0:3]
        self._ori_task.update_desired(foot_rot_des, foot_ang_vel_des,
                                      np.zeros(3))

    def initialize_swing_foot_trajectory(self, start_time, swing_duration,
                                         landing_foot):
        # TODO
        pass

    def update_swing_foot_desired(self, curr_Time):
        # TODO
        pass

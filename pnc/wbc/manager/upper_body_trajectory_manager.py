import numpy as np


class UpperBodyTrajectoryManager(object):
    def __init__(self, upper_body_task, robot):
        self._upper_body_task = upper_body_task
        self._robot = robot

    def use_initial_joint_pos(self):
        pass

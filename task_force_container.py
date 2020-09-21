import abc


class TaskForceContainer(abc.ABC):
    def __init__(self, robot):
        self._robot = robot
        self._task_list = []
        self._contact_list = []

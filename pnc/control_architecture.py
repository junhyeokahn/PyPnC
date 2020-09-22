import abc


class ControlArchitecture(abc.ABC):
    def __init__(self, robot):
        self._robot = robot
        self._state = 0
        self._prev_state = 0
        self._state_machine = dict()
        self._trajectory_managers = dict()
        self._hierarchy_manangers = dict()
        self._reaction_force_managers = dict()

    @property
    def state(self):
        return self._state

    @property
    def prev_state(self):
        return self._prev_state

    @abc.abstractmethod
    def get_command(self):
        pass

import abc


class Interface(abc.ABC):
    def __init__(self):
        self._count = 0
        self._running_time = 0.

    @abc.abstractmethod
    def get_command(self, sensor_data):
        pass

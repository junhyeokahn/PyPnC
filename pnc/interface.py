import abc


class Interface(abc.ABC):
    def __init__(self):
        _count = 0
        _running_tim = 0.

    @abc.abstractmethod
    def get_command(self, sensor_data):
        pass

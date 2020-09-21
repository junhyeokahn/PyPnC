import numpy as np

from pnc.control_architecture import ControlArchitecture
from pnc.atlas_pnc.atlas_task_force_container import AtlasTaskForceContainer
from pnc.atlas_pnc.atlas_controller import AtlasController


class AtlasControlArchitecture(ControlArchitecture):
    def __init__(self, robot):
        super(AtlasControlArchitecture, self).__init__(robot)

        self._taf_container = AtlasTaskForceContainer(robot)
        self._main_controller = AtlasController(self._taf_container, robot)

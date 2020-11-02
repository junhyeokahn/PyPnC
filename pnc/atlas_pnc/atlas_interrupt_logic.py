import numpy as np

from pnc.interrupt_logic import InterruptLogic
from config.atlas_config import WalkingState


class AtlasInterruptLogic(InterruptLogic):
    def __init__(self, ctrl_arch):
        super(AtlasInterruptLogic, self).__init__()
        self._control_architecture = ctrl_arch

    def process_interrupts(self):
        if self._b_interrupt_button_eight:
            print("[Interrupt Logic] button {} pressed".format(8))
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.walk_forward()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].b_state_switch_trigger = True

        self._reset_flags()

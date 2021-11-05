import numpy as np

from pnc.interrupt_logic import InterruptLogic
# from config.laikago_config import WalkingState


class LaikagoInterruptLogic(InterruptLogic):
    def __init__(self, ctrl_arch):
        super(LaikagoInterruptLogic, self).__init__()
        self._control_architecture = ctrl_arch

    def process_interrupts(self):
        if self._b_interrupt_button_eight:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(8))
            print("=" * 80)
        if self._b_interrupt_button_five:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(5))
            print("=" * 80)
        if self._b_interrupt_button_four:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(4))
            print("=" * 80)
        if self._b_interrupt_button_six:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(6))
            print("=" * 80)
        if self._b_interrupt_button_two:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(2))
            print("=" * 80)
        if self._b_interrupt_button_seven:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(7))
            print("=" * 80)
        if self._b_interrupt_button_nine:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: ".format(9))
            print("=" * 80)

        self._reset_flags()

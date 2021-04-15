import numpy as np

from pnc.interrupt_logic import InterruptLogic
from config.draco3_config import WalkingState


class Draco3InterruptLogic(InterruptLogic):
    def __init__(self, ctrl_arch):
        super(Draco3InterruptLogic, self).__init__()
        self._control_architecture = ctrl_arch

    def process_interrupts(self):
        if self._b_interrupt_button_eight:
            print("=" * 80)
            print(
                "[Interrupt Logic] button {} pressed: Walk Forward".format(8))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.walk_forward()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_five:
            print("=" * 80)
            print(
                "[Interrupt Logic] button {} pressed: Walk In Place".format(5))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.walk_in_place()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_four:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: Walk Left".format(4))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.strafe_left()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_six:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: Walk Right".format(6))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.strafe_right()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_two:
            print("=" * 80)
            print(
                "[Interrupt Logic] button {} pressed: Walk Backward".format(2))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.walk_backward()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_seven:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: Turn Left".format(7))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.turn_left()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_nine:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: Turn Right".format(9))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.dcm_tm.turn_right()
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_zero:
            print("=" * 80)
            print("[Interrupt Logic] button {} pressed: Swaying".format(0))
            print("=" * 80)
            if self._control_architecture.state == WalkingState.BALANCE:
                self._control_architecture.state_machine[
                    WalkingState.BALANCE].swaying_trigger = True

        self._reset_flags()

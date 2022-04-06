import numpy as np

from pnc.interrupt_logic import InterruptLogic
from config.draco_manipulation_config import LocomanipulationState

COM_VEL_THRE = 0.01


class DracoManipulationInterruptLogic(InterruptLogic):
    def __init__(self, ctrl_arch):
        super(DracoManipulationInterruptLogic, self).__init__()
        self._control_architecture = ctrl_arch

        self._lh_target_pos = np.array([0., 0., 0.])
        self._lh_waypoint_pos = np.array([0., 0., 0.])
        self._lh_target_quat = np.array([0., 0., 0., 1.])
        self._rh_target_pos = np.array([0., 0., 0.])
        self._rh_waypoint_pos = np.array([0., 0., 0.])
        self._rh_target_quat = np.array([0., 0., 0., 1.])

        self._com_displacement_x = 0.
        self._com_displacement_y = 0.

        self._b_walk_in_progress = False
        self._b_walk_ready = False
        self._b_left_hand_ready = False
        self._b_right_hand_ready = False

    @property
    def com_displacement_x(self):
        return self._com_displacement_x

    @com_displacement_x.setter
    def com_displacement_x(self, value):
        self._com_displacement_x = value

    @property
    def com_displacement_y(self):
        return self._com_displacement_y

    @com_displacement_y.setter
    def com_displacement_y(self, value):
        self._com_displacement_y = value

    @property
    def lh_target_pos(self):
        return self._lh_target_pos

    @lh_target_pos.setter
    def lh_target_pos(self, value):
        self._lh_target_pos = value

    @property
    def rh_target_pos(self):
        return self._rh_target_pos

    @rh_target_pos.setter
    def rh_target_pos(self, value):
        self._rh_target_pos = value

    @property
    def rh_waypoint_pos(self):
        return self._rh_waypoint_pos

    @rh_waypoint_pos.setter
    def rh_waypoint_pos(self, value):
        self._rh_waypoint_pos = value

    @property
    def lh_waypoint_pos(self):
        return self._lh_waypoint_pos

    @lh_waypoint_pos.setter
    def lh_waypoint_pos(self, value):
        self._lh_waypoint_pos = value

    @property
    def lh_target_quat(self):
        return self._lh_target_quat

    @lh_target_quat.setter
    def lh_target_quat(self, value):
        self._lh_target_quat = value

    @property
    def rh_target_quat(self):
        return self._rh_target_quat

    @rh_target_quat.setter
    def rh_target_quat(self, value):
        self._rh_target_quat = value

    @property
    def b_walk_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if np.linalg.norm(
                com_vel
        ) < COM_VEL_THRE and self._control_architecture.state == LocomanipulationState.BALANCE:
            self._b_walk_ready = True
        else:
            self._b_walk_ready = False
        return self._b_walk_ready

    @property
    def b_left_hand_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if np.linalg.norm(
                com_vel
        ) < COM_VEL_THRE and self._control_architecture.state == LocomanipulationState.BALANCE:
            self._b_left_hand_ready = True
        else:
            self._b_left_hand_ready = False
        return self._b_left_hand_ready

    @property
    def b_right_hand_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if np.linalg.norm(
                com_vel
        ) < COM_VEL_THRE and self._control_architecture.state == LocomanipulationState.BALANCE:
            self._b_right_hand_ready = True
        else:
            self._b_right_hand_ready = False
        return self._b_right_hand_ready

    def process_interrupts(self):
        if self._b_interrupt_button_eight:
            # print("=" * 80)
            # print(
            # "[Interrupt Logic] button {} pressed: Walk Forward".format(8))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.walk_forward()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_five:
            # print("=" * 80)
            # print(
            # "[Interrupt Logic] button {} pressed: Walk In Place".format(5))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.walk_in_place()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_four:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Walk Left".format(4))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.strafe_left()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_six:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Walk Right".format(6))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.strafe_right()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_two:
            # print("=" * 80)
            # print(
            # "[Interrupt Logic] button {} pressed: Walk Backward".format(2))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.walk_backward()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_seven:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Turn Left".format(7))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.turn_left()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_nine:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Turn Right".format(9))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.turn_right()
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_m:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed".format('m'))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.walk_in_x(
                    self._com_displacement_x)
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_n:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed".format('n'))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.dcm_tm.walk_in_y(
                    self._com_displacement_y)
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE].walking_trigger = True

        if self._b_interrupt_button_one:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Left Hand Reaching".
            # format(1))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                # Set target pos and goal here
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    LH_HANDREACH].lh_target_pos = self._lh_target_pos
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    LH_HANDREACH].lh_waypoint_pos = self._lh_waypoint_pos
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    LH_HANDREACH].lh_target_quat = self._lh_target_quat
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    BALANCE].lhand_task_trans_trigger = True

        if self._b_interrupt_button_three:
            # print("=" * 80)
            # print("[Interrupt Logic] button {} pressed: Right Hand Reaching".
            # format(3))
            # print("=" * 80)
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    BALANCE].rhand_task_trans_trigger = True
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    RH_HANDREACH].rh_target_pos = self._rh_target_pos
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    RH_HANDREACH].rh_waypoint_pos = self._rh_waypoint_pos
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    RH_HANDREACH].rh_target_quat = self._rh_target_quat

        if self._b_interrupt_button_r:
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    BALANCE].rhand_task_return_trigger = True

        if self._b_interrupt_button_e:
            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.state_machine[
                    LocomanipulationState.
                    BALANCE].lhand_task_return_trigger = True

        self._reset_flags()

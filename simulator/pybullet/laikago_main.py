import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import time
import copy

import pybullet as p
import numpy as np

from util import pybullet_util
from config.laikago_config import SimConfig
from pnc.laikago_pnc.laikago_interface import LaikagoInterface

p.connect(p.GUI)
plane = p.loadURDF(cwd + "/robot_model/ground/plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(SimConfig.CONTROLLER_DT)

quadruped = p.loadURDF(cwd + "/robot_model/laikago/laikago_toes.urdf",
                       SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT,
                       useFixedBase=False)

nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    quadruped, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
    SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

p.resetJointState(quadruped, joint_id["FR_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["FL_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["RR_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["RL_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)

pybullet_util.set_link_damping(quadruped, link_id.values(), 0., 0.)
pybullet_util.set_joint_friction(quadruped, joint_id, 0)

interface = LaikagoInterface()

nominal_sensor_data = pybullet_util.get_sensor_data(quadruped, joint_id,
                                                    link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

t = 0
count = 0

while (1):

    # Get sensor data
    sensor_data = pybullet_util.get_sensor_data(quadruped, joint_id, link_id,
                                                pos_basejoint_to_basecom,
                                                rot_basejoint_to_basecom)

    # Get Keyboard Event
    keys = p.getKeyboardEvents()
    if pybullet_util.is_key_triggered(keys, '8'):
        interface.interrupt_logic.b_interrupt_button_eight = True
    elif pybullet_util.is_key_triggered(keys, '5'):
        interface.interrupt_logic.b_interrupt_button_five = True
    elif pybullet_util.is_key_triggered(keys, '4'):
        interface.interrupt_logic.b_interrupt_button_four = True
    elif pybullet_util.is_key_triggered(keys, '2'):
        interface.interrupt_logic.b_interrupt_button_two = True
    elif pybullet_util.is_key_triggered(keys, '6'):
        interface.interrupt_logic.b_interrupt_button_six = True
    elif pybullet_util.is_key_triggered(keys, '7'):
        interface.interrupt_logic.b_interrupt_button_seven = True
    elif pybullet_util.is_key_triggered(keys, '9'):
        interface.interrupt_logic.b_interrupt_button_nine = True

    command = interface.get_command(copy.deepcopy(sensor_data))

    # Set motor cmd
    pybullet_util.set_motor_pos(quadruped, joint_id,
                                nominal_sensor_data['joint_pos'])
    # pybullet_util.set_motor_trq(quadruped, joint_id, command['joint_trq'])
    # pybullet_util.set_motor_pos(quadruped, joint_id, command['joint_pos'])

    p.stepSimulation()
    t += SimConfig.CONTROLLER_DT
    count += 1
    time.sleep(SimConfig.CONTROLLER_DT)

    # print("toeFL")
    # print(pybullet_util.get_link_iso(quadruped, link_id['toeFL']))
    # print("toeRR")
    # print(pybullet_util.get_link_iso(quadruped, link_id['toeRR']))
    # print("toeRL")
    # print(pybullet_util.get_link_iso(quadruped, link_id['toeRL']))
    # print("base pos")
    # print(sensor_data["base_com_pos"])
    # print("base quat")
    # print(sensor_data["base_com_quat"])

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import cv2
import pybullet as p
import numpy as np

np.set_printoptions(precision=2)

from config.draco_manipulation_config import SimConfig
from pnc.draco_manipulation_pnc.draco_manipulation_interface import DracoManipulationInterface
from util import pybullet_util
from util import util
from util import liegroup

p.connect(p.GUI)
p.resetDebugVisualizerCamera(cameraDistance=1.0,
                             cameraYaw=120,
                             cameraPitch=-30,
                             cameraTargetPosition=[1, 0.5, 1.0])

p.setGravity(0, 0, -9.8)

p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
xOffset = 1.0

p.loadURDF(cwd + "/robot_model/bookcase/bookshelf.urdf",
           useFixedBase=1,
           basePosition=[0 + xOffset, 0, 0.025],
           baseOrientation=[0, 0, 0.7068252, 0.7068252])
p.loadURDF(cwd + "/robot_model/bookcase/red_can.urdf",
           useFixedBase=0,
           basePosition=[0 + xOffset, 0.75, 1.05])
p.loadURDF(cwd + "/robot_model/bookcase/green_can.urdf",
           useFixedBase=0,
           basePosition=[0 + xOffset, -0.7, 1.35])
p.loadURDF(cwd + "/robot_model/bookcase/blue_can.urdf",
           useFixedBase=0,
           basePosition=[0 + xOffset, 0, 0.7])

# gripper = p.loadURDF(cwd + "/robot_model/ezgripper/gripper.urdf",
# useFixedBase=1,
# basePosition=[0 + xOffset - 0.2, 0, 0.7])
gripper = p.loadURDF(cwd + "/robot_model/ezgripper/gripper.urdf",
                     useFixedBase=1,
                     basePosition=[0 + xOffset - 0.2, -0.7, 1.35])

gripper_joints = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2"
]

gripper_floating = {"basePosX": 0, "basePosY": 1, "basePosZ": 2}

nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    gripper, [0, 0, 1], [0, 0, 0, 1], True)

pybullet_util.set_joint_friction(gripper, gripper_floating, 0.)
pybullet_util.set_link_damping(gripper, link_id.values(), 0., 0.)

nominal_sensor_data = pybullet_util.get_sensor_data(gripper, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

gripper_pos = nominal_sensor_data['base_com_pos']
gripper_quat = nominal_sensor_data['base_com_quat']

gripper_command = dict()
for gripper_joint in gripper_joints:
    gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
        gripper_joint]

for gripper_fjoint in gripper_floating.keys():
    gripper_command[gripper_fjoint] = nominal_sensor_data['joint_pos'][
        gripper_fjoint]

#GRIPPER_DELTA_ANGLE = 1.94 / 3
GRIPPER_DELTA_ANGLE = 1.94 / 4

while (1):

    keys = p.getKeyboardEvents()
    if pybullet_util.is_key_triggered(keys, 'c'):
        for k in gripper_joints:
            gripper_command[k] += GRIPPER_DELTA_ANGLE
    elif pybullet_util.is_key_triggered(keys, 'o'):
        for k in gripper_joints:
            gripper_command[k] -= GRIPPER_DELTA_ANGLE
    elif pybullet_util.is_key_triggered(keys, 'u'):
        # gripper_pos[2] += 0.02  # z
        gripper_command['basePosZ'] += 0.02
    elif pybullet_util.is_key_triggered(keys, 'd'):
        # gripper_pos[2] -= 0.02  # z
        gripper_command['basePosZ'] -= 0.02
    elif pybullet_util.is_key_triggered(keys, 'f'):
        # gripper_pos[0] += 0.02  # x
        gripper_command['basePosX'] += 0.02
    elif pybullet_util.is_key_triggered(keys, 'b'):
        # gripper_pos[0] -= 0.02  # x
        gripper_command['basePosX'] -= 0.02
    elif pybullet_util.is_key_triggered(keys, 'l'):
        # gripper_pos[1] += 0.02  # y
        gripper_command['basePosY'] += 0.02
    elif pybullet_util.is_key_triggered(keys, 'r'):
        # gripper_pos[1] -= 0.02  # y
        gripper_command['basePosY'] -= 0.02
    elif pybullet_util.is_key_triggered(keys, 's'):
        # gripper_pos[0] = 0.82  # x
        # gripper_pos[1] = 0.0  # y
        # gripper_pos[2] = 0.719  # z
        gripper_command['basePosX'] = 0.82  # x
        gripper_command['basePosY'] = 0.0  # y
        gripper_command['basePosZ'] = 0.019  # z

    # print("-" * 80)
    # print(gripper_command)
    pybullet_util.set_motor_pos(gripper, joint_id, gripper_command)

    p.stepSimulation()

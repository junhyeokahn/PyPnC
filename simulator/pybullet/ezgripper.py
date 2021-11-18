import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np
from util import pybullet_util
from util import util
from util import liegroup

INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 0.5]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]
CONTROLLER_DT = 0.01

if __name__ == "__main__":
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=0.05,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.0])
    p.setGravity(0, 0, 9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=CONTROLLER_DT)

    gripper = p.loadURDF(cwd +
                         "/robot_model/ezgripper/ezgripper_standalone.urdf",
                         INITIAL_POS_WORLD_TO_BASEJOINT,
                         INITIAL_QUAT_WORLD_TO_BASEJOINT,
                         useFixedBase=True)
    print("Loaded")

    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        gripper, INITIAL_POS_WORLD_TO_BASEJOINT,
        INITIAL_QUAT_WORLD_TO_BASEJOINT, True)

    c = p.createConstraint(gripper,
                           link_id['left_ezgripper_finger_L1_1'],
                           gripper,
                           link_id['left_ezgripper_finger_L1_2'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])

    p.changeConstraint(c, gearRatio=-1, maxForce=1000, erp=2)

    # c = p.createConstraint(gripper,
    # link_id['left_ezgripper_finger_L2_1'],
    # gripper,
    # link_id['left_ezgripper_finger_L2_2'],
    # jointType=p.JOINT_GEAR,
    # jointAxis=[0, 1, 0],
    # parentFramePosition=[0, 0, 0],
    # childFramePosition=[0, 0, 0])
    # p.changeConstraint(c, gearRatio=-1, maxForce=1000, erp=2)

    print("Constrain added")

    # pybullet_util.set_link_damping(gripper, link_id.values(), 1., 1.)
    # pybullet_util.set_joint_friction(gripper, joint_id, 5)

    t = 0
    dt = CONTROLLER_DT
    count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        gripper, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    # Add debug parameters
    param_ids = []
    for k, v in joint_id.items():
        param_ids.append(
            p.addUserDebugParameter(k, -4, 4,
                                    nominal_sensor_data['joint_pos'][k]))

    while (1):
        for i in range(len(param_ids)):
            c = param_ids[i]
            target_pos = p.readUserDebugParameter(c)
            # if list(joint_id.values()
            # )[i] == joint_id['left_ezgripper_knuckle_palm_L1_2']:
            # pass
            # elif list(joint_id.values()
            # )[i] == joint_id['left_ezgripper_knuckle_L1_L2_2']:
            # pass
            # else:
            # p.setJointMotorControl2(gripper,
            # list(joint_id.values())[i],
            # p.POSITION_CONTROL, target_pos)

        p.stepSimulation()
        time.sleep(CONTROLLER_DT)

        t += dt
        count += 1

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

from util import pybullet_util
from util import util

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=4.0,
                                 cameraYaw=0,
                                 cameraPitch=-45,
                                 cameraTargetPosition=[1.5, 0., 0.])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=0.01, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/manipulator/rolling_joint.urdf",
                       useFixedBase=True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, [0., 0., 0.], [0., 0., 0., 1.], True)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.1)

    # Rolling Joint
    c = p.createConstraint(robot,
                           link_id['l2'],
                           robot,
                           link_id['l3'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000, erp=10)

    pybullet_util.draw_link_frame(robot, link_id['l1'], text="l1")
    pybullet_util.draw_link_frame(robot, link_id['l2'], text="l2")
    pybullet_util.draw_link_frame(robot, link_id['l3'], text="l3")
    pybullet_util.draw_link_frame(robot, link_id['ee'], text="ee")

    # Run Sim
    t = 0
    dt = 0.01
    count = 0
    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        # Apply Trq
        p.setJointMotorControlArray(robot, [0, 1],
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[-0.1, 0.2])

        p.stepSimulation()

        if count % 10 == 0:
            print("+" * 80)
            print("count: ", count)
            print(sensor_data['joint_pos'])
            print(sensor_data['joint_vel'])

        time.sleep(dt)
        t += dt
        count += 1

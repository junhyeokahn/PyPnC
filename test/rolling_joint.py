import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + '/utils/python_utils')
sys.path.append(cwd + '/simulator/pybullet')
sys.path.append(cwd + '/build/lib')

import time, math
from collections import OrderedDict
import copy
import signal
import shutil

from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

import cv2
import pybullet as p
import numpy as np

np.set_printoptions(precision=2)

from util import pybullet_util
from util import util

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                 cameraYaw=0,
                                 cameraPitch=0,
                                 cameraTargetPosition=[0., 0., -1.5])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=0.01, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint.urdf',
        useFixedBase=True)

    viz_model, viz_collision_model, viz_visual_model = pin.buildModelsFromUrdf(
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint.urdf',
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf')
    viz = MeshcatVisualizer(viz_model, viz_collision_model, viz_visual_model)
    try:
        viz.initViewer(open=True)
    except Exception as e:
        print(e)
        exit()
    viz.loadViewerModel()
    vis_q = pin.neutral(viz_model)

    pin_robot = PinocchioRobotSystem(
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint.urdf',
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf',
        True, True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, [0., 0., 0.], [0., 0., 0., 1.], True)

    # Initial Config
    p.resetJointState(robot, joint_id["jp"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["jd"], np.pi / 6, 0.)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.)

    # Rolling Joint
    c = p.createConstraint(robot,
                           link_id['lp'],
                           robot,
                           link_id['ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000, erp=10)

    pybullet_util.draw_link_frame(robot, link_id['lp'], text="proximal")
    pybullet_util.draw_link_frame(robot, link_id['ld'], text="distal")
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

        pin_robot.update_system(np.zeros(3), np.zeros(4), np.zeros(3),
                                np.zeros(3), np.zeros(3), np.zeros(4),
                                np.zeros(3), np.zeros(3),
                                sensor_data['joint_pos'],
                                sensor_data['joint_vel'], True)

        # Apply Trq
        p.setJointMotorControlArray(robot, [joint_id['jp'], joint_id['jd']],
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=[0., 1.])

        for i in range(2):
            vis_q[i] = pin_robot.get_q()[i]
        viz.display(vis_q)

        if count % 10 == 0:
            print("+" * 80)
            print("count: ", count)
            # print("joint pos")
            # print(sensor_data['joint_pos'])
            # print("joint vel")
            # print(sensor_data['joint_vel'])
            print(pybullet_util.get_link_iso(robot, link_id['ee'])[0:3, 3])
            print(pin_robot.get_link_iso('ee')[0:3, 3])

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

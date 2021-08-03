import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + '/utils/python_utils')
sys.path.append(cwd + '/simulator/pybullet')
sys.path.append(cwd + '/build/lib')

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

np.set_printoptions(precision=5)

from util import pybullet_util
from util import util

DT = 0.001

jac_i = np.array([[0, 0, 0, 1, -1, 0, 0]])
s_a = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])


def IsClose(a, b, threshold=0.0001):
    if np.abs(a - b) < threshold:
        return True
    else:
        return False


def UpdateRobotSystem(robot, q, qdot):

    robot.update_system(np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3),
                        np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3), q,
                        qdot, True)
    mass = robot.get_mass_matrix()
    mass_inv = np.linalg.inv(mass)
    b = robot.get_coriolis()
    g = robot.get_gravity()

    lambda_i = np.linalg.inv(np.dot(jac_i, np.dot(mass_inv,
                                                  jac_i.transpose())))
    jac_i_bar = np.dot(np.dot(mass_inv, jac_i.transpose()), lambda_i)
    null_i = np.eye(jac_i.shape[1]) - np.dot(jac_i_bar, jac_i)

    return mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i


def PnCPjInvDyn(robot, q, qdot, qddot):
    """
    q: np.array, qdot: np.array, qddot: np.array
    """
    assert IsClose(qddot[1], qddot[2])

    mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i = UpdateRobotSystem(
        robot, {
            'jb00': q[0],
            'jb0': q[1],
            'jb1': q[2],
            'jp': q[3],
            'jd': q[4],
            'ja1': q[5],
            'ja2': q[6]
        }, {
            'jb00': qdot[0],
            'jb0': qdot[1],
            'jb1': qdot[2],
            'jp': qdot[3],
            'jd': qdot[4],
            'ja1': qdot[5],
            'ja2': qdot[6]
        })

    tau = np.dot(np.linalg.pinv(np.dot(s_a, null_i).transpose()),
                 np.dot(mass, qddot) + np.dot(null_i.transpose(), b + g))

    # print("ee pos : ", robot.get_link_iso('ee')[0:3, 3])
    # print("gravity comp command : ", tau[0], ", ", tau[1], ", ", tau[2], ", ",
    # tau[3] / 2, ", ", tau[4], ", ", tau[5])
    # print("full grav")
    # print(g)

    return tau


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                 cameraYaw=0,
                                 cameraPitch=0,
                                 cameraTargetPosition=[0., 0., -1.5])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint2.urdf',
        useFixedBase=True)

    # for i in range(3):
    # p.changeDynamics(0, i, localInertiaDiagonal=[1., 1., 1.])

    pin_robot = PinocchioRobotSystem(
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint2.urdf',
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf',
        True, True)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, [0., 0., 0.], [0., 0., 0., 1.], True)

    # Initial Config
    p.resetJointState(robot, joint_id["jb00"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["jb0"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["jb1"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["jp"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["jd"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["ja1"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["ja2"], -np.pi / 4, 0.)

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
    p.changeConstraint(c, gearRatio=-1, maxForce=10000, erp=2)

    pybullet_util.draw_link_frame(robot, link_id['lb00'], text="lb00")
    pybullet_util.draw_link_frame(robot, link_id['lb0'], text="lb0")
    pybullet_util.draw_link_frame(robot, link_id['lb1'], text="lb1")
    pybullet_util.draw_link_frame(robot, link_id['lp'], text="proximal")
    pybullet_util.draw_link_frame(robot, link_id['ld'], text="distal")
    pybullet_util.draw_link_frame(robot, link_id['la1'], text="la1")
    pybullet_util.draw_link_frame(robot, link_id['la2'], text="la2")
    pybullet_util.draw_link_frame(robot, link_id['ee'], text="ee")

    # Run Sim
    t = 0
    count = 0
    while (1):
        if count % 100:
            print("count : ", count)

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)
        q = np.array([
            sensor_data["joint_pos"]["jb00"],
            sensor_data["joint_pos"]["jb0"],
            sensor_data["joint_pos"]["jb1"],
            sensor_data["joint_pos"]["jp"],
            sensor_data["joint_pos"]["jd"],
            sensor_data["joint_pos"]["ja1"],
            sensor_data["joint_pos"]["ja2"],
        ])
        tau = PnCPjInvDyn(pin_robot, q, np.zeros(7), np.zeros(7))

        trq_cmd = dict()
        trq_cmd["jb00"] = tau[0]
        trq_cmd["jb0"] = tau[1]
        trq_cmd["jb1"] = tau[2]
        trq_cmd["jd"] = tau[3]
        trq_cmd["ja1"] = tau[4]
        trq_cmd["ja2"] = tau[5]
        pybullet_util.set_motor_trq(robot, joint_id, trq_cmd)

        p.stepSimulation()

        # time.sleep(DT)
        t += DT
        count += 1

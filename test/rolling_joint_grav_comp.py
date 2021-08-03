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

jac_i = np.array([[1, -1]])
s_a = np.array([[0, 1]])


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
    null_i = np.eye(2) - np.dot(jac_i_bar, jac_i)

    return mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i


def PnCPjFwdDyn(robot, q, qdot, tau):
    """
    q: np.array, qdot: np.array, tau: np.array
    """
    assert IsClose(q[0], q[1])
    assert IsClose(qdot[0], qdot[1])

    mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i = UpdateRobotSystem(
        robot, {
            'jp': q[0],
            'jd': q[1]
        }, {
            'jp': qdot[0],
            'jd': qdot[1]
        })

    qddot = np.dot(
        mass_inv,
        np.dot(np.dot(s_a, null_i).transpose(), tau) -
        np.dot(null_i.transpose(), (b + g)))

    assert IsClose(qddot[0], qddot[1])
    return qddot


def PnCPjInvDyn(robot, q, qdot, qddot):
    """
    q: np.array, qdot: np.array, qddot: np.array
    """
    assert IsClose(qddot[0], qddot[1])

    mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i = UpdateRobotSystem(
        robot, {
            'jp': q[0],
            'jd': q[1]
        }, {
            'jp': qdot[0],
            'jd': qdot[1]
        })

    tau = np.dot(np.linalg.pinv(np.dot(s_a, null_i).transpose()),
                 np.dot(mass, qddot) + np.dot(null_i.transpose(), b + g))

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
        '/home/apptronik/catkin_ws/src/draco_3/models/draco_3_model/urdf/rolling_joint.urdf',
        useFixedBase=True)

    # for i in range(3):
    # p.changeDynamics(0, i, localInertiaDiagonal=[1., 1., 1.])

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
    p.changeConstraint(c, gearRatio=-1, maxForce=10000, erp=2)

    pybullet_util.draw_link_frame(robot, link_id['lp'], text="proximal")
    pybullet_util.draw_link_frame(robot, link_id['ld'], text="distal")
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

        # Compute gravity compensation torque
        trq = PnCPjInvDyn(
            pin_robot,
            np.array([
                sensor_data["joint_pos"]["jp"], sensor_data["joint_pos"]["jd"]
            ]), np.zeros(2), np.zeros(2))

        p.setJointMotorControl2(robot,
                                joint_id['jd'],
                                controlMode=p.TORQUE_CONTROL,
                                force=trq)

        p.stepSimulation()

        # time.sleep(DT)
        t += DT
        count += 1

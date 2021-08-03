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

pin_robot = PinocchioRobotSystem(
    '/home/apptronik/Repository/PnC/robot_model/draco/draco_one_leg.urdf',
    '/home/apptronik/Repository/PnC/robot_model/draco', True, True)
l_jp_idx, l_jd_idx = pin_robot.get_q_dot_idx(['l_knee_fe_jp', 'l_knee_fe_jd'])
act_list = [False] * pin_robot.n_floating + [True] * pin_robot.n_a
act_list[l_jd_idx] = False
n_q_dot = len(act_list)
n_active = np.count_nonzero(np.array(act_list))
n_passive = n_q_dot - n_active - 6

s_a = np.zeros((n_active, n_q_dot))
j, k = 0, 0
for i in range(n_q_dot):
    if act_list[i]:
        s_a[j, i] = 1.
        j += 1

jac_i = np.zeros((1, n_q_dot))
jac_i[0, l_jp_idx] = 1.
jac_i[0, l_jd_idx] = -1.


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


def PnCPjGrav(robot, q):
    """
    q: np.array, qdot: np.array, qddot: np.array
    """
    qdot = dict()
    for k, v in q.items():
        qdot[k] = 0.

    mass, mass_inv, b, g, lambda_i, jac_i_bar, null_i = UpdateRobotSystem(
        robot, q, qdot)

    tau = np.dot(np.linalg.pinv(np.dot(s_a, null_i).transpose()),
                 np.dot(null_i.transpose(), g))
    tau_extended = np.dot(s_a.transpose(), tau)
    cmd = robot.create_cmd_ordered_dict(tau_extended, tau_extended,
                                        tau_extended)

    return cmd


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=0.25,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, -0.])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        '/home/apptronik/Repository/PnC/robot_model/draco/draco_one_leg.urdf',
        useFixedBase=True)

    # for i in range(3):
    # p.changeDynamics(0, i, localInertiaDiagonal=[1., 1., 1.])

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, [0., 0., 0.], [0., 0., 0., 1.], True)

    # Initial Config
    # Upperbody
    # p.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    # p.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    # p.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["l_wrist_ps"], np.pi / 6, 0.)
    # p.resetJointState(robot, joint_id["r_wrist_ps"], -np.pi / 6, 0.)

    # Lowerbody
    hip_yaw_angle = 5
    p.resetJointState(robot, joint_id["l_hip_aa"], np.radians(hip_yaw_angle),
                      0.)
    p.resetJointState(robot, joint_id["l_hip_fe"], -np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jp"], np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jd"], np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["l_ankle_fe"], -np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["l_ankle_ie"],
                      np.radians(-hip_yaw_angle), 0.)

    # p.resetJointState(robot, joint_id["r_hip_aa"], np.radians(-hip_yaw_angle),
    # 0.)
    # p.resetJointState(robot, joint_id["r_hip_fe"], -np.pi / 4, 0.)
    # p.resetJointState(robot, joint_id["r_knee_fe_jp"], np.pi / 4, 0.)
    # p.resetJointState(robot, joint_id["r_knee_fe_jd"], np.pi / 4, 0.)
    # p.resetJointState(robot, joint_id["r_ankle_fe"], -np.pi / 4, 0.)
    # p.resetJointState(robot, joint_id["r_ankle_ie"], np.radians(hip_yaw_angle),
    # 0.)

    # Link Damping
    # pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.)

    # Rolling Joint

    c = p.createConstraint(robot,
                           link_id['l_knee_fe_lp'],
                           robot,
                           link_id['l_knee_fe_ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=5)

    # c = p.createConstraint(robot,
    # link_id['r_knee_fe_lp'],
    # robot,
    # link_id['r_knee_fe_ld'],
    # jointType=p.JOINT_GEAR,
    # jointAxis=[0, 1, 0],
    # parentFramePosition=[0, 0, 0],
    # childFramePosition=[0, 0, 0])
    # p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=2)

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
        cmd = PnCPjGrav(pin_robot, sensor_data["joint_pos"])
        del cmd['joint_pos']['l_knee_fe_jd']
        # del cmd['joint_pos']['r_knee_fe_jd']
        del cmd['joint_vel']['l_knee_fe_jd']
        # del cmd['joint_vel']['r_knee_fe_jd']
        del cmd['joint_trq']['l_knee_fe_jd']
        # del cmd['joint_trq']['r_knee_fe_jd']

        pybullet_util.set_motor_trq(robot, joint_id, cmd["joint_trq"])

        p.stepSimulation()

        # time.sleep(DT)
        t += DT
        count += 1

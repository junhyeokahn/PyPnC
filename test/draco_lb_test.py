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
import numpy as np

np.set_printoptions(precision=5)

from util import util

if __name__ == "__main__":

    q = {
        'r_hip_ie': 0.1,
        'r_hip_aa': 0.3,
        'r_hip_fe': -0.2,
        'r_knee_fe_jp': 0.2,
        'r_knee_fe_jd': 0.2,
        'r_ankle_ie': -0.2,
        'r_ankle_fe': -0.3,
        'l_hip_ie': 0.1,
        'l_hip_aa': 0.3,
        'l_hip_fe': -0.2,
        'l_knee_fe_jp': 0.2,
        'l_knee_fe_jd': 0.2,
        'l_ankle_ie': -0.2,
        'l_ankle_fe': -0.3
    }

    qdot = {
        'r_hip_ie': 0.,
        'r_hip_aa': 0.,
        'r_hip_fe': -0.,
        'r_knee_fe_jp': 0.,
        'r_knee_fe_jd': 0.,
        'r_ankle_ie': -0.,
        'r_ankle_fe': -0.,
        'l_hip_ie': 0.,
        'l_hip_aa': 0.,
        'l_hip_fe': -0.,
        'l_knee_fe_jp': 0.,
        'l_knee_fe_jd': 0.,
        'l_ankle_ie': -0.,
        'l_ankle_fe': -0.
    }

    robot = PinocchioRobotSystem(
        '/home/apptronik/Repository/PnC/robot_model/draco/draco_lb.urdf',
        '/home/apptronik/Repository/PnC/robot_model/draco', True, True)

    j_i = np.zeros((2, robot.n_q_dot))
    l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = robot.get_q_dot_idx(
        ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])
    j_i[0, l_jp_idx] = 1.
    j_i[0, l_jd_idx] = -1.
    j_i[1, r_jp_idx] = 1.
    j_i[1, r_jd_idx] = -1.

    s_a = np.zeros((robot.n_a - 2, robot.n_a))
    act_list = [False] * robot.n_floating + [True] * robot.n_a
    act_list[l_jp_idx] = False
    act_list[r_jp_idx] = False
    j, k = 0, 0
    for i in range(robot.n_q_dot):
        if act_list[i]:
            s_a[j, i] = 1.
            j += 1

    # print(j_i)
    # print(s_a)
    # exit()

    robot.update_system(np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3),
                        np.zeros(3), np.zeros(4), np.zeros(3), np.zeros(3), q,
                        qdot, True)

    mass = robot.get_mass_matrix()
    mass_inv = np.linalg.inv(mass)
    g = robot.get_gravity()

    lambda_i = np.linalg.inv(np.dot(j_i, np.dot(mass_inv, j_i.transpose())))
    j_i_bar = np.dot(np.dot(mass_inv, j_i.transpose()), lambda_i)
    null_i = np.eye(robot.n_q_dot) - np.dot(j_i_bar, j_i)

    grav_comp_cmd = np.dot(
        np.dot(np.linalg.pinv(np.dot(s_a, null_i).transpose()),
               null_i.transpose()), g)

    grav_comp_cmd[3] /= 2.
    grav_comp_cmd[9] /= 2.

    print(grav_comp_cmd)

    iso = robot.get_link_iso('r_foot_contact')
    print("linear")
    print(iso[0:3, 0:3])
    print("translation")
    print(iso[0:3, 3])

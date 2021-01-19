import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
import math
import copy
from collections import OrderedDict

import pybullet as p
import numpy as np
# np.set_printoptions(precision=3)

from config.atlas_config import KinSimConfig
from util.util import *
from util.liegroup import *
from util.robot_kinematics import *


def get_robot_config(robot):
    nq, nv, na, joint_id, link_id = 0, 0, 0, OrderedDict(), OrderedDict()
    link_id[(p.getBodyInfo(robot)[0]).decode("utf-8")] = -1
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if info[2] != p.JOINT_FIXED:
            joint_id[info[1].decode("utf-8")] = info[0]
        link_id[info[12].decode("utf-8")] = info[0]
        nq = max(nq, info[3])
        nv = max(nv, info[4])
    nq += 1
    nv += 1
    na = len(joint_id)

    if KinSimConfig.PRINT_ROBOT_INFO:
        print("=" * 80)
        print("SimulationRobot")
        print("nq: ", nq, ", nv: ", nv, ", na: ", na)
        print("+" * 80)
        print("Joint Infos")
        pretty_print(joint_id)
        print("+" * 80)
        print("Link Infos")
        pretty_print(link_id)

    return nq, nv, na, joint_id, link_id


def set_initial_config(robot, joint_id):
    # shoulder_x
    p.resetJointState(robot, joint_id["l_arm_shx"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_arm_shx"], np.pi / 4, 0.)
    # elbow_y
    p.resetJointState(robot, joint_id["l_arm_ely"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_ely"], np.pi / 2, 0.)
    # elbow_x
    p.resetJointState(robot, joint_id["l_arm_elx"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_elx"], -np.pi / 2, 0.)
    # hip_y
    p.resetJointState(robot, joint_id["l_leg_hpy"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_hpy"], -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, joint_id["l_leg_kny"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_leg_kny"], np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, joint_id["l_leg_aky"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_aky"], -np.pi / 4, 0.)


def set_config(robot, joint_id, link_id, base_pos, base_quat, joint_pos):
    p.resetBasePositionAndOrientation(robot, base_pos, base_quat)
    for k, v in joint_pos.items():
        p.resetJointState(robot, joint_id[k], v, 0.)


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def get_sensor_data(robot, joint_id, link_id):
    """
    Parameters
    ----------
    joint_id (dict):
        Joint ID Dict
    Returns
    -------
    sensor_data (dict):
        base_pos (np.array):
            Base CoM Pos
        base_quat (np.array):
            Base CoM Quat
        joint_pos (dict):
            Joint pos
        joint_vel (dict):
            Joint vel
        b_rf_contact (bool):
            Right Foot Contact Switch
        b_lf_contact (bool):
            Left Foot Contact Switch
    """
    sensor_data = OrderedDict()

    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    sensor_data['base_pos'] = np.asarray(base_pos)
    sensor_data['base_quat'] = np.asarray(base_quat)

    base_lin_vel, base_ang_vel = p.getBaseVelocity(robot)
    sensor_data['base_lin_vel'] = np.asarray(base_lin_vel)
    sensor_data['base_ang_vel'] = np.asarray(base_ang_vel)

    sensor_data['joint_pos'] = OrderedDict()
    sensor_data['joint_vel'] = OrderedDict()
    for k, v in joint_id.items():
        js = p.getJointState(robot, v)
        sensor_data['joint_pos'][k] = js[0]
        sensor_data['joint_vel'][k] = js[1]

    rf_info = p.getLinkState(robot, link_id["r_sole"], False, True)
    rf_pos = rf_info[0]
    lf_info = p.getLinkState(robot, link_id["l_sole"], False, True)
    lf_pos = lf_info[0]

    # pretty_print(sensor_data['joint_pos'])
    # print('base_pos: ', sensor_data['base_pos'])
    # print('base_quat: ',sensor_data['base_quat'])
    # print('rf_pos: ', rf_pos)
    # print('lf_pos: ', lf_pos)
    # exit()

    return sensor_data


def get_qdot(robot, sensor_data):
    ## Angular first, Linear Later
    jvel = list(sensor_data['joint_vel'].values())
    ret = np.zeros(6 + len(jvel))
    ret[3:6] = np.copy(sensor_data['base_lin_vel'])
    ret[0:3] = np.copy(sensor_data['base_ang_vel'])
    ret[6:] = np.array(jvel)

    return ret


def get_jacobian(robot, link_idx, sensor_data):
    jpos = list(sensor_data['joint_pos'].values())
    jvel = list(sensor_data['joint_vel'].values())
    zeros = [0.] * len(jvel)
    jac = p.calculateJacobian(robot, link_idx, [0., 0., 0.], jpos, zeros,
                              zeros)
    nv = len(jac[0][0])
    ret = np.zeros((6, nv))
    for row in range(3):
        for col in range(nv):
            ret[row, col] = jac[1][row][col]  # angular
            ret[row + 3, col] = jac[0][row][col]  # linear

    return ret


def get_spatial_velocities(robot, link_idx):
    link_state = p.getLinkState(robot, link_idx, 1, 1)
    ret = np.zeros(6)
    for i in range(3):
        ret[i] = link_state[7][i]  # ang vel
        ret[i + 3] = link_state[6][i]  # lin vel

    return ret


def is_key_triggered(keys, key):
    o = ord(key)
    if o in keys:
        return keys[ord(key)] & p.KEY_WAS_TRIGGERED
    return False


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=KinSimConfig.DT, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        [0, 0, 1.5 - 0.761])

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Robot Configuration
    nq, nv, na, joint_id, link_id = get_robot_config(robot)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 2)

    # Run Sim
    t = 0
    dt = KinSimConfig.DT
    count = 0

    nominal_sensor_data = get_sensor_data(robot, joint_id, link_id)
    base_pos = np.copy(nominal_sensor_data['base_pos'])
    base_quat = np.copy(nominal_sensor_data['base_quat'])
    joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])
    while (1):

        # Get SensorData
        sensor_data = get_sensor_data(robot, joint_id, link_id)

        # Get Keyboard Event
        keys = p.getKeyboardEvents()

        # Reset base_pos, base_quat, joint_pos here for visualization
        if is_key_triggered(keys, '8'):
            pass
        elif is_key_triggered(keys, '5'):
            pass
        elif is_key_triggered(keys, '4'):
            pass
        elif is_key_triggered(keys, '2'):
            pass
        elif is_key_triggered(keys, '6'):
            pass
        elif is_key_triggered(keys, '7'):
            pass
        elif is_key_triggered(keys, '9'):
            pass
        elif is_key_triggered(keys, '1'):

            ## TEST
            r_sole_jac = get_jacobian(robot, link_id['r_sole'], sensor_data)
            r_sole_vel = get_spatial_velocities(robot, link_id['r_sole'])
            qdot = get_qdot(robot, sensor_data)
            # qdot = np.zeros(36)
            # qdot[3] = 1.
            print("-" * 80)
            print("r_sole_vel")
            print(r_sole_vel)
            print("jac times qdot")
            print(np.dot(r_sole_jac, qdot))
            ## TEST
            # torso_jac = get_jacobian(robot, -1, sensor_data)
            # print(torso_jac)

            p.stepSimulation()
            pass

        # Visualize config
        # set_config(robot, joint_id, link_id, base_pos, base_quat, joint_pos)

        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

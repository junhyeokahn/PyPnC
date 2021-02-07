import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
import math
import copy
from collections import OrderedDict

from ruamel.yaml import YAML
import pybullet as p
import numpy as np
np.set_printoptions(precision=3)

from util import pybullet_util
from util import util
from util import liegroup
from util import robot_kinematics

## Configs
DT = 0.01

PRINT_ROBOT_INFO = False
VIDEO_RECORD = False

INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 0.342]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

## Parse file
file = args.file
vis_idx = 0
if file is not None:
    with open(file, 'r') as stream:
        data = YAML().load(stream)
        vis_time = data["trajectory"]["time"]
        vis_base_lin = np.array(data["trajectory"]["base_lin"])
        vis_base_ang = np.array(data["trajectory"]["base_ang"])
        vis_ee_motion_lin = dict()
        vis_ee_motion_ang = dict()
        vis_ee_wrench_lin = dict()
        vis_ee_wrench_ang = dict()
        for ee in range(2):
            vis_ee_motion_lin[ee] = np.array(
                data["trajectory"]["ee_motion_lin"][ee])
            vis_ee_motion_ang[ee] = np.array(
                data["trajectory"]["ee_motion_ang"][ee])
            vis_ee_wrench_lin[ee] = np.array(
                data["trajectory"]["ee_wrench_lin"][ee])
            vis_ee_wrench_ang[ee] = np.array(
                data["trajectory"]["ee_wrench_ang"][ee])


def set_initial_config(robot, joint_id):
    p.resetJointState(robot, joint_id["LShoulderPitch"], 1.5, 0.)
    p.resetJointState(robot, joint_id["RShoulderPitch"], 1.5, 0.)
    p.resetJointState(robot, joint_id["LShoulderRoll"], 0.5, 0.)
    p.resetJointState(robot, joint_id["RShoulderRoll"], -0.5, 0.)

    p.resetJointState(robot, joint_id["LHipPitch"], -0.572, 0.)
    p.resetJointState(robot, joint_id["RHipPitch"], -0.572, 0.)
    p.resetJointState(robot, joint_id["LKneePitch"], 1.174, 0.)
    p.resetJointState(robot, joint_id["RKneePitch"], 1.174, 0.)
    p.resetJointState(robot, joint_id["LAnklePitch"], -0.602, 0.)
    p.resetJointState(robot, joint_id["RAnklePitch"], -0.602, 0.)


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, .5])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSubSteps=1)
    if VIDEO_RECORD:
        if not os.path.exists('video'):
            os.makedirs('video')
        for f in os.listdir('video'):
            if f == 'nao_kin.mp4':
                os.remove('video/' + f)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/nao_kin.mp4")

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/nao/nao.urdf",
                       INITIAL_POS_WORLD_TO_BASEJOINT,
                       INITIAL_QUAT_WORLD_TO_BASEJOINT)
    if file == "data/nao_block.yaml":
        block = p.loadURDF(cwd + "/robot_model/ground/block.urdf",
                           [0, 0, 0.15],
                           useFixedBase=True)
    elif file == "data/nao_stair.yaml":
        stair = p.loadURDF(cwd + "/robot_model/ground/stair.urdf",
                           [0.2, 0, 0.],
                           useFixedBase=True)
    elif file == "data/nao_slope.yaml":
        gap = p.loadURDF(cwd + "/robot_model/ground/slope.urdf",
                         [0.325, 0, -0.125],
                         useFixedBase=True)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Robot Configuration : 0 << Left Foot, 1 << Right Foot
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT)

    joint_screws_in_ee_at_home, ee_SE3_at_home = dict(), dict()
    open_chain_joints, base_link, ee_link = dict(), dict(), dict()
    base_link[0] = 'torso'
    ee_link[0] = 'l_sole'
    open_chain_joints[0] = [
        'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch',
        'LAnkleRoll'
    ]
    base_link[1] = 'torso'
    ee_link[1] = 'r_sole'
    open_chain_joints[1] = [
        'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch',
        'RAnkleRoll'
    ]

    for ee in range(2):
        joint_screws_in_ee_at_home[ee], ee_SE3_at_home[
            ee] = pybullet_util.get_kinematics_config(robot, joint_id, link_id,
                                                      open_chain_joints[ee],
                                                      base_link[ee],
                                                      ee_link[ee])

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    # Run Sim
    t = 0
    dt = DT
    count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)
    base_pos = np.copy(nominal_sensor_data['base_com_pos'])
    base_quat = np.copy(nominal_sensor_data['base_com_quat'])
    joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])
    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        # Parse data
        vis_t = vis_time[vis_idx]
        vis_base_lin_pos = vis_base_lin[vis_idx, 0:3]
        vis_base_ang_pos = vis_base_ang[vis_idx, 0:3]
        vis_ee_motion_lin_pos = dict()
        vis_ee_motion_ang_pos = dict()
        vis_ee_wrench_lin_pos = dict()
        vis_ee_wrench_ang_pos = dict()
        for ee in range(2):
            vis_ee_motion_lin_pos[ee] = vis_ee_motion_lin[ee][vis_idx, 0:3]
            vis_ee_motion_ang_pos[ee] = vis_ee_motion_ang[ee][vis_idx, 0:3]
            vis_ee_wrench_lin_pos[ee] = vis_ee_wrench_lin[ee][vis_idx, 0:3]
            vis_ee_wrench_ang_pos[ee] = vis_ee_wrench_ang[ee][vis_idx, 0:3]

        # Solve Inverse Kinematics
        base_pos = np.copy(vis_base_lin_pos)
        base_quat = np.copy(
            util.rot_to_quat(util.euler_to_rot(vis_base_ang_pos)))
        for ee in range(2):
            q_guess = np.array([
                nominal_sensor_data['joint_pos'][j_name]
                for j_name in open_chain_joints[ee]
            ])
            T_w_base = liegroup.RpToTrans(util.euler_to_rot(vis_base_ang_pos),
                                          vis_base_lin_pos)
            T_w_ee = liegroup.RpToTrans(
                util.euler_to_rot(vis_ee_motion_ang_pos[ee]),
                vis_ee_motion_lin_pos[ee])
            des_T = np.dot(liegroup.TransInv(T_w_base), T_w_ee)  # << T_base_ee
            q_sol, done = robot_kinematics.IKinBody(
                joint_screws_in_ee_at_home[ee], ee_SE3_at_home[ee], des_T,
                q_guess)
            for j_id, j_name in enumerate(open_chain_joints[ee]):
                joint_pos[j_name] = q_sol[j_id]

            if not done:
                print("====================================")
                print("Sovling inverse kinematics for ee-{} at time {}".format(
                    ee, vis_t))
                print("success: {}".format(done))
                print("T_w_base")
                print(T_w_base)
                print("T_w_ee")
                print(T_w_ee)
                print("q_guess")
                print(q_guess)
                print("q_sol")
                print(q_sol)
                __import__('ipdb').set_trace()
                print("====================================")

        # Handle timings
        if vis_idx == len(vis_time) - 1:
            vis_idx = 0
        else:
            vis_idx += 1

        # Visualize config
        pybullet_util.set_config(robot, joint_id, link_id, base_pos, base_quat,
                                 joint_pos)

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

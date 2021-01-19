import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
import math
import copy
from collections import OrderedDict

import yaml
import pybullet as p
import numpy as np
# np.set_printoptions(precision=3)

from config.atlas_config import KinSimConfig
from util.util import *
from util.liegroup import *
from util.robot_kinematics import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str)
args = parser.parse_args()

## Parse file if exist
file = args.file
b_has_file = False
b_auto_progress = False
vis_idx = 0
if file is not None:
    with open(file, 'r') as stream:
        try:
            data = yaml.load(stream, Loader=yaml.FullLoader)
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
        except yaml.YAMLError as exc:
            print(exc)
    b_has_file = True


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
    if KinSimConfig.VIDEO_RECORD:
        if not os.path.exists('video'):
            os.makedirs('video')
        for f in os.listdir('video'):
            os.remove('video/' + f)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/atlas_kin.mp4")

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        [0, 0, 1.5 - 0.761])

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Robot Configuration : 0 << Left Foot, 1 << Right Foot
    nq, nv, na, joint_id, link_id = get_robot_config(robot)
    joint_screws_in_ee_at_home, ee_SE3_at_home = dict(), dict()
    open_chain_joints, base_link, ee_link = dict(), dict(), dict()
    base_link[0] = 'pelvis'
    ee_link[0] = 'l_sole'
    open_chain_joints[0] = [
        'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky',
        'l_leg_akx'
    ]
    base_link[1] = 'pelvis'
    ee_link[1] = 'r_sole'
    open_chain_joints[1] = [
        'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',
        'r_leg_akx'
    ]

    for ee in range(2):
        joint_screws_in_ee_at_home[ee], ee_SE3_at_home[
            ee] = get_kinematics_config(robot, joint_id, link_id,
                                        open_chain_joints[ee], base_link[ee],
                                        ee_link[ee])

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 2)

    # RobotSystem
    # if KinSimConfig.DYN_LIB == "dart":
    # from pnc.robot_system.dart_robot_system import DartRobotSystem
    # robot_system = DartRobotSystem(
    # cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
    # ['rootJoint'], PnCConfig.PRINT_ROBOT_INFO)
    # else:
    # raise ValueError

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
        # Update RobotSystem
        # robot_system.update_system(sensor_data["base_pos"],
        # sensor_data["base_quat"],
        # sensor_data["base_lin_vel"],
        # sensor_data["base_ang_vel"],
        # sensor_data["joint_pos"],
        # sensor_data["joint_vel"],
        # b_cent=True)

        # Get Keyboard Event
        keys = p.getKeyboardEvents()

        # Set base_pos, base_quat, joint_pos here for visualization
        if is_key_triggered(keys, '8'):
            pass
        elif is_key_triggered(keys, '5'):
            pass
        elif is_key_triggered(keys, '4'):
            pass
        elif is_key_triggered(keys, '0'):
            if b_auto_progress:
                print("Pressed 0: Trun Off Auto-Progression")
                b_auto_progress = False
            else:
                print("Pressed 0: Turn on Auto-Progression")
                b_auto_progress = True
        elif is_key_triggered(keys, '2') or b_auto_progress:
            if not b_auto_progress:
                print("Pressed 2: Progress Trajectory Visualization")
            assert (b_has_file)

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
            base_quat = np.copy(rot_to_quat(euler_to_rot(vis_base_ang_pos)))
            for ee in range(2):
                q_guess = np.array([
                    nominal_sensor_data['joint_pos'][j_name]
                    for j_name in open_chain_joints[ee]
                ])
                T_w_base = RpToTrans(euler_to_rot(vis_base_ang_pos),
                                     vis_base_lin_pos)
                T_w_ee = RpToTrans(euler_to_rot(vis_ee_motion_ang_pos[ee]),
                                   vis_ee_motion_lin_pos[ee])
                des_T = np.dot(TransInv(T_w_base), T_w_ee)  # << T_base_ee
                q_sol, done = IKinBody(joint_screws_in_ee_at_home[ee],
                                       ee_SE3_at_home[ee], des_T, q_guess)
                for j_id, j_name in enumerate(open_chain_joints[ee]):
                    joint_pos[j_name] = q_sol[j_id]

                if (not done) or (not b_auto_progress):
                    print("====================================")
                    print("Sovling inverse kinematics for ee-{} at time {}".
                          format(ee, vis_t))
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

        elif is_key_triggered(keys, '6'):
            pass
        elif is_key_triggered(keys, '7'):
            pass
        elif is_key_triggered(keys, '9'):
            pass
        elif is_key_triggered(keys, '1'):
            # Nominal Pos
            print("Pressed 1: Reset to Nominal Pos")
            base_pos = np.copy(nominal_sensor_data['base_pos'])
            base_quat = np.copy(nominal_sensor_data['base_quat'])
            joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])

        # Visualize config
        set_config(robot, joint_id, link_id, base_pos, base_quat, joint_pos)

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

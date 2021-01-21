import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np

from config.atlas_config import DynSimConfig
from util import util
from util import liegroup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dyn_lib", default=None, type=str)
args = parser.parse_args()


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

    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    rot_world_com = util.quat_to_rot(base_quat)
    rot_world_basejoint = util.quat_to_rot(
        np.array(DynSimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT))
    pos_basejoint_to_basecom = base_pos - np.array(
        DynSimConfig.INITIAL_POS_WORLD_TO_BASEJOINT)
    rot_basejoint_to_basecom = np.dot(rot_world_basejoint.transpose(),
                                      rot_world_com)

    if DynSimConfig.PRINT_ROBOT_INFO:
        print("=" * 80)
        print("SimulationRobot")
        print("nq: ", nq, ", nv: ", nv, ", na: ", na)
        print("Vector from base joint frame to base com frame")
        print(pos_basejoint_to_basecom)
        print("Rotation from base joint frame to base com frame")
        print(rot_basejoint_to_basecom)
        print("+" * 80)
        print("Joint Infos")
        util.pretty_print(joint_id)
        print("+" * 80)
        print("Link Infos")
        util.pretty_print(link_id)

    return nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom


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


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def set_motor_trq(robot, joint_id, command):
    assert len(joint_id) == len(command['joint_trq'])
    trq_applied = OrderedDict()
    for (joint_name, pos_des), (_, vel_des), (_, trq_des) in zip(
            command['joint_pos'].items(), command['joint_vel'].items(),
            command['joint_trq'].items()):
        joint_state = p.getJointState(robot, joint_id[joint_name])
        joint_pos, joint_vel = joint_state[0], joint_state[1]
        trq_applied[joint_id[joint_name]] = (trq_des + DynSimConfig.KP *
                                             (pos_des - joint_pos) +
                                             DynSimConfig.KD *
                                             (vel_des - joint_vel))
    p.setJointMotorControlArray(robot,
                                trq_applied.keys(),
                                controlMode=p.TORQUE_CONTROL,
                                forces=trq_applied.values())


def get_sensor_data(robot, joint_id, link_id, pos_basejoint_to_basecom,
                    rot_basejoint_to_basecom):
    """
    Parameters
    ----------
    joint_id (dict):
        Joint ID Dict
    link_id (dict):
        Link ID Dict
    pos_basejoint_to_basecom (np.ndarray):
        3d vector from base joint frame to base com frame
    rot_basejoint_to_basecom (np.ndarray):
        SO(3) from base joint frame to base com frame
    Returns
    -------
    sensor_data (dict):
        base_com_pos (np.array):
            base com pos in world
        base_com_quat (np.array):
            base com quat in world
        base_com_lin_vel (np.array):
            base com lin vel in world
        base_com_ang_vel (np.array):
            base com ang vel in world
        base_joint_pos (np.array):
            base pos in world
        base_joint_quat (np.array):
            base quat in world
        base_joint_lin_vel (np.array):
            base lin vel in world
        base_joint_ang_vel (np.array):
            base ang vel in world
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

    # Handle Base Frame Quantities
    base_com_pos, base_com_quat = p.getBasePositionAndOrientation(robot)
    sensor_data['base_com_pos'] = np.asarray(base_com_pos)
    sensor_data['base_com_quat'] = np.asarray(base_com_quat)

    base_com_lin_vel, base_com_ang_vel = p.getBaseVelocity(robot)
    sensor_data['base_com_lin_vel'] = np.asarray(base_com_lin_vel)
    sensor_data['base_com_ang_vel'] = np.asarray(base_com_ang_vel)

    rot_world_com = util.quat_to_rot(np.copy(sensor_data['base_com_quat']))
    rot_world_joint = np.dot(rot_world_com,
                             rot_basejoint_to_basecom.transpose())
    sensor_data['base_joint_pos'] = sensor_data['base_com_pos'] - np.dot(
        rot_world_joint, pos_basejoint_to_basecom)
    sensor_data['base_joint_quat'] = util.rot_to_quat(rot_world_joint)
    trans_joint_com = liegroup.RpToTrans(rot_basejoint_to_basecom,
                                         pos_basejoint_to_basecom)
    adT_joint_com = liegroup.Adjoint(trans_joint_com)
    twist_com_in_world = np.zeros(6)
    twist_com_in_world[0:3] = np.copy(sensor_data['base_com_ang_vel'])
    twist_com_in_world[3:6] = np.copy(sensor_data['base_com_lin_vel'])
    augrot_com_world = np.zeros((6, 6))
    augrot_com_world[0:3, 0:3] = rot_world_com.transpose()
    augrot_com_world[3:6, 3:6] = rot_world_com.transpose()
    twist_com_in_com = np.dot(augrot_com_world, twist_com_in_world)
    twist_joint_in_joint = np.dot(adT_joint_com, twist_com_in_com)
    rot_world_joint = np.dot(rot_world_com,
                             rot_basejoint_to_basecom.transpose())
    augrot_world_joint = np.zeros((6, 6))
    augrot_world_joint[0:3, 0:3] = rot_world_joint
    augrot_world_joint[3:6, 3:6] = rot_world_joint
    twist_joint_in_world = np.dot(augrot_world_joint, twist_joint_in_joint)
    sensor_data['base_joint_lin_vel'] = np.copy(twist_joint_in_world[3:6])
    sensor_data['base_joint_ang_vel'] = np.copy(twist_joint_in_world[0:3])

    # Joint Quantities
    sensor_data['joint_pos'] = OrderedDict()
    sensor_data['joint_vel'] = OrderedDict()
    for k, v in joint_id.items():
        js = p.getJointState(robot, v)
        sensor_data['joint_pos'][k] = js[0]
        sensor_data['joint_vel'][k] = js[1]

    # Contact Sensing
    rf_info = p.getLinkState(robot, link_id["r_sole"])
    rf_pos = rf_info[0]
    if rf_pos[2] <= 0.01:
        sensor_data["b_rf_contact"] = True
    else:
        sensor_data["b_rf_contact"] = False

    lf_info = p.getLinkState(robot, link_id["l_sole"])
    lf_pos = lf_info[0]
    if lf_pos[2] <= 0.01:
        sensor_data["b_lf_contact"] = True
    else:
        sensor_data["b_lf_contact"] = False

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
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=DynSimConfig.CONTROLLER_DT,
                                numSubSteps=DynSimConfig.N_SUBSTEP)
    if DynSimConfig.VIDEO_RECORD:
        if not os.path.exists('video'):
            os.makedirs('video')
        for f in os.listdir('video'):
            os.remove('video/' + f)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/atlas.mp4")

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        DynSimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        DynSimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = get_robot_config(
        robot)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 2)

    # RobotSystem
    if args.dyn_lib == 'dart':
        from pnc.robot_system.dart_robot_system import DartRobotSystem
        robot_sys = DartRobotSystem(
            cwd +
            "/robot_model/atlas/atlas_v4_with_multisense_relative_path.urdf",
            False, True)
    elif args.dyn_lib == 'pinocchio':
        from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
        robot_sys = PinocchioRobotSystem(
            cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
            cwd + "/robot_model/atlas", False, True)
    else:
        raise ValueError

    # Run Sim
    t = 0
    dt = DynSimConfig.CONTROLLER_DT
    count = 0
    while (1):

        # Get SensorData
        sensor_data = get_sensor_data(robot, joint_id, link_id,
                                      pos_basejoint_to_basecom,
                                      rot_basejoint_to_basecom)

        # Update Robot
        robot_sys.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        ## TEST
        target_link = 'r_sole'
        ls = p.getLinkState(robot, link_id[target_link], 1, 1)
        print("Link Pos from PyBullet")
        print(np.array(ls[0]))
        print("Link Rot from PyBullet")
        print(util.quat_to_rot(np.array(ls[1])))

        pnc_iso = robot_sys.get_link_iso(target_link)
        print("Link Pos from PnC")
        print(pnc_iso[0:3, 3])
        print("Link Rot from PnC")
        print(pnc_iso[0:3, 0:3])

        print("=" * 80)
        ## TEST

        # p.stepSimulation()
        keys = p.getKeyboardEvents()
        if is_key_triggered(keys, '1'):
            p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

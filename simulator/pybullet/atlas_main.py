import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import util


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

    # print("=" * 80)
    # print("SimulationRobot")
    # print("nq: ", nq, ", nv: ", nv, ", na: ", na)
    # print("+" * 80)
    # print("Joint Infos")
    # util.pretty_print(joint_id)
    # print("+" * 80)
    # print("Link Infos")
    # util.pretty_print(link_id)

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


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def set_motor_trq(robot, joint_id, command):
    assert len(joint_id) == len(command['joint_trq'])

    for (joint_name, pos_des), (_, vel_des), (_, trq_des) in zip(
            command['joint_pos'].items(), command['joint_vel'].items(),
            command['joint_trq'].items()):
        joint_state = p.getJointState(robot, joint_id[joint_name])
        joint_pos, joint_vel = joint_state[0], joint_state[1]
        trq_applied = trq_des + SimConfig.KP * (
            pos_des - joint_pos) + SimConfig.KD * (vel_des - joint_vel)
        p.setJointMotorControl2(robot, joint_id[joint_name], p.TORQUE_CONTROL,
                                trq_applied)


def debug(robot, joint_id, link_id):
    print("-" * 80)
    print("Sim")
    print("-" * 80)
    for (k, v) in link_id.items():
        if k != "pelvis":
            link_info = p.getLinkState(robot, v, 1, 1)
            print(k, np.asarray(link_info[0]), np.asarray(link_info[1]),
                  np.asarray(link_info[6]), np.asarray(link_info[7]))


def get_sensor_data(robot, joint_id):
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

    # util.pretty_print(sensor_data['joint_pos'])
    # print(sensor_data['base_pos'])
    # print(sensor_data['base_quat'])

    return sensor_data


def get_camera_image(robot, link_id, projection_matrix):
    link_info = p.getLinkState(robot, link_id['head'])  #Get head link info
    link_pos = link_info[0]  #Get link com pos wrt world
    link_ori = link_info[1]  #Get link com ori wrt world
    rot = p.getMatrixFromQuaternion(link_ori)
    rot = np.array(rot).reshape(3, 3)

    global_camera_x_unit = np.array([1, 0, 0])
    global_camera_z_unit = np.array([0, 0, 1])

    camera_eye_pos = link_pos + np.dot(rot, 0.1 * global_camera_x_unit)
    camera_target_pos = link_pos + np.dot(rot, 1.0 * global_camera_x_unit)
    camera_up_vector = np.dot(rot, global_camera_z_unit)
    view_matrix = p.computeViewMatrix(camera_eye_pos, camera_target_pos,
                                      camera_up_vector)
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        50,  #image width
        10,  #image height
        view_matrix,
        projection_matrix)
    return width, height, rgb_img, depth_img, seg_img


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        [0, 0, 1.5 - 0.761])

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id = get_robot_config(robot)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 1)

    #camera intrinsic parameter
    fov, aspect, nearval, farval = 60.0, 2.0, 0.1, 10
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearval,
                                                     farval)

    # Construct Interface
    interface = AtlasInterface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0
    while (1):

        if count % (SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
            camera_img = get_camera_image(robot, link_id, projection_matrix)

        sensor_data = get_sensor_data(robot, joint_id)
        # debug(robot, joint_id, link_id)
        command = interface.get_command(sensor_data)

        set_motor_trq(robot, joint_id, command)

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np
# np.set_printoptions(precision=2, threshold=sys.maxsize)

from config.atlas_config import DynSimConfig
from util import pybullet_util
from util import util
from util import liegroup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dyn_lib", default=None, type=str)
args = parser.parse_args()


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
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, DynSimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        DynSimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT,
        DynSimConfig.PRINT_ROBOT_INFO)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 2)

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
    step = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)
    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)
        # Update Robot
        # for k, v in sensor_data['joint_pos'].items():
        # sensor_data['joint_pos'][k] = 0.2
        # sensor_data['joint_vel'][k] = -0.2
        robot_sys.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"], True)

        print("=" * 80)
        print("Step : ", step)
        print("-" * 80)
        target_link = 'r_sole'
        ls = p.getLinkState(robot, link_id[target_link], 1, 1)
        pb_lin_vel = np.array(ls[6])
        pb_ang_vel = np.array(ls[7])
        pb_spat_vel = np.concatenate([pb_ang_vel, pb_lin_vel], axis=0)
        print("Link Pos from PyBullet")
        print(np.array(ls[0]))
        print("Link Rot from PyBullet")
        print(util.quat_to_rot(np.array(ls[1])))
        print("Link Vel from PyBullet")
        print(pb_spat_vel)

        pnc_iso = robot_sys.get_link_iso(target_link)
        pnc_spat_vel = robot_sys.get_link_vel(target_link)
        pnc_qdot = robot_sys.get_q_dot()
        pnc_jac = robot_sys.get_link_jacobian(target_link)
        pnc_jacdot = robot_sys.get_link_jacobian_dot(target_link)
        pnc_com = robot_sys.get_com_pos()
        pnc_com_vel = robot_sys.get_com_lin_vel()
        pnc_com_jac = robot_sys.get_com_lin_jacobian()
        pnc_com_jac_dot = robot_sys.get_com_lin_jacobian_dot()
        pnc_grav = robot_sys.get_gravity()
        pnc_cor = robot_sys.get_coriolis()
        pnc_mass = robot_sys.get_mass_matrix()
        print("Link Pos from PnC")
        print(pnc_iso[0:3, 3])
        print("Link Rot from PnC")
        print(pnc_iso[0:3, 0:3])
        print("Link Vel from PnC")
        print(pnc_spat_vel)
        print("Link Vel from J * qdot")
        print(np.dot(pnc_jac, pnc_qdot))

        print("Jdot * qdot")
        print(np.dot(pnc_jacdot, pnc_qdot))
        print("com pos")
        print(pnc_com)
        print("com vel")
        print(pnc_com_vel)
        print("com jac*qdot")
        print(np.dot(pnc_com_jac, pnc_qdot))
        print("com jacdot*qdot")
        print(np.dot(pnc_com_jac_dot, pnc_qdot))
        print("grav")
        print(pnc_grav)
        print("cori")
        print(pnc_cor)
        print("total mass")
        print(robot_sys.total_mass)
        print("hg")
        print(robot_sys.hg)
        print("Ag*qdot")
        print(np.dot(robot_sys.Ag, pnc_qdot))
        print("Ig")
        print(robot_sys.Ig)

        print("-" * 80)

        p.stepSimulation()

        __import__('ipdb').set_trace()

        keys = p.getKeyboardEvents()

        time.sleep(dt)
        t += dt
        count += 1

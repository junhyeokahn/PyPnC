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
np.set_printoptions(precision=3)
from tqdm import tqdm

from util import pybullet_util
from util import util
from util import liegroup
from util import robot_kinematics

## Configs
DT = 0.01
PRINT_ROBOT_INFO = False
INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.761]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

## Motion Boundaries
RFOOT_POS_LB = np.array([-0.15, -0.1, -0.05])
RFOOT_POS_UB = np.array([0.15, 0.05, 0.05])
LFOOT_POS_LB = np.array([-0.15, -0.05, -0.05])
LFOOT_POS_UB = np.array([0.15, 0.1, 0.05])

FOOT_EA_LB = np.array([np.deg2rad(-5.), np.deg2rad(-15.), -np.pi / 6.])
FOOT_EA_UB = np.array([np.deg2rad(5.), np.deg2rad(15.), np.pi / 6.])

SWING_HEIGHT_LB, SWING_HEIGHT_UB = 0.03, 0.20

SWING_TIME_LB, SWING_TIME_UB = 0.35, 0.75

BASE_HEIGHT_LB, BASE_HEIGHT_UB = 0.7, 0.8


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


def ik_feet(base_pos, base_quat, lf_pos, lf_quat, rf_pos, rf_quat,
            nominal_sensor_data, joint_screws_in_ee_at_home, ee_SE3_at_home,
            open_chain_joints):
    joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])
    T_w_base = liegroup.RpToTrans(util.quat_to_rot(base_quat), base_pos)

    # left foot
    lf_q_guess = np.array([
        nominal_sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints[0]
    ])
    T_w_lf = liegroup.RpToTrans(util.quat_to_rot(lf_quat), lf_pos)
    T_base_lf = np.dot(liegroup.TransInv(T_w_base), T_w_lf)
    lf_q_sol, lf_done = robot_kinematics.IKinBody(
        joint_screws_in_ee_at_home[0], ee_SE3_at_home[0], T_base_lf,
        lf_q_guess)
    for j_id, j_name in enumerate(open_chain_joints[0]):
        joint_pos[j_name] = lf_q_sol[j_id]

    # right foot
    rf_q_guess = np.array([
        nominal_sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints[1]
    ])
    T_w_rf = liegroup.RpToTrans(util.quat_to_rot(rf_quat), rf_pos)
    T_base_rf = np.dot(liegroup.TransInv(T_w_base), T_w_rf)
    rf_q_sol, rf_done = robot_kinematics.IKinBody(
        joint_screws_in_ee_at_home[1], ee_SE3_at_home[1], T_base_rf,
        rf_q_guess)
    for j_id, j_name in enumerate(open_chain_joints[1]):
        joint_pos[j_name] = rf_q_sol[j_id]

    return joint_pos, lf_done, rf_done


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Robot Configuration : 0 << Left Foot, 1 << Right Foot
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT)

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
    nominal_lf_iso = pybullet_util.get_link_iso(robot, link_id['l_sole'])
    nominal_rf_iso = pybullet_util.get_link_iso(robot, link_id['r_sole'])
    base_pos = np.copy(nominal_sensor_data['base_com_pos'])
    base_quat = np.copy(nominal_sensor_data['base_com_quat'])
    joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])
    s = 0.
    b_ik = False
    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        # Get Keyboard Event
        keys = p.getKeyboardEvents()

        # Set base_pos, base_quat, joint_pos here for visualization
        if pybullet_util.is_key_triggered(keys, '8'):
            pass
        elif pybullet_util.is_key_triggered(keys, '5'):
            pass
        elif pybullet_util.is_key_triggered(keys, '4'):
            pass
        elif pybullet_util.is_key_triggered(keys, '3'):
            # Left Foot Swing, Right Foot Stance
            print("-" * 80)
            print("Pressed 3: Sample Motion for Left Foot Swing")

            # Sample swing time
            swing_time = np.random.uniform(SWING_TIME_LB, SWING_TIME_UB)

            # Sample rfoot config
            rfoot_ini_iso = np.copy(nominal_rf_iso)
            rfoot_mid_iso = np.copy(nominal_rf_iso)
            rfoot_fin_iso = np.copy(nominal_rf_iso)

            rfoot_mid_vel = (rfoot_ini_iso[0:3, 3] -
                             rfoot_ini_iso[0:3, 3]) / swing_time
            rfoot_mid_vel[2] = 0.

            # Sample lfoot config
            lfoot_ini_pos = np.copy(nominal_lf_iso)[
                0:3, 3] + np.random.uniform(LFOOT_POS_LB, LFOOT_POS_UB)
            lfoot_ini_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
            lfoot_ini_rot = util.euler_to_rot(lfoot_ini_ea)
            lfoot_ini_iso = liegroup.RpToTrans(lfoot_ini_rot, lfoot_ini_pos)
            lfoot_fin_pos = np.copy(nominal_lf_iso)[
                0:3, 3] + np.random.uniform(LFOOT_POS_LB, LFOOT_POS_UB)
            lfoot_fin_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
            lfoot_fin_rot = util.euler_to_rot(lfoot_fin_ea)
            lfoot_fin_iso = liegroup.RpToTrans(lfoot_fin_rot, lfoot_fin_pos)
            lfoot_mid_iso = util.iso_interpolate(lfoot_ini_iso, lfoot_fin_iso,
                                                 0.5)
            lfoot_mid_iso[2, 3] = (lfoot_ini_pos[2] +
                                   lfoot_fin_pos[2]) / 2.0 + np.random.uniform(
                                       SWING_HEIGHT_LB, SWING_HEIGHT_UB)

            lfoot_mid_vel = (lfoot_fin_pos - lfoot_ini_pos) / swing_time
            lfoot_mid_vel[2] = 0.

            # Sample base config
            base_ini_pos = (rfoot_ini_iso[0:3, 3] +
                            lfoot_ini_iso[0:3, 3]) / 2.0
            base_ini_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
            base_ini_rot = util.euler_to_rot(
                np.array([0., 0., lfoot_ini_ea[2] / 2.]))
            base_ini_iso = liegroup.RpToTrans(base_ini_rot, base_ini_pos)
            base_fin_pos = (rfoot_fin_iso[0:3, 3] +
                            lfoot_fin_iso[0:3, 3]) / 2.0
            base_fin_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
            base_fin_rot = util.euler_to_rot(
                np.array([0., 0., lfoot_fin_ea[2] / 2.]))
            base_fin_iso = liegroup.RpToTrans(base_fin_rot, base_fin_pos)

            s = 0.  # << Reset progression variable
            b_ik = True  # << IK flag

            # print("Base Ini Config")
            # print(base_ini_iso)
            # print("Base Fin Config")
            # print(base_fin_iso)
            # print("Left Foot Config")
            # print(lfoot_ini_iso)
            # print("Right Foot Ini Config")
            # print(rfoot_ini_iso)
            # print("Right Foot Mid Config")
            # print(rfoot_mid_iso)
            # print("Right Foot Fin Config")
            # print(rfoot_fin_iso)

        elif pybullet_util.is_key_triggered(keys, '2'):
            # Left Foot Stance, Right Foot Swing
            print("-" * 80)
            print("Pressed 2: Sample Motion for Right Foot Swing")

            # Sample swing time
            swing_time = np.random.uniform(SWING_TIME_LB, SWING_TIME_UB)

            # Sample lfoot config
            lfoot_ini_iso = np.copy(nominal_lf_iso)
            lfoot_mid_iso = np.copy(nominal_lf_iso)
            lfoot_fin_iso = np.copy(nominal_lf_iso)

            lfoot_mid_vel = (lfoot_ini_iso[0:3, 3] -
                             lfoot_ini_iso[0:3, 3]) / swing_time
            lfoot_mid_vel[2] = 0.

            # Sample rfoot config
            rfoot_ini_pos = np.copy(nominal_rf_iso)[
                0:3, 3] + np.random.uniform(RFOOT_POS_LB, RFOOT_POS_UB)
            rfoot_ini_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
            rfoot_ini_rot = util.euler_to_rot(rfoot_ini_ea)
            rfoot_ini_iso = liegroup.RpToTrans(rfoot_ini_rot, rfoot_ini_pos)
            rfoot_fin_pos = np.copy(nominal_rf_iso)[
                0:3, 3] + np.random.uniform(RFOOT_POS_LB, RFOOT_POS_UB)
            rfoot_fin_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
            rfoot_fin_rot = util.euler_to_rot(rfoot_fin_ea)
            rfoot_fin_iso = liegroup.RpToTrans(rfoot_fin_rot, rfoot_fin_pos)
            rfoot_mid_iso = util.iso_interpolate(rfoot_ini_iso, rfoot_fin_iso,
                                                 0.5)
            rfoot_mid_iso[2, 3] = (rfoot_ini_pos[2] +
                                   rfoot_fin_pos[2]) / 2.0 + np.random.uniform(
                                       SWING_HEIGHT_LB, SWING_HEIGHT_UB)

            rfoot_mid_vel = (rfoot_fin_pos - rfoot_ini_pos) / swing_time
            rfoot_mid_vel[2] = 0.

            # Sample base config
            base_ini_pos = (rfoot_ini_pos + lfoot_ini_iso[0:3, 3]) / 2.0
            base_ini_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
            base_ini_rot = util.euler_to_rot(
                np.array([0., 0., rfoot_ini_ea[2] / 2.]))
            base_ini_iso = liegroup.RpToTrans(base_ini_rot, base_ini_pos)
            base_fin_pos = (rfoot_fin_pos + lfoot_fin_iso[0:3, 3]) / 2.0
            base_fin_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
            base_fin_rot = util.euler_to_rot(
                np.array([0., 0., rfoot_fin_ea[2] / 2.]))
            base_fin_iso = liegroup.RpToTrans(base_fin_rot, base_fin_pos)

            s = 0.  # << Reset progression variable
            b_ik = True  # << IK flag

            # print("Base Ini Config")
            # print(base_ini_iso)
            # print("Base Fin Config")
            # print(base_fin_iso)
            # print("Left Foot Config")
            # print(lfoot_ini_iso)
            # print("Right Foot Ini Config")
            # print(rfoot_ini_iso)
            # print("Right Foot Mid Config")
            # print(rfoot_mid_iso)
            # print("Right Foot Fin Config")
            # print(rfoot_fin_iso)

        elif pybullet_util.is_key_triggered(keys, '6'):
            pass
        elif pybullet_util.is_key_triggered(keys, '7'):
            pass
        elif pybullet_util.is_key_triggered(keys, '9'):
            pass
        elif pybullet_util.is_key_triggered(keys, '1'):
            # Nominal Pos
            print("-" * 80)
            print("Pressed 1: Reset to Nominal Pos")
            base_pos = np.copy(nominal_sensor_data['base_com_pos'])
            base_quat = np.copy(nominal_sensor_data['base_com_quat'])
            joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])

        # Solve IK if needed to define base_pos, base_quat, joint_pos
        if b_ik:
            if s == 0.:
                # Create trajectories
                lfoot_pos_curve_ini_to_mid = util.HermiteCurveVec(
                    lfoot_ini_iso[0:3, 3], np.zeros(3), lfoot_mid_iso[0:3, 3],
                    lfoot_mid_vel)
                lfoot_pos_curve_mid_to_fin = util.HermiteCurveVec(
                    lfoot_mid_iso[0:3, 3], lfoot_mid_vel, lfoot_fin_iso[0:3,
                                                                        3],
                    np.zeros(3))
                lfoot_quat_curve = util.HermiteCurveQuat(
                    util.rot_to_quat(lfoot_ini_iso[0:3, 0:3]), np.zeros(3),
                    util.rot_to_quat(lfoot_fin_iso[0:3, 0:3]), np.zeros(3))
                rfoot_pos_curve_ini_to_mid = util.HermiteCurveVec(
                    rfoot_ini_iso[0:3, 3], np.zeros(3), rfoot_mid_iso[0:3, 3],
                    rfoot_mid_vel)
                rfoot_pos_curve_mid_to_fin = util.HermiteCurveVec(
                    rfoot_mid_iso[0:3, 3], rfoot_mid_vel, rfoot_fin_iso[0:3,
                                                                        3],
                    np.zeros(3))
                rfoot_quat_curve = util.HermiteCurveQuat(
                    util.rot_to_quat(rfoot_ini_iso[0:3, 0:3]), np.zeros(3),
                    util.rot_to_quat(rfoot_fin_iso[0:3, 0:3]), np.zeros(3))
                base_pos_curve = util.HermiteCurveVec(base_ini_iso[0:3, 3],
                                                      np.zeros(3),
                                                      base_fin_iso[0:3, 3],
                                                      np.zeros(3))
                base_quat_curve = util.HermiteCurveQuat(
                    util.rot_to_quat(base_ini_iso[0:3, 0:3]), np.zeros(3),
                    util.rot_to_quat(base_fin_iso[0:3, 0:3]), np.zeros(3))

                pbar = tqdm(total=math.floor(swing_time / DT))
            if s >= 1.:
                # Done
                b_ik = False
                s = 0
                pbar.close()
            else:
                pbar.update(1)
                # Solve IK and set base_pos, base_quat, joint_pos
                base_pos = base_pos_curve.evaluate(s)
                base_quat = base_quat_curve.evaluate(s)

                if s <= 0.5:
                    sprime = 2.0 * s
                    lf_pos = lfoot_pos_curve_ini_to_mid.evaluate(sprime)
                    rf_pos = rfoot_pos_curve_ini_to_mid.evaluate(sprime)
                else:
                    sprime = 2.0 * (s - 0.5)
                    lf_pos = lfoot_pos_curve_mid_to_fin.evaluate(sprime)
                    rf_pos = rfoot_pos_curve_mid_to_fin.evaluate(sprime)
                lf_quat = lfoot_quat_curve.evaluate(s)
                rf_quat = rfoot_quat_curve.evaluate(s)

                joint_pos, lf_done, rf_done = ik_feet(
                    base_pos, base_quat, lf_pos, lf_quat, rf_pos, rf_quat,
                    nominal_sensor_data, joint_screws_in_ee_at_home,
                    ee_SE3_at_home, open_chain_joints)

                if not (lf_done and rf_done):
                    print("====================================")
                    __import__('ipdb').set_trace()
                    print("====================================")

                s += DT / swing_time

        # Visualize config
        pybullet_util.set_config(robot, joint_id, link_id, base_pos, base_quat,
                                 joint_pos)

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

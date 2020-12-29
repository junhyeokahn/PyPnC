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


def inv_kin(robot, sensor_data):
    pass


def _ik_sanity_check(robot, joint_id, link_id, open_chain_joints, sensor_data,
                     nominal_sensor_data):
    lf_state = p.getLinkState(robot, link_id['l_sole'], False, True)
    T_w_lf = RpToTrans(quat_to_rot(np.array(lf_state[1])),
                       np.array(lf_state[0]))
    T_w_base = RpToTrans(quat_to_rot(sensor_data['base_quat']),
                         sensor_data['base_pos'])
    T_base_lf = np.dot(TransInv(T_w_base), T_w_lf)
    gt_lf_joints = np.array([
        sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['left_foot']
    ])
    guess_lf_joints = np.array([
        nominal_sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['left_foot']
    ])
    sol_lf_joints, done = IKinBody(joint_screws_in_ee_at_home['left_foot'],
                                   ee_SE3_at_home['left_foot'], T_base_lf,
                                   guess_lf_joints)

    left_foot_SE3_from_ik = FKinBody(ee_SE3_at_home['left_foot'],
                                     joint_screws_in_ee_at_home['left_foot'],
                                     sol_lf_joints)
    if not done:
        print("ik fail")
        __import__('ipdb').set_trace()

    if not (np.isclose(left_foot_SE3_from_ik, T_base_lf, rtol=0.,
                       atol=1.e-2)).all():
        print("--" * 40)
        print((np.isclose(left_foot_SE3_from_ik,
                          T_base_lf,
                          rtol=0.,
                          atol=1.e-2)))
        print("left foot SE3 error")
        print(T_base_lf - left_foot_SE3_from_ik)
        __import__('ipdb').set_trace()
    else:
        print("left foot ik SE3 pass")

    if not (np.isclose(gt_lf_joints, sol_lf_joints, rtol=0.,
                       atol=5.e-2)).all():
        print("--" * 40)
        print((np.isclose(gt_lf_joints, sol_lf_joints)))
        print("left foot joint error")
        print(gt_lf_joints - sol_lf_joints)
        __import__('ipdb').set_trace()
    else:
        print("left foot ik joint pass")

    rf_state = p.getLinkState(robot, link_id['r_sole'], False, True)
    T_w_rf = RpToTrans(quat_to_rot(np.array(rf_state[1])),
                       np.array(rf_state[0]))
    T_base_rf = np.dot(TransInv(T_w_base), T_w_rf)
    tf_rf_joints = np.array([
        sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['right_foot']
    ])
    guess_rf_joints = np.array([
        nominal_sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['right_foot']
    ])
    sol_rf_joints, done = IKinBody(joint_screws_in_ee_at_home['right_foot'],
                                   ee_SE3_at_home['right_foot'], T_base_rf,
                                   guess_rf_joints)

    right_foot_SE3_from_ik = FKinBody(ee_SE3_at_home['right_foot'],
                                      joint_screws_in_ee_at_home['right_foot'],
                                      sol_rf_joints)
    if not done:
        print("ik fail")
        __import__('ipdb').set_trace()

    if not (np.isclose(right_foot_SE3_from_ik, T_base_rf, rtol=0.,
                       atol=1.e-2)).all():
        print("--" * 40)
        print((np.isclose(right_foot_SE3_from_ik,
                          T_base_rf,
                          rtol=0.,
                          atol=1.e-2)))
        print("right foot SE3 error")
        print(T_base_rf - right_foot_SE3_from_ik)
        __import__('ipdb').set_trace()
    else:
        print("right foot ik SE3 pass")

    if not (np.isclose(tf_rf_joints, sol_rf_joints, rtol=0.,
                       atol=5.e-2)).all():
        print("--" * 40)
        print((np.isclose(tf_rf_joints, sol_rf_joints)))
        print("right foot joint error")
        print(tf_rf_joints - sol_rf_joints)
        __import__('ipdb').set_trace()
    else:
        print("right foot ik joint pass")


def _fk_sanity_check(robot, joint_id, link_id, open_chain_joints, sensor_data):
    lfoot_joints = np.array([
        sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['left_foot']
    ])
    left_foot_SE3 = FKinBody(ee_SE3_at_home['left_foot'],
                             joint_screws_in_ee_at_home['left_foot'],
                             lfoot_joints)
    T_w_base = RpToTrans(quat_to_rot(sensor_data['base_quat']),
                         sensor_data['base_pos'])
    my_fwd_kin = np.dot(T_w_base, left_foot_SE3)
    l_sole_link_state = p.getLinkState(robot, link_id['l_sole'], False, True)
    pb_fwd_kin = RpToTrans(quat_to_rot(np.array(l_sole_link_state[1])),
                           np.array(l_sole_link_state[0]))

    if not (np.isclose(my_fwd_kin, pb_fwd_kin, rtol=0., atol=1.e-5)).all():
        print("--" * 40)
        print((np.isclose(my_fwd_kin, pb_fwd_kin)))
        print("left foot error")
        print(my_fwd_kin - pb_fwd_kin)
        __import__('ipdb').set_trace()
    else:
        print("Pass Left Foot")

    rfoot_joints = np.array([
        sensor_data['joint_pos'][j_name]
        for j_name in open_chain_joints['right_foot']
    ])
    right_foot_SE3 = FKinBody(ee_SE3_at_home['right_foot'],
                              joint_screws_in_ee_at_home['right_foot'],
                              rfoot_joints)
    my_fwd_kin = np.dot(T_w_base, right_foot_SE3)
    r_sole_link_state = p.getLinkState(robot, link_id['r_sole'], False, True)
    pb_fwd_kin = RpToTrans(quat_to_rot(np.array(r_sole_link_state[1])),
                           np.array(r_sole_link_state[0]))

    if not (np.isclose(my_fwd_kin, pb_fwd_kin, rtol=0., atol=1.e-5)).all():
        print("--" * 40)
        print((np.isclose(my_fwd_kin, pb_fwd_kin)))
        print("right foot error")
        print(my_fwd_kin - pb_fwd_kin)
    else:
        print("Pass Right Foot")


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

    # Robot Configuration
    nq, nv, na, joint_id, link_id = get_robot_config(robot)
    joint_screws_in_ee_at_home, ee_SE3_at_home = dict(), dict()
    open_chain_joints, base_link, ee_link = dict(), dict(), dict()
    base_link['left_foot'] = 'pelvis'
    ee_link['left_foot'] = 'l_sole'
    open_chain_joints['left_foot'] = [
        'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky',
        'l_leg_akx'
    ]
    base_link['right_foot'] = 'pelvis'
    ee_link['right_foot'] = 'r_sole'
    open_chain_joints['right_foot'] = [
        'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',
        'r_leg_akx'
    ]

    for ee in ['left_foot', 'right_foot']:
        joint_screws_in_ee_at_home[ee], ee_SE3_at_home[
            ee] = get_kinematics_config(robot, joint_id, link_id,
                                        open_chain_joints[ee], base_link[ee],
                                        ee_link[ee])
    ## TEST

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 2)

    #camera intrinsic parameter
    fov, aspect, nearval, farval = 60.0, 2.0, 0.1, 10
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearval,
                                                     farval)

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
            print("Pressed 9: Testing Batch IK and Batch FK")
            # Testing batch_ik
            l_sole_state = p.getLinkState(robot, link_id['l_sole'], False,
                                          True)
            sol_guess = np.array([
                sensor_data['joint_pos'][j_name]
                for j_name in open_chain_joints['left_foot']
            ])
            N = 1000
            l_sole_SE3_list = [
                RpToTrans(
                    quat_to_rot(np.array(l_sole_state[1])),
                    np.random.normal(np.array(l_sole_state[0]),
                                     np.array([0.05, 0.02, 0.])) -
                    sensor_data['base_pos']) for i in range(N)
            ]
            t0 = time.time()
            sol_list, success_list = batch_ik(
                joint_screws_in_ee_at_home['left_foot'],
                ee_SE3_at_home['left_foot'], l_sole_SE3_list, sol_guess)
            t1 = time.time()
            print("batck ik takes {}".format(t1 - t0))
            sol_list2, success_list2 = [], []
            t0 = time.time()
            for des_SE3 in l_sole_SE3_list:
                _sol, _suc = IKinBody(joint_screws_in_ee_at_home['left_foot'],
                                      ee_SE3_at_home['left_foot'], des_SE3,
                                      sol_guess)
                sol_list2.append(_sol)
                success_list2.append(_suc)
            t1 = time.time()
            print("single ik takes {}".format(t1 - t0))
            t0 = time.time()
            T_list = batch_fk(ee_SE3_at_home['left_foot'],
                              joint_screws_in_ee_at_home['left_foot'],
                              sol_list)
            t1 = time.time()
            print("batch fk takes {}".format(t1 - t0))
            T_list2 = []
            t0 = time.time()
            for q in sol_list2:
                _T = FKinBody(ee_SE3_at_home['left_foot'],
                              joint_screws_in_ee_at_home['left_foot'], q)
                T_list2.append(_T)
            t1 = time.time()
            print("single fk takes {}".format(t1 - t0))
            for i in range(N):
                if np.isclose(sol_list[i], sol_list2[i], rtol=0.,
                              atol=1.e-5).all():
                    pass
                else:
                    print("Batch IK is wrong")
                    __import__('ipdb').set_trace()

                if np.isclose(T_list2[i], T_list[i], rtol=0.,
                              atol=1.e-5).all():
                    pass
                else:
                    print("Batch FK is wrong")
                    __import__('ipdb').set_trace()

                if np.isclose(T_list[i],
                              l_sole_SE3_list[i],
                              rtol=0.,
                              atol=1.e-2).all():
                    pass
                else:
                    print("IK FK is not matched")
                    __import__('ipdb').set_trace()
            print("Batch FK Test and IK Test Passed")

        elif is_key_triggered(keys, '1'):
            # Nominal Pos
            print("Pressed 1: Reset to Nominal Pos")
            base_pos = np.copy(nominal_sensor_data['base_pos'])
            base_quat = np.copy(nominal_sensor_data['base_quat'])
            joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])

        # Visualize config
        set_config(robot, joint_id, link_id, base_pos, base_quat, joint_pos)

        # Forward Kinematics Sanity Check : To check this, comoment out set_config and comment in stepSimulation to make robot fall down
        # _fk_sanity_check(robot, joint_id, link_id, open_chain_joints,
        # sensor_data)

        # Inverse Kinemtaics Sanity Check : To check this, comoment out set_config and comment in stepSimulation to make robot fall down
        # _ik_sanity_check(robot, joint_id, link_id, open_chain_joints,
        # sensor_data, nominal_sensor_data)

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
from datetime import datetime
import math
import copy
from collections import OrderedDict
import shutil

import pybullet as p
import numpy as np
np.set_printoptions(precision=3)
import tensorflow as tf
from tqdm import tqdm
from ruamel.yaml import YAML
from casadi import *

from pnc.data_saver import DataSaver
from util import pybullet_util
from util import util
from util import interpolation
from util import liegroup
from util import robot_kinematics

## Configs
VIDEO_RECORD = False
PRINT_FREQ = 10
DT = 0.01
PRINT_ROBOT_INFO = False
INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 0.342]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]
DYN_LIB = "pinocchio"  # "dart"

## Motion Boundaries
RFOOT_POS_LB = np.array([-0.01, -0.005, -0.001])
RFOOT_POS_UB = np.array([0.01, 0.005, 0.001])
LFOOT_POS_LB = np.array([-0.01, -0.005, -0.001])
LFOOT_POS_UB = np.array([0.01, 0.005, 0.001])

FOOT_EA_LB = np.array([np.deg2rad(0.), np.deg2rad(0.), -np.pi / 10.])
FOOT_EA_UB = np.array([np.deg2rad(0.), np.deg2rad(0.), np.pi / 10.])

SWING_HEIGHT_LB, SWING_HEIGHT_UB = 0.02, 0.03

SWING_TIME_LB, SWING_TIME_UB = 0.45, 0.55

BASE_HEIGHT_LB, BASE_HEIGHT_UB = 0.342, 0.362

## Dataset Generation
N_CPU_DATA_GEN = 5
N_MOTION_PER_LEG = 1e4
N_DATA_PER_MOTION = 15

N_LAYER_OUTPUT = [64, 64]
ACTIVATION = [tf.keras.activations.tanh, tf.keras.activations.tanh]
LR = 0.01
MOMENTUM = 0.
N_EPOCH = 20
BATCH_SIZE = 32


def inertia_from_one_hot_vec(vec):
    ret = np.zeros((3, 3))

    ret[0, 0] = vec[0]
    ret[1, 1] = vec[1]
    ret[2, 2] = vec[2]

    ret[0, 1] = vec[3]
    ret[1, 0] = vec[3]
    ret[0, 2] = vec[4]
    ret[2, 0] = vec[4]
    ret[1, 2] = vec[5]
    ret[2, 1] = vec[5]

    return ret


def inertia_to_one_hot_vec(inertia):
    ret = np.zeros(6)
    ret[0] = inertia[0, 0]
    ret[1] = inertia[1, 1]
    ret[2] = inertia[2, 2]
    ret[3] = inertia[0, 1]
    ret[4] = inertia[0, 2]
    ret[5] = inertia[1, 2]
    return ret


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


def sample_swing_config(nominal_lf_iso, nominal_rf_iso, side):

    swing_time = np.random.uniform(SWING_TIME_LB, SWING_TIME_UB)
    if side == "left":
        # Sample rfoot config
        rfoot_ini_iso = np.copy(nominal_rf_iso)
        rfoot_mid_iso = np.copy(nominal_rf_iso)
        rfoot_fin_iso = np.copy(nominal_rf_iso)

        rfoot_mid_vel = (rfoot_ini_iso[0:3, 3] -
                         rfoot_ini_iso[0:3, 3]) / swing_time
        rfoot_mid_vel[2] = 0.

        # Sample lfoot config
        lfoot_ini_pos = np.copy(nominal_lf_iso)[0:3, 3] + np.random.uniform(
            LFOOT_POS_LB, LFOOT_POS_UB)
        lfoot_ini_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
        lfoot_ini_rot = util.euler_to_rot(lfoot_ini_ea)
        lfoot_ini_iso = liegroup.RpToTrans(lfoot_ini_rot, lfoot_ini_pos)
        lfoot_fin_pos = np.copy(nominal_lf_iso)[0:3, 3] + np.random.uniform(
            LFOOT_POS_LB, LFOOT_POS_UB)
        lfoot_fin_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
        lfoot_fin_rot = util.euler_to_rot(lfoot_fin_ea)
        lfoot_fin_iso = liegroup.RpToTrans(lfoot_fin_rot, lfoot_fin_pos)
        lfoot_mid_iso = util.iso_interpolate(lfoot_ini_iso, lfoot_fin_iso, 0.5)
        lfoot_mid_iso[2, 3] = (lfoot_ini_pos[2] +
                               lfoot_fin_pos[2]) / 2.0 + np.random.uniform(
                                   SWING_HEIGHT_LB, SWING_HEIGHT_UB)

        lfoot_mid_vel = (lfoot_fin_pos - lfoot_ini_pos) / swing_time
        lfoot_mid_vel[2] = 0.

        # Sample base config
        base_ini_pos = (rfoot_ini_iso[0:3, 3] + lfoot_ini_iso[0:3, 3]) / 2.0
        base_ini_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
        base_ini_rot = util.euler_to_rot(
            np.array([0., 0., lfoot_ini_ea[2] / 2.]))
        base_ini_iso = liegroup.RpToTrans(base_ini_rot, base_ini_pos)
        base_fin_pos = (rfoot_fin_iso[0:3, 3] + lfoot_fin_iso[0:3, 3]) / 2.0
        base_fin_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
        base_fin_rot = util.euler_to_rot(
            np.array([0., 0., lfoot_fin_ea[2] / 2.]))
        base_fin_iso = liegroup.RpToTrans(base_fin_rot, base_fin_pos)

    elif side == "right":
        # Sample lfoot config
        lfoot_ini_iso = np.copy(nominal_lf_iso)
        lfoot_mid_iso = np.copy(nominal_lf_iso)
        lfoot_fin_iso = np.copy(nominal_lf_iso)

        lfoot_mid_vel = (lfoot_ini_iso[0:3, 3] -
                         lfoot_ini_iso[0:3, 3]) / swing_time
        lfoot_mid_vel[2] = 0.

        # Sample rfoot config
        rfoot_ini_pos = np.copy(nominal_rf_iso)[0:3, 3] + np.random.uniform(
            RFOOT_POS_LB, RFOOT_POS_UB)
        rfoot_ini_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
        rfoot_ini_rot = util.euler_to_rot(rfoot_ini_ea)
        rfoot_ini_iso = liegroup.RpToTrans(rfoot_ini_rot, rfoot_ini_pos)
        rfoot_fin_pos = np.copy(nominal_rf_iso)[0:3, 3] + np.random.uniform(
            RFOOT_POS_LB, RFOOT_POS_UB)
        rfoot_fin_ea = np.random.uniform(FOOT_EA_LB, FOOT_EA_UB)
        rfoot_fin_rot = util.euler_to_rot(rfoot_fin_ea)
        rfoot_fin_iso = liegroup.RpToTrans(rfoot_fin_rot, rfoot_fin_pos)
        rfoot_mid_iso = util.iso_interpolate(rfoot_ini_iso, rfoot_fin_iso, 0.5)
        rfoot_mid_iso[2, 3] = (rfoot_ini_pos[2] +
                               rfoot_fin_pos[2]) / 2.0 + np.random.uniform(
                                   SWING_HEIGHT_LB, SWING_HEIGHT_UB)

        rfoot_mid_vel = (rfoot_fin_pos - rfoot_ini_pos) / swing_time
        rfoot_mid_vel[2] = 0.

        # Sample base config
        base_ini_pos = (rfoot_ini_iso[0:3, 3] + lfoot_ini_iso[0:3, 3]) / 2.0
        base_ini_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
        base_ini_rot = util.euler_to_rot(
            np.array([0., 0., rfoot_ini_ea[2] / 2.]))
        base_ini_iso = liegroup.RpToTrans(base_ini_rot, base_ini_pos)
        base_fin_pos = (rfoot_fin_iso[0:3, 3] + lfoot_fin_iso[0:3, 3]) / 2.0
        base_fin_pos[2] = np.random.uniform(BASE_HEIGHT_LB, BASE_HEIGHT_UB)
        base_fin_rot = util.euler_to_rot(
            np.array([0., 0., rfoot_fin_ea[2] / 2.]))
        base_fin_iso = liegroup.RpToTrans(base_fin_rot, base_fin_pos)

    else:
        raise ValueError

    return swing_time, lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel, rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel, base_ini_iso, base_fin_iso


def create_curves(lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel,
                  rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel,
                  base_ini_iso, base_fin_iso):
    lfoot_pos_curve_ini_to_mid = interpolation.HermiteCurveVec(
        lfoot_ini_iso[0:3, 3], np.zeros(3), lfoot_mid_iso[0:3, 3],
        lfoot_mid_vel)
    lfoot_pos_curve_mid_to_fin = interpolation.HermiteCurveVec(
        lfoot_mid_iso[0:3, 3], lfoot_mid_vel, lfoot_fin_iso[0:3, 3],
        np.zeros(3))
    lfoot_quat_curve = interpolation.HermiteCurveQuat(
        util.rot_to_quat(lfoot_ini_iso[0:3, 0:3]), np.zeros(3),
        util.rot_to_quat(lfoot_fin_iso[0:3, 0:3]), np.zeros(3))
    rfoot_pos_curve_ini_to_mid = interpolation.HermiteCurveVec(
        rfoot_ini_iso[0:3, 3], np.zeros(3), rfoot_mid_iso[0:3, 3],
        rfoot_mid_vel)
    rfoot_pos_curve_mid_to_fin = interpolation.HermiteCurveVec(
        rfoot_mid_iso[0:3, 3], rfoot_mid_vel, rfoot_fin_iso[0:3, 3],
        np.zeros(3))
    rfoot_quat_curve = interpolation.HermiteCurveQuat(
        util.rot_to_quat(rfoot_ini_iso[0:3, 0:3]), np.zeros(3),
        util.rot_to_quat(rfoot_fin_iso[0:3, 0:3]), np.zeros(3))
    base_pos_curve = interpolation.HermiteCurveVec(base_ini_iso[0:3, 3],
                                                   np.zeros(3),
                                                   base_fin_iso[0:3, 3],
                                                   np.zeros(3))
    base_quat_curve = interpolation.HermiteCurveQuat(
        util.rot_to_quat(base_ini_iso[0:3, 0:3]), np.zeros(3),
        util.rot_to_quat(base_fin_iso[0:3, 0:3]), np.zeros(3))

    return lfoot_pos_curve_ini_to_mid, lfoot_pos_curve_mid_to_fin, lfoot_quat_curve, rfoot_pos_curve_ini_to_mid, rfoot_pos_curve_mid_to_fin, rfoot_quat_curve, base_pos_curve, base_quat_curve


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


def _do_generate_data(n_data,
                      nominal_lf_iso,
                      nominal_rf_iso,
                      nominal_sensor_data,
                      side,
                      rseed=None,
                      cpu_idx=0):
    if rseed is not None:
        np.random.seed(rseed)

    from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
    robot_sys = PinocchioRobotSystem(cwd + "/robot_model/nao/nao.urdf",
                                     cwd + "/robot_model/nao", False, False)

    data_x, data_y = [], []

    text = "#" + "{}".format(cpu_idx).zfill(3)
    with tqdm(total=n_data,
              desc=text + ': Generating data',
              position=cpu_idx + 1) as pbar:
        for i in range(n_data):

            swing_time, lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel, rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel, base_ini_iso, base_fin_iso = sample_swing_config(
                nominal_lf_iso, nominal_rf_iso, side)

            lfoot_pos_curve_ini_to_mid, lfoot_pos_curve_mid_to_fin, lfoot_quat_curve, rfoot_pos_curve_ini_to_mid, rfoot_pos_curve_mid_to_fin, rfoot_quat_curve, base_pos_curve, base_quat_curve = create_curves(
                lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel,
                rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel,
                base_ini_iso, base_fin_iso)

            for s in np.linspace(0, 1, N_DATA_PER_MOTION):
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

                if lf_done and rf_done:
                    rot_world_com = util.quat_to_rot(np.copy(base_quat))
                    rot_world_joint = np.dot(
                        rot_world_com, rot_basejoint_to_basecom.transpose())
                    base_joint_pos = base_pos - np.dot(
                        rot_world_joint, pos_basejoint_to_basecom)
                    base_joint_quat = util.rot_to_quat(rot_world_joint)
                    robot_sys.update_system(base_pos, base_quat, np.zeros(3),
                                            np.zeros(3), base_joint_pos,
                                            base_joint_quat, np.zeros(3),
                                            np.zeros(3), joint_pos,
                                            nominal_sensor_data['joint_vel'],
                                            True)
                    world_I = robot_sys.Ig[0:3, 0:3]
                    local_I = np.dot(
                        np.dot(rot_world_com.transpose(), world_I),
                        rot_world_com)
                    # append to data
                    data_x.append(
                        np.concatenate([lf_pos - base_pos, rf_pos - base_pos],
                                       axis=0))
                    data_y.append(
                        np.array([
                            local_I[0, 0], local_I[1, 1], local_I[2, 2],
                            local_I[0, 1], local_I[0, 2], local_I[1, 2]
                        ]))
            pbar.update(1)

    return data_x, data_y


def _generate_data(arg_list):
    return _do_generate_data(*arg_list)


def generate_data(n_data, nominal_lf_iso, nominal_rf_iso, nominal_sensor_data,
                  side, num_cpu):
    rollout_per_cpu = int(max(n_data // num_cpu, 1))
    args_list = [
        rollout_per_cpu, nominal_lf_iso, nominal_rf_iso, nominal_sensor_data,
        side
    ]
    results = util.try_multiprocess(args_list, num_cpu, _generate_data)

    data_x, data_y = [], []
    for result in results:
        data_x += result[0]
        data_y += result[1]

    return data_x, data_y


def save_weights_to_yaml(tf_model):
    model_path = 'data/tf_model/nao_crbi'
    mlp_model = dict()
    mlp_model['num_layer'] = len(tf_model.layers)
    for l_id, l in enumerate(tf_model.layers):
        mlp_model['w' + str(l_id)] = l.weights[0].numpy().tolist()
        mlp_model['b' + str(l_id)] = l.weights[1].numpy().reshape(
            1, l.weights[1].shape[0]).tolist()
        # Activation Fn Idx: None: 0, Tanh: 1
        if (l_id == (len(tf_model.layers) - 1)):
            mlp_model['act_fn' + str(l_id)] = 0
        else:
            mlp_model['act_fn' + str(l_id)] = 1
    with open(model_path + '/mlp_model.yaml', 'w') as f:
        yml = YAML()
        yml.dump(mlp_model, f)


def generate_casadi_func(tf_model,
                         input_mean,
                         input_std,
                         output_mean,
                         output_std,
                         generate_c_code=True):
    c_code_path = cwd + "/data/tf_model/nao_crbi"
    ## Computational Graph
    b = MX.sym('b', 3)
    l = MX.sym('l', 3)
    r = MX.sym('r', 3)
    # Input
    l_minus_b = l - b
    r_minus_b = r - b
    inp = vertcat(l_minus_b, r_minus_b)
    normalized_inp = (inp - input_mean) / input_std  # (6, 1)
    # MLP (Somewhat manual)
    w0 = tf_model.layers[0].weights[0].numpy()  # (6, 64)
    b0 = tf_model.layers[0].weights[1].numpy().reshape(1, -1)  # (1, 64)
    w1 = tf_model.layers[1].weights[0].numpy()  # (64, 64)
    b1 = tf_model.layers[1].weights[1].numpy().reshape(1, -1)  # (1, 64)
    w2 = tf_model.layers[2].weights[0].numpy()  # (64, 6)
    b2 = tf_model.layers[2].weights[1].numpy().reshape(1, -1)  # (6)
    output = mtimes(
        tanh(mtimes(tanh(mtimes(normalized_inp.T, w0) + b0), w1) + b1),
        w2) + b2
    denormalized_output = (output.T * output_std) + output_mean

    # Define casadi function
    func = Function('nao_crbi_helper', [b, l, r], [denormalized_output])
    jac_func = func.jacobian()
    print(func)
    print(jac_func)

    if generate_c_code:
        # Code generator
        code_gen = CodeGenerator('nao_crbi_helper.c', dict(with_header=True))
        code_gen.add(func)
        code_gen.add(jac_func)
        code_gen.generate()
        shutil.move(
            cwd + '/nao_crbi_helper.h', cwd +
            "/pnc/planner/locomotion/towr_plus/include/towr_plus/models/examples/nao_crbi_helper.h"
        )
        shutil.move(
            cwd + '/nao_crbi_helper.c',
            cwd + "/pnc/planner/locomotion/towr_plus/src/nao_crbi_helper.c")

    return func, jac_func


def evaluate_crbi_model_using_casadi(cas_func, b, l, r):
    out = cas_func(b, l, r)
    return out


def evaluate_crbi_model_using_tf(tf_model, b, l, r, input_mean, input_std,
                                 output_mean, output_std):
    inp1 = l - b
    inp2 = r - b
    inp = np.concatenate([inp1, inp2], axis=0)
    normalized_inp = np.expand_dims(util.normalize(inp, input_mean, input_std),
                                    axis=0)
    output = tf_model(normalized_inp)
    d_output = util.denormalize(np.squeeze(output), output_mean, output_std)
    return d_output, output


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
            if f == "nao_crbi.mp4":
                os.remove('video/' + f)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/nao_crbi.mp4")

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/nao/nao.urdf",
                       INITIAL_POS_WORLD_TO_BASEJOINT,
                       INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Robot Configuration : 0 << Left Foot, 1 << Right Foot
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT,
        True)

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

    if DYN_LIB == 'dart':
        from pnc.robot_system.dart_robot_system import DartRobotSystem
        robot_sys = DartRobotSystem(cwd + "/robot_model/nao/nao_rel_path.urdf",
                                    False, True)
    elif DYN_LIB == 'pinocchio':
        from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
        robot_sys = PinocchioRobotSystem(cwd + "/robot_model/nao/nao.urdf",
                                         cwd + "/robot_model/nao", False, True)
    else:
        raise ValueError

    # DataSaver
    data_saver = DataSaver("nao_crbi_validation.pkl")

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
    b_regressor_trained = False
    while (1):

        # Get Keyboard Event
        keys = p.getKeyboardEvents()

        # Set base_pos, base_quat, joint_pos here for visualization
        if pybullet_util.is_key_triggered(keys, '8'):
            # f, jac_f = generate_casadi_func(crbi_model, input_mean, input_std,
            # output_mean, output_std, True)
            pass
        elif pybullet_util.is_key_triggered(keys, '5'):
            # Generate Dataset
            print("-" * 80)
            print("Pressed 5: Train CRBI Regressor")

            l_data_x, l_data_y = generate_data(N_MOTION_PER_LEG,
                                               nominal_lf_iso, nominal_rf_iso,
                                               nominal_sensor_data, "left",
                                               N_CPU_DATA_GEN)
            r_data_x, r_data_y = generate_data(N_MOTION_PER_LEG,
                                               nominal_lf_iso, nominal_rf_iso,
                                               nominal_sensor_data, "right",
                                               N_CPU_DATA_GEN)
            data_x = l_data_x + r_data_x
            data_y = l_data_y + r_data_y

            print("{} data is collected".format(len(data_x)))
            input_mean, input_std, normalized_data_x = util.normalize_data(
                data_x)
            output_mean, output_std, normalized_data_y = util.normalize_data(
                data_y)

            log_dir = "data/tensorboard/nao_crbi"
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq='batch')

            crbi_model = tf.keras.Sequential()
            for l_id, (n_layer_output,
                       act) in enumerate(zip(N_LAYER_OUTPUT, ACTIVATION)):
                crbi_model.add(
                    tf.keras.layers.Dense(n_layer_output, activation=act))
            crbi_model.add(tf.keras.layers.Dense(6, activation=None))
            opt = tf.keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM)
            crbi_model.compile(optimizer='sgd', loss='mse')
            crbi_model.fit(x=np.array(normalized_data_x, dtype=np.float32),
                           y=np.array(normalized_data_y, dtype=np.float32),
                           batch_size=BATCH_SIZE,
                           epochs=N_EPOCH,
                           verbose=1,
                           validation_split=0.1,
                           shuffle=True,
                           workers=4,
                           use_multiprocessing=True,
                           callbacks=[tensorboard_callback])
            model_path = 'data/tf_model/nao_crbi'
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            crbi_model.save("data/tf_model/nao_crbi")
            data_stats = {
                'input_mean': input_mean.tolist(),
                'input_std': input_std.tolist(),
                'output_mean': output_mean.tolist(),
                'output_std': output_std.tolist()
            }
            with open(model_path + '/data_stat.yaml', 'w') as f:
                yml = YAML()
                yml.dump(data_stats, f)
            save_weights_to_yaml(crbi_model)

            cas_func, cas_jac_func = generate_casadi_func(
                crbi_model, input_mean, input_std, output_mean, output_std,
                True)

            b_regressor_trained = True

            # Don't wanna mess up the visualizer
            base_pos = np.copy(nominal_sensor_data['base_com_pos'])
            base_quat = np.copy(nominal_sensor_data['base_com_quat'])
            joint_pos = copy.deepcopy(nominal_sensor_data['joint_pos'])

        elif pybullet_util.is_key_triggered(keys, '4'):
            print("-" * 80)
            print("Pressed 4: Load Pre-trained CRBI Model")
            crbi_model = tf.keras.models.load_model("data/tf_model/nao_crbi")
            model_path = 'data/tf_model/nao_crbi'
            with open(model_path + '/data_stat.yaml', 'r') as f:
                yml = YAML().load(f)
                input_mean = np.array(yml['input_mean'])
                input_std = np.array(yml['input_std'])
                output_mean = np.array(yml['output_mean'])
                output_std = np.array(yml['output_std'])

            b_regressor_trained = True

        elif pybullet_util.is_key_triggered(keys, '3'):
            # Left Foot Swing, Right Foot Stance
            print("-" * 80)
            print("Pressed 3: Sample Motion for Left Foot Swing")

            swing_time, lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel, rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel, base_ini_iso, base_fin_iso = sample_swing_config(
                nominal_lf_iso, nominal_rf_iso, "left")
            s = 0.
            b_ik = True

        elif pybullet_util.is_key_triggered(keys, '2'):
            # Left Foot Stance, Right Foot Swing
            print("-" * 80)
            print("Pressed 2: Sample Motion for Right Foot Swing")

            swing_time, lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel, rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel, base_ini_iso, base_fin_iso = sample_swing_config(
                nominal_lf_iso, nominal_rf_iso, "right")
            s = 0.
            b_ik = True

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
                lfoot_pos_curve_ini_to_mid, lfoot_pos_curve_mid_to_fin, lfoot_quat_curve, rfoot_pos_curve_ini_to_mid, rfoot_pos_curve_mid_to_fin, rfoot_quat_curve, base_pos_curve, base_quat_curve = create_curves(
                    lfoot_ini_iso, lfoot_mid_iso, lfoot_fin_iso, lfoot_mid_vel,
                    rfoot_ini_iso, rfoot_mid_iso, rfoot_fin_iso, rfoot_mid_vel,
                    base_ini_iso, base_fin_iso)

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
                    print("base pos")
                    print(base_pos)
                    print("base_quat")
                    print(base_quat)
                    print("lf pos")
                    print(lf_pos)
                    print("lf quat")
                    print(lf_quat)
                    print("rf pos")
                    print(rf_pos)
                    print("rf quat")
                    print(rf_quat)
                    __import__('ipdb').set_trace()
                    print("====================================")

                s += DT / swing_time

        # Visualize config
        pybullet_util.set_config(robot, joint_id, link_id, base_pos, base_quat,
                                 joint_pos)

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)
        if b_regressor_trained:
            robot_sys.update_system(
                sensor_data["base_com_pos"], sensor_data["base_com_quat"],
                sensor_data["base_com_lin_vel"],
                sensor_data["base_com_ang_vel"], sensor_data["base_joint_pos"],
                sensor_data["base_joint_quat"],
                sensor_data["base_joint_lin_vel"],
                sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
                sensor_data["joint_vel"], True)
            rot_world_base = util.quat_to_rot(sensor_data['base_com_quat'])
            world_I = robot_sys.Ig[0:3, 0:3]
            local_I = np.dot(np.dot(rot_world_base.transpose(), world_I),
                             rot_world_base)
            rf_iso = pybullet_util.get_link_iso(robot, link_id["r_sole"])
            lf_iso = pybullet_util.get_link_iso(robot, link_id["l_sole"])

            denormalized_output, output = evaluate_crbi_model_using_tf(
                crbi_model, sensor_data["base_com_pos"], lf_iso[0:3, 3],
                rf_iso[0:3, 3], input_mean, input_std, output_mean, output_std)

            local_I_est = inertia_from_one_hot_vec(denormalized_output)
            if b_ik:
                data_saver.add('gt_inertia', inertia_to_one_hot_vec(local_I))
                data_saver.add(
                    'gt_inertia_normalized',
                    util.normalize(inertia_to_one_hot_vec(local_I),
                                   output_mean, output_std))
                data_saver.add('est_inertia_normalized',
                               np.copy(np.squeeze(output)))
                data_saver.add('est_inertia', np.copy(denormalized_output))
                data_saver.advance()

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

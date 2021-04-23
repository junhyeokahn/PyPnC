import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
import math
import copy
import shutil
from collections import OrderedDict

from tqdm import tqdm
import cv2
import imageio
from ruamel.yaml import YAML
import pybullet as p
import numpy as np
np.set_printoptions(precision=3)

from pnc.data_saver import DataSaver
from util import pybullet_util
from util import util
from util import liegroup
from util import robot_kinematics

## Configs
DT = 0.01

PRINT_ROBOT_INFO = False
VIDEO_RECORD = True
RECORD_FREQ = 1

INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.761]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--crbi", type=bool, default=False)
args = parser.parse_args()
if args.crbi:
    import tensorflow as tf
    b_crbi = True
    crbi_model = tf.keras.models.load_model('data/tf_model/atlas_crbi')
    with open('data/tf_model/atlas_crbi/data_stat.yaml', 'r') as f:
        yml = YAML().load(f)
        input_mean = np.array(yml['input_mean'])
        input_std = np.array(yml['input_std'])
        output_mean = np.array(yml['output_mean'])
        output_std = np.array(yml['output_std'])

    from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
    robot_sys = PinocchioRobotSystem(cwd + "/robot_model/atlas/atlas.urdf",
                                     cwd + "/robot_model/atlas", False, True)

    def evaluate_crbi_model_using_tf(tf_model, b, l, r, input_mean, input_std,
                                     output_mean, output_std):
        inp1 = l - b
        inp2 = r - b
        inp = np.concatenate([inp1, inp2], axis=0)
        normalized_inp = np.expand_dims(util.normalize(inp, input_mean,
                                                       input_std),
                                        axis=0)
        output = tf_model(normalized_inp)
        d_output = util.denormalize(np.squeeze(output), output_mean,
                                    output_std)
        return d_output, output

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
else:
    b_crbi = False

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
    p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                 cameraYaw=120,
                                 cameraPitch=-15,
                                 cameraTargetPosition=[1, 0.5, 1.])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSubSteps=1)
    if VIDEO_RECORD:
        video_dir = 'video/' + file.split('/')[-1].split('.')[0]
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
                       INITIAL_POS_WORLD_TO_BASEJOINT,
                       INITIAL_QUAT_WORLD_TO_BASEJOINT)
    if file == "data/atlas_block.yaml":
        block = p.loadURDF(cwd + "/robot_model/ground/block.urdf",
                           [0, 0, 0.15],
                           useFixedBase=True)
    elif file == "data/atlas_stair.yaml":
        stair = p.loadURDF(cwd + "/robot_model/ground/stair.urdf",
                           [0.2, 0, 0.],
                           useFixedBase=True)
    elif file == "data/atlas_slope.yaml":
        gap = p.loadURDF(cwd + "/robot_model/ground/slope.urdf",
                         [0.325, 0, -0.125],
                         useFixedBase=True)
    elif file == 'data/atlas_chimney.yaml':
        lr_chimney = p.loadURDF(cwd + "/robot_model/ground/chimney.urdf",
                                [0, 0, 0],
                                useFixedBase=True)
    elif file == 'data/atlas_lr_chimney_jump.yaml':
        lr_chimney = p.loadURDF(cwd + "/robot_model/ground/lr_chimney.urdf",
                                [0, 0, 0],
                                useFixedBase=True)

    if (file != 'data/atlas_lr_chimney_jump.yaml') and (
            file != 'data/atlas_chimney.yaml'):
        p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
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

    # DataSaver
    data_saver = DataSaver("atlas_crbi_validation.pkl")

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
            if VIDEO_RECORD:
                pybullet_util.make_video(video_dir, False)
            p.disconnect()
            exit()
        else:
            vis_idx += 1

        # Visualize config
        pybullet_util.set_config(robot, joint_id, link_id, base_pos, base_quat,
                                 joint_pos)

        if b_crbi and VIDEO_RECORD:
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
            data_saver.add('gt_inertia', inertia_to_one_hot_vec(local_I))
            data_saver.add(
                'gt_inertia_normalized',
                util.normalize(inertia_to_one_hot_vec(local_I), output_mean,
                               output_std))
            data_saver.add('est_inertia_normalized',
                           np.copy(np.squeeze(output)))
            data_saver.add('est_inertia', np.copy(denormalized_output))
            data_saver.advance()

        # Save Image
        if (VIDEO_RECORD) and (count % RECORD_FREQ == 0):
            frame = pybullet_util.get_camera_image([1.4, 0.5, 1.], 2.0, 120,
                                                   -15, 0, 60, 1920, 1080, 0.1,
                                                   500.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % count
            cv2.imwrite(filename, frame)

        # Disable forward step
        # p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

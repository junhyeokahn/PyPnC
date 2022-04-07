import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil
from tqdm import tqdm

import cv2
import pybullet as p
import numpy as np
import pickle

np.set_printoptions(precision=2)

from config.draco_manipulation_config import SimConfig, WalkingConfig, ManipulationConfig
from pnc.draco_manipulation_pnc.draco_manipulation_interface import DracoManipulationInterface
from util import pybullet_util
from util import util
from util import liegroup

import ipdb

import logging

for f in os.listdir('data'):
    if f == "d3_manipulation_data_gen.log":
        os.remove('data/' + f)

logging.basicConfig(filename='data/d3_manipulation_data_gen.log',
                    filemode='w',
                    format='%(message)s',
                    level=logging.DEBUG)

B_VISUALIZE = False

GRIPPER_JOINTS = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2",
    "right_ezgripper_knuckle_palm_L1_1", "right_ezgripper_knuckle_L1_L2_1",
    "right_ezgripper_knuckle_palm_L1_2", "right_ezgripper_knuckle_L1_L2_2"
]
RIGHTUP_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
KEYPOINT_OFFSET = 0.1

TRACKING_ERROR_THRESHOLD = [0.02, 0.02, 0.04, 0.1]
NOMINAL_BASE_COM_HEIGHT = 0.9


################### SAFETY
def is_safe(robot, link_id, sensor_data):
    safe = True
    rfoot_pos = pybullet_util.get_link_iso(robot,
                                           link_id['r_foot_contact'])[0:3, 3]
    lfoot_pos = pybullet_util.get_link_iso(robot,
                                           link_id['l_foot_contact'])[0:3, 3]

    if sensor_data['base_com_pos'][1] <= rfoot_pos[1]:
        logging.info("reasoning: com falling to the right")
        safe = False
    if sensor_data['base_com_pos'][1] >= lfoot_pos[1]:
        safe = False
        logging.info("reasoning: com falling to the left")
    if sensor_data['base_com_pos'][0] >= 0.2:
        safe = False
    if sensor_data['base_com_pos'][0] <= -0.2:
        logging.info("reasoning: com soared")
        safe = False
    if np.abs(sensor_data['base_com_pos'][2] - NOMINAL_BASE_COM_HEIGHT) >= 0.1:
        logging.info("reasoning: com flopped down")
        safe = False

    return safe


def is_tracking_error_safe(goal_pos, angle, robot, link_id):
    safe = True
    lh_pos = pybullet_util.get_link_iso(robot, link_id['l_hand_contact'])[0:3,
                                                                          3]
    lh_ori = pybullet_util.get_link_iso(robot, link_id['l_hand_contact'])[0:3,
                                                                          0:3]
    for i in range(3):
        if np.abs(lh_pos[i] - goal_pos[i]) >= TRACKING_ERROR_THRESHOLD[i]:
            logging.info("reasoning: bad tracking, {}th error is {}".format(
                i, lh_pos[i] - goal_pos[i]))
            safe = False
    # lh_angle =

    return safe


################### SAFETY


def x_rot(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def generate_keypoint(target_iso):
    ret = np.copy(target_iso)
    R_wl = np.copy(target_iso[0:3, 0:3])
    local_z_offset = np.array([0., 0., KEYPOINT_OFFSET])
    global_z_offset = np.dot(R_wl, local_z_offset)
    ret[0:3, 3] += global_z_offset

    return ret


def set_initial_config(robot, joint_id):
    # Upperbody
    p.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)

    # Lowerbody
    hip_yaw_angle = 5
    p.resetJointState(robot, joint_id["l_hip_aa"], np.radians(hip_yaw_angle),
                      0.)
    p.resetJointState(robot, joint_id["l_hip_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jp"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_knee_fe_jd"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_ankle_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["l_ankle_ie"],
                      np.radians(-hip_yaw_angle), 0.)

    p.resetJointState(robot, joint_id["r_hip_aa"], np.radians(-hip_yaw_angle),
                      0.)
    p.resetJointState(robot, joint_id["r_hip_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_knee_fe_jp"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_knee_fe_jd"], np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_ankle_fe"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_ankle_ie"], np.radians(hip_yaw_angle),
                      0.)


def run_sim(inp):
    goal_pos = np.copy(inp)
    angle = 0.

    if not p.isConnected():
        if not B_VISUALIZE:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                         cameraYaw=180 + 45,
                                         cameraPitch=-15,
                                         cameraTargetPosition=[0.5, 0.5, 0.6])
    p.resetSimulation()

    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                numSubSteps=SimConfig.N_SUBSTEP)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
        SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, False)

    if B_VISUALIZE:
        lh_target_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                     [0., 0, 0.], [0, 0, 0, 1])
        lh_target_pos = np.array([0., 0., 0.])
        lh_target_quat = np.array([0., 0., 0., 1.])

        lh_waypoint_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                       [0., 0, 0.], [0, 0, 0, 1])
        lh_waypoint_pos = np.array([0., 0., 0.])
        lh_waypoint_quat = np.array([0., 0., 0., 1.])

    # Add Gear constraint
    c = p.createConstraint(robot,
                           link_id['l_knee_fe_lp'],
                           robot,
                           link_id['l_knee_fe_ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

    c = p.createConstraint(robot,
                           link_id['r_knee_fe_lp'],
                           robot,
                           link_id['r_knee_fe_ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.)
    gripper_attached_joint_id = OrderedDict()
    gripper_attached_joint_id["l_wrist_pitch"] = joint_id["l_wrist_pitch"]
    gripper_attached_joint_id["r_wrist_pitch"] = joint_id["r_wrist_pitch"]
    pybullet_util.set_joint_friction(robot, gripper_attached_joint_id, 0.1)

    # Construct Interface
    interface = DracoManipulationInterface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    if B_VISUALIZE:
        pybullet_util.draw_link_frame(robot,
                                      link_id['l_hand_contact'],
                                      text="lh")
        pybullet_util.draw_link_frame(lh_target_frame, -1, text="lh_target")
        pybullet_util.draw_link_frame(lh_waypoint_frame, -1)

    gripper_command = dict()
    for gripper_joint in GRIPPER_JOINTS:
        gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
            gripper_joint]

    waiting_command = True
    time_command_recved = 0.

    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        for gripper_joint in GRIPPER_JOINTS:
            del sensor_data['joint_pos'][gripper_joint]
            del sensor_data['joint_vel'][gripper_joint]

        rf_height = pybullet_util.get_link_iso(robot,
                                               link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot,
                                               link_id['l_foot_contact'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        if t >= WalkingConfig.INIT_STAND_DUR + 0.1 and waiting_command:
            global_goal_pos = np.copy(goal_pos)
            global_goal_pos[0] += sensor_data['base_com_pos'][0]
            global_goal_pos[1] += sensor_data['base_com_pos'][1]
            lh_target_pos = np.copy(global_goal_pos)
            lh_target_rot = np.dot(RIGHTUP_GRIPPER, x_rot(angle))
            lh_target_quat = util.rot_to_quat(lh_target_rot)
            lh_target_iso = liegroup.RpToTrans(lh_target_rot, lh_target_pos)
            lh_waypoint_pos = generate_keypoint(lh_target_iso)[0:3, 3]

            interface.interrupt_logic.lh_target_pos = lh_target_pos
            interface.interrupt_logic.lh_waypoint_pos = lh_waypoint_pos
            interface.interrupt_logic.lh_target_quat = lh_target_quat
            interface.interrupt_logic.b_interrupt_button_one = True

            waiting_command = False
            time_command_recved = t

        command = interface.get_command(copy.deepcopy(sensor_data))

        if B_VISUALIZE:
            p.resetBasePositionAndOrientation(lh_target_frame, lh_target_pos,
                                              lh_target_quat)
            p.resetBasePositionAndOrientation(lh_waypoint_frame,
                                              lh_waypoint_pos, lh_target_quat)

        # Exclude Knee Proximal Joints Command
        del command['joint_pos']['l_knee_fe_jp']
        del command['joint_pos']['r_knee_fe_jp']
        del command['joint_vel']['l_knee_fe_jp']
        del command['joint_vel']['r_knee_fe_jp']
        del command['joint_trq']['l_knee_fe_jp']
        del command['joint_trq']['r_knee_fe_jp']

        # Apply Command
        pybullet_util.set_motor_trq(robot, joint_id, command['joint_trq'])
        pybullet_util.set_motor_pos(robot, joint_id, gripper_command)

        p.stepSimulation()
        t += dt
        count += 1

        ## Safety On the run
        if is_safe(robot, link_id, sensor_data):
            pass
        else:
            success = False
            del interface
            break

        ## Safety at the end
        if t >= time_command_recved + ManipulationConfig.T_REACHING_DURATION + 0.2:
            if is_tracking_error_safe(global_goal_pos, angle, robot, link_id):
                success = True
            else:
                success = False
            del interface
            break

    return success


N = 1024

prior_rollouts = []

for i in tqdm(range(N)):
    inp = np.random.uniform([0.3, -0.2, 0.7], [0.7, 0.5, 1.0])
    logging.info('*' * 80)
    logging.info('input: {}'.format(inp))
    success = run_sim(inp)
    logging.info('label: {}'.format(success))
    prior_rollouts.append([inp, success])

for f in os.listdir('data'):
    if f == "prior_rollouts.pkl":
        os.remove('data/' + f)
with open('data/prior_rollouts.pkl', 'ab') as f:
    pickle.dump(prior_rollouts, f)
"""

run_sim(np.array([0.6, 0.6, 0.85, 0.]))
print("Done")
run_sim(np.array([0.6, 0.6, 0.85, 0.]))
print("Done")
run_sim(np.array([0.6, 0.6, 0.85, 0.]))
"""

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import cv2
import pybullet as p
import numpy as np

np.set_printoptions(precision=2)

from config.draco_manipulation_config import SimConfig
from pnc.draco_manipulation_pnc.draco_manipulation_interface import DracoManipulationInterface
from util import pybullet_util
from util import util
from util import liegroup
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

import ipdb
import pickle

######################## Key stroke ########################
############## Gripper
# z: close left gripper
# x: open left gripper
# /: close right gripper
# ,: open right gripper
############## Manipulation
## 1: move left hand
# interface.interrupt_logic.lh_target_pos = lh_target_pos
# interface.interrupt_logic.lh_waypoint_pos = lh_waypoint_pos
# interface.interrupt_logic.lh_target_quat = lh_target_quat
# interface.interrupt_logic.b_interrupt_button_one = True
## 3: move right hand
# interface.interrupt_logic.rh_target_pos = rh_target_pos
# interface.interrupt_logic.rh_waypoint_pos = rh_waypoint_pos
# interface.interrupt_logic.rh_target_quat = rh_target_quat
# interface.interrupt_logic.b_interrupt_button_three = True
############## Locomotion
## m: move in x
# interface.interrupt_logic.com_displacement_x = com_displacement_x
# interface.interrupt_logic.b_interrupt_button_m = True
## n: move in y
# interface.interrupt_logic.com_displacement_y = com_displacement_y
# interface.interrupt_logic.b_interrupt_button_n = True
## 5: walk in place
## 4: walk left
## 6: walk right
## 2: walk backward
## 8: walk forward
############################################################

######################## Variables ########################
## Standby
b_left_gripper_ready = False
t_left_gripper_command_recv = 0.
b_right_gripper_ready = False
t_right_gripper_command_recv = 0.
t_gripper_stab_dur = 2.
b_left_hand_ready = False
b_right_hand_ready = False
b_walk_ready = False
############################################################

gripper_joints = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2",
    "right_ezgripper_knuckle_palm_L1_1", "right_ezgripper_knuckle_L1_L2_1",
    "right_ezgripper_knuckle_palm_L1_2", "right_ezgripper_knuckle_L1_L2_2"
]

KEYPOINT_OFFSET = 0.1
RIGHTUP_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

SAFETY_THRESHOLD = 0.5

grid_location = util.GridLocation(np.array([0.05, 0.05, 0.05]))
saf_list = [None] * 7
for i in range(7):
    with open('saf/saf_{}.pkl'.format(i), 'rb') as f:
        saf_list[i] = pickle.load(f)


def is_lh_reachable(sensor_data, global_goal):
    """
    0: falling right
    1: falling left
    2: falling forward
    3: falling backward
    4: falling down
    5: falling up
    6: no reaching
    """
    local_goal = global_goal - sensor_data['base_com_pos']
    local_goal[2] = global_goal[2]
    normalized_goal = local_goal - np.array([0.6, 0.25, 0.85])
    grid_idx = grid_location.get_grid_idx(normalized_goal)
    labels = [False] * 7
    print("lhand")
    print("grid: ", grid_idx)
    for i in range(7):
        print("{}th prob: {}".format(i, saf_list[i][grid_idx]))
        labels[i] = False if saf_list[i][grid_idx] < SAFETY_THRESHOLD else True

    return labels


def is_rh_reachable(sensor_data, global_goal):
    """
    0: falling right
    1: falling left
    2: falling forward
    3: falling backward
    4: falling down
    5: falling up
    6: no reaching
    """
    local_goal = global_goal - sensor_data['base_com_pos']
    local_goal[1] *= -1
    local_goal[2] = global_goal[2]
    normalized_goal = local_goal - np.array([0.6, 0.25, 0.85])
    grid_idx = grid_location.get_grid_idx(normalized_goal)
    labels = [False] * 7
    print("rhand")
    print("grid: ", grid_idx)
    for i in range(7):
        print("{}th prob: {}".format(i, saf_list[i][grid_idx]))
        labels[i] = False if saf_list[i][grid_idx] < SAFETY_THRESHOLD else True

    return labels


def is_colliding(start, waypoint, goal, obs_min, obs_max, threshold, n):
    """
    start, waypoint, goal, OBS_MIN, OBS_MAX, THRESHOLD: 3d np.array
    N: number of check point
    """
    if util.is_colliding_3d(start, waypoint, obs_min, obs_max, threshold, n):
        return false
    if util.is_colliding_3d(waypoint, goal, obs_min, obs_max, threshold, n):
        return false
    return true


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


def signal_handler(signal, frame):
    if SimConfig.VIDEO_RECORD:
        pybullet_util.make_video(video_dir, False)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # Environment Setup
    if SimConfig.B_USE_MESHCAT:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                     cameraYaw=180 + 45,
                                     cameraPitch=-15,
                                     cameraTargetPosition=[0.5, 0.5, 0.6])

    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                numSubSteps=SimConfig.N_SUBSTEP)
    if SimConfig.VIDEO_RECORD:
        video_dir = 'video/draco3_pnc'
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
        SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    lh_target_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                 [0., 0, 0.], [0, 0, 0, 1])
    lh_target_pos = np.array([0., 0., 0.])
    lh_target_quat = np.array([0., 0., 0., 1.])
    rh_target_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf", [0, 0, 0],
                                 [0, 0, 0, 1])
    rh_target_pos = np.array([0., 0., 0.])
    rh_target_quat = np.array([0., 0., 0., 1.])

    lh_waypoint_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                   [0., 0, 0.], [0, 0, 0, 1])
    lh_waypoint_pos = np.array([0., 0., 0.])
    lh_waypoint_quat = np.array([0., 0., 0., 1.])
    rh_waypoint_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                   [0, 0, 0], [0, 0, 0, 1])
    rh_waypoint_pos = np.array([0., 0., 0.])
    rh_waypoint_quat = np.array([0., 0., 0., 1.])

    com_target_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                  [0., 0, 0.], [0, 0, 0, 1])
    com_target_x = 0.
    com_target_y = 0.

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    xOffset = 1.0

    p.loadURDF(cwd + "/robot_model/bookcase/bookshelf.urdf",
               useFixedBase=1,
               basePosition=[0 + xOffset, 0, 0.025],
               baseOrientation=[0, 0, 0.7068252, 0.7068252])
    p.loadURDF(cwd + "/robot_model/bookcase/red_can.urdf",
               useFixedBase=0,
               basePosition=[0 + xOffset, 0.75, 1.05])
    p.loadURDF(cwd + "/robot_model/bookcase/green_can.urdf",
               useFixedBase=0,
               basePosition=[0 + xOffset, -0.7, 1.35])
    p.loadURDF(cwd + "/robot_model/bookcase/blue_can.urdf",
               useFixedBase=0,
               basePosition=[0 + xOffset, 0, 0.7])

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

    if SimConfig.B_USE_MESHCAT:
        # Create Robot for Meshcat Visualization
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        try:
            viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        viz.loadViewerModel()
        vis_q = pin.neutral(model)

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
    jpg_count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    # Draw Frames
    pybullet_util.draw_link_frame(robot, link_id['r_hand_contact'], text="rh")
    pybullet_util.draw_link_frame(robot, link_id['l_hand_contact'], text="lh")
    pybullet_util.draw_link_frame(robot, link_id['camera'], text="camera")
    pybullet_util.draw_link_frame(lh_target_frame, -1, text="lh_target")
    pybullet_util.draw_link_frame(rh_target_frame, -1, text="rh_target")
    pybullet_util.draw_link_frame(lh_waypoint_frame, -1)
    pybullet_util.draw_link_frame(rh_waypoint_frame, -1)
    pybullet_util.draw_link_frame(com_target_frame, -1, text="com_target")

    gripper_command = dict()
    for gripper_joint in gripper_joints:
        gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
            gripper_joint]

    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        for gripper_joint in gripper_joints:
            del sensor_data['joint_pos'][gripper_joint]
            del sensor_data['joint_vel'][gripper_joint]

        rf_height = pybullet_util.get_link_iso(robot,
                                               link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot,
                                               link_id['l_foot_contact'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, '8'):
            interface.interrupt_logic.b_interrupt_button_eight = True
        elif pybullet_util.is_key_triggered(keys, '5'):
            interface.interrupt_logic.b_interrupt_button_five = True
        elif pybullet_util.is_key_triggered(keys, '4'):
            interface.interrupt_logic.b_interrupt_button_four = True
        elif pybullet_util.is_key_triggered(keys, '2'):
            interface.interrupt_logic.b_interrupt_button_two = True
        elif pybullet_util.is_key_triggered(keys, '6'):
            interface.interrupt_logic.b_interrupt_button_six = True
        elif pybullet_util.is_key_triggered(keys, '7'):
            interface.interrupt_logic.b_interrupt_button_seven = True
        elif pybullet_util.is_key_triggered(keys, '9'):
            interface.interrupt_logic.b_interrupt_button_nine = True
        elif pybullet_util.is_key_triggered(keys, '0'):
            interface.interrupt_logic.b_interrupt_button_zero = True
        elif pybullet_util.is_key_triggered(keys, '1'):
            ## Update target pos and quat here
            lh_target_pos = np.array([0.6, 0.02, 0.83])
            lh_target_rot = np.dot(RIGHTUP_GRIPPER, x_rot(0.))
            lh_target_quat = util.rot_to_quat(lh_target_rot)
            lh_target_iso = liegroup.RpToTrans(lh_target_rot, lh_target_pos)
            lh_waypoint_pos = generate_keypoint(lh_target_iso)[0:3, 3]

            print("lh reachability: ",
                  is_lh_reachable(sensor_data, lh_target_pos))

            interface.interrupt_logic.lh_target_pos = lh_target_pos
            interface.interrupt_logic.lh_waypoint_pos = lh_waypoint_pos
            interface.interrupt_logic.lh_target_quat = lh_target_quat
            interface.interrupt_logic.b_interrupt_button_one = True
        elif pybullet_util.is_key_triggered(keys, '3'):
            ## Update target pos and quat here
            rh_target_pos = np.array([0.4, 0., 0.83])
            rh_target_rot = np.dot(RIGHTUP_GRIPPER, x_rot(0.))
            rh_target_quat = util.rot_to_quat(rh_target_rot)
            rh_target_iso = liegroup.RpToTrans(rh_target_rot, rh_target_pos)
            rh_waypoint_pos = generate_keypoint(rh_target_iso)[0:3, 3]

            print("rh reachability: ",
                  is_rh_reachable(sensor_data, rh_target_pos))

            interface.interrupt_logic.rh_target_pos = rh_target_pos
            interface.interrupt_logic.rh_waypoint_pos = rh_waypoint_pos
            interface.interrupt_logic.rh_target_quat = rh_target_quat
            interface.interrupt_logic.b_interrupt_button_three = True
        elif pybullet_util.is_key_triggered(keys, 't'):
            interface.interrupt_logic.b_interrupt_button_t = True
        elif pybullet_util.is_key_triggered(keys, 'z'):
            for k, v in gripper_command.items():
                if k.split('_')[0] == "left":
                    gripper_command[k] += 1.94 / 3.
            t_left_gripper_command_recv = t
        elif pybullet_util.is_key_triggered(keys, 'x'):
            for k, v in gripper_command.items():
                if k.split('_')[0] == "left":
                    gripper_command[k] -= 1.94 / 3.
            t_left_gripper_command_recv = t
        elif pybullet_util.is_key_triggered(keys, '/'):
            for k, v in gripper_command.items():
                if k.split('_')[0] == "right":
                    gripper_command[k] += 1.94 / 3.
            t_right_gripper_command_recv = t
        elif pybullet_util.is_key_triggered(keys, '.'):
            for k, v in gripper_command.items():
                if k.split('_')[0] == "right":
                    gripper_command[k] -= 1.94 / 3.
            t_right_gripper_command_recv = t
        elif pybullet_util.is_key_triggered(keys, 'm'):
            com_target_x = 0.35
            interface.interrupt_logic.com_displacement_x = com_target_x
            interface.interrupt_logic.b_interrupt_button_m = True
        elif pybullet_util.is_key_triggered(keys, 'n'):
            com_target_y = 0.23
            interface.interrupt_logic.com_displacement_y = com_target_y
            interface.interrupt_logic.b_interrupt_button_n = True
        elif pybullet_util.is_key_triggered(keys, 'r'):
            interface.interrupt_logic.b_interrupt_button_r = True
        elif pybullet_util.is_key_triggered(keys, 'e'):
            interface.interrupt_logic.b_interrupt_button_e = True

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        p.resetBasePositionAndOrientation(lh_target_frame, lh_target_pos,
                                          lh_target_quat)
        p.resetBasePositionAndOrientation(rh_target_frame, rh_target_pos,
                                          rh_target_quat)
        p.resetBasePositionAndOrientation(lh_waypoint_frame, lh_waypoint_pos,
                                          lh_target_quat)
        p.resetBasePositionAndOrientation(rh_waypoint_frame, rh_waypoint_pos,
                                          rh_target_quat)
        p.resetBasePositionAndOrientation(com_target_frame,
                                          [com_target_x, com_target_y, 0.02],
                                          [0., 0., 0., 1.])

        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Exclude Knee Proximal Joints Command
        del command['joint_pos']['l_knee_fe_jp']
        del command['joint_pos']['r_knee_fe_jp']
        del command['joint_vel']['l_knee_fe_jp']
        del command['joint_vel']['r_knee_fe_jp']
        del command['joint_trq']['l_knee_fe_jp']
        del command['joint_trq']['r_knee_fe_jp']

        # Apply Command
        pybullet_util.set_motor_trq(robot, joint_id, command['joint_trq'])
        # pybullet_util.set_motor_impedance(robot, joint_id, command,
        # SimConfig.KP, SimConfig.KD)

        pybullet_util.set_motor_pos(robot, joint_id, gripper_command)

        # Update Stanby variables
        if t >= t_left_gripper_command_recv + t_gripper_stab_dur:
            b_left_gripper_ready = True
        else:
            b_left_gripper_ready = False
        if t >= t_right_gripper_command_recv + t_gripper_stab_dur:
            b_right_gripper_ready = True
        else:
            b_right_gripper_ready = False
        b_left_hand_ready = interface.interrupt_logic.b_left_hand_ready
        b_right_hand_ready = interface.interrupt_logic.b_right_hand_ready
        b_walk_ready = interface.interrupt_logic.b_walk_ready

        # Save Image
        if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
            frame = pybullet_util.get_camera_image([1., 0.5, 1.], 1.0, 120,
                                                   -15, 0, 60., 1920, 1080,
                                                   0.1, 100.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % jpg_count
            cv2.imwrite(filename, frame)
            jpg_count += 1

        if SimConfig.B_USE_MESHCAT:
            vis_q[0:3] = sensor_data['base_joint_pos']
            vis_q[3:7] = sensor_data['base_joint_quat']
            for i, (k, v) in enumerate(sensor_data['joint_pos'].items()):
                idx = interface._robot.get_q_idx(k)
                vis_q[idx] = v
            viz.display(vis_q)

        p.stepSimulation()
        # time.sleep(dt)
        t += dt
        count += 1

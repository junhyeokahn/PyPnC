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

from rosnode.draco_manipulation_rosnode import DracoManipulationRosnode

np.set_printoptions(precision=2)

from config.draco_manipulation_config import SimConfig
from pnc.draco_manipulation_pnc.draco_manipulation_interface import DracoManipulationInterface
from util import pybullet_util
from util import util
from util import liegroup
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

import ipdb

gripper_joints = [
    "left_ezgripper_knuckle_palm_L1_1", "left_ezgripper_knuckle_L1_L2_1",
    "left_ezgripper_knuckle_palm_L1_2", "left_ezgripper_knuckle_L1_L2_2",
    "right_ezgripper_knuckle_palm_L1_1", "right_ezgripper_knuckle_L1_L2_1",
    "right_ezgripper_knuckle_palm_L1_2", "right_ezgripper_knuckle_L1_L2_2"
]

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
    robot = p.loadURDF(cwd + "/robot_model/draco3/draco3_gripper.urdf",
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

    com_target_frame = p.loadURDF(cwd + "/robot_model/etc/ball.urdf",
                                  [0., 0, 0.], [0, 0, 0, 1])
    com_target_x = 0.
    com_target_y = 0.

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    xOffset = 1.5

    p.loadURDF(cwd + "/robot_model/bookcase/bookshelf.urdf",
               useFixedBase=1,
               basePosition=[0 + xOffset, 0, 0.025],
               baseOrientation=[0, 0, 0.7068252, 0.7068252])
    p.loadURDF(cwd + "/robot_model/bookcase/red_can.urdf",
               useFixedBase=0,
               basePosition=[-0.4 + xOffset, 0.75, 1.05])
    p.loadURDF(cwd + "/robot_model/bookcase/green_can.urdf",
               useFixedBase=0,
               basePosition=[-0.4 + xOffset, -0.7, 1.35])
    p.loadURDF(cwd + "/robot_model/bookcase/blue_can.urdf",
               useFixedBase=0,
               basePosition=[-0.4 + xOffset, 0.1, 0.7])

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

    # Construct Rosnode
    rosnode = DracoManipulationRosnode(robot, link_id)

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
    # pybullet_util.draw_link_frame(com_target_frame, -1, text="com_target")

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
        # command = self._rosnode.apply_command(self._interface)
        keys = p.getKeyboardEvents()
        command = -1
        if pybullet_util.is_key_triggered(keys, '3'):
            command = 3
        # command = rosnode.get_command()
        if command == 8:
            interface.interrupt_logic.b_interrupt_button_eight = True
        elif command == 5:
            interface.interrupt_logic.b_interrupt_button_five = True
        elif command == 4:
            interface.interrupt_logic.b_interrupt_button_four = True
        elif command == 2:
            interface.interrupt_logic.b_interrupt_button_two = True
        elif command == 6:
            interface.interrupt_logic.b_interrupt_button_six = True
        elif command == 7:
            interface.interrupt_logic.b_interrupt_button_seven = True
        elif command == 9:
            interface.interrupt_logic.b_interrupt_button_nine = True
        elif command == 0:
            interface.interrupt_logic.b_interrupt_button_zero = True
        elif command == 1:
            ## Update target pos and quat here
            lh_target_pos = np.array([0.29, 0.23, 0.96])
            lh_target_quat = np.array([0.2, -0.64, -0.21, 0.71])
            interface.interrupt_logic.lh_target_pos = lh_target_pos
            interface.interrupt_logic.lh_target_quat = lh_target_quat
            interface.interrupt_logic.b_interrupt_button_one = True
        elif command == 3:
            ## Update target pos and quat here
            # rh_target_pos = np.array([0.29, -0.24, 0.96])
            # rh_target_pos = np.array([1.1438913892327487, 0.5728717559681102, 0.9297155125446818])
            rh_target_pos = np.array([1.06, .088, 0.787])
            # rh_target_quat = np.array([-0.14, -0.66, 0.15, 0.72])
            # rh_target_quat = np.array([0.7982853216171371, 0.13284627061082785, -0.273157764732058, 0.5200742728130325])
            rh_target_quat = np.array([0.7793, 0.2959338171752087, -0.17152155322599372, 0.5250648597243313])
            interface.interrupt_logic.rh_target_pos = rh_target_pos
            interface.interrupt_logic.rh_target_quat = rh_target_quat
            interface.interrupt_logic.b_interrupt_button_three = True
        elif command == 10:
            interface.interrupt_logic.b_interrupt_button_t = True
        elif command == 11:
            for k, v in gripper_command.items():
                gripper_command[k] += 1.94 / 3.
        elif command == 12:
            for k, v in gripper_command.items():
                gripper_command[k] -= 1.94 / 3.

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        p.resetBasePositionAndOrientation(lh_target_frame, lh_target_pos,
                                          lh_target_quat)
        p.resetBasePositionAndOrientation(rh_target_frame, rh_target_pos,
                                          rh_target_quat)
        # p.resetBasePositionAndOrientation(com_target_frame,
        #                                   [com_target_x, com_target_y, 0.02],
        #                                   [0., 0., 0., 1.])

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
        pybullet_util.set_motor_pos(robot, joint_id, gripper_command)

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

        rosnode.publish_data(sensor_data)

        p.stepSimulation()
        # time.sleep(dt)
        t += dt
        count += 1

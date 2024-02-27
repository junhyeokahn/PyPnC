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

from util import pybullet_util
from util import util
from util import liegroup

CONTROLLER_DT = 0.01
N_SUBSTEP = 10
CAMERA_DT = 0.05
KP = 0.
KD = 0.

INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

PRINT_ROBOT_INFO = True
VIDEO_RECORD = False
RECORD_FREQ = 10


def set_initial_config(robot, joint_id):
    # Upperbody
    p.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["l_wrist_ps"], np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["r_wrist_ps"], -np.pi / 6, 0.)

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
    if VIDEO_RECORD:
        pybullet_util.make_video(video_dir)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=0.25,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    # p.setGravity(0, 0, -9.8)
    p.setGravity(0, 0, 0)
    p.setPhysicsEngineParameter(fixedTimeStep=CONTROLLER_DT,
                                numSubSteps=N_SUBSTEP)
    if VIDEO_RECORD:
        video_dir = 'video/draco3_kinematics'
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/draco3/draco3_pin_model.urdf",
                       INITIAL_POS_WORLD_TO_BASEJOINT,
                       INITIAL_QUAT_WORLD_TO_BASEJOINT,
                       useFixedBase=True)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT,
        PRINT_ROBOT_INFO)

    # Add Gear constraint
    c = p.createConstraint(robot,
                           link_id['l_knee_fe_lp'],
                           robot,
                           link_id['l_knee_fe_ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=2)

    c = p.createConstraint(robot,
                           link_id['r_knee_fe_lp'],
                           robot,
                           link_id['r_knee_fe_ld'],
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 1, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=2)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 1., 1.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 5)

    # Run Sim
    t = 0
    dt = CONTROLLER_DT
    count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    # Draw Camera Link
    pybullet_util.draw_link_frame(robot, link_id['r_camera'], text="r_camera")
    pybullet_util.draw_link_frame(robot, link_id['l_camera'], text="l_camera")

    # Add debug parameters
    param_ids = []
    for k, v in joint_id.items():
        param_ids.append(
            p.addUserDebugParameter(k, -4, 4,
                                    nominal_sensor_data['joint_pos'][k]))

    while (1):

        # Get SensorData
        if count % (CAMERA_DT / CONTROLLER_DT) == 0:
            l_camera_img = pybullet_util.get_camera_image_from_link(
                robot, link_id['l_camera'], 50, 10, 60., 0.1, 10)
            # r_camera_img = pybullet_util.get_camera_image_from_link(
            # robot, link_id['r_camera'], 50, 10, 60., 0.1, 10)

        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        rf_height = pybullet_util.get_link_iso(robot,
                                               link_id['r_foot_contact'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot,
                                               link_id['l_foot_contact'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, '8'):
            pass
        elif pybullet_util.is_key_triggered(keys, '5'):
            pass
        elif pybullet_util.is_key_triggered(keys, '4'):
            pass
        elif pybullet_util.is_key_triggered(keys, '2'):
            pass
        elif pybullet_util.is_key_triggered(keys, '6'):
            pass
        elif pybullet_util.is_key_triggered(keys, '7'):
            pass
        elif pybullet_util.is_key_triggered(keys, '9'):
            pass

        # Save Image
        if (VIDEO_RECORD) and (count % RECORD_FREQ == 0):
            frame = pybullet_util.get_camera_image([1.2, 0.5, 1.], 2.0, 120,
                                                   -15, 0, 60., 1920, 1080,
                                                   0.1, 100.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % count
            cv2.imwrite(filename, frame)

        for i in range(len(param_ids)):
            c = param_ids[i]
            target_pos = p.readUserDebugParameter(c)
            # distal joint is just mimicking the proximal joint
            if list(joint_id.values())[i] == joint_id['l_knee_fe_jd']:
                pass
            elif list(joint_id.values())[i] == joint_id['r_knee_fe_jd']:
                pass
            else:
                p.setJointMotorControl2(robot,
                                        list(joint_id.values())[i],
                                        p.POSITION_CONTROL,
                                        target_pos,
                                        force=200.)

        p.stepSimulation()

        # time.sleep(dt)
        t += dt
        count += 1

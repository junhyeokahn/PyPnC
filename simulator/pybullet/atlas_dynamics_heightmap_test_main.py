import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import matplotlib.pyplot as plt
import cv2
import pybullet as p
import numpy as np
np.set_printoptions(precision=2)

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import pybullet_util
from util import util
from util import liegroup
from vision.height_map import HeightMap
from util.pybullet_camera_util import Camera


def set_initial_config(robot, joint_id):
    # shoulder_x
    p.resetJointState(robot, joint_id["l_arm_shx"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_arm_shx"], np.pi / 4, 0.)
    # elbow_y
    # p.resetJointState(robot, joint_id["l_arm_ely"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["r_arm_ely"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["l_arm_ely"], 0., 0.)
    p.resetJointState(robot, joint_id["r_arm_ely"], 0., 0.)
    # elbow_x
    # p.resetJointState(robot, joint_id["l_arm_elx"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["r_arm_elx"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["l_arm_elx"], 0., 0.)
    p.resetJointState(robot, joint_id["r_arm_elx"], 0., 0.)
    # hip_y
    p.resetJointState(robot, joint_id["l_leg_hpy"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_hpy"], -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, joint_id["l_leg_kny"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_leg_kny"], np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, joint_id["l_leg_aky"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_aky"], -np.pi / 4, 0.)
    # head
    p.resetJointState(robot, joint_id["neck_ry"], np.pi / 4, 0.)


def signal_handler(signal, frame):
    if SimConfig.VIDEO_RECORD:
        pybullet_util.make_video(video_dir)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                numSubSteps=SimConfig.N_SUBSTEP)
    if SimConfig.VIDEO_RECORD:
        video_dir = 'video/atlas_pnc'

        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
                       SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)
    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.loadURDF(cwd + "/robot_model/ground/stair.urdf", [0.5, 0, 0],
               useFixedBase=True)
    # p.loadURDF(cwd + "/robot_model/ground/stair.urdf",[1,0,0],useFixedBase=True)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    # Construct Interface
    interface = AtlasInterface()

    # Construct Heightmap
    heightmap = HeightMap(1000, 100, 15, 1.5)

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0
    camera_img_count = 0

    while (1):

        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        rf_height = pybullet_util.get_link_iso(robot, link_id['r_sole'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot, link_id['l_sole'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Get cameradata
        if count % (SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
            # fov = 60
            # nearval = 0.01
            # farval = 1000
            # camera_img = pybullet_util.get_camera_image_from_link(
                # robot, link_id['head'], 128, 128, fov, nearval, farval)
            # depth_buffer = camera_img[3]
            # view_matrix = camera_img[5]
            # projection_matrix = camera_img[6]
            # camera_pos = camera_img[7]
            
            cam = Camera(128,128)
            cam.set_view_matrix_from_robot_link(robot, link_id['head'])
            cam.set_projection_matrix(fovy=60, aspect=1, near=0.01, far=10)

            rgb_img, depth_img, seg_img = cam.get_pybullet_image()
            pcl_points, pcl_colors = cam.unproject_canvas_to_pointcloud(rgb_img,
                    depth_img)

            #show the depth image
            cam.show_image(depth_img, RGB=False, title="Image 1")

            print("pcl data",pcl_points)
            __import__('ipdb').set_trace()

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

        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Apply Trq
        pybullet_util.set_motor_trq(robot, joint_id, command['joint_trq'])

        # Save Image
        if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
            frame = pybullet_util.get_camera_image([1.2, 0.5, 1.], 2.0, 120,
                                                   -15, 0, 60., 1920, 1080,
                                                   0.1, 100.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % count
            cv2.imwrite(filename, frame)

        p.stepSimulation()

        # time.sleep(dt)
        t += dt
        count += 1

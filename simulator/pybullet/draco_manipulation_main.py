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
    # if SimConfig.VIDEO_RECORD:
    #     pybullet_util.make_video(video_dir, False)
    p.disconnect()
    sys.exit(0)
    
    
signal.signal(signal.SIGINT, signal_handler)

# EW: No longer need this to be a class after actually implementing the rosnode, can return
#   to original loop in __main__
class DracoManipulationMain():
    def __init__(self):
        self._rosnode = DracoManipulationRosnode()

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
        # robot = p.loadURDF(
        # cwd + "/robot_model/draco3/draco3.urdf",
        self._robot = p.loadURDF(cwd + "/robot_model/draco3/draco3_gripper.urdf",
                           SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                           SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

        p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._nq, self._nv, self._na, self._joint_id, self._link_id, self._pos_basejoint_to_basecom, self._rot_basejoint_to_basecom = pybullet_util.get_robot_config(
            self._robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
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
        c = p.createConstraint(self._robot,
                               self._link_id['l_knee_fe_lp'],
                               self._robot,
                               self._link_id['l_knee_fe_ld'],
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=500, erp=10)

        c = p.createConstraint(self._robot,
                               self._link_id['r_knee_fe_lp'],
                               self._robot,
                               self._link_id['r_knee_fe_ld'],
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
            self._viz = MeshcatVisualizer(model, collision_model, visual_model)
            try:
                self._viz.initViewer(open=True)
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            self._viz.loadViewerModel()
            self._vis_q = pin.neutral(model)

        # Initial Config
        set_initial_config(self._robot, self._joint_id)

        # Link Damping
        pybullet_util.set_link_damping(self._robot, self._link_id.values(), 0., 0.)

        # Joint Friction
        pybullet_util.set_joint_friction(self._robot, self._joint_id, 0.)
        gripper_attached_joint_id = OrderedDict()
        gripper_attached_joint_id["l_wrist_pitch"] = self._joint_id["l_wrist_pitch"]
        gripper_attached_joint_id["r_wrist_pitch"] = self._joint_id["r_wrist_pitch"]
        pybullet_util.set_joint_friction(self._robot, gripper_attached_joint_id, 0.1)

        # Construct Interface
        self._interface = DracoManipulationInterface()

        # Run Sim
        self._t = 0
        self._dt = SimConfig.CONTROLLER_DT
        self._count = 0
        self._jpg_count = 0

        nominal_sensor_data = pybullet_util.get_sensor_data(
            self._robot, self._joint_id, self._link_id, self._pos_basejoint_to_basecom,
            self._rot_basejoint_to_basecom)

        self._gripper_command = dict()
        for gripper_joint in gripper_joints:
            self._gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
                gripper_joint]

    def main(self):
        while (1):

            # Get SensorData
            sensor_data = pybullet_util.get_sensor_data(self._robot, self._joint_id, self._link_id,
                                                        self._pos_basejoint_to_basecom,
                                                        self._rot_basejoint_to_basecom)

            for gripper_joint in gripper_joints:
                del sensor_data['joint_pos'][gripper_joint]
                del sensor_data['joint_vel'][gripper_joint]

            rf_height = pybullet_util.get_link_iso(self._robot,
                                                   self._link_id['r_foot_contact'])[2, 3]
            lf_height = pybullet_util.get_link_iso(self._robot,
                                                   self._link_id['l_foot_contact'])[2, 3]
            sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
            sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

            # TODO: Once draco interface is updated, modify this to match.
            #   Look at gazebo_scorpio_plugin to see how it worked
            #       one command at a time or multiple?
            #       override commands or queue?
            # Seems as though it was handled by scorpiointerface methods isReadyToMove
            #   and isReadyToGrasp. If not ready then seems as though command was just discarded
            command = self._rosnode.apply_command(self._interface)
            #
            if command == 11:
                for k, v in self._gripper_command.items():
                    self._gripper_command[k] += 1.94 / 3.
            elif command == 12:
                for k, v in self._gripper_command.items():
                    self._gripper_command[k] -= 1.94 / 3.

            # Compute Command
            if SimConfig.PRINT_TIME:
                start_time = time.time()
            command = self._interface.get_command(copy.deepcopy(sensor_data))

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
            pybullet_util.set_motor_trq(self._robot, self._joint_id, command['joint_trq'])
            pybullet_util.set_motor_pos(self._robot, self._joint_id, self._gripper_command)

            # Save Image
            if (SimConfig.VIDEO_RECORD) and (self._count % SimConfig.RECORD_FREQ == 0):
                frame = pybullet_util.get_camera_image([1., 0.5, 1.], 1.0, 120,
                                                       -15, 0, 60., 1920, 1080,
                                                       0.1, 100.)
                frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
                filename = self._video_dir + '/step%06d.jpg' % self._jpg_count
                cv2.imwrite(filename, frame)
                self._jpg_count += 1

            if SimConfig.B_USE_MESHCAT:
                self._vis_q[0:3] = sensor_data['base_joint_pos']
                self._vis_q[3:7] = sensor_data['base_joint_quat']
                for i, (k, v) in enumerate(sensor_data['joint_pos'].items()):
                    idx = self._interface._robot.get_q_idx(k)
                    self._vis_q[idx] = v
                self._viz.display(self._vis_q)

            self._rosnode.publish_data(sensor_data)

            p.stepSimulation()
            # time.sleep(dt)
            self._t += self._dt
            self._count += 1

if __name__ == "__main__":
    main_class = DracoManipulationMain()
    main_class.main()
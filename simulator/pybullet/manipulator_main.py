import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict

import pybullet as p
import numpy as np

from config.manipulator_config import SimConfig


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

    if SimConfig.PRINT_ROBOT_INFO:
        print("=" * 80)
        print("SimulationRobot")
        print("nq: ", nq, ", nv: ", nv, ", na: ", na)
        print("+" * 80)
        print("Joint Infos")
        util.pretty_print(joint_id)
        print("+" * 80)
        print("Link Infos")
        util.pretty_print(link_id)

    return nq, nv, na, joint_id, link_id


def set_initial_config(robot, joint_id):
    ############################################################################
    ## TODO (Cassie)
    ## Set initial positions for your robot
    ## You need to use resetJointState API from pybullet.
    ## Ex) p.resetJointState(robot, joint_id["l_arm_shx"], -np.pi / 4, 0.)
    ## See the documentation for more info
    ############################################################################
    pass


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def set_motor_trq(robot, joint_id, command):
    assert len(joint_id) == len(command['joint_trq'])
    trq_applied = OrderedDict()
    for (joint_name, pos_des), (_, vel_des), (_, trq_des) in zip(
            command['joint_pos'].items(), command['joint_vel'].items(),
            command['joint_trq'].items()):
        joint_state = p.getJointState(robot, joint_id[joint_name])
        joint_pos, joint_vel = joint_state[0], joint_state[1]
        trq_applied[joint_id[joint_name]] = (trq_des + SimConfig.KP *
                                             (pos_des - joint_pos) +
                                             SimConfig.KD *
                                             (vel_des - joint_vel))
    p.setJointMotorControlArray(robot,
                                trq_applied.keys(),
                                controlMode=p.TORQUE_CONTROL,
                                forces=trq_applied.values())


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
        if not os.path.exists('video'):
            os.makedirs('video')
        for f in os.listdir('video'):
            os.remove('video/' + f)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/manipulator.mp4")

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    ############################################################################
    ## TODO (Cassie)
    ## Write your own urdf file and import.
    ## You need to use optional flag useFixedBase for the manipulator.
    ############################################################################
    # robot = p.loadURDF(
    # cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
    # [0, 0, 1.5 - 0.761])

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id = get_robot_config(robot)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 2)

    # Construct Interface
    ############################################################################
    ## TODO (Cassie & Junhyeok)
    ## We will make this interface class later
    ## I haven't explain this much yet
    ############################################################################
    # interface = ManipulatorInterface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0
    while (1):

        # Get SensorData
        sensor_data = get_sensor_data(robot, joint_id, link_id)

        # Compute Command
        command = interface.get_command(sensor_data)

        # Apply Trq
        set_motor_trq(robot, joint_id, command)

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1

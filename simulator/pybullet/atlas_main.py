import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p
import numpy as np

from config.atlas_config import SimConfig


def print_joint_info(robot):
    for i in range(p.getNumJoints(robot)):
        print(p.getJointInfo(robot, i))


def set_initial_config(robot):
    # shoulder_x
    p.resetJointState(robot, 4, -0.785, 0.)
    p.resetJointState(robot, 12, 0.785, 0.)
    # elbow_y
    p.resetJointState(robot, 5, -1.57, 0.)
    p.resetJointState(robot, 13, 1.57, 0.)
    # elbow_x
    p.resetJointState(robot, 6, -1.57, 0.)
    p.resetJointState(robot, 14, -1.57, 0.)
    # hip_y
    p.resetJointState(robot, 20, -np.pi / 4, 0.)
    p.resetJointState(robot, 27, -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, 21, np.pi / 2, 0.)
    p.resetJointState(robot, 28, np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, 22, -np.pi / 4, 0.)
    p.resetJointState(robot, 29, -np.pi / 4, 0.)
    # soles
    # print(p.getLinkState(robot, 24))
    # print(p.getLinkState(robot, 31))


def set_joint_friction(robot, max_force=0):
    num_joint = p.getNumJoints(robot)
    p.setJointMotorControlArray(robot, [i for i in range(num_joint)],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * num_joint)


def set_motor_trq(robot, trq):
    num_joint = p.getNumJoints(robot)
    assert num_joint == trq.shape[0]
    p.setJointMotorControlArray(robot, [i for i in range(num_joint)],
                                trq.tolist())


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=2,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.dt, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        [0, 0, 1.5 - 0.761])
    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Print Robot Info
    print_joint_info(robot)

    # Initial Config
    set_initial_config(robot)

    # Joint Friction
    set_joint_friction(robot, 1)

    # Run Sim
    t = 0
    dt = SimConfig.dt
    while (1):
        time.sleep(dt)
        t += dt
        # interface.get_command()
        # set_motor_trq(robot, trq)
        p.stepSimulation()

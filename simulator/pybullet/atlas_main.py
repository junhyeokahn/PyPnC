import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math

import pybullet as p
import numpy as np

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface


def get_robot_config(robot):
    nq, nv, na, joint_id, link_id = 0, 0, 0, dict(), dict()
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        print(info)
        if info[2] != p.JOINT_FIXED:
            joint_id[info[1].decode("utf-8")] = info[0]
        link_id[info[12].decode("utf-8")] = info[0]
        nq = max(nq, info[3])
        nv = max(nv, info[4])
    nq += 1
    nv += 1
    na = len(joint_id)
    link_id[(p.getBodyInfo(robot)[0]).decode("utf-8")] = -1

    return nq, nv, na, joint_id, link_id


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


def set_joint_friction(robot, joint_id, max_force=0):
    p.setJointMotorControlArray(robot, [*joint_id.values()],
                                p.VELOCITY_CONTROL,
                                forces=[max_force] * len(joint_id))


def set_motor_trq(robot, joint_id, trq):
    assert len(joint_id) == trq.shape[0]
    p.setJointMotorControlArray(robot, [*joint_id.values()], trq.tolist())


def get_sensor_data(robot, joint_id):
    sensor_data = dict()

    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    sensor_data['base_pos'] = np.asarray(base_pos)
    sensor_data['base_quat'] = np.asarray(base_quat)

    base_lin_vel, base_ang_vel = p.getBaseVelocity(robot)
    sensor_data['base_lin_vel'] = np.asarray(base_lin_vel)
    sensor_data['base_ang_vel'] = np.asarray(base_ang_vel)

    joint_states = p.getJointStates(robot, [*joint_id.values()])
    sensor_data['joint_pos'] = np.array([js[0] for js in joint_states])
    sensor_data['joint_vel'] = np.array([js[1] for js in joint_states])

    return sensor_data


if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=120,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[1, 0.5, 1.5])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.DT, numSubSteps=1)

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(
        cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf",
        [0, 0, 1.5 - 0.761])

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id = get_robot_config(robot)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Joint Friction
    set_joint_friction(robot, joint_id, 1)

    # Construct Interface
    interface = AtlasInterface()

    # Run Sim
    t = 0
    dt = SimConfig.DT
    while (1):
        time.sleep(dt)
        t += dt
        sensor_data = get_sensor_data(robot, joint_id)
        # command = interface.get_command(sensor_data)
        # set_motor_trq(robot, joint_id, command['joint_trq'])
        p.stepSimulation()

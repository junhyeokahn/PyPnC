import pybullet as p
import numpy as np
import time

np.set_printoptions(precision=2)

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from util import pybullet_util
from util import util

INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.761]
INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]
PRINT_ROBOT_INFO = False


def set_initial_config(robot, joint_id):
    # shoulder_x
    p.resetJointState(robot, joint_id["l_arm_shx"], -1.57, 0.)
    # p.resetJointState(robot, joint_id["r_arm_shx"], np.pi / 4, 0.)
    # elbow_y
    # p.resetJointState(robot, joint_id["l_arm_ely"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["r_arm_ely"], np.pi / 2, 0.)
    # elbow_x
    # p.resetJointState(robot, joint_id["l_arm_elx"], -np.pi / 2, 0.)
    # p.resetJointState(robot, joint_id["r_arm_elx"], -np.pi / 2, 0.)
    # hip_y
    p.resetJointState(robot, joint_id["l_leg_hpy"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_hpy"], -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, joint_id["l_leg_kny"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_leg_kny"], np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, joint_id["l_leg_aky"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_aky"], -np.pi / 4, 0.)

    # back_bkx
    p.resetJointState(robot, joint_id["back_bkx"], 0.5235, 0.)


if __name__ == "__main__":

    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                 cameraYaw=90,
                                 cameraPitch=0,
                                 cameraTargetPosition=[0, 0, 1.])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf",
                       INITIAL_POS_WORLD_TO_BASEJOINT,
                       INITIAL_QUAT_WORLD_TO_BASEJOINT,
                       useFixedBase=1)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, INITIAL_POS_WORLD_TO_BASEJOINT, INITIAL_QUAT_WORLD_TO_BASEJOINT,
        PRINT_ROBOT_INFO)

    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)

    set_initial_config(robot, joint_id)
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    print("initial joint_pos: ", nominal_sensor_data['joint_pos'])

    p.setRealTimeSimulation(1)

    paramIds = []
    for k in joint_id.keys():
        paramIds.append(
            p.addUserDebugParameter(k, -4, 4,
                                    nominal_sensor_data['joint_pos'][k]))

    while (1):
        p.setGravity(0, 0, -9.81)
        for i in range(len(paramIds)):
            c = paramIds[i]
            targetPos = p.readUserDebugParameter(c)
            p.setJointMotorControl2(robot,
                                    list(joint_id.values())[i],
                                    p.POSITION_CONTROL,
                                    targetPos,
                                    force=5 * 240.)
        time.sleep(0.01)

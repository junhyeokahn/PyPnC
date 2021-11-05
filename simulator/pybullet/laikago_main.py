import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import time

import pybullet as p
import numpy as np

from util import pybullet_util

p.connect(p.GUI)
plane = p.loadURDF(cwd + "/robot_model/ground/plane.urdf")
p.setGravity(0, 0, -9.8)
dt = 0.002
p.setTimeStep(dt)

ini_pos = [0., 0., 0.5]
ini_quat = [0.5, 0.5, 0.5, 0.5]

quadruped = p.loadURDF(cwd + "/robot_model/laikago/laikago_toes.urdf",
                       ini_pos,
                       ini_quat,
                       useFixedBase=False)

nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    quadruped, ini_pos, ini_quat, True)

p.resetJointState(quadruped, joint_id["FR_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["FL_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["RR_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)
p.resetJointState(quadruped, joint_id["RL_lower_leg_2_upper_leg_joint"], -0.56,
                  0.)

pybullet_util.set_link_damping(quadruped, link_id.values(), 0., 0.)
pybullet_util.set_joint_friction(quadruped, joint_id, 0)

nominal_sensor_data = pybullet_util.get_sensor_data(quadruped, joint_id,
                                                    link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

t = 0
count = 0

while (1):

    # Get sensor data
    sensor_data = pybullet_util.get_sensor_data(quadruped, joint_id, link_id,
                                                pos_basejoint_to_basecom,
                                                rot_basejoint_to_basecom)

    # Set motor cmd
    pybullet_util.set_motor_pos(quadruped, joint_id,
                                nominal_sensor_data['joint_pos'])

    p.stepSimulation()
    t += dt
    count += 1
    time.sleep(dt)

    print("-" * 80)
    print("toeFR")
    print(pybullet_util.get_link_iso(quadruped, link_id['toeFR']))
    print("toeFL")
    print(pybullet_util.get_link_iso(quadruped, link_id['toeFL']))
    print("toeRR")
    print(pybullet_util.get_link_iso(quadruped, link_id['toeRR']))
    print("toeRL")
    print(pybullet_util.get_link_iso(quadruped, link_id['toeRL']))
    print("base pos")
    print(sensor_data["base_com_pos"])
    print("base quat")
    print(sensor_data["base_com_quat"])

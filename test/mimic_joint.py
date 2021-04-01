import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from util import pybullet_util
#a mimic joint can act as a gear between two joints
#you can control the gear ratio in magnitude and sign (>0 reverses direction)

import pybullet as p
import time
import pybullet_data

print(pybullet_data.getDataPath())
p.connect(p.GUI)
p.resetDebugVisualizerCamera(cameraDistance=0.2,
                             cameraYaw=0,
                             cameraPitch=-45,
                             cameraTargetPosition=[0., 0., 0.])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", 0, 0, -2)
wheelA = p.loadURDF("differential/diff_ring.urdf", [0, 0, 0])

nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
    wheelA, [0., 0., 0.], [0., 0., 0., 1.], True)

for i in range(p.getNumJoints(wheelA)):
    print(p.getJointInfo(wheelA, i))
    p.setJointMotorControl2(wheelA,
                            i,
                            p.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=0)

c = p.createConstraint(wheelA,
                       1,
                       wheelA,
                       3,
                       jointType=p.JOINT_GEAR,
                       jointAxis=[0, 1, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=10000)

# c = p.createConstraint(wheelA,
# 2,
# wheelA,
# 4,
# jointType=p.JOINT_GEAR,
# jointAxis=[0, 1, 0],
# parentFramePosition=[0, 0, 0],
# childFramePosition=[0, 0, 0])
# p.changeConstraint(c, gearRatio=-1, maxForce=10000)

# c = p.createConstraint(wheelA,
# 1,
# wheelA,
# 4,
# jointType=p.JOINT_GEAR,
# jointAxis=[0, 1, 0],
# parentFramePosition=[0, 0, 0],
# childFramePosition=[0, 0, 0])
# p.changeConstraint(c, gearRatio=-1, maxForce=10000)

p.setRealTimeSimulation(1)
count = 0
while (1):
    p.setGravity(0, 0, -10)
    time.sleep(0.01)

    sensor_data = pybullet_util.get_sensor_data(wheelA, joint_id, link_id,
                                                pos_basejoint_to_basecom,
                                                rot_basejoint_to_basecom)
    if count % 10 == 0:
        print("+" * 80)
        print("count: ", count)
        print(sensor_data['joint_pos'])
        print(sensor_data['joint_vel'])

    # p.setJointMotorControlArray(wheelA, [1, 3],
    # controlMode=p.VELOCITY_CONTROL,
    # targetVelocities=[0.5, 0.5])

    p.setJointMotorControlArray(wheelA, [1],
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=[0.5])

    count += 1

# p.removeConstraint(c)

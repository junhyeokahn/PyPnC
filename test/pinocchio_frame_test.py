import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import pinocchio as pin
import numpy as np

urdf_file = cwd + "/robot_model/manipulator/three_link_manipulator.urdf"
model = pin.buildModelFromUrdf(urdf_file)
data = model.createData()
print(model)

q = np.array([np.pi / 2., 0., 0.])
# q = np.zeros(3)
qdot = np.ones(3)

pin.forwardKinematics(model, data, q, qdot)

## Print Frame Names
print([frame.name for frame in model.frames])

## Calculate j2 placement
j2_frame = model.getFrameId('j1')
j2_translation = pin.updateFramePlacement(model, data, j2_frame)
print("j2 translation")
print(j2_translation)

## Calculate l2 placement
l2_frame = model.getFrameId('l2')
l2_translation = pin.updateFramePlacement(model, data, l2_frame)
print("l2 translation")
print(l2_translation)

## Calculate j2 jacobian
pin.computeJointJacobians(model, data, q)
j2_jacobian = pin.getFrameJacobian(model, data, j2_frame,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print("j2 jacobian")
print(j2_jacobian)

## Calculate l2 jacobian
l2_jacobian = pin.getFrameJacobian(model, data, l2_frame,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print("l2 jacobian")
print(l2_jacobian)

## Calculate j2 spatial velocity
j2_vel = pin.getFrameVelocity(model, data, j2_frame)
print("j2 vel")
print(j2_vel)

## Calculate l2 spatial velocity
l2_vel = pin.getFrameVelocity(model, data, l2_frame,
                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print("l2 vel")
print(l2_vel)
print(np.dot(l2_jacobian, qdot))

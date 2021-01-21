import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time
import math
import copy
from collections import OrderedDict

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

from pinocchio.visualize import MeshcatVisualizer

urdf_filename = cwd + "/robot_model/atlas/atlas_v4_with_multisense.urdf"
package_dir = cwd + "/robot_model/atlas/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_filename, package_dir, pin.JointModelFreeFlyer())
data, collision_data, visual_data = pin.createDatas(model, collision_model,
                                                    visual_model)
q0 = pin.neutral(model)
q0[0] = 0.1

pin.forwardKinematics(model, data, q0)
for name, oMi in zip(model.names, data.oMi):
    print("{}: {} {} {}".format(name, *oMi.translation.T.flat))

__import__('ipdb').set_trace()
print("Done")

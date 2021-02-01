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

# urdf_filename = cwd + "/robot_model/atlas/atlas.urdf"
# package_dir = cwd + "/robot_model/atlas/"
urdf_filename = cwd + "/robot_model/valkyrie/valkyrie.urdf"
package_dir = cwd + "/robot_model/valkyrie/"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_filename, package_dir, pin.JointModelFreeFlyer())

viz = MeshcatVisualizer(model, collision_model, visual_model)

try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

viz.loadViewerModel()

q0 = pin.neutral(model)
for i in range(1000):
    q0[0] += 0.001
    # print(i, q0)
    viz.display(q0)

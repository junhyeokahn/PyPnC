import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

import meshcat.geometry as g

# Display Robot in Meshcat Visualizer
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
    cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
viz = MeshcatVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer(open=True)
    viz.viewer.wait()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
viz.loadViewerModel(rootNodeName="draco3")
vis_q = pin.neutral(model)
viz.display(vis_q)

# Display Reachable spaces
ee_list = ['LF', 'RF', 'LH', 'RH']
for ee_name in ee_list:
    meshPath = cwd + "/pnc/reachability_map/output/draco3_" + ee_name + ".stl"
    filename = os.path.join(meshPath)
    obj = g.Mesh(g.StlMeshGeometry.from_file(filename))
    obj.material.transparent = True
    obj.material.opacity = 0.6
    viz.viewer["reachability"][ee_name].set_object(obj)


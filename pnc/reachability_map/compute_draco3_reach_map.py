"""
Constructs the reachable space of the end effectors of Draco3
assuming the base is fixed. The reachable space is randomly sampled,
then a convex hull is created using these points, and finally
the polytopes are simplified using a decimation process.

This follows more or less the approach from
Tonneau, Steve, et al. "An efficient acyclic contact planner for
multiped robots." IEEE Transactions on Robotics 34.3 (2018): 586-601.
"""
import os
import sys
import numpy as np
import open3d as o3d
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

# Robot model libraries
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem

b_visualize_hulls = False
decimation_triangles = 30
N_samples = 10000

def generate_random_joint_dict(joint_id):
    q_rand = np.random.uniform(low=q_llim, high=q_ulim, size=nq)
    for k, v in joint_id.items():
        q_dict[k] = q_rand[v]
        zero_qdot_dict[k] = 0.


# Load Draco3 robot model
robot = PinocchioRobotSystem(
                cwd + "/robot_model/draco3/draco3.urdf",
                cwd + "/robot_model/draco3", b_fixed_base=True)

joint_id = robot.joint_id
nq = robot.n_q
q_dict = OrderedDict()
zero_qdot_dict = OrderedDict()
q_lims = robot.joint_pos_limit
q_llim, q_ulim = q_lims[:, 0],  q_lims[:, 1]

# Loop to generate end effector points
reach_space = OrderedDict()
# ee_links = OrderedDict()
end_effectors = ['LF', 'RF', 'LH', 'RH']
end_effector_names = ['l_foot_contact', 'r_foot_contact',
                      'l_hand_contact', 'r_hand_contact']
for ee in end_effectors:
    reach_space[ee] = np.zeros((N_samples, 3))      # [x,y,z] for all samples

for n in range(N_samples):

    # generate new random state, q_dict
    generate_random_joint_dict(joint_id)

    # update robot model and get corresponding end effector position
    robot.update_system(None, None, None, None, None, None, None, None,
                        q_dict, zero_qdot_dict)
    for ee, ee_link in zip(end_effectors, end_effector_names):
        reach_space[ee][n] = robot.get_link_iso(ee_link)[:3, 3]

    # Print progress every 10% increment
    if (n != 0) and n % (0.1*N_samples) == 0:
        print(f'Reachability space generation complete: {(n/N_samples)*100}%')

# Define path to save convex hull
save_loc = cwd + '/pnc/reachability_map/output/'

# Create convex hulls
mesh_simplified = OrderedDict()
for ee_name, rs in reach_space.items():
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(rs)
    hull, _ = pcl.compute_convex_hull()

    # Decimation
    mesh_simplified[ee_name] = hull.simplify_quadric_decimation(decimation_triangles)
    if b_visualize_hulls:
        o3d.visualization.draw_geometries([mesh_simplified[ee_name]],
                                          mesh_show_wireframe=True)

    # compute triangle normals and save meshes as STL
    filename = 'draco3_' + ee_name + '.stl'
    mesh_simplified[ee_name] = o3d.geometry.TriangleMesh.\
        compute_triangle_normals(mesh_simplified[ee_name])
    o3d.io.write_triangle_mesh(save_loc + filename, mesh_simplified[ee_name])

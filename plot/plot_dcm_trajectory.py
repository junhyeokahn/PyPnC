import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib.pyplot as plt
import math, matplotlib
matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


'Conversion from quaternion to rotation matrix'


def quat2mat(quat):
    quat = np.squeeze(np.asarray(quat))
    x, y, z, w = quat
    return np.matrix([[
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w
    ], [
        2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w
    ], [
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
    ]])


def compute_arrow_vec(quats):
    n_points = quats.shape[0]
    arrow_ends = np.zeros([n_points, 3])
    for i in range(n_points):
        R = quat2mat(quats[i])
        arrow_ends[i] = np.dot(R, np.array([1., 0., 0.]))
    return arrow_ends


def plot_foot(ax, pos, ori, color, text):
    if text:
        ax.text(pos[0], pos[1] + 0.03, pos[2] + 0.05, text, color=color)
    rmat = quat2mat(ori)
    normal = np.array([rmat[0, 2], rmat[1, 2], rmat[2, 2]])
    d = -pos.dot(normal)
    xx, yy = np.meshgrid(np.linspace(pos[0] - 0.11, pos[0] + 0.11, 2),
                         np.linspace(pos[1] - 0.065, pos[1] + 0.065, 2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_wireframe(xx, yy, z, color=color, linewidth=1.5)
    ax.plot_surface(xx, yy, z, edgecolors=color, color=color, alpha=0.5)
    ax.scatter(xs=pos[0],
               ys=pos[1],
               zs=pos[2],
               zdir='z',
               s=50,
               c=color,
               depthshade=True)


def main(args):
    file = args.file

    with open(file, 'rb') as file:
        while True:
            try:
                d = pickle.load(file)

                initial_time = d["temporal_parameters"]["initial_time"]
                final_time = d["temporal_parameters"]["final_time"]
                time_step = d["temporal_parameters"]["time_step"]
                t_ds = d["temporal_parameters"]["t_ds"]
                t_ss = d["temporal_parameters"]["t_ss"]
                t_transfer = d["temporal_parameters"]["t_transfer"]

                curr_rfoot_contact_pos = d["contact"]["curr_right_foot"]["pos"]
                curr_rfoot_contact_ori = d["contact"]["curr_right_foot"]["ori"]
                curr_lfoot_contact_pos = d["contact"]["curr_left_foot"]["pos"]
                curr_lfoot_contact_ori = d["contact"]["curr_left_foot"]["ori"]

                rfoot_contact_pos = d["contact"]["right_foot"]["pos"]
                rfoot_contact_ori = d["contact"]["right_foot"]["ori"]
                lfoot_contact_pos = d["contact"]["left_foot"]["pos"]
                lfoot_contact_ori = d["contact"]["left_foot"]["ori"]

                t = d["reference"]["time"]
                com_pos_ref = d["reference"]["com_pos"]
                com_vel_ref = d["reference"]["com_vel"]
                base_ori_ref = d["reference"]["base_ori"]
            except EOFError:
                break

    arrow_ends = compute_arrow_vec(base_ori_ref)
    line_styles = {0: '--', 1: '-', 2: '--', 3: '-'}
    colors = {0: 'red', 1: 'magenta', 2: 'blue', 3: 'cyan'}
    line_colors = {
        0: 'cornflowerblue',
        1: 'sandybrown',
        2: 'seagreen',
        3: 'gold'
    }
    # ==========================================================================
    # Plot Motion
    # ==========================================================================
    offset = 0.05
    axis_tick_size = 10
    axis_label_size = 14
    axis_tick_color = '#434440'
    axis_label_color = '#373834'
    comref_linewidth = 2
    comref_linecolor = 'darkorange'
    dcmref_linewidth = 4
    dcmref_linecolor = 'cornflowerblue'

    fig1 = plt.figure()
    com_motion = Axes3D(fig1)

    # plot com
    com_motion.plot(xs=com_pos_ref[:, 0],
                    ys=com_pos_ref[:, 1],
                    zs=com_pos_ref[:, 2],
                    linewidth=comref_linewidth,
                    color=comref_linecolor)
    num_interval = 50
    com_motion.quiver(com_pos_ref[::num_interval, 0],
                      com_pos_ref[::num_interval, 1],
                      com_pos_ref[::num_interval, 2],
                      arrow_ends[::num_interval, 0],
                      arrow_ends[::num_interval, 1],
                      arrow_ends[::num_interval, 2],
                      length=0.1,
                      linewidth=comref_linewidth,
                      color='red')
    # plot foot
    plot_foot(com_motion, np.squeeze(curr_rfoot_contact_pos),
              np.squeeze(curr_rfoot_contact_ori), colors[0], "InitRF")
    plot_foot(com_motion, np.squeeze(curr_lfoot_contact_pos),
              np.squeeze(curr_lfoot_contact_ori), colors[1], "InitLF")
    for i, (pos, ori) in enumerate(zip(rfoot_contact_pos, rfoot_contact_ori)):
        plot_foot(com_motion, pos, ori, colors[0], "RF" + str(i))
    for i, (pos, ori) in enumerate(zip(lfoot_contact_pos, lfoot_contact_ori)):
        plot_foot(com_motion, pos, ori, colors[1], "LF" + str(i))

    com_motion.tick_params(labelsize=axis_tick_size, colors=axis_tick_color)
    com_motion.set_xlabel("x",
                          fontsize=axis_label_size,
                          color=axis_label_color)
    com_motion.set_ylabel("y",
                          fontsize=axis_label_size,
                          color=axis_label_color)
    com_motion.set_zlabel("z",
                          fontsize=axis_label_size,
                          color=axis_label_color)
    set_axes_equal(com_motion)

    # ==========================================================================
    # Plot Trajectory
    # ==========================================================================

    fig, axes = plt.subplots(3, 2)
    for i in range(3):
        axes[i, 0].plot(t, com_pos_ref[:, i], color='k', linewidth=3)
        axes[i, 1].plot(t, com_vel_ref[:, i], color='k', linewidth=3)
        axes[i, 0].grid(True)
        axes[i, 1].grid(True)
    axes[0, 0].set_title('com pos')
    axes[0, 1].set_title('com vel')

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    main(args)

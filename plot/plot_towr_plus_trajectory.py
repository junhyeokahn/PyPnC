import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle
import itertools

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
plt.rcParams.update({
    'axes.grid': True,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True
})
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

line_styles = {0: '--', 1: '-', 2: '--', 3: '-'}
colors = {0: 'red', 1: 'magenta', 2: 'blue', 3: 'cyan'}
line_colors = {0: 'cornflowerblue', 1: 'sandybrown', 2: 'seagreen', 3: 'gold'}
facecolors = [
    'cornflowerblue', 'darkorange', 'lightgray', 'rosybrown', 'seagreen',
    'gold', 'lightpink', 'slategrey'
]
motion_label = [
    r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$'
]
xyz_label = [r'$x$', r'$y$', r'$z$']
dxdydz_label = [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
frc_label = [r'$f_x$', r'$f_y$', r'$f_z$']
trq_label = ['$\tau_x$', r'$\tau_y$', r'$\tau_z$']


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


def euler_to_rot(seq, euler_angle, degrees=False):
    return (R.from_euler(seq, euler_angle, degrees=degrees)).as_matrix()


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


def compute_arrow_vec(euler_xyz):
    n_points = euler_xyz.shape[0]
    arrow_ends = np.zeros([n_points, 3])
    for i in range(n_points):
        R = euler_to_rot("xyz", euler_xyz[i], False)
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


def fill_phase(ax, ph_durations, facecolor_iter):
    t_start = 0
    shading = 0.2
    lb, ub = ax.get_ylim()
    for time_dur in ph_durations:
        t_end = t_start + time_dur
        ax.fill_between([t_start, t_end],
                        lb,
                        ub,
                        facecolor=next(facecolor_iter),
                        alpha=shading)
        t_start = t_end


def add_twinx(ax, x, y, color, linewidth):
    ax2 = ax.twinx()
    ax.spines['right'].set_color(color)
    ax2.plot(x, y, color=color, linewidth=linewidth)
    ax2.tick_params('y', colors=color)
    ax2.yaxis.label.set_color(color)

    def _adjust_grid(event):
        ylim1 = ax.get_ylim()
        len1 = ylim1[1] - ylim1[0]
        yticks1 = ax.get_yticks()
        rel_dist = [(y - ylim1[0]) / len1 for y in yticks1]
        ylim2 = ax2.get_ylim()
        len2 = ylim2[1] - ylim2[0]
        yticks2 = [ry * len2 + ylim2[0] for ry in rel_dist]
        ax2.set_yticks(yticks2)
        ax2.set_ylim(ylim2)

    fig = ax.get_figure()
    fig.canvas.mpl_connect('resize_event', _adjust_grid)

    return ax2


def main(args):
    file = args.file

    with open(file, 'r') as stream:
        try:
            # Read Trajectory
            data = yaml.load(stream, Loader=yaml.FullLoader)
            time = data["trajectory"]["time"]
            base_lin = np.array(data["trajectory"]["base_lin"])
            base_ang = np.array(data["trajectory"]["base_ang"])
            ee_motion_lin = dict()
            ee_wrench_lin = dict()
            for ee in range(2):
                ee_motion_lin[ee] = np.array(
                    data["trajectory"]["ee_motion_lin"][ee])
                ee_wrench_lin[ee] = np.array(
                    data["trajectory"]["ee_wrench_lin"][ee])

            # Read Node and Contact Schedule
            node_base_lin = np.array(data["node"]["base_lin"]["value"])
            node_base_lin_time = data["node"]["base_lin"]["duration"]
            node_base_lin_time = np.array([0] + [
                sum(node_base_lin_time[0:i + 1])
                for i in range(len(node_base_lin_time))
            ])

            node_base_ang = np.array(data["node"]["base_ang"]["value"])
            node_base_ang_time = data["node"]["base_lin"]["duration"]
            node_base_ang_time = np.array([0] + [
                sum(node_base_ang_time[0:i + 1])
                for i in range(len(node_base_ang_time))
            ])

            node_ee_motion_lin = dict()
            node_ee_motion_lin_time = dict()
            node_ee_wrench_lin = dict()
            node_ee_wrench_lin_time = dict()
            contact_schedule = dict()
            for ee in range(2):
                node_ee_motion_lin[ee] = np.array(
                    data["node"]["ee_motion_lin"][ee]["value"])
                node_ee_motion_lin_time[ee] = data["node"]["ee_motion_lin"][
                    ee]["duration"]
                node_ee_motion_lin_time[ee] = np.array([0] + [
                    sum(node_ee_motion_lin_time[ee][0:i + 1])
                    for i in range(len(node_ee_motion_lin_time[ee]))
                ])
                node_ee_wrench_lin[ee] = np.array(
                    data["node"]["ee_wrench_lin"][ee]["value"])
                node_ee_wrench_lin_time[ee] = data["node"]["ee_wrench_lin"][
                    ee]["duration"]
                node_ee_wrench_lin_time[ee] = np.array([0] + [
                    sum(node_ee_wrench_lin_time[ee][0:i + 1])
                    for i in range(len(node_ee_wrench_lin_time[ee]))
                ])

                contact_schedule[ee] = np.array(data["contact_schedule"][ee])

            # Read Parameter
            force_polynomials_per_stance_phase = np.array(
                data["parameter"]["force_polynomials_per_stance_phase"])
            ee_polynomials_per_swing_phase = np.array(
                data["parameter"]["ee_polynomials_per_swing_phase"])

        except yaml.YAMLError as exc:
            print(exc)

    arrow_ends = compute_arrow_vec(base_ang[:, 0:3])

    # ==========================================================================
    # Plot Motion
    # ==========================================================================
    fig1 = plt.figure()
    com_motion = Axes3D(fig1)

    # plot com
    com_motion.plot(xs=base_lin[:, 0],
                    ys=base_lin[:, 1],
                    zs=base_lin[:, 2],
                    linewidth=3,
                    color='darkorange')
    num_interval = 50
    com_motion.quiver(base_lin[::num_interval, 0],
                      base_lin[::num_interval, 1],
                      base_lin[::num_interval, 2],
                      arrow_ends[::num_interval, 0],
                      arrow_ends[::num_interval, 1],
                      arrow_ends[::num_interval, 2],
                      length=0.07,
                      linewidth=3,
                      color='red')
    # plot foot
    for ee in range(2):
        com_motion.plot(xs=ee_motion_lin[ee][:, 0],
                        ys=ee_motion_lin[ee][:, 1],
                        zs=ee_motion_lin[ee][:, 2],
                        linewidth=3,
                        color='cornflowerblue')
    # plot_foot(com_motion, np.squeeze(curr_rfoot_contact_pos),
    # np.squeeze(curr_rfoot_contact_ori), colors[0], "InitRF")
    # plot_foot(com_motion, np.squeeze(curr_lfoot_contact_pos),
    # np.squeeze(curr_lfoot_contact_ori), colors[1], "InitLF")
    # for i, (pos, ori) in enumerate(zip(rfoot_contact_pos, rfoot_contact_ori)):
    # plot_foot(com_motion, pos, ori, colors[0], "RF" + str(i))
    # for i, (pos, ori) in enumerate(zip(lfoot_contact_pos, lfoot_contact_ori)):
    # plot_foot(com_motion, pos, ori, colors[1], "LF" + str(i))

    com_motion.set_xlabel(r"$x$")
    com_motion.set_ylabel(r"$y$")
    com_motion.set_zlabel(r"$z$")
    set_axes_equal(com_motion)

    # ==========================================================================
    # Plot Trajectory
    # ==========================================================================

    fig, axes = plt.subplots(6, 4, constrained_layout=True)
    axes[0, 0].set_title('base lin')
    axes[0, 1].set_title('base ang')
    axes[0, 2].set_title('left foot')
    axes[0, 3].set_title('right foot')
    for i in range(6):
        axes[i, 0].plot(time, base_lin[:, i], color='k', linewidth=3)
        axes[i, 0].scatter(node_base_lin_time,
                           node_base_lin[:, i],
                           s=50,
                           c='lightgray',
                           linewidths=2,
                           edgecolors='k')
        axes[i, 0].set_ylabel(motion_label[i])
        axes[i, 1].plot(time, base_ang[:, i], color='k', linewidth=3)
        axes[i, 1].scatter(node_base_ang_time,
                           node_base_ang[:, i],
                           s=50,
                           c='lightgray',
                           linewidths=2,
                           edgecolors='k')
        axes[i, 1].set_ylabel(motion_label[i])
        for ee in range(2):
            if i < 3:
                pass
            else:
                # Draw EE Lin Motion
                axes[i, 2 + ee].plot(time,
                                     ee_motion_lin[ee][:, i - 3],
                                     color='b',
                                     linewidth=3)
                axes[i, 2 + ee].scatter(node_ee_motion_lin_time[ee],
                                        node_ee_motion_lin[ee][:, i - 3],
                                        s=50,
                                        c='skyblue',
                                        linewidths=2,
                                        edgecolors='b')
                axes[i, 2 + ee].tick_params('y', colors='b')
                axes[i, 2 + ee].spines['left'].set_color('b')
                axes[i, 2 + ee].set_ylabel(xyz_label[i - 3])
                axes[i, 2 + ee].yaxis.label.set_color('b')
                # Draw EE Lin Wrench
                frc_ax = add_twinx(axes[i, 2 + ee], time,
                                   ee_wrench_lin[ee][:, i - 3], 'r', 3)
                frc_ax.scatter(node_ee_wrench_lin_time[ee],
                               node_ee_wrench_lin[ee][:, i - 3],
                               s=50,
                               c='lightpink',
                               linewidths=2,
                               edgecolors='r')
                frc_ax.set_ylabel(frc_label[i - 3])

                # Fill Phase
                fill_phase(axes[i, 2 + ee], contact_schedule[ee],
                           itertools.cycle(facecolors[0:2]))

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    main(args)

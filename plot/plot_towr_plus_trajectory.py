import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle
import itertools

import yaml  ## TODO Replace this to ruamel.yaml
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
    'darkorange', 'cornflowerblue', 'lightgray', 'rosybrown', 'seagreen',
    'gold', 'lightpink', 'slategrey'
]
motion_label = [
    r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$'
]
xyz_label = [r'$x$', r'$y$', r'$z$']
dxdydz_label = [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
frc_label = [r'$f_x$', r'$f_y$', r'$f_z$']
trq_label = [r'$\tau_x$', r'$\tau_y$', r'$\tau_z$']


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


def euler_to_rot(angles):
    # Euler ZYX to Rot
    # Note that towr has (x, y, z) order
    x = angles[0]
    y = angles[1]
    z = angles[2]
    ret = np.array([
        np.cos(y) * np.cos(z),
        np.cos(z) * np.sin(x) * np.sin(y) - np.cos(x) * np.sin(z),
        np.sin(x) * np.sin(z) + np.cos(x) * np.cos(z) * np.sin(y),
        np.cos(y) * np.sin(z),
        np.cos(x) * np.cos(z) + np.sin(x) * np.sin(y) * np.sin(z),
        np.cos(x) * np.sin(y) * np.sin(z) - np.cos(z) * np.sin(x), -np.sin(y),
        np.cos(y) * np.sin(x),
        np.cos(x) * np.cos(y)
    ]).reshape(3, 3)
    return ret


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
        R = euler_to_rot(euler_xyz[i])
        arrow_ends[i] = np.dot(R, np.array([1., 0., 0.]))
    return arrow_ends


def plot_foot(ax, pos, ori, color, text):
    foot_half_len = 0.11
    foot_half_wid = 0.065
    if text:
        ax.text(pos[0], pos[1] + 0.03, pos[2] + 0.05, text, color=color)
    rmat = euler_to_rot(ori)
    normal = np.array([rmat[0, 2], rmat[1, 2], rmat[2, 2]])
    d = -pos.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-foot_half_len, foot_half_len, 2),
                         np.linspace(-foot_half_wid, foot_half_wid, 2))
    xx, yy = np.einsum('ji, mni->jmn', rmat[0:2, 0:2], np.dstack([xx, yy]))
    xx += pos[0]
    yy += pos[1]
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


def get_idx(time, t):
    idx = 0
    for _t in time:
        if _t >= t:
            break
        idx += 1
    return idx


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
            ee_motion_ang = dict()
            ee_wrench_lin = dict()
            ee_wrench_ang = dict()
            for ee in range(2):
                ee_motion_lin[ee] = np.array(
                    data["trajectory"]["ee_motion_lin"][ee])
                ee_motion_ang[ee] = np.array(
                    data["trajectory"]["ee_motion_ang"][ee])
                ee_wrench_lin[ee] = np.array(
                    data["trajectory"]["ee_wrench_lin"][ee])
                ee_wrench_ang[ee] = np.array(
                    data["trajectory"]["ee_wrench_ang"][ee])

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
            node_ee_motion_ang = dict()
            node_ee_motion_ang_time = dict()
            node_ee_wrench_lin = dict()
            node_ee_wrench_lin_time = dict()
            node_ee_wrench_ang = dict()
            node_ee_wrench_ang_time = dict()
            contact_schedule = dict()
            contact_time_instance = dict()
            contact_pos = dict()
            contact_ori = dict()
            for ee in range(2):
                node_ee_motion_lin[ee] = np.array(
                    data["node"]["ee_motion_lin"][ee]["value"])
                node_ee_motion_lin_time[ee] = data["node"]["ee_motion_lin"][
                    ee]["duration"]
                node_ee_motion_lin_time[ee] = np.array([0] + [
                    sum(node_ee_motion_lin_time[ee][0:i + 1])
                    for i in range(len(node_ee_motion_lin_time[ee]))
                ])
                node_ee_motion_ang[ee] = np.array(
                    data["node"]["ee_motion_ang"][ee]["value"])
                node_ee_motion_ang_time[ee] = data["node"]["ee_motion_ang"][
                    ee]["duration"]
                node_ee_motion_ang_time[ee] = np.array([0] + [
                    sum(node_ee_motion_ang_time[ee][0:i + 1])
                    for i in range(len(node_ee_motion_ang_time[ee]))
                ])

                node_ee_wrench_lin[ee] = np.array(
                    data["node"]["ee_wrench_lin"][ee]["value"])
                node_ee_wrench_lin_time[ee] = data["node"]["ee_wrench_lin"][
                    ee]["duration"]
                node_ee_wrench_lin_time[ee] = np.array([0] + [
                    sum(node_ee_wrench_lin_time[ee][0:i + 1])
                    for i in range(len(node_ee_wrench_lin_time[ee]))
                ])
                node_ee_wrench_ang[ee] = np.array(
                    data["node"]["ee_wrench_ang"][ee]["value"])
                node_ee_wrench_ang_time[ee] = data["node"]["ee_wrench_ang"][
                    ee]["duration"]
                node_ee_wrench_ang_time[ee] = np.array([0] + [
                    sum(node_ee_wrench_ang_time[ee][0:i + 1])
                    for i in range(len(node_ee_wrench_ang_time[ee]))
                ])

                contact_schedule[ee] = np.array(data["contact_schedule"][ee])
                contact_time_instance[ee] = []
                for idx, dur in enumerate(contact_schedule[ee]):
                    if idx % 2 == 0:
                        contact_time_instance[ee].append(
                            contact_schedule[ee][0:idx].sum() +
                            contact_schedule[ee][idx] / 2.)
                contact_pos[ee] = []
                contact_ori[ee] = []
                for ct in contact_time_instance[ee]:
                    idx = get_idx(time, ct)
                    contact_pos[ee].append(ee_motion_lin[ee][idx][0:3])
                    contact_ori[ee].append(ee_motion_ang[ee][idx][0:3])

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
                    color='black')
    num_interval = 50
    com_motion.quiver(base_lin[::num_interval, 0],
                      base_lin[::num_interval, 1],
                      base_lin[::num_interval, 2],
                      arrow_ends[::num_interval, 0],
                      arrow_ends[::num_interval, 1],
                      arrow_ends[::num_interval, 2],
                      length=0.07,
                      linewidth=3,
                      color='slategrey')

    # plot left foot
    for i, (pos, ori) in enumerate(zip(contact_pos[0], contact_ori[0])):
        plot_foot(com_motion, pos, ori, 'b', "LF" + str(i))
    com_motion.plot(xs=ee_motion_lin[0][:, 0],
                    ys=ee_motion_lin[0][:, 1],
                    zs=ee_motion_lin[0][:, 2],
                    linewidth=3,
                    color='cornflowerblue')
    # plot right foot
    for i, (pos, ori) in enumerate(zip(contact_pos[1], contact_ori[1])):
        plot_foot(com_motion, pos, ori, 'r', "RF" + str(i))
    com_motion.plot(xs=ee_motion_lin[1][:, 0],
                    ys=ee_motion_lin[1][:, 1],
                    zs=ee_motion_lin[1][:, 2],
                    linewidth=3,
                    color='tomato')

    com_motion.set_xlabel(r"$x$")
    com_motion.set_ylabel(r"$y$")
    com_motion.set_zlabel(r"$z$")
    set_axes_equal(com_motion)

    # ==========================================================================
    # Plot Trajectory
    # ==========================================================================

    fig, axes = plt.subplots(6, 6, constrained_layout=True)
    axes[0, 0].set_title('base lin')
    axes[0, 1].set_title('base ang')
    axes[0, 2].set_title('lf motion')
    axes[0, 3].set_title('rf motion')
    axes[0, 4].set_title('lf wrench')
    axes[0, 5].set_title('rf wrench')
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
                # Draw EE Ang Motion
                axes[i, 2 + ee].plot(time,
                                     ee_motion_ang[ee][:, i],
                                     color='k',
                                     linewidth=3)
                axes[i, 2 + ee].scatter(node_ee_motion_ang_time[ee],
                                        node_ee_motion_ang[ee][:, i],
                                        s=50,
                                        c='lightgray',
                                        linewidths=2,
                                        edgecolors='k')
                axes[i, 2 + ee].set_ylabel(xyz_label[i])
                # Draw EE Ang Wrench
                axes[i, 4 + ee].plot(time,
                                     ee_wrench_ang[ee][:, i],
                                     color='k',
                                     linewidth=3)
                axes[i, 4 + ee].scatter(node_ee_wrench_ang_time[ee],
                                        node_ee_wrench_ang[ee][:, i],
                                        s=50,
                                        c='lightgray',
                                        linewidths=2,
                                        edgecolors='k')
                axes[i, 4 + ee].set_ylabel(trq_label[i])

                # Fill Phase
                fill_phase(axes[i, 2 + ee], contact_schedule[ee],
                           itertools.cycle(facecolors[0:2]))
                fill_phase(axes[i, 4 + ee], contact_schedule[ee],
                           itertools.cycle(facecolors[0:2]))
            else:
                # Draw EE Lin Motion
                axes[i, 2 + ee].plot(time,
                                     ee_motion_lin[ee][:, i - 3],
                                     color='k',
                                     linewidth=3)
                axes[i, 2 + ee].scatter(node_ee_motion_lin_time[ee],
                                        node_ee_motion_lin[ee][:, i - 3],
                                        s=50,
                                        c='lightgray',
                                        linewidths=2,
                                        edgecolors='k')
                axes[i, 2 + ee].set_ylabel(xyz_label[i - 3])
                # Draw EE Ang Motion
                axes[i, 4 + ee].plot(time,
                                     ee_wrench_lin[ee][:, i - 3],
                                     color='k',
                                     linewidth=3)
                axes[i, 4 + ee].scatter(node_ee_wrench_lin_time[ee],
                                        node_ee_wrench_lin[ee][:, i - 3],
                                        s=50,
                                        c='lightgray',
                                        linewidths=2,
                                        edgecolors='k')
                axes[i, 4 + ee].set_ylabel(frc_label[i - 3])

                # Fill Phase
                fill_phase(axes[i, 2 + ee], contact_schedule[ee],
                           itertools.cycle(facecolors[0:2]))
                fill_phase(axes[i, 4 + ee], contact_schedule[ee],
                           itertools.cycle(facecolors[0:2]))

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    main(args)

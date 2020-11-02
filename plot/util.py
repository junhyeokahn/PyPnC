import random

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

xyz_label = ['x', 'y', 'z']
quat_label = ['x', ' y', ' z', 'w']
markers = ['*', '+', 'h', 'x', 'o', 'v', 'd']

def plot_task(time, pos_des, pos, vel_des, vel, suptitle):
    if pos_des.shape[1] == 3:

        fig, axes = plt.subplots(3, 2)
        for i in range(3):
            axes[i, 0].plot(time,
                            pos_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 0].plot(time,
                            pos[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 0].grid(True)
            axes[i, 0].set_ylabel(xyz_label[i])
            axes[i, 1].plot(time,
                            vel_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 1].plot(time,
                            vel[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 1].grid(True)
            axes[i, 1].set_ylabel(xyz_label[i] + 'dot')
        axes[2, 0].set_xlabel('time')
        axes[2, 1].set_xlabel('time')
        fig.suptitle(suptitle)

    elif pos_des.shape[1] == 4:
        fig, axes = plt.subplots(4, 2)
        for i in range(4):
            axes[i, 0].plot(time,
                            pos_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 0].plot(time,
                            pos[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 0].grid(True)
            axes[i, 0].set_ylabel(quat_label[i])
        for i in range(3):
            axes[i, 1].plot(time,
                            vel_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 1].plot(time,
                            vel[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 1].grid(True)
            axes[i, 1].set_ylabel(xyz_label[i] + 'dot')
        axes[3, 0].set_xlabel('time')
        axes[3, 1].set_xlabel('time')
        fig.suptitle(suptitle)

    else:
        dim = pos_des.shape[1]
        fig, axes = plt.subplots(dim, 2)
        for i in range(dim):
            axes[i, 0].plot(time,
                            pos_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 0].plot(time,
                            pos[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 0].grid(True)
            axes[i, 1].plot(time,
                            vel_des[:, i],
                            color='r',
                            linestyle='dashed',
                            linewidth=4)
            axes[i, 1].plot(time,
                            vel[:, i],
                            color='b',
                            linestyle='dashed',
                            linewidth=2)
            axes[i, 1].grid(True)
        axes[dim - 1, 0].set_xlabel('time')
        axes[dim - 1, 1].set_xlabel('time')
        fig.suptitle(suptitle)


def plot_weights(time, weights_dict):
    fig, ax = plt.subplots()
    for i, (k, v) in enumerate(weights_dict.items()):
        ax.plot(time,
                v,
                label=k,
                marker=markers[i],
                markersize=10,
                markevery=random.randint(50, 150))
        ax.grid(True)
    ax.set_xlabel('time')
    ax.legend()
    fig.suptitle('task weights')


def plot_rf_z_max(time, rf_z_max):
    fig, ax = plt.subplots()
    for i, (k, v) in enumerate(rf_z_max.items()):
        ax.plot(time,
                v,
                label=k,
                marker=markers[i],
                markersize=10,
                markevery=random.randint(50, 150))
        ax.grid(True)
    ax.set_xlabel('time')
    ax.legend()
    fig.suptitle('rf_z_max')

def plot_vector_traj(time, vector, suptitle):
    dim = vector.shape[1]
    fig, axes = plt.subplots(dim, 1)
    for i in range(dim):
        axes[i].plot(time, vector[:,i], color='k', linewidth=3)
        axes[i].grid(True)
    axes[dim-1].set_xlabel('time')
    fig.suptitle(suptitle)

def plot_rf(time, rfs):
    fig, axes = plt.subplots(6, 2)
    for i in range(6):
        axes[i, 0].plot(time, rfs[:,i], color='k', linewidth=3)
        axes[i, 1].plot(time, rfs[:,i+6], color='k', linewidth=3)
        axes[i, 0].grid(True)
        axes[i, 1].grid(True)
    axes[5,0].set_xlabel('time')
    axes[5,1].set_xlabel('time')
    axes[0,0].set_title('Right Foot')
    axes[0,1].set_title('Left Foot')
    fig.suptitle('Reaction Force Command')


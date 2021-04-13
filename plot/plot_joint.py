import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot.helper import plot_phase

time = []
phase = []
joint_pos = []
joint_vel = []
joint_trq_cmd = []
joint_pos_limit = []
joint_vel_limit = []
joint_trq_limit = []

with open('data/pnc.pkl', 'rb') as file:
    while True:
        try:
            d = pickle.load(file)
            time.append(d['time'])
            phase.append(d['phase'])
            joint_pos.append(d['joint_pos'])
            joint_vel.append(d['joint_vel'])
            joint_trq_cmd.append(d['joint_trq_cmd'])
            joint_pos_limit.append(d['joint_pos_limit'])
            joint_vel_limit.append(d['joint_vel_limit'])
            joint_trq_limit.append(d['joint_trq_limit'])
        except EOFError:
            break

    phase = np.stack(phase, axis=0)
    joint_pos = np.stack(joint_pos, axis=0)
    joint_vel = np.stack(joint_vel, axis=0)
    joint_trq_cmd = np.stack(joint_trq_cmd, axis=0)
    joint_pos_limit = np.array(joint_pos_limit[-1])
    joint_vel_limit = np.array(joint_vel_limit[-1])
    joint_trq_limit = np.array(joint_trq_limit[-1])

    n_data, n_joint = joint_pos.shape

    fig, axes = plt.subplots(n_joint, 2)

    for i in range(n_joint):
        axes[i, 0].plot(time, joint_pos[:, i], color='k', linewidth=3)
        axes[i, 1].plot(time, joint_vel[:, i], color='k', linewidth=3)
        plot_phase(axes[i, 0], time, phase)
        plot_phase(axes[i, 1], time, phase)
        axes[i, 0].axhline(joint_pos_limit[i, 0], color='r', linewidth=3)
        axes[i, 0].axhline(joint_pos_limit[i, 1], color='r', linewidth=3)
        axes[i, 1].axhline(joint_vel_limit[i, 0], color='r', linewidth=3)
        axes[i, 1].axhline(joint_vel_limit[i, 1], color='r', linewidth=3)
        axes[i, 0].grid(True)
        axes[i, 1].grid(True)
    axes[0, 0].set_title('joint positions')
    axes[0, 1].set_title('joint velocities')

    fig, axes = plt.subplots(n_joint, 1)
    for i in range(n_joint):
        axes[i].plot(time, joint_trq_cmd[:, i], color='k', linewidth=3)
        plot_phase(axes[i], time, phase)
        axes[i].grid(True)
        axes[i].axhline(joint_trq_limit[i, 0], color='r', linewidth=3)
        axes[i].axhline(joint_trq_limit[i, 1], color='r', linewidth=3)
    fig.suptitle('joint trq command')

    plt.show()

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

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

ylabel_left = ['Ixx', 'Iyy', 'Izz']
ylabel_right = ['Ixy', 'Ixz', 'Iyz']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

gt_inertia = []
gt_inertia_normalized = []
est_inertia = []
est_inertia_normalized = []
with open(args.file, 'rb') as file:
    while True:
        try:
            d = pickle.load(file)
            gt_inertia.append(d['gt_inertia'])
            gt_inertia_normalized.append(d['gt_inertia_normalized'])
            est_inertia.append(d['est_inertia'])
            est_inertia_normalized.append(d['est_inertia_normalized'])
        except EOFError:
            break

gt_inertia = np.stack(gt_inertia, axis=0)
gt_inertia_normalized = np.stack(gt_inertia_normalized, axis=0)
est_inertia = np.stack(est_inertia, axis=0)
est_inertia_normalized = np.stack(est_inertia_normalized, axis=0)
x = np.arange(gt_inertia.shape[0])

fig, axes = plt.subplots(3, 2, constrained_layout=True)
for i in range(3):
    axes[i, 0].plot(x, gt_inertia[:, i], color='k', linewidth=3)
    axes[i, 0].plot(x,
                    est_inertia[:, i],
                    color='gray',
                    linewidth=4,
                    linestyle="--")
    axes[i, 0].set_ylabel(ylabel_left[i])
    axes[i, 1].plot(x, gt_inertia[:, i + 3], color='k', linewidth=3)
    axes[i, 1].plot(x,
                    est_inertia[:, i + 3],
                    color='gray',
                    linewidth=4,
                    linestyle="--")
    axes[i, 1].set_ylabel(ylabel_right[i])

fig, axes = plt.subplots(3, 2, constrained_layout=True)
for i in range(3):
    axes[i, 0].plot(x, gt_inertia_normalized[:, i], color='k', linewidth=3)
    axes[i, 0].plot(x,
                    est_inertia_normalized[:, i],
                    color='gray',
                    linewidth=4,
                    linestyle="--")
    axes[i, 0].set_ylabel(ylabel_left[i])
    axes[i, 1].plot(x, gt_inertia_normalized[:, i + 3], color='k', linewidth=3)
    axes[i, 1].plot(x,
                    est_inertia_normalized[:, i + 3],
                    color='gray',
                    linewidth=4,
                    linestyle="--")
    axes[i, 1].set_ylabel(ylabel_right[i])

plt.show()

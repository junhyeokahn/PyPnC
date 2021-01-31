import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot.helper import plot_task, plot_weights, plot_rf_z_max, plot_rf, plot_vector_traj

tasks = [
    'com_pos', 'com_vel', 'pelvis_quat', 'pelvis_ang_vel', 'joint_pos',
    'joint_vel', 'leftCOP_Frame_pos', 'leftCOP_Frame_vel',
    'leftCOP_Frame_quat', 'leftCOP_Frame_ang_vel', 'rightCOP_Frame_pos',
    'rightCOP_Frame_vel', 'rightCOP_Frame_quat', 'rightCOP_Frame_ang_vel'
]

weights = [
    'w_com', 'w_pelvis_ori', 'w_joint', 'w_leftCOP_Frame',
    'w_leftCOP_Frame_ori', 'w_rightCOP_Frame', 'w_rightCOP_Frame_ori'
]

rf_z = ['rf_z_max_rightCOP_Frame', 'rf_z_max_leftCOP_Frame']

time = []

rf_cmd = []

joint_trq_cmd = []

joint_acc_cmd = []

des, act = dict(), dict()
for topic in tasks:
    des[topic] = []
    act[topic] = []
w = dict()
for topic in weights:
    w[topic] = []
rf_z_max = dict()
for topic in rf_z:
    rf_z_max[topic] = []

with open('data/history.pkl', 'rb') as file:
    while True:
        try:
            d = pickle.load(file)
            time.append(d['time'])
            for topic in tasks:
                des[topic].append(d[topic + '_des'])
                act[topic].append(d[topic])
            for topic in weights:
                w[topic].append(d[topic])
            for topic in rf_z:
                rf_z_max[topic].append(d[topic])
            rf_cmd.append(d['rf_cmd'])
        except EOFError:
            break

for k, v in des.items():
    des[k] = np.stack(v, axis=0)
for k, v in act.items():
    act[k] = np.stack(v, axis=0)
rf_cmd = np.stack(rf_cmd, axis=0)

## =============================================================================
## Plot Task
## =============================================================================

plot_task(time, des['com_pos'], act['com_pos'], des['com_vel'], act['com_vel'],
          'com lin')

plot_task(time, des['pelvis_quat'], act['pelvis_quat'], des['pelvis_ang_vel'],
          act['pelvis_ang_vel'], 'pelvis ori')

plot_task(time, des['joint_pos'], act['joint_pos'], des['joint_vel'],
          act['joint_vel'], 'upperbody joint')

# plot_task(time, des['leftCOP_Frame_pos'], act['leftCOP_Frame_pos'],
# des['leftCOP_Frame_vel'], act['leftCOP_Frame_vel'], 'left foot lin')

# plot_task(time, des['leftCOP_Frame_quat'], act['leftCOP_Frame_quat'],
# des['leftCOP_Frame_ang_vel'], act['leftCOP_Frame_ang_vel'],
# 'left foot ori')

# plot_task(time, des['rightCOP_Frame_pos'], act['rightCOP_Frame_pos'],
# des['rightCOP_Frame_vel'], act['rightCOP_Frame_vel'],
# 'right foot lin')

# plot_task(time, des['rightCOP_Frame_quat'], act['rightCOP_Frame_quat'],
# des['rightCOP_Frame_ang_vel'], act['rightCOP_Frame_ang_vel'],
# 'right foot ori')

## =============================================================================
## Plot Weights and Max Reaction Force Z
## =============================================================================
# plot_weights(time, w)

# plot_rf_z_max(time, rf_z_max)

## =============================================================================
## Plot WBC Solutions
## =============================================================================
plot_rf(time, rf_cmd)

plt.show()

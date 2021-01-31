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
    'com_pos', 'com_vel', 'pelvis_com_quat', 'pelvis_com_ang_vel', 'joint_pos',
    'joint_vel', 'l_sole_pos', 'l_sole_vel', 'l_sole_quat', 'l_sole_ang_vel',
    'r_sole_pos', 'r_sole_vel', 'r_sole_quat', 'r_sole_ang_vel'
]

weights = [
    'w_com', 'w_pelvis_com_ori', 'w_joint', 'w_l_sole', 'w_l_sole_ori',
    'w_r_sole', 'w_r_sole_ori'
]

rf_z = ['rf_z_max_r_sole', 'rf_z_max_l_sole']

time = []

phase = []

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

with open('data/pnc.pkl', 'rb') as file:
    while True:
        try:
            d = pickle.load(file)
            time.append(d['time'])
            phase.append(d['phase'])
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
phase = np.stack(phase, axis=0)

## =============================================================================
## Plot Task
## =============================================================================

plot_task(time, des['com_pos'], act['com_pos'], des['com_vel'], act['com_vel'],
          phase, 'com lin')

plot_task(time, des['pelvis_com_quat'], act['pelvis_com_quat'],
          des['pelvis_com_ang_vel'], act['pelvis_com_ang_vel'], phase,
          'pelvis ori')

plot_task(time, des['joint_pos'], act['joint_pos'], des['joint_vel'],
          act['joint_vel'], phase, 'upperbody joint')

plot_task(time, des['l_sole_pos'], act['l_sole_pos'], des['l_sole_vel'],
          act['l_sole_vel'], phase, 'left foot lin')

plot_task(time, des['l_sole_quat'], act['l_sole_quat'], des['l_sole_ang_vel'],
          act['l_sole_ang_vel'], phase, 'left foot ori')

plot_task(time, des['r_sole_pos'], act['r_sole_pos'], des['r_sole_vel'],
          act['r_sole_vel'], phase, 'right foot lin')

plot_task(time, des['r_sole_quat'], act['r_sole_quat'], des['r_sole_ang_vel'],
          act['r_sole_ang_vel'], phase, 'right foot ori')

## =============================================================================
## Plot WBC Solutions
## =============================================================================
plot_rf(time, rf_cmd, phase)

## =============================================================================
## Plot Weights and Max Reaction Force Z
## =============================================================================
plot_weights(time, w, phase)

plot_rf_z_max(time, rf_z_max, phase)

plt.show()

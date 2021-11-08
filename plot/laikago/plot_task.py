import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot.helper import plot_task, plot_weights, plot_rf_z_max, plot_rf_quad, plot_vector_traj

tasks = [
    'com_pos', 'com_vel', 'chassis_quat', 'chassis_ang_vel', 'toeFL_pos',
    'toeFL_vel', 'toeFR_pos', 'toeFR_vel', 'toeRR_pos', 'toeRR_vel',
    'toeRL_pos', 'toeRL_vel'
]

weights = [
    'w_com', 'w_chassis_ori', 'w_toeFL', 'w_toeFR', 'w_toeRR', 'w_toeRL'
]

rf_z = ['rf_z_max_toeFL', 'rf_z_max_toeFR', 'rf_z_max_toeRR', 'rf_z_max_toeRL']

time = []

phase = []

rf_cmd = []

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

plot_task(time, des['chassis_quat'], act['chassis_quat'],
          des['chassis_ang_vel'], act['chassis_ang_vel'], phase, 'pelvis ori')

plot_task(time, des['toeFL_pos'], act['toeFL_pos'], des['toeFL_vel'],
          act['toeFL_vel'], phase, 'left foot lin')

plot_task(time, des['toeFR_pos'], act['toeFR_pos'], des['toeFR_vel'],
          act['toeFR_vel'], phase, 'left foot ori')

plot_task(time, des['toeRR_pos'], act['toeRR_pos'], des['toeRR_vel'],
          act['toeRR_vel'], phase, 'right foot lin')

plot_task(time, des['toeRL_pos'], act['toeRL_pos'], des['toeRL_vel'],
          act['toeRL_vel'], phase, 'right foot ori')

## =============================================================================
## Plot WBC Solutions
## =============================================================================
plot_rf_quad(time, rf_cmd, phase)

## =============================================================================
## Plot Weights and Max Reaction Force Z
## =============================================================================
plot_weights(time, w, phase)

plot_rf_z_max(time, rf_z_max, phase)

plt.show()

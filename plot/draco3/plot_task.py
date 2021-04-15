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
    'com_pos', 'com_vel', 'torso_com_link_quat', 'torso_com_link_ang_vel',
    'selected_joint_pos', 'selected_joint_vel', 'l_foot_contact_pos',
    'l_foot_contact_vel', 'l_foot_contact_quat', 'l_foot_contact_ang_vel',
    'r_foot_contact_pos', 'r_foot_contact_vel', 'r_foot_contact_quat',
    'r_foot_contact_ang_vel'
]

weights = [
    'w_com', 'w_torso_com_link_ori', 'w_selected_joint', 'w_l_foot_contact',
    'w_l_foot_contact_ori', 'w_r_foot_contact', 'w_r_foot_contact_ori'
]

rf_z = ['rf_z_max_r_foot_contact', 'rf_z_max_l_foot_contact']

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

plot_task(time, des['torso_com_link_quat'], act['torso_com_link_quat'],
          des['torso_com_link_ang_vel'], act['torso_com_link_ang_vel'], phase,
          'torso ori')

plot_task(time, des['selected_joint_pos'], act['selected_joint_pos'],
          des['selected_joint_vel'], act['selected_joint_vel'], phase,
          'upperbody joint')

plot_task(time, des['l_foot_contact_pos'], act['l_foot_contact_pos'],
          des['l_foot_contact_vel'], act['l_foot_contact_vel'], phase,
          'left foot lin')

plot_task(time, des['l_foot_contact_quat'], act['l_foot_contact_quat'],
          des['l_foot_contact_ang_vel'], act['l_foot_contact_ang_vel'], phase,
          'left foot ori')

plot_task(time, des['r_foot_contact_pos'], act['r_foot_contact_pos'],
          des['r_foot_contact_vel'], act['r_foot_contact_vel'], phase,
          'right foot lin')

plot_task(time, des['r_foot_contact_quat'], act['r_foot_contact_quat'],
          des['r_foot_contact_ang_vel'], act['r_foot_contact_ang_vel'], phase,
          'right foot ori')

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

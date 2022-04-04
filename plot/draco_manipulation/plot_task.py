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
    'r_foot_contact_ang_vel', 'l_hand_contact_pos', 'l_hand_contact_vel',
    'l_hand_contact_quat', 'l_hand_contact_ang_vel', 'r_hand_contact_pos',
    'r_hand_contact_vel', 'r_hand_contact_quat', 'r_hand_contact_ang_vel'
]

weights = [
    'w_com', 'w_torso_com_link_ori', 'w_selected_joint', 'w_l_foot_contact',
    'w_l_foot_contact_ori', 'w_r_foot_contact', 'w_r_foot_contact_ori',
    'w_l_hand_contact', 'w_l_hand_contact_ori', 'w_r_hand_contact',
    'w_r_hand_contact_ori'
]

quat_err = ['l_hand_contact_quat_err', 'r_hand_contact_quat_err']

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
quat_err_list = dict()
for topic in quat_err:
    quat_err_list[topic] = []

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
            for topic in quat_err:
                quat_err_list[topic].append(d[topic])
            rf_cmd.append(d['rf_cmd'])
        except EOFError:
            break

for k, v in des.items():
    des[k] = np.stack(v, axis=0)
for k, v in act.items():
    act[k] = np.stack(v, axis=0)
for k, v in quat_err_list.items():
    quat_err_list[k] = np.stack(v, axis=0)
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

# plot_task(time, des['selected_joint_pos'], act['selected_joint_pos'],
# des['selected_joint_vel'], act['selected_joint_vel'], phase,
# 'neck joint')
"""
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
"""

plot_task(time, des['l_hand_contact_pos'], act['l_hand_contact_pos'],
          des['l_hand_contact_vel'], act['l_hand_contact_vel'], phase,
          'left hand lin')

plot_task(time, des['l_hand_contact_quat'], act['l_hand_contact_quat'],
          des['l_hand_contact_ang_vel'], act['l_hand_contact_ang_vel'], phase,
          'left hand ori')

plot_task(time, des['r_hand_contact_pos'], act['r_hand_contact_pos'],
          des['r_hand_contact_vel'], act['r_hand_contact_vel'], phase,
          'right hand lin')

plot_task(time, des['r_hand_contact_quat'], act['r_hand_contact_quat'],
          des['r_hand_contact_ang_vel'], act['r_hand_contact_ang_vel'], phase,
          'right hand ori')

## =============================================================================
## Plot WBC Solutions
## =============================================================================
plot_rf(time, rf_cmd, phase)

## =============================================================================
## Plot Weights and Max Reaction Force Z
## =============================================================================
# plot_weights(time, w, phase)

plot_rf_z_max(time, rf_z_max, phase)

# plot_vector_traj(time, quat_err_list['l_hand_contact_quat_err'],
# 'lhand_quat_err')
# plot_vector_traj(time, quat_err_list['r_hand_contact_quat_err'],
# 'rhand_quat_err')

plt.show()

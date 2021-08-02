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

time = []
qddot_des = []
ee_acc_des = []
ee_pos_des, ee_pos_act = [], []
ee_vel_des, ee_vel_act = [], []
jpos_des, jpos_act = [], []
jvel_des, jvel_act = [], []

with open('data/pnc.pkl', 'rb') as file:
    while True:
        try:
            d = pickle.load(file)
            time.append(d['time'])
            ee_pos_des.append(d['ee_pos_des'])
            ee_pos_act.append(d['ee_pos_act'])
            ee_vel_des.append(d['ee_vel_des'])
            ee_vel_act.append(d['ee_vel_act'])
            jpos_des.append(d['jpos_des'])
            jpos_act.append(d['jpos_act'])
            jvel_des.append(d['jvel_des'])
            jvel_act.append(d['jvel_act'])
            qddot_des.append(d['qddot_des'])
            ee_acc_des.append(d['ee_acc_des'])
        except EOFError:
            break

time = np.stack(time, axis=0)
ee_pos_des = np.stack(ee_pos_des, axis=0)
ee_pos_act = np.stack(ee_pos_act, axis=0)
ee_vel_des = np.stack(ee_vel_des, axis=0)
ee_vel_act = np.stack(ee_vel_act, axis=0)
jpos_des = np.stack(jpos_des, axis=0)
jpos_act = np.stack(jpos_act, axis=0)
jvel_des = np.stack(jvel_des, axis=0)
jvel_act = np.stack(jvel_act, axis=0)
qddot_des = np.stack(qddot_des, axis=0)
ee_acc_des = np.stack(ee_acc_des, axis=0)

plot_task(time, ee_pos_des, ee_pos_act, ee_vel_des, ee_vel_act,
          [1] * ee_pos_des.shape[0], 'ee')
plot_task(time, jpos_des, jpos_act, jvel_des, jvel_act,
          [1] * ee_pos_des.shape[0], 'joint')
plot_vector_traj(time, ee_acc_des, "ee ff acc")
plot_vector_traj(time, qddot_des, "qddot des")

plt.show()

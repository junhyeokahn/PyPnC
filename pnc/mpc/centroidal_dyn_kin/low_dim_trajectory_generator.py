from enum import Enum
import numpy as np


class Task(Enum):
    STAND = 1
    SQUAT = 2


def get_desired_mpc_trajectory(task, s0, u_guess, model, N_horizon, w=1, A=0.0):

    ns = model.ns
    na = model.na
    dt = model.dt

    s_traj = np.zeros((ns, N_horizon))
    if task == Task.STAND:
        s_traj = np.repeat(np.reshape(s0, [ns, 1]), N_horizon+1, axis=1)
    elif task == Task.SQUAT:
        s_traj[:, 0] = s0
        # change desired CoM-z position and velocity entries
        for n in range(N_horizon):
            s_traj[2, n] = s0[2] + A*np.sin(w * n*dt)
            s_traj[5, n] = s0[5] + A*np.cos(w * n*dt)
    u_guess_traj = np.repeat(np.reshape(u_guess, [na, 1]), N_horizon, axis=1)

    return s_traj, u_guess_traj
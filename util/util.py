from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml


def pretty_print(ob):
    if type(ob) is dict:
        print(yaml.dump(ob, default_flow_style=False))


def quat_to_rot(quat):
    """
    Parameters
    ----------
    quat (np.array): scalar last quaternion

    Returns
    -------
    ret (np.array): SO3
    """
    return (R.from_quat(quat)).as_matrix()


def rot_to_quat(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return R.from_matrix(rot).as_quat()


def smooth_changing(ini, end, dur, curr_time):
    ret = ini + (end - ini) * 0.5 * (1 - np.cos(curr_time / dur * np.pi))
    if curr_time > dur:
        ret = end

    return ret


def smooth_changing_vel(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * np.sin(curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.

    return ret


def smooth_changing_acc(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * (np.pi / dur) * np.cos(
        curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.

    return ret


def quat_to_exp(quat):
    img_vec = np.array([quat[0], quat[1], quat[2]])
    w = quat[3]
    theta = 2.0 * np.arcsin(
        np.sqrt(img_vec[0] * img_vec[0] + img_vec[1] * img_vec[1] +
                img_vec[2] * img_vec[2]))

    if np.abs(theta) < 1e-4:
        return np.zeros(3)
    ret = img_vec / np.sin(theta / 2.0)

    return ret * theta


def exp_to_quat(exp):
    theta = np.square(exp[0] * exp[0] + exp[1] * exp[1] + exp[2] * exp[2])
    ret = np.zeros(4)
    if theta > 1e-4:
        ret[0] = sin(theta / 2.0) * exp[0] / theta
        ret[1] = sin(theta / 2.0) * exp[1] / theta
        ret[2] = sin(theta / 2.0) * exp[2] / theta
        ret[3] = cos(theta / 2.0)
    else:
        ret[0] = 0.5 * exp[0]
        ret[1] = 0.5 * exp[1]
        ret[2] = 0.5 * exp[2]
        ret[3] = 1.0
    return ret

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

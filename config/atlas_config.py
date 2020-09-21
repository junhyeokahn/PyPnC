import numpy as np


class SimConfig(object):
    DT = 0.001


class PnCConfig(object):
    DYN_LIB = "dart"


class WBCConfig(object):
    # Max normal force per contact
    MAX_Z_FORCE = 2000.0

    # Task Hierarchy Weights
    W_COM = 10.0
    W_PELVIS = 10.0
    W_UPPER_BODY = 20.0
    W_CONTACT_FOOT = 40.0
    W_SWING_FOOT = 40.0

    # Task Gains
    KP_COM = np.array([50., 50., 50])
    KD_COM = np.array([5., 5., 5.])

    KP_PELVIS = np.array([50., 50., 50])
    KD_PELVIS = np.array([5., 5., 5.])

    KP_UPPER_BODY = 50.
    KD_UPPER_BODY = 5.

    KP_FOOT = np.array([100., 100., 100.])
    KD_FOOT = np.array([10., 10., 10.])

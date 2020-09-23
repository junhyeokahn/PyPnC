import numpy as np


class SimConfig(object):
    DT = 0.001


class PnCConfig(object):
    DYN_LIB = "dart"
    DT = SimConfig.DT


class WBCConfig(object):
    # Max normal force per contact
    RF_Z_MAX = 2000.0

    # Task Hierarchy Weights
    W_COM = 10.0
    W_PELVIS = 10.0
    W_UPPER_BODY = 20.0
    W_CONTACT_FOOT = 40.0
    W_SWING_FOOT = 20.0

    # Task Gains
    KP_COM = np.array([50., 50., 50])
    KD_COM = np.array([5., 5., 5.])

    KP_PELVIS = np.array([50., 50., 50])
    KD_PELVIS = np.array([5., 5., 5.])

    KP_UPPER_BODY = 50.
    KD_UPPER_BODY = 5.

    KP_FOOT = np.array([100., 100., 100.])
    KD_FOOT = np.array([10., 10., 10.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-6
    LAMBDA_RF = 1e-6

    B_TRQ_LIMIT = True

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians


class WalkingConfig(object):
    # STAND
    INIT_STAND_DUR = 1.0
    RF_Z_MAX_TIME = 0.1

    COM_HEIGHT = 1.02

    SWING_HEIGHT = 0.05  #cm

import numpy as np


class SimConfig(object):
    CONTROLLER_DT = 0.001
    CAMERA_DT = 0.05
    KP = 0.
    KD = 0.


class PnCConfig(object):
    DYN_LIB = "dart"
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = False  #True
    SAVE_FREQ = 10


class WBCConfig(object):
    # Max normal force per contact
    RF_Z_MAX = 1500.

    # Task Hierarchy Weights
    W_COM = 10.0
    W_PELVIS = 20.0
    W_UPPER_BODY = 20.0
    W_CONTACT_FOOT = 40.0
    W_SWING_FOOT = 20.0

    # Task Gains
    KP_COM = np.array([100., 100., 100])
    KD_COM = np.array([10., 10., 10.])

    KP_PELVIS = np.array([100., 100., 100])
    KD_PELVIS = np.array([10., 10., 10.])

    KP_UPPER_BODY = 100.
    KD_UPPER_BODY = 10.

    KP_FOOT = np.array([400., 400., 400.])
    KD_FOOT = np.array([40., 40., 40.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-8
    LAMBDA_RF = 1e-8

    B_TRQ_LIMIT = True

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians


class WalkingConfig(object):
    # STAND
    INIT_STAND_DUR = 1.0
    RF_Z_MAX_TIME = 0.1

    COM_HEIGHT = 1.015

    SWING_HEIGHT = 0.05  #cm


class WalkingState(object):
    STAND = 0
    BALANCE = 1

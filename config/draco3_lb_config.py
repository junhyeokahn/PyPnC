import numpy as np


class SimConfig(object):
    CONTROLLER_DT = 0.01
    N_SUBSTEP = 10
    CAMERA_DT = 0.05
    KP = 0.
    KD = 0.

    INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.757]
    INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = True
    VIDEO_RECORD = False
    RECORD_FREQ = 5


class PnCConfig(object):
    DYN_LIB = "pinocchio"  # "dart"
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = True
    SAVE_FREQ = 1

    PRINT_ROBOT_INFO = SimConfig.PRINT_ROBOT_INFO


class WBCConfig(object):
    # Max normal force per contact
    RF_Z_MAX = 1000.0

    # Task Hierarchy Weights
    W_COM = 20.0
    W_TORSO = 20.0
    W_UPPER_BODY = 40.0
    W_CONTACT_FOOT = 60.0
    W_SWING_FOOT = 40.0

    # Task Gains
    KP_COM = np.array([400., 400., 400])
    KD_COM = np.array([40., 40., 40.])

    KP_TORSO = np.array([100., 100., 100])
    KD_TORSO = np.array([10., 10., 10.])

    KP_FOOT_POS = np.array([300., 300., 300.])
    KD_FOOT_POS = np.array([30., 30., 30.])
    KP_FOOT_ORI = np.array([300., 300., 300.])
    KD_FOOT_ORI = np.array([30., 30., 30.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-8
    LAMBDA_RF = 1e-7

    B_TRQ_LIMIT = True

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians


class WalkingConfig(object):
    # STAND
    INIT_STAND_DUR = 1.0
    RF_Z_MAX_TIME = 0.1

    COM_HEIGHT = 0.60  # m
    SWING_HEIGHT = 0.03  # m

    # SWAYING_AMP = np.array([0., 0.08, 0.])
    # SWAYING_FREQ = np.array([0., 0.3, 0.])
    SWAYING_AMP = np.array([0., 0., 0.05])
    SWAYING_FREQ = np.array([0., 0., 0.3])

    T_ADDITIONAL_INI_TRANS = 0.
    T_CONTACT_TRANS = 0.15
    T_SWING = 0.25
    PERCENTAGE_SETTLE = 0.9
    ALPHA_DS = 0.5

    NOMINAL_FOOTWIDTH = 0.25
    NOMINAL_FORWARD_STEP = 0.08
    NOMINAL_BACKWARD_STEP = -0.08
    NOMINAL_TURN_RADIANS = np.pi / 10
    NOMINAL_STRAFE_DISTANCE = 0.03


class WalkingState(object):
    STAND = 0
    BALANCE = 1
    RF_CONTACT_TRANS_START = 2
    RF_CONTACT_TRANS_END = 3
    RF_SWING = 4
    LF_CONTACT_TRANS_START = 5
    LF_CONTACT_TRANS_END = 6
    LF_SWING = 7
    SWAYING = 10

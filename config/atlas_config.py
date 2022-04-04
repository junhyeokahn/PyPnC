import numpy as np


class SimConfig(object):
    # CONTROLLER_DT = 0.00125
    CONTROLLER_DT = 0.01
    N_SUBSTEP = 10
    CAMERA_DT = 0.05
    KP = 0.
    KD = 0.

    INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.761]
    INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = False
    VIDEO_RECORD = False
    RECORD_FREQ = 10
    SIMULATE_CAMERA = False


class PnCConfig(object):
    DYN_LIB = "pinocchio"  # "dart"
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = True
    SAVE_FREQ = 1

    PRINT_ROBOT_INFO = SimConfig.PRINT_ROBOT_INFO


class WBCConfig(object):
    VERBOSE = False

    # Max normal force per contact
    RF_Z_MAX = 2000.

    # Task Hierarchy Weights
    W_COM = 10.0
    W_PELVIS = 20.0
    W_UPPER_BODY = 0.1
    W_CONTACT_FOOT = 60.0
    W_SWING_FOOT = 40.0
    W_HAND_POS_MIN = 0.
    W_HAND_POS_MAX = 50.

    # Task Gains
    KP_COM = np.array([100., 100., 100])
    KD_COM = np.array([10., 10., 10.])

    KP_PELVIS = np.array([100., 100., 100])
    KD_PELVIS = np.array([10., 10., 10.])

    KP_UPPER_BODY = 100.
    KD_UPPER_BODY = 10.

    KP_HAND_POS = np.array([70., 70., 70.])
    KD_HAND_POS = np.array([5., 5., 5.])

    KP_FOOT_POS = np.array([400., 400., 400.])
    KD_FOOT_POS = np.array([40., 40., 40.])
    KP_FOOT_ORI = np.array([400., 400., 400.])
    KD_FOOT_ORI = np.array([40., 40., 40.])

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

    COM_HEIGHT = 1.02  # m
    SWING_HEIGHT = 0.05  # m

    T_ADDITIONAL_INI_TRANS = 0.  # sec
    T_CONTACT_TRANS = 0.45
    T_SWING = 0.75
    PERCENTAGE_SETTLE = 0.9
    ALPHA_DS = 0.5

    NOMINAL_FOOTWIDTH = 0.25
    NOMINAL_FORWARD_STEP = 0.15
    NOMINAL_BACKWARD_STEP = -0.15
    NOMINAL_TURN_RADIANS = np.pi / 6
    NOMINAL_STRAFE_DISTANCE = 0.1

    T_TRANS_DURATION = 1
    T_REACHING_DURATION = 3.

    BLUE_CAN = np.array([0.7, 0.03, 0.72])
    RED_CAN = np.array([0.7, 0.65, 1.1])

    RH_TARGET_POS = np.array([0.62, -0.54,
                              0.96])  # initial: [0.42, -0.54, 0.96]
    LH_TARGET_POS = np.array([0.52, 0.54, 0.96])  # initial: [0.41, 0.53, 0.96]

    # LH_TARGET_POS = np.copy(BLUE_CAN)
    LH_TARGET_POS = np.copy(RED_CAN)
    RH_TARGET_POS = np.copy(RED_CAN) + np.array([0., 0.1, -0.04])
    # RH_TARGET_POS = np.copy(RED_CAN) + np.array([0., 0., -0.04])


class WalkingState(object):
    STAND = 0
    BALANCE = 1
    RF_CONTACT_TRANS_START = 2
    RF_CONTACT_TRANS_END = 3
    RF_SWING = 4
    LF_CONTACT_TRANS_START = 5
    LF_CONTACT_TRANS_END = 6
    LF_SWING = 7
    RH_HANDREACH = 8
    LH_HANDREACH = 9

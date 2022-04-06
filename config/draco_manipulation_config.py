import numpy as np


class SimConfig(object):
    CONTROLLER_DT = 0.01
    N_SUBSTEP = 10
    CAMERA_DT = 0.05

    INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.757]
    INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = False
    VIDEO_RECORD = False
    RECORD_FREQ = 5000000000000
    SIMULATE_CAMERA = False
    SAVE_CAMERA_DATA = False

    B_USE_MESHCAT = False

    KP, KD = dict(), dict()

    KP["l_hip_ie"] = 1.0
    KP["l_hip_aa"] = 1.0
    KP["l_hip_fe"] = 1.0
    KP["l_knee_fe_jp"] = 1.0
    KP["l_knee_fe_jd"] = 1.0
    KP["l_ankle_fe"] = 1.0
    KP["l_ankle_ie"] = 1.0
    KP["l_shoulder_fe"] = 1.0
    KP["l_shoulder_aa"] = 1.0
    KP["l_shoulder_ie"] = 1.0
    KP["l_elbow_fe"] = 1.0
    KP["l_wrist_ps"] = 1.0
    KP["l_wrist_pitch"] = 1.0
    KP["neck_pitch"] = 1.0
    KP["r_hip_ie"] = 1.0
    KP["r_hip_aa"] = 1.0
    KP["r_hip_fe"] = 1.0
    KP["r_knee_fe_jp"] = 1.0
    KP["r_knee_fe_jd"] = 1.0
    KP["r_ankle_fe"] = 1.0
    KP["r_ankle_ie"] = 1.0
    KP["r_shoulder_fe"] = 1.0
    KP["r_shoulder_aa"] = 1.0
    KP["r_shoulder_ie"] = 1.0
    KP["r_elbow_fe"] = 1.0
    KP["r_wrist_ps"] = 1.0
    KP["r_wrist_pitch"] = 1.0

    KD["l_hip_ie"] = 0.0
    KD["l_hip_aa"] = 0.0
    KD["l_hip_fe"] = 0.0
    KD["l_knee_fe_jp"] = 0.0
    KD["l_knee_fe_jd"] = 0.0
    KD["l_ankle_fe"] = 0.0
    KD["l_ankle_ie"] = 0.0
    KD["l_shoulder_fe"] = 0.0
    KD["l_shoulder_aa"] = 0.0
    KD["l_shoulder_ie"] = 0.0
    KD["l_elbow_fe"] = 0.0
    KD["l_wrist_ps"] = 0.0
    KD["l_wrist_pitch"] = 0.0
    KD["neck_pitch"] = 0.0
    KD["r_hip_ie"] = 0.0
    KD["r_hip_aa"] = 0.0
    KD["r_hip_fe"] = 0.0
    KD["r_knee_fe_jp"] = 0.0
    KD["r_knee_fe_jd"] = 0.0
    KD["r_ankle_fe"] = 0.0
    KD["r_ankle_ie"] = 0.0
    KD["r_shoulder_fe"] = 0.0
    KD["r_shoulder_aa"] = 0.0
    KD["r_shoulder_ie"] = 0.0
    KD["r_elbow_fe"] = 0.0
    KD["r_wrist_ps"] = 0.0
    KD["r_wrist_pitch"] = 0.0


class PnCConfig(object):
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = True
    SAVE_FREQ = 1

    PRINT_ROBOT_INFO = SimConfig.PRINT_ROBOT_INFO


class WBCConfig(object):
    VERBOSE = False

    # Max normal force per contact
    RF_Z_MAX = 1000.0

    # Task Hierarchy Weights
    W_COM = 20.0
    W_TORSO = 20.0
    W_UPPER_BODY = 0.1
    W_HAND_POS_MIN = 0.
    W_HAND_POS_MAX = 100.0
    W_HAND_ORI_MIN = 0.
    W_HAND_ORI_MAX = 100.0
    W_CONTACT_FOOT = 60.0
    W_SWING_FOOT = 40.0

    # Task Gains
    KP_COM = np.array([400., 400., 400])
    KD_COM = np.array([20., 20., 20.])

    KP_TORSO = np.array([100., 100., 100])
    KD_TORSO = np.array([10., 10., 10.])

    # ['neck_pitch', 'l_shoulder_fe', 'l_shoulder_aa', 'l_shoulder_ie',
    # 'l_elbow_fe', 'l_wrist_ps', 'l_wrist_pitch', 'r_shoulder_fe',
    # 'r_shoulder_aa', 'r_shoulder_ie', 'r_elbow_fe', 'r_wrist_ps',
    # 'r_wrist_pitch'
    # ]
    # KP_UPPER_BODY = np.array([
    # 40., 100., 100., 100., 50., 40., 40., 100., 100., 100., 50., 40., 40.
    # ])

    KP_UPPER_BODY = np.array(
        [20., 50., 50., 50., 25., 20., 20., 50., 50., 50., 25., 20., 20.])

    KD_UPPER_BODY = np.array(
        # [2., 8., 8., 8., 3., 2., 2., 8., 8., 8., 3., 2., 2.])
        # [4., 15., 15., 15., 6., 4., 4., 15., 15., 15., 6., 4., 4.])
        [8., 20., 20., 20., 12., 8., 8., 20., 20., 20., 12., 8., 8.])

    KP_HAND_POS = np.array([400., 400., 400.])
    KD_HAND_POS = np.array([60., 60., 60.])
    # KP_HAND_POS = np.array([250., 250., 250.])
    # KD_HAND_POS = np.array([5., 5., 5.])
    KP_HAND_ORI = np.array([250., 250., 250.])
    KD_HAND_ORI = np.array([10., 10., 10.])

    KP_FOOT_POS = np.array([300., 300., 300.])
    KD_FOOT_POS = np.array([30., 30., 30.])
    KP_FOOT_ORI = np.array([300., 300., 300.])
    KD_FOOT_ORI = np.array([30., 30., 30.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-8
    LAMBDA_RF = 1e-7

    # B_TRQ_LIMIT = True
    B_TRQ_LIMIT = False

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians


class WalkingConfig(object):
    # STAND
    # INIT_STAND_DUR = 1.0
    INIT_STAND_DUR = 0.1
    RF_Z_MAX_TIME = 0.05

    # COM_HEIGHT = 0.73  # m
    COM_HEIGHT = 0.65  # m
    SWING_HEIGHT = 0.04  # m

    T_ADDITIONAL_INI_TRANS = 0.  # sec
    T_CONTACT_TRANS = 1.0
    T_SWING = 1.0
    PERCENTAGE_SETTLE = 0.9
    ALPHA_DS = 0.5

    ## !! This will be overwritten in main !! ##
    NOMINAL_FOOTWIDTH = 0.28
    NOMINAL_FORWARD_STEP = 0.1
    NOMINAL_BACKWARD_STEP = -0.1
    NOMINAL_TURN_RADIANS = np.pi / 10
    NOMINAL_STRAFE_DISTANCE = 0.05


class ManipulationConfig(object):
    T_REACHING_TRANS_DURATION = 0.75
    T_REACHING_DURATION = 1.5

    T_RETURNING_TRANS_DURATION = 0.75

    ## !! This will be overwritten in main !! ##
    LH_TARGET_POS = np.array([0.29, 0.23, 0.96])
    LH_TARGET_QUAT = np.array([0.2, -0.64, -0.21, 0.71])
    # LH_TARGET_QUAT = np.array([0., 0., 0., 1.])
    RH_TARGET_POS = np.array([0.29, -0.24, 0.96])
    RH_TARGET_QUAT = np.array([-0.14, -0.66, 0.15, 0.72])
    # RH_TARGET_QUAT = np.array([0., 0., 0., 1.])


class LocomanipulationState(object):
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
    RH_HANDRETURN = 10
    LH_HANDRETURN = 11

import numpy as np

from util import util


class ManipulatorConfig(object):
    DT = 0.01
    N_SUBSTEP = 1

    PRINT_ROBOT_INFO = True
    VIDEO_RECORD = False
    DYN_LIB = "pinocchio"

    DES_EE_POS = np.array([1., 1., 0.])
    KP = 0.5
    KD = 0.

import numpy as np

from util import util
from pnc.data_saver import DataSaver
from config.draco3_lb_config import PnCConfig, WBCConfig
from pnc.draco3_lb_pnc.draco3_lb_ihwbc import Draco3LBIHWBC
from pnc.wbc.ihwbc.joint_integrator import JointIntegrator


class Draco3LBController(object):
    def __init__(self, tf_container, robot):
        self._tf_container = tf_container
        self._robot = robot

        # Initialize WBC
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
            ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])

        jac_int = np.zeros((2, self._robot.n_q_dot))
        jac_int[0, l_jp_idx] = 1.
        jac_int[0, l_jd_idx] = -1.
        jac_int[1, r_jp_idx] = 1.
        jac_int[1, r_jd_idx] = -1.

        act_list = [False] * robot.n_floating + [True] * robot.n_a
        act_list[l_jd_idx] = False
        act_list[r_jd_idx] = False

        self._wbc = Draco3LBIHWBC(act_list, jac_int, PnCConfig.SAVE_DATA)

        self._full_to_active = self._wbc._sa[:, 6:]  # TODO

        if WBCConfig.B_TRQ_LIMIT:
            self._wbc.trq_limit = np.dot(self._full_to_active,
                                         self._robot.joint_trq_limit)
        self._wbc.lambda_q_ddot = WBCConfig.LAMBDA_Q_DDOT
        self._wbc.lambda_rf = WBCConfig.LAMBDA_RF
        self._wbc.lambda_if = WBCConfig.LAMBDA_IF

        # Initialize Joint Integrator
        self._joint_integrator = JointIntegrator(robot.n_a,
                                                 PnCConfig.CONTROLLER_DT)
        self._joint_integrator.pos_cutoff_freq = WBCConfig.POS_CUTOFF_FREQ
        self._joint_integrator.vel_cutoff_freq = WBCConfig.VEL_CUTOFF_FREQ
        self._joint_integrator.max_pos_err = WBCConfig.MAX_POS_ERR
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit

        self._b_first_visit = True

        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()

    def get_command(self):
        if self._b_first_visit:
            self.first_visit()

        # Dynamics properties
        mass_matrix = self._robot.get_mass_matrix()
        mass_matrix_inv = np.linalg.inv(mass_matrix)
        coriolis = self._robot.get_coriolis()
        gravity = self._robot.get_gravity()
        self._wbc.update_setting(mass_matrix, mass_matrix_inv, coriolis,
                                 gravity)
        # Task and Contact Setup
        w_hierarchy_list = []
        for task in self._tf_container.task_list:
            task.update_jacobian()
            task.update_cmd()
            w_hierarchy_list.append(task.w_hierarchy)
        self._wbc.w_hierarchy = np.array(w_hierarchy_list)
        for contact in self._tf_container.contact_list:
            contact.update_contact()

        # WBC commands
        joint_trq_cmd, joint_acc_cmd, rf_cmd = self._wbc.solve(
            self._tf_container.task_list, self._tf_container.contact_list,
            False)

        joint_trq_cmd = np.dot(self._full_to_active.transpose(), joint_trq_cmd)
        joint_acc_cmd = np.dot(self._full_to_active.transpose(), joint_acc_cmd)

        # Double integration
        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            joint_acc_cmd, self._robot.joint_velocities,
            self._robot.joint_positions)

        if PnCConfig.SAVE_DATA:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)

        command = self._robot.create_cmd_ordered_dict(joint_pos_cmd,
                                                      joint_vel_cmd,
                                                      joint_trq_cmd)
        return command

    def first_visit(self):
        joint_pos_ini = self._robot.joint_positions
        self._joint_integrator.initialize_states(np.zeros(self._robot.n_a),
                                                 joint_pos_ini)

        self._b_first_visit = False

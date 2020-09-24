import numpy as np

from config.atlas_config import PnCConfig, WBCConfig
from pnc.wbc.wbc import WBC
from pnc.wbc.joint_integrator import JointIntegrator


class AtlasController(object):
    def __init__(self, tf_container, robot):
        self._tf_container = tf_container
        self._robot = robot

        # Initialize WBC
        act_list = [False] * robot.n_virtual + [True] * robot.n_a
        self._wbc = WBC(act_list)
        if WBCConfig.B_TRQ_LIMIT:
            self._wbc.trq_limit = self._robot.joint_trq_limit
        else:
            self._wbc.trq_limit = None
        self._wbc.lambda_q_ddot = WBCConfig.LAMBDA_Q_DDOT
        self._wbc.lambda_rf = WBCConfig.LAMBDA_RF
        # Initialize Joint Integrator
        self._joint_integrator = JointIntegrator(robot.n_a,
                                                 PnCConfig.CONTROLLER_DT)
        self._joint_integrator.pos_cutoff_freq = WBCConfig.POS_CUTOFF_FREQ
        self._joint_integrator.vel_cutoff_freq = WBCConfig.VEL_CUTOFF_FREQ
        self._joint_integrator.max_pos_err = WBCConfig.MAX_POS_ERR
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit

    def get_command(self):
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
            self._tf_container.task_list, self._tf_container.contact_list)
        # Double integration
        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            joint_acc_cmd,
            self._robot.get_q_dot()[self._robot.n_virtual:],
            self._robot.get_q()[self._robot.n_virtual:])

        command = dict()
        command["joint_pos"] = joint_pos_cmd
        command["joint_vel"] = joint_vel_cmd
        command["joint_trq"] = joint_trq_cmd

        return command

    def first_visit(self):
        joint_pos_ini = self._robot.get_q()[self._robot.n_virtual:self._robot.
                                            n_a]
        self._joint_integrator.intialize(np.zeros(self._robot.n_a),
                                         joint_pos_ini)

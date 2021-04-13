import sys
import numpy as np
np.set_printoptions(precision=6, threshold=sys.maxsize)
from scipy.linalg import block_diag
from qpsolvers import solve_qp

from pnc.data_saver import DataSaver


class Draco3IHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """
    def __init__(self, robot, act_list, data_save=False):
        self._robot = robot
        self._n_q_dot = len(act_list)
        self._n_active = np.count_nonzero(np.array(act_list))
        self._n_passive = self._n_q_dot - self._n_active

        # Selection matrix
        self._sa = np.zeros((self._n_active, self._n_q_dot))
        self._sv = np.zeros((self._n_passive, self._n_q_dot))
        j, k = 0, 0
        for i in range(self._n_q_dot):
            if act_list[i]:
                self._sa[j, i] = 1.
                j += 1
            else:
                self._sv[k, i] = 1.
                k += 1

        # Assume first six is floating
        self._sf = np.zeros((6, self._n_q_dot))
        self._sf[0:6, 0:6] = np.eye(6)

        # Internal constraint
        self._n_int = 2
        self._jac_int = np.zeros((self._n_int, self._n_q_dot))
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
            ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])
        self._jac_int[0, l_jp_idx] = 1.
        self._jac_int[0, l_jd_idx] = -1.
        self._jac_int[1, r_jp_idx] = 1.
        self._jac_int[1, r_jd_idx] = -1.

        self._trq_limit = None
        self._lambda_q_ddot = 0.
        self._lambda_if = 0.
        self._lambda_rf = 0.
        self._w_hierarchy = 0.

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def trq_limit(self):
        return self._trq_limit

    @property
    def lambda_q_ddot(self):
        return self._lambda_q_ddot

    @property
    def lambda_rf(self):
        return self._lambda_rf

    @property
    def lambda_if(self):
        return self._lambda_if

    @property
    def w_hierarchy(self):
        return self._w_hierarchy

    @trq_limit.setter
    def trq_limit(self, val):
        assert val.shape[0] == self._n_active
        self._trq_limit = np.copy(val)

    @lambda_q_ddot.setter
    def lambda_q_ddot(self, val):
        self._lambda_q_ddot = val

    @lambda_rf.setter
    def lambda_rf(self, val):
        self._lambda_rf = val

    @lambda_if.setter
    def lambda_if(self, val):
        self._lambda_if = val

    @w_hierarchy.setter
    def w_hierarchy(self, val):
        self._w_hierarchy = val

    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = np.copy(mass_matrix)
        self._mass_matrix_inv = np.copy(mass_matrix_inv)
        self._coriolis = np.copy(coriolis)
        self._gravity = np.copy(gravity)

    def solve(self, task_list, contact_list, verbose=False):
        """
        Parameters
        ----------
        task_list (list of Task):
            Task list
        contact_list (list of Contact):
            Contact list
        verbose (bool):
            Printing option

        Returns
        -------
        joint_trq_cmd (np.array):
            Joint trq cmd
        joint_acc_cmd (np.array):
            Joint acc cmd
        sol_rf (np.array):
            Reaction force
        sol_if (np.array):
            Internal force
        """

        # ======================================================================
        # Cost
        # ======================================================================
        cost_t_mat = np.zeros((self._n_q_dot, self._n_q_dot))
        cost_t_vec = np.zeros(self._n_q_dot)

        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print(i, " th task")
                task.debug()

            cost_t_mat += self._w_hierarchy[i] * np.dot(j.transpose(), j)
            cost_t_vec += self._w_hierarchy[i] * np.dot(
                (j_dot_q_dot - x_ddot).transpose(), j)
        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        cost_t_mat += self._lambda_q_ddot * self._mass_matrix

        cost_if_mat = self._lambda_if * np.eye(self._n_int)
        cost_if_vec = np.zeros(self._n_int)

        if contact_list is not None:
            uf_mat = np.array(
                block_diag(
                    *[contact.cone_constraint_mat
                      for contact in contact_list]))
            uf_vec = np.concatenate(
                [contact.cone_constraint_vec for contact in contact_list])
            contact_jacobian = np.concatenate(
                [contact.jacobian for contact in contact_list], axis=0)

            assert uf_mat.shape[0] == uf_vec.shape[0]
            assert uf_mat.shape[1] == contact_jacobian.shape[0]
            dim_cone_constraint, dim_contacts = uf_mat.shape

            cost_rf_mat = self._lambda_rf * np.eye(dim_contacts)
            cost_rf_vec = np.zeros(dim_contacts)

            cost_mat = np.array(
                block_diag(cost_t_mat, cost_if_mat,
                           cost_rf_mat))  # (nqdot+nint+nc, nqdot+nint+nc)
            cost_vec = np.concatenate([cost_t_vec, cost_if_vec,
                                       cost_rf_vec])  # (nqdot+nint+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = np.array(block_diag(cost_t_mat, cost_if_mat))
            cost_vec = np.concatenate([cost_t_vec, cost_if_vec])

        if verbose:
            print("cost_t_mat")
            print(cost_t_mat)
            print("cost_t_vec")
            print(cost_t_vec)
            print("cost_rf_mat")
            print(cost_rf_mat)
            print("cost_rf_vec")
            print(cost_rf_vec)

        # ======================================================================
        # Equality Constraint
        # ======================================================================

        if contact_list is not None:
            eq_floating_mat = np.concatenate(
                (np.dot(self._sf, self._mass_matrix), np.zeros(
                    (6, self._n_int)),
                 -np.dot(self._sf, contact_jacobian.transpose())),
                axis=1)  # (6, nqdot+nint+nc)
            eq_int_mat = np.concatenate(
                (self._jac_int, np.zeros(
                    (2, self._n_int)), np.zeros((2, dim_contacts))),
                axis=1)  # (2, nqdot+nint+nc)
        else:
            eq_floating_mat = np.concatenate(
                (np.dot(self._sf, self._mass_matrix), np.zeros(
                    (6, self._n_int))),
                axis=1)  # (6, nqdot+nint)
            eq_int_mat = np.concatenate(
                (self._jac_int, np.zeros(
                    (2, self._n_int))), axis=1)  # (2, nqdot+nint)
        eq_floating_vec = -np.dot(self._sf, (self._coriolis + self._gravity))
        eq_int_vec = np.zeros(2)

        eq_trq_mat = np.concatenate(
            (np.dot(self._jac_int, self._mass_matrix),
             -np.dot(self._jac_int, self._jac_int.transpose()),
             -np.dot(self._jac_int, contact_jacobian.transpose())),
            axis=1)
        eq_trq_vec = -np.dot(self._jac_int, (self._coriolis + self._gravity))

        eq_mat = np.concatenate((eq_floating_mat, eq_trq_mat, eq_int_mat),
                                axis=0)
        eq_vec = np.concatenate((eq_floating_vec, eq_trq_vec, eq_int_vec),
                                axis=0)

        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = np.concatenate((np.zeros(
                    (dim_cone_constraint, self._n_q_dot + self._n_int)),
                                           -uf_mat),
                                          axis=1)
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            if contact_list is not None:
                ineq_mat = np.concatenate(
                    (np.concatenate((np.zeros(
                        (dim_cone_constraint, self._n_q_dot)),
                                     -np.dot(self._sa, self._mass_matrix),
                                     np.dot(self._sa, self._mass_matrix)),
                                    axis=0),
                     np.concatenate(
                         (np.zeros((dim_cone_constraint, self._n_int)),
                          np.dot(self._sa, self._jac_int.transpose()),
                          -np.dot(self._sa, self._jac_int.transpose())),
                         axis=0),
                     np.concatenate(
                         (-uf_mat,
                          np.dot(self._sa, contact_jacobian.transpose()),
                          -np.dot(self._sa, contact_jacobian.transpose())),
                         axis=0)),
                    axis=1)
                ineq_vec = np.concatenate(
                    (-uf_vec,
                     np.dot(self._sa, self._coriolis + self._gravity) -
                     self._trq_limit[:, 0],
                     -np.dot(self._sa, self._coriolis + self._gravity) +
                     self._trq_limit[:, 1]))
            else:
                ineq_mat = np.concatenate(
                    (np.concatenate((-np.dot(self._sa, self._mass_matrix),
                                     np.dot(self._sa, self._mass_matrix)),
                                    axis=0),
                     np.concatenate(
                         (np.dot(self._sa, self._jac_int.transpose()),
                          -np.dot(self._sa, self._jac_int.transpose())),
                         axis=0)),
                    axis=1)
                ineq_vec = np.concatenate(
                    (np.dot(self._sa, self._coriolis + self._gravity) -
                     self._trq_limit[:, 0],
                     -np.dot(self._sa, self._coriolis + self._gravity) +
                     self._trq_limit[:, 1]))

        if verbose:
            print("eq_mat")
            print(eq_mat)
            print("eq_vec")
            print(eq_vec)

            print("ineq_mat")
            print(ineq_mat)
            print("ineq_vec")
            print(ineq_vec)

        sol = solve_qp(cost_mat,
                       cost_vec,
                       ineq_mat,
                       ineq_vec,
                       eq_mat,
                       eq_vec,
                       solver="quadprog",
                       verbose=True)

        if contact_list is not None:
            sol_q_ddot, sol_if, sol_rf = sol[:self._n_q_dot], sol[
                self._n_q_dot:self._n_q_dot + self._n_int], sol[self._n_q_dot +
                                                                self._n_int:]
        else:
            sol_q_ddot, sol_if, sol_rf = sol[:self._n_q_dot], sol[
                self._n_q_dot:self._n_q_dot + self._n_int], None

        if contact_list is not None:
            joint_trq_cmd = np.dot(
                self._sa,
                np.dot(self._mass_matrix, sol_q_ddot) + self._coriolis +
                self._gravity - np.dot(contact_jacobian.transpose(), sol_rf) -
                np.dot(self._jac_int.transpose(), sol_if))
        else:
            joint_trq_cmd = np.dot(
                self._sa,
                np.dot(self._mass_matrix, sol_q_ddot) + self._coriolis +
                self._gravity - np.dot(self._jac_int.transpose(), sol_if))

        joint_acc_cmd = np.dot(self._sa, sol_q_ddot)

        if verbose:
            print("joint_trq_cmd: ", joint_trq_cmd)
            print("sol_q_ddot: ", sol_q_ddot)
            print("sol_rf: ", sol_rf)
            print("sol_if: ", sol_if)

        if self._b_data_save:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)
            self._data_saver.add('joint_acc_cmd', joint_acc_cmd)
            self._data_saver.add('if_cmd', sol_if)
            self._data_saver.add('rf_cmd', sol_rf)

        return joint_trq_cmd, joint_acc_cmd, sol_rf, sol_if

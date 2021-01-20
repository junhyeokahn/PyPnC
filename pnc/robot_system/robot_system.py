import abc
from collections import OrderedDict

import numpy as np


class RobotSystem(abc.ABC):
    def __init__(self,
                 filepath,
                 floating_joint_list,
                 b_print_robot_info=False):
        """
        Base RobotSystem Class

        Parameters
        ----------
        filepath (str):
            urdf path
        floating_joint_list (list of str):
            list of floating joint name
        """

        self._b_print_robot_info = b_print_robot_info

        self._n_virtual = 0
        self._n_q = 0
        self._n_q_dot = 0
        self._n_a = 0
        self._total_mass = 0.
        self._joint_pos_limit = []
        self._joint_vel_limit = []
        self._joint_trq_limit = []
        self._joint_id = OrderedDict()
        self._floating_id = OrderedDict()
        self._link_id = OrderedDict()

        # Centroidal Quantities
        self._I_cent = np.zeros((6, 6))
        self._J_cent = np.zeros((6, self._n_q_dot))
        self._A_cent = np.zeros((6, self._n_q_dot))

        self._config_robot(filepath, floating_joint_list)

    @property
    def n_virtual(self):
        return self._n_virtual

    @property
    def n_q(self):
        return self._n_q

    @property
    def n_q_dot(self):
        return self._n_q_dot

    @property
    def n_a(self):
        return self._n_a

    @property
    def total_mass(self):
        return self._total_mass

    @property
    def joint_pos_limit(self):
        return self._joint_pos_limit

    @property
    def joint_vel_limit(self):
        return self._joint_vel_limit

    @property
    def joint_trq_limit(self):
        return self._joint_trq_limit

    @property
    def joint_id(self):
        return self._joint_id

    @property
    def link_id(self):
        return self._link_id

    @property
    def I_cent(self):
        return self._I_cent

    @property
    def J_cent(self):
        return self._J_cent

    @property
    def A_cent(self):
        return self._A_cent

    @abc.abstractmethod
    def _update_centroidal_quantities(self):
        """
        Update I_cent, A_cent, J_cent
        , where
        centroid_momentum = I_cent * centroid_velocity = A_cent * qdot
                  J_cent = inv(I_cent) * A_cent
        centroid_velocity = J_cent * qdot
        """
        pass

    @abc.abstractmethod
    def _config_robot(self, filepath):
        """
        Set properties.

        Parameters
        ----------
        filepath (str): urdf path
        """
        pass

    @abc.abstractmethod
    def get_q_idx(self, joint_id):
        """
        Get joint index in generalized coordinate

        Parameters
        ----------
        joint_id (str or list of str)

        Returns
        -------
        joint_idx (int or list of int)
        """
        pass

    @abc.abstractmethod
    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd,
                                joint_trq_cmd):
        """
        Create command ordered dict

        Parameters
        ----------
        joint_pos_cmd (np.array):
            Joint Pos Cmd
        joint_vel_cmd (np.array):
            Joint Vel Cmd
        joint_trq_cmd (np.array):
            Joint Trq Cmd

        Returns
        -------
        command (OrderedDict)
        """
        pass

    @abc.abstractmethod
    def update_system(self,
                      base_pos,
                      base_quat,
                      base_lin_vel,
                      base_ang_vel,
                      joint_pos,
                      joint_vel,
                      b_cent=False):
        """
        Update generalized coordinate

        Parameters
        ----------
        base_pos (np.array): Root pos, None if the robot is fixed in the world
        base_quat (np.array): Root quat
        base_lin_vel (np.array): Root linear velocity
        base_ang_vel (np.array): Root angular velocity
        joint_pos (OrderedDict): Actuator pos
        joint_vel (OrderedDict): Actuator vel
        b_cent (Bool): Whether updating centroidal frame or not
        """
        pass

    @abc.abstractmethod
    def get_q(self):
        """
        Returns
        -------
        q (np.array): positions in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_q_dot(self):
        """
        Returns
        -------
        qdot (np.array): velocities in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_mass_matrix(self):
        """
        Returns
        -------
        A (np.array): Mass matrix in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_gravity(self):
        """
        Returns
        -------
        g (np.array): Gravity forces in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_coriolis(self):
        """
        Returns
        -------
        c (np.array): Coriolis forces in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_com_pos(self):
        """
        Returns
        -------
        com_pos (np.array): COM position
        """
        pass

    @abc.abstractmethod
    def get_com_lin_vel(self):
        """
        Returns
        -------
        com_lin_vel (np.array): COM linear velocity
        """
        pass

    @abc.abstractmethod
    def get_com_lin_jacobian(self):
        """
        Returns
        -------
        com_lin_jac (np.array): COM linear jacobian
        """
        pass

    @abc.abstractmethod
    def get_com_lin_jacobian_dot(self):
        """
        Returns
        -------
        com_lin_jac_dot (np.array): COM linear jacobian dot
        """
        pass

    @abc.abstractmethod
    def get_link_iso(self, link_id):
        """
        Returns
        -------
        link_iso (np.array): Link SE(3)
        """
        pass

    @abc.abstractmethod
    def get_link_vel(self, link_id):
        """
        Returns
        -------
        link_vel (np.array): Link Screw
        """
        pass

    @abc.abstractmethod
    def get_link_jacobian(self, link_id):
        """
        Returns
        -------
        link_jac (np.array): Link jacobian
        """
        pass

    @abc.abstractmethod
    def get_link_jacobian_dot(self, link_id):
        """
        Returns
        -------
        link_jac_dot (np.array): Link jacobian dot
        """
        pass

import abc


class RobotSystem(abc.ABC):
    def __init__(self, n_virtual, filepath):
        """
        Base RobotSystem Class

        Parameters
        ----------
        n_virtual (int): Number of DOF for Root
        filepath (str): urdf path
        """
        self._n_virtual = n_virtual
        self._n_q = 0
        self._n_qdot = 0
        self._n_a = 0
        self._total_mass = 0.
        self._joint_pos_limit = []
        self._joint_vel_limit = []
        self._joint_trq_limit = []
        self._joint_id = dict()
        self._link_id = dict()

        self.config_robot(filepath)

    @property
    def n_virtual(self):
        return self._n_virtual

    @property
    def n_q(self):
        return self._n_q

    @property
    def n_qdot(self):
        return self._n_qdot

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

    @abc.abstractmethod
    def config_robot(self, filepath):
        """
        Set properties.

        Parameters
        ----------
        filepath (str): urdf path
        """
        pass

    @abc.abstractmethod
    def update_system(self, base_pos, base_quat, base_lin_vel, base_ang_vel,
                      joint_pos, joint_vel):
        """
        Update generalized coordinate

        Parameters
        ----------
        base_pos (np.array): Root pos, None if the robot is fixed in the world
        base_quat (np.array): Root quat
        base_lin_vel (np.array): Root linear velocity
        base_ang_vel (np.array): Root angular velocity
        joint_pos (dict): Actuator pos
        joint_vel (dict): Actuator vel
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
    def get_qdot(self):
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
    def get_com_lin_jac(self):
        """
        Returns
        -------
        com_lin_jac (np.array): COM linear jacobian
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
    def get_link_jac(self, link_id):
        """
        Returns
        -------
        link_jac (np.array): Link jacobian
        """
        pass

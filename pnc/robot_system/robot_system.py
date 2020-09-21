import abc


class RobotSystem(abc.ABC):
    def __init__(self, filepath):
        self._nq = 0
        self._nqdot = 0
        self._na = 0
        self._total_mass = 0.
        self._joint_pos_limit = []
        self._joint_vel_limit = []
        self._joint_trq_limit = []
        self._joint_id = dict()
        self._link_id = dict()

        self.config_robot(filepath)

    @property
    def nq(self):
        return self._nq

    @property
    def nqdot(self):
        return self._nqdot

    @property
    def na(self):
        return self._na

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
        pass

    @abc.abstractmethod
    def update_system(self, base_pos, base_quat, base_lin_vel, base_ang_vel,
                      joint_pos, joint_vel):
        pass

    @abc.abstractmethod
    def get_q(self):
        pass

    @abc.abstractmethod
    def get_qdot(self):
        pass

    @abc.abstractmethod
    def get_mass_matrix(self):
        pass

    @abc.abstractmethod
    def get_gravity(self):
        pass

    @abc.abstractmethod
    def get_coriolis(self):
        pass

    @abc.abstractmethod
    def get_com_pos(self):
        pass

    @abc.abstractmethod
    def get_com_vel(self):
        pass

    @abc.abstractmethod
    def get_com_jac(self):
        pass

    @abc.abstractmethod
    def get_body_iso(self, body_id):
        pass

    @abc.abstractmethod
    def get_body_vel(self, body_id):
        pass

    @abc.abstractmethod
    def get_body_jac(self, body_id):
        pass

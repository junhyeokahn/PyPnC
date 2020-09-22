import abc
import numpy as np

class ContactSpec():
    def __init__(self, robot, dim):
        self._robot = robot
        self._dim_contact = dim
        self._b_set_contact = False
        self._idx_fz = self._dim_contact - 1
        self._jc = np.zeros((self._dim_contact, self._robot.n_q))
        self._jcdot_qdot = np.zeors(self._dim_contact)

        self._max_fz = 0.
        self._uf = None
        self._ieq_vec = []

    @property
    def jc(self):
        return self._jc

    @property
    def jcdot_qdot(self):
        return self._jcdot_qdot

    @property
    def dim_contact(self):
        return self._dim_contact

    @b_set_contact.setter
    def b_set_contact(self, value):
        self._b_set_contact = value

    @max_fz.setter
    def max_fz(self, value):
        return self._max_fz = value

    @property
    def uf(self):
        return self._uf

    @property
    def ieq_vec(self):
        return self._ieq_vec

    @property
    def idx_fz(self):
        return self._idx_fz

    def get_dim_rf_constraint(self):
        return np.shape(self._uf)[0]

    def update_contact_spec(self):
        self.update_jc()
        self.update_jdot_qdot()
        self.update_uf()
        self.update_ieq_vec()
        self._b_set_contact = True
        return True

    @abc.abstractmethod
    def update_jc(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def update_jdot_qdot(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def update_uf(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def update_ieq_vec(self):
        """
        Returns
        -------
        boolean 
        """
        pass


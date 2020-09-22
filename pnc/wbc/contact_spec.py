import abc
import numpy as np

class ContactSpec():
    def __init__(self, robot, dim):
        self._robot = robot
        self._dim_contact = dim
        self._b_set_contact = False
        self._idx_Fz = self._dim_contact - 1
        self._Jc = np.zeros(self._dim_contact, self._robot.n_q)
        self._JcDotQdot = np.zeors(self._dim_contact)

        self._Uf = np.array([])
        self._ieq_vec = []
        ##TODO: self._Uf
        ##TODO: self._ieq_vec

    @property
    def getContactJacobian(self):
        return self._Jc

    @property
    def getJcDotQdot(self):
        return self._JcDotQdot

    @property
    def getDim(self):
        return self._dim_contact

    @unsetContact.setter
    def unsetContact(self):
        self._b_set_contact = False

    ##TODO: virtual void setMaxFz(double _max_fz) {}
    ##TODO: virtual double getMaxFz() { return 0. }
    @property
    def getMaxFz(self):
        return 0.

    def updateContactSpec(self):
        self.UpdateJc()
        self.UpdateJcDotQdot()
        self.UpdateUf()
        self.UpdateInequalityVector()
        self._b_set_contact = True
        return True

    @property
    def getDimRFConstraint(self):
        return np.shape(self._Uf)[0]

    @property
    def getRFConstraintMtx(self):
        return self._Uf

    @property
    def getRFConstraintVec(self):
        return self._ieq_vec

    @property
    def getFzIndex(self):
        return self._idx_Fz

    @abc.abstractmethod
    def UpdateJc(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def UpdateJcDotQdot(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def UpdateUf(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def UpdateInequalityVector(self):
        """
        Returns
        -------
        boolean 
        """
        pass


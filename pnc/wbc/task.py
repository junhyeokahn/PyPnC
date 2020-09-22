import abc
import numpy as np

class Task():
    def __init__(self, robot, dim):
        self._robot = robot 
        self._b_set_task = False
        self._dim_task = dim

        self._w_hierarchy = 1.0
        self._kp = np.zeros(self._dim)
        self._kd = np.zeros(self._dim)
        self._JtDotQdot = np.zeors(self._dim)
        self._Jt = np.zeros(self._dim, self._robot.n_q)

        ## For Dyn WBC
        self._op_cmd = np.zeros(self._dim)

        ## For Kin WBC
        self._pos_err = np.zeros(self._dim)
        self._vel_des = np.zeros(self._dim)
        self._acc_des = np.zeros(self._dim)

        ## Store for reuse old command
        self._pos_des_old = np.zeros(self._dim)
        self._vel_des_old = np.zeros(self._dim)
        self._acc_des_old = np.zeros(self._dim)
    
    ##TODO: virrtual int getLinkID(){}

    @property
    def getCommand(self):
        return self._op_cmd

    @property
    def getTaskJacobian(self):
        return self._Jt

    @property
    def getTaskJacobianDotQdot(self):
        return self._JtDotQdot

    @setGain.setter
    def setGain(self, kp, kd):
        self._kp = kp
        self._kd = kd

    @setHierarchyWeight.setter
    def setHierarchyWeight(self, w_hierarchy):
        self._w_hierarchy = w_hierarchy

    @property
    def getHierarchyWeight(self):
        return self._w_hierarchy

    def updateJacobians(self):
        self.UpdateTaskJacobian()
        self.UpdateTaskJDotQdot()

    @updateDesired.setter
    def updateDesired(self, pos_des, vel_des, acc_des):
        self._pos_des_old = pos_des
        self._vel_des_old = vel_des
        self._acc_des_old = acc_des

    def computeCommands(self):
        self.UpdateCommand(self._pos_des_old, self._vel_des_old, self._acc_des_old)

    def updateTask(self, pos_des, vel_des, acc_des):
        self.updateJacobians()
        self.updateDesired(pos_des, vel_des, acc_des)
        self.computeCommands()
        self._b_set_task = True
        return True

    def isTaskSet(self):
        return self._b_set_task

    @property
    def getDim(self):
        return self._dim_task

    def unsetTask(self):
        self._b_set_task = False

    @abc.abstractmethod
    def UpdateCommand(self, pos_des, vel_des, acc_des):
        """
        Returns
        -------
        boolean 
        """
        pass
    
    @abc.abstractmethod
    def UpdateTaskJacobian(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def UpdateTaskJDotQdot(self):
        """
        Returns
        -------
        boolean 
        """
        pass




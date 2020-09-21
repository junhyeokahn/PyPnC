import abc
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

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
        self._Jt = np.zeros(self._dim, self._robot._n_q)

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
    
    def getCommand(self, op_cmd):
        op_cmd = self._op_cmd

    def getTaskJacobian(self, Jt):
        Jt = self._Jt

    def getTaskJacobianDotQdot(self, JtDotQdot):
        JtDotQdot = self._JtDotQdot

    def setGain(self, kp, kd):
        self._kp = kp
        self._kd = kd

    def setHierarchyWeight(self, w_hierarchy):
        self._w_hierarchy = w_hierarchy

    def getHierarchyWeight(self):
        return self._w_hierarchy

    def updateJacobians(self):
        UpdateTaskJacobian(self)
        UpdateTaskJDotQdot(self)

    def updateDesired(self, pos_des, vel_des, acc_des):
        self.pos_des_old = pos_des
        self.vel_des_old = vel_des
        self.acc_des_old = acc_des

    def computeCommands(self):
        UpdateCommand(self, pos_des, vel_des, acc_des)

    def updateTask(self, pos_des, vel_des, acc_des):
        updateJacobians(self)
        updateDesired(self, pos_des, vel_des, acc_des)
        computeCommands(self)
        self._b_set_task = True
        return True

    def isTaskSet(self):
        return self._b_set_task

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




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
        self._jtdot_qdot = np.zeors(self._dim)
        self._jt = np.zeros((self._dim, self._robot.n_q))

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
    def op_cmd(self):
        return self._op_cmd

    @property
    def jt(self):
        return self._Jt

    @property
    def jtdot_qdot(self):
        return self._jtdot_qdot

    @kp.setter
    def kp(self, value):
        self._kp = value

    @kd.setter
    def kd(self, value):
        self._kd = value

    @w_hierarcy.setter
    def w_hierarcy(self, value):
        self._w_hierarchy = value

    @property
    def w_hierarcy(self):
        return self._w_hierarchy

    @property
    def b_set_task(self):
        return self._b_set_task

    @b_set_task.setter
    def b_set_task(self, value):
        self._b_set_task = value

    @property
    def dim_task(self):
        return self._dim_task

    def update_jacobian(self):
        self.update_task_jacobian()
        self.update_task_jdot_qdot()

    def update_desired(self, pos_des, vel_des, acc_des):
        self._pos_des_old = pos_des
        self._vel_des_old = vel_des
        self._acc_des_old = acc_des

    def compute_command(self):
        self.update_commnad(self._pos_des_old, self._vel_des_old, self._acc_des_old)

    def update_task(self, pos_des, vel_des, acc_des):
        self.update_jacobian()
        self.update_desired(pos_des, vel_des, acc_des)
        self.compute_command()
        self._b_set_task = True
        return True

    @abc.abstractmethod
    def update_commnad(self, pos_des, vel_des, acc_des):
        """
        Returns
        -------
        boolean 
        """
        pass
    
    @abc.abstractmethod
    def update_task_jacobian(self):
        """
        Returns
        -------
        boolean 
        """
        pass

    @abc.abstractmethod
    def update_task_jdot_qdot(self):
        """
        Returns
        -------
        boolean 
        """
        pass




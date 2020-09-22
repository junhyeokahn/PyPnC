import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np

from pnc.wbc.contact_spec import ContactSpec

class PointContactSpec(ContactSpec):
    def __init__(self, robot, link_id, mu):
        super(PointContactSpec, self).__init__(robot, 3)

        print("Point Contact Spec")
        
        self._link_id = link_id
        self._max_fz = 500.
        self._mu = mu

    def update_jc(self):
        jt_temp = self._robot.get_link_jac(self._link_id)
        self._jc = jt_temp[self._dim_contact:, :]
        return true

    def update_jcdot_qdot(self):
        jcdot = self._robot.get_link_jacobian_dot(self._link_id)[self._dim_contact:,:]
        self.jcdot_qdot = np.dot(jcdot, self._robot.get_qdot())
        return true

    def update_uf(self):
        rot = self._robot.get_link_iso(self._link_id)[0:3,0:3].T
        self._uf = np.zeros((6, self._dim_contact))
        self._uf[0,2] = 1.

        self._uf[1,0] = 1.
        self._uf[1,2] = self._mu
        self._uf[2,0] = -1.
        self._uf[2,2] = self._mu


        self._uf[3,1] = 1.
        self._uf[3,2] = self._mu
        self._uf[4,1] = -1.
        self._uf[4,2] = self._mu
        
        self._uf[5,2] = -1.

        self._uf = np.dot(self._uf, rot)
        return true

    def update_ieq_vec(self):
        self._ieq_vec = np.zeors(6)
        self._ieq_vec[5] = -self._max_fz
        return true

class SurfaceContactSpec(ContactSpec):
    def __init__(self, robot, link_id, x, y, mu):
        super(SurfaceContactSpec, self).__init__(robot, 6)

        print("Surface Contact Spec")
        
        self._link_id = link_id
        self._max_fz = 1500.
        self._x = x
        self._y = y
        self._mu = mu

    def update_jc(self):
        self._jc = self._robot.get_link_jac(self._link_id)
        return true

    def update_jcdot_qdot(self):
        self.jcdot_qdot = np.dot(self._robot.get_jacobian_dot(self._link_id), self.get_qdot())
        return true

    def update_uf(self):
        self._uf = np.zeros((16+2, self._dim_contact))

        u = self.get_u(self._x, self._y, self._mu)
        rot = self._robot.get_link_iso(self._link_id)[0:3, 0:3]
        rot_foot = np.zeros((6,6))
        rot_foot[0:3, 0:3] = rot.T
        rot_foot[3:6, 3:6] = rot.T

        self._uf = np.dot(u, rot_foot)
        return true

    def update_ieq_vec(self):
        self._ieq_vec = np.zeors(16+2)
        self._ieq_vec[17] = -self._max_fz
        return true

    def get_u(self, x, y, mu):
        u = np.zeors((16+2, 6))

        u[0,5] = 1.

        u[1,3] = 1.
        u[1,5] = mu
        u[2,3] = -1.
        u[2,5] = mu

        u[3,4] = 1.
        u[3,5] = mu
        u[4,4] = -1.
        u[4,5] = mu

        u[5,0] = 1.
        u[5,5] = y
        u[6,0] = -1.
        u[6,5] = y

        u[7,1] = 1.
        u[7,5] = x
        u[8,1] = -1.
        u[8,5] = x

        ##tau
        u[9,0] = -mu
        u[9,1] = -mu
        u[9,2] = 1.
        u[9,3] = y
        u[9,4] = x 
        u[9,5] = (x + y) * mu

        u[10,0] = -mu
        u[10,1] = mu
        u[10,2] = 1.
        u[10,3] = y
        u[10,4] = -x 
        u[10,5] = (x + y) * mu


        u[11,0] = mu
        u[11,1] = -mu
        u[11,2] = 1.
        u[11,3] = -y
        u[11,4] = x 
        u[11,5] = (x + y) * mu

        u[12,0] = mu
        u[12,1] = mu
        u[12,2] = 1.
        u[12,3] = -y
        u[12,4] = -x 
        u[12,5] = (x + y) * mu


        u[13,0] = -mu
        u[13,1] = -mu
        u[13,2] = -1.
        u[13,3] = -y
        u[13,4] = -x 
        u[13,5] = (x + y) * mu

        u[14,0] = -mu
        u[14,1] = mu
        u[14,2] = -1.
        u[14,3] = -y
        u[14,4] = x 
        u[14,5] = (x + y) * mu

        u[15,0] = mu
        u[15,1] = -mu
        u[15,2] = -1.
        u[15,3] = y
        u[15,4] = -x 
        u[15,5] = (x + y) * mu

        u[16,0] = mu
        u[16,1] = mu
        u[16,2] = -1.
        u[16,3] = y
        u[16,4] = x 
        u[16,5] = (x + y) * mu

        u[17,5] = -1.
    return u

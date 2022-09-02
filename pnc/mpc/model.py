import numpy as np
from casadi import Function, jacobian


class Model:

    def __init__(self, s_expr, a_expr, f_expr, s_steady_state, a_steady_state, name='model', dt=1):

        self.name = name

        self.dt = dt

        self.s_steady_state = s_steady_state
        self.a_steady_state = a_steady_state
        
        self.s_expr = s_expr
        self.a_expr = a_expr

        self.f_expr = f_expr
        self.f = Function('f', [s_expr, a_expr], [f_expr])

        self.J_x = Function('J_x', [s_expr, a_expr], [jacobian(f_expr, s_expr)])
        self.J_u = Function('J_u', [s_expr, a_expr], [jacobian(f_expr, a_expr)])

        self.ns = s_expr.shape[0]
        self.na = a_expr.shape[0]


    def simulate(self, s0, a0, w0=0.0):

        s1 = np.reshape(self.f(s0, a0).full(), (self.ns, 1)) + np.reshape(w0, [np.size(w0), 1])

        return s1

    def simulate_traj(self, s0, a_traj):

        N_sim = a_traj.shape[1]

        s_traj = np.zeros((self.ns, N_sim+1))
        s_traj[:, [0]] = s0

        for n in range(N_sim):
            s1 = self.simulate(s_traj[:, n], a_traj[:, n])
            s_traj[:, [n+1]] = s1 

        return s_traj

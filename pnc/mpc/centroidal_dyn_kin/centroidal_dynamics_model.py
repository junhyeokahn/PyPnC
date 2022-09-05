from casadi import *
from pnc.mpc.mpc_utils import integrate_RK4
from pnc.mpc.model import Model


class CentroidalDynamics(Model):
    def __init__(self, robot, lf_pos_vec, rf_pos_vec, name='centroidal_model'):
        # MPC states
        nx = 9
        nc = 2
        nu = 3 * nc
        p_com = SX.sym('p_com', 3, 1)  # CoM position
        v_com = SX.sym('v_com', 3, 1)  # CoM velocity
        L = SX.sym('L', 3, 1)  # angular momentum about CoM
        x_expr = vertcat(p_com, v_com, L)

        # MPC control
        F_lfoot = SX.sym('F_lfoot', 3, 1)
        m_lfoot = SX.sym('m_lfoot', 3, 1)
        F_rfoot = SX.sym('F_rfoot', 3, 1)
        m_rfoot = SX.sym('m_rfoot', 3, 1)
        # F_larm = SX.sym('F_larm', 3, 1)
        # m_larm = SX.sym('m_larm', 3, 1)
        # F_rarm = SX.sym('F_rarm', 3, 1)
        # m_rarm = SX.sym('m_rarm', 3, 1)
        # u_expr = vertcat(F_lfoot, m_lfoot, F_rfoot, m_rfoot, F_larm, m_larm, F_rarm, m_rarm)
        u_expr = vertcat(F_lfoot, F_rfoot)

        # MPC control limits
        self.u_max = 500 * np.ones(nu)
        self.u_min = -self.u_max
        self.u_min[2] = 0.
        self.u_min[5] = 0.

        # dynamics
        g = 9.81
        mass = robot.total_mass

        g_vec = np.array([0., 0., -g])
        # a_com = g_vec + (1/M) * (F_lfoot + F_rfoot + F_larm + F_rarm)
        a_com = g_vec + (1 / mass) * (F_lfoot + F_rfoot)
        L_dot = cross(lf_pos_vec - p_com, F_lfoot) + cross(rf_pos_vec - p_com, F_rfoot)

        x_dot = vertcat(v_com,
                        a_com,
                        L_dot
                        )

        # discrete dynamics
        n_steps = 5
        dt = 0.01
        F_discrete = integrate_RK4(x_expr, u_expr, x_dot, dt, n_steps)

        # steady state
        s_steady_state = np.zeros((nx, 1))
        s_steady_state[0:3, 0] = robot.get_com_pos()
        a_steady_state = np.zeros((nu, 1))
        a_steady_state[2] = mass * g / 2.
        a_steady_state[5] = mass * g / 2.
        super().__init__(x_expr, u_expr, F_discrete, s_steady_state, a_steady_state, name, dt)

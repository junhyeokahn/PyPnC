import numpy as np
from casadi import *
from pnc.mpc.mpc_utils import integrate_RK4
from pnc.mpc.model import Model


class CMMJointDynamics(Model):
    def __init__(self, robot, r_com_lf, r_com_rf, A_cmm, name='cmm_joint_model'):
        # MPC states
        na = robot.n_a
        nx = 12 + na
        nc = 2
        nu = 6 * nc + na
        h_lin = SX.sym('h_lin', 3, 1)       # linear momentum
        h_ang = SX.sym('h_ang', 3, 1)       # angular momentum
        q_base = SX.sym('q_base', 6, 1)     # base pose (position, rpy)
        q_joints = SX.sym('q_joints', na, 1)
        x_expr = vertcat(h_lin, h_ang, q_base, q_joints)

        # MPC control
        F_lfoot = SX.sym('F_lfoot', 3, 1)
        m_lfoot = SX.sym('m_lfoot', 3, 1)
        F_rfoot = SX.sym('F_rfoot', 3, 1)
        m_rfoot = SX.sym('m_rfoot', 3, 1)
        # F_larm = SX.sym('F_larm', 3, 1)
        # m_larm = SX.sym('m_larm', 3, 1)
        # F_rarm = SX.sym('F_rarm', 3, 1)
        # m_rarm = SX.sym('m_rarm', 3, 1)
        v_joint = SX.sym('v_joint', na, 1)
        u_expr = vertcat(F_lfoot, m_lfoot, F_rfoot, m_rfoot, v_joint)

        # MPC control limits
        feet_force_ulim = np.array([500., 500., 0.])
        feet_moment_ulim = np.array([100., 100., 0.])
        v_joint_ulim = robot.joint_vel_limit[:, 1]
        feet_force_llim = -feet_force_ulim
        feet_moment_llim = -feet_moment_ulim
        v_joint_llim = robot.joint_vel_limit[:, 0]
        self.u_max = np.append(feet_force_ulim, feet_moment_ulim, v_joint_ulim)
        self.u_min = np.append(feet_force_llim, feet_moment_llim, v_joint_llim)

        # dynamics
        g = 9.81
        mass = robot.total_mass

        g_vec = np.array([0., 0., -g])
        A_b_inv = np.linalg.inv(A_cmm[:, :6])
        A_j = A_cmm[:, 6:]
        hdot_lin = mass * g_vec + (F_lfoot + F_rfoot)
        hdot_ang = cross(r_com_lf, F_lfoot) + m_lfoot + cross(r_com_rf, F_rfoot) + m_rfoot
        qdot_base = A_b_inv * (vertcat(h_lin, h_ang) - A_j * v_joint)

        x_dot = vertcat(hdot_lin,
                        hdot_ang,
                        qdot_base,
                        v_joint
                        )

        # discrete dynamics
        n_steps = 5
        dt = 0.01
        F_discrete = integrate_RK4(x_expr, u_expr, x_dot, dt, n_steps)

        # steady state
        s_steady_state = np.zeros((nx, 1))
        a_steady_state = np.zeros((nu, 1))
        a_steady_state[2] = mass * g / 2.
        a_steady_state[2+6] = mass * g / 2.
        super().__init__(x_expr, u_expr, F_discrete, s_steady_state, a_steady_state, name, dt)

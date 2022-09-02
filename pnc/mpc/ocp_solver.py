from casadi import *


class OCPsolver:

    def __init__(self, model, stage_cost, terminal_cost, N, a_max, a_min, s_des_traj, a_guess_traj):

        self.N = N
        self.model = model

        ns = model.ns
        na = model.na

        # setup OCP

        # optimization variables
        s_traj = SX.sym('s_traj', ns * (N + 1), 1)
        a_traj = SX.sym('a_traj', na * N, 1)

        # split trajectories to obtain x and u for each stage
        s_ns = vertsplit(s_traj, np.arange(0, ns * (N + 1) + 1, ns))
        a_ns = vertsplit(a_traj, np.arange(0, na * N + 1, na))

        # initial state (will be a parameter of the optimization problem)
        s0_bar = SX.sym('s0_bar', ns, 1)

        cost = 0

        # initial constraint
        constraints = [s_ns[0] - s0_bar]

        for n in range(N):
            s0 = s_ns[n]
            a0 = a_ns[n]
            s1 = s_ns[n + 1]

            f0 = model.f(s0, a0)
            sbar_n = s_des_traj[:, n]
            ubar_n = a_guess_traj[:, n]

            # add stage cost to objective
            cost += stage_cost(s0, sbar_n, a0, ubar_n)

            # add continuity constraint
            constraints += [s1 - f0]

        # add terminal cost
        cost += terminal_cost(s1, sbar_n)

        # continuity constraints and bounds on u
        constraints = vertcat(*constraints, a_traj)

        # set upper and lower bounds
        # zeros for continuity constraints, a_max/a_min from control bounds
        self.ubg = vertcat(np.zeros((ns * (N + 1))), repmat(a_max, N, 1))
        self.lbg = vertcat(np.zeros((ns * (N + 1))), repmat(a_min, N, 1))

        self.ocp = {'f': cost, 'x': vertcat(s_traj, a_traj), 'g': constraints, 'p': s0_bar}
        self.solver = nlpsol('solver', 'ipopt', self.ocp)

        # initialize current solution guess
        self.s_current = np.zeros((ns * (N + 1), 1))
        self.a_current = a_guess_traj.flatten()
        # self.a_current = np.zeros((na * N, 1))

    def solve(self, s0, s_traj_init=None, a_traj_init=None):

        if s_traj_init is not None:
            self.s_current = s_traj_init

        if a_traj_init is not None:
            self.a_current = a_traj_init

        # solve the NLP
        sol = self.solver(x0=vertcat(self.s_current, self.a_current), lbg=self.lbg, ubg=self.ubg, p=s0)

        w_opt = sol['x'].full()

        s_opt = w_opt[:(self.N + 1) * self.model.ns]
        a_opt = w_opt[(self.N + 1) * self.model.ns:]

        cost = sol['f'].full()

        self.s_current = s_opt
        self.a_current = a_opt

        a0_opt = a_opt[:self.model.na]
        return a0_opt

    def reset(self):
        self.s_current = np.zeros((self.model.ns * (self.N + 1), 1))
        self.a_current = np.zeros((self.model.na * self.N, 1))

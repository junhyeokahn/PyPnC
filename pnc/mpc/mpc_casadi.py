from pnc.mpc import ocp_solver


class MPCCasadi:
    def __init__(self, model, mpc_cost, u_max, u_min, x_des_traj, u_guess_traj,
                N_horizon=80, mpc_hold=5):
        self.N_horizon = N_horizon  # horizon in seconds is: N_horizon * dt
        self.mpc_hold = mpc_hold       # dt's for which the previous control input is held

        self.model = model
        self.stage_cost = mpc_cost.stage_cost
        self.terminal_cost = mpc_cost.terminal_cost

        self.u_max = u_max
        self.u_min = u_min

        self.x_des_traj = x_des_traj
        self.u_des_traj = u_guess_traj

        self.ocp_solver = ocp_solver.OCPsolver(self.model, self.stage_cost, self.terminal_cost,
                                        self.N_horizon, u_max, u_min, x_des_traj, u_guess_traj)

    def solve(self, s0, s_traj_init=None, a_traj_init=None):
        return self.ocp_solver.solve(s0, s_traj_init, a_traj_init)


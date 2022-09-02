from casadi import *
from collections import OrderedDict


class MPCQuadraticCost:
    def __init__(self, model, Q, R, P):
        x_bar = SX.sym('x_bar', model.ns, 1)
        u_bar = SX.sym('u_bar', model.na, 1)
        stage_cost_expr = (model.s_expr - x_bar).T @ Q @ (model.s_expr - x_bar) + \
                          (model.a_expr - u_bar).T @ R @ (model.a_expr - u_bar)
        terminal_cost_expr = (model.s_expr - x_bar).T @ P @ (model.s_expr - x_bar)

        self.stage_cost = Function('stage_cost', [model.s_expr, x_bar, model.a_expr, u_bar], [stage_cost_expr])
        self.terminal_cost = Function('terminal_cost', [model.s_expr, x_bar], [terminal_cost_expr])

    def get_cost(self):
        cost_terms = OrderedDict()
        cost_terms["stage_cost"] = self.stage_cost
        cost_terms["terminal_cost"] = self.terminal_cost
        return cost_terms

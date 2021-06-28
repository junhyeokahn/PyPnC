import numpy as np
from qpsolvers import solve_qp
from ruamel.yaml import YAML

with open('/home/junhyeok/Repository/PnC/ExperimentData/py.yaml', 'r') as f:
    yml = YAML().load(f)
    cost_mat = np.array(yml["cost_mat"])
    cost_vec = np.array(yml["cost_vec"])
    eq_mat = np.array(yml["eq_mat"])
    eq_vec = np.array(yml["eq_vec"])
    ineq_mat = np.array(yml["ineq_mat"])
    ineq_vec = np.array(yml["ineq_vec"])

# cost_mat = np.array([[4., -2.], [-2., 4.]])
# cost_vec = np.array([6., 0.])
# eq_mat = np.array([[1., 1.]])
# eq_vec = np.array([3.])
# ineq_mat = -np.array([[1., 0.], [0., 1.], [1., 1.]])
# ineq_vec = -np.array([0., 0., 2.])

sol = solve_qp(cost_mat,
               cost_vec,
               ineq_mat,
               ineq_vec,
               eq_mat,
               eq_vec,
               solver="quadprog",
               verbose=True)

print(sol)

from casadi import Function

import numpy as np

from scipy.linalg import solve_discrete_are

def integrate_RK4(s_expr, a_expr, sdot_expr, dt, N_steps=1):
    h = dt / N_steps

    s_end = s_expr

    xdot_fun = Function('xdot', [s_expr, a_expr], [sdot_expr])

    for _ in range(N_steps):
        k_1 = xdot_fun(s_end, a_expr)
        k_2 = xdot_fun(s_end + 0.5 * h * k_1, a_expr)
        k_3 = xdot_fun(s_end + 0.5 * h * k_2, a_expr)
        k_4 = xdot_fun(s_end + k_3 * h, a_expr)

        s_end = s_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    F_expr = s_end

    return F_expr


def get_LQR_gain(A, B, Q, R):

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve((B.T @ P @ B + R), B.T @ P @ A)
    return (K, P)


def get_solve_info(output_file):
    with open(output_file, 'r') as f:
        for line in f:
            if line.startswith('Total CPU secs in IPOPT (w/o function evaluations)   ='):
                tokens = line.split()
                solve_times = 1000*float(tokens[9])    # in ms

            elif line.startswith('Number of Iterations....:'):
                tokens = line.split()
                number_of_iterations = float(tokens[3])

    return solve_times, number_of_iterations
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

from util import interpolation

import matplotlib.pyplot as plt

if __name__ == "__main__":

    p0 = 0.02
    v0 = 0.
    p1 = 0.07
    v1 = 0.
    test_curve = interpolation.HermiteCurve(p0, v0, p1, v1)

    s = 0
    N = 100
    s_list, pos_list, vel_list = [], [], []
    for s_id in range(N):
        s = s_id * 1 / N
        pos = test_curve.evaluate(s)
        vel = test_curve.evaluate_first_derivative(s)
        s_list.append(s)
        pos_list.append(pos)
        vel_list.append(vel)
        print("s: ", s, " pos: ", pos, " vel: ", vel)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(s_list, pos_list)
    axes[0].grid(True)
    axes[1].plot(s_list, vel_list)
    axes[1].grid(True)

    plt.show()

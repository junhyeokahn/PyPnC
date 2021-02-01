import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

file_path = os.getcwd() + "/build/dubin.txt"
data = np.genfromtxt(file_path, delimiter=',', dtype=float)

fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1])
ax.grid(True)
ax.axis('equal')

plt.show()

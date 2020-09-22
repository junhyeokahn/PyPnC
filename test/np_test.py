import numpy as np
from scipy.linalg import block_diag

a1 = np.ones((2,3))
a2 = np.ones((5,2))
a3 = np.ones((3,3))
print(block_diag(*[a1, a2, a3]))
__import__('ipdb').set_trace()

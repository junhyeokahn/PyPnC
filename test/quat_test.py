import numpy as np
from scipy.spatial.transform import Rotation as R
import dartpy as dart

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

r0 = R.from_quat([0., 0., 0., 1.])
r1 = R.from_quat([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
r2 = R.from_quat([0., np.sin(np.pi/4), 0, np.cos(np.pi/4)])

r0_ = dart.math.Quaternion()
r0_.set_wxyz(1, 0, 0, 0)
r1_ = dart.math.Quaternion()
r1_.set_wxyz(np.cos(np.pi/4), 0, 0, np.sin(np.pi/4))
r2_ = dart.math.Quaternion()
r2_.set_wxyz(np.cos(np.pi/4), 0, np.sin(np.pi/4), 0)

r0__ = [1, 0, 0, 0]
r1__ = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
r2__ = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]

print("r1*r2: ", (r1*r2).as_quat())
print("quaternion_multiply(r1__, r2__): ", quaternion_multiply(r1__, r2__))
print("r1_.multiply(r2_)", r1_.multiply(r2_))
__import__('ipdb').set_trace()

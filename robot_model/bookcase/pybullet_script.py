import pybullet as p
import time

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.loadURDF("bookcase.urdf", useFixedBase = 1)
p.loadURDF("red_can.urdf", useFixedBase = 0, basePosition = [0.3, 0, 0.91])
p.loadURDF("green_can.urdf", useFixedBase = 0, basePosition = [-0.2, 0, 1.32])
p.loadURDF("blue_can.urdf", useFixedBase = 0, basePosition = [0, 0, 1.70])


# for cans toggle fixed base to 0
# use base position variables to place bookcase and can

time.sleep(100)

p.disconnect()
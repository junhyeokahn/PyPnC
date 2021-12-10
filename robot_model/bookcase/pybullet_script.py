import pybullet as p
import time

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

p.loadURDF("../robot_model/ground/plane.urdf", [0, 0, 0])
p.loadURDF("bookshelf.urdf", useFixedBase = 1, basePosition = [0, 0, 0.025])
p.loadURDF("red_can.urdf", useFixedBase = 0, basePosition = [0.75, 0, 0.95])
p.loadURDF("green_can.urdf", useFixedBase = 0, basePosition = [-0.7, 0, 1.25])
p.loadURDF("blue_can.urdf", useFixedBase = 0, basePosition = [0, 0, 0.65])


# for cans toggle fixed base to 0
# use base position variables to place bookcase and can

while (1):
	p.stepSimulation();
	time.sleep(0.01);

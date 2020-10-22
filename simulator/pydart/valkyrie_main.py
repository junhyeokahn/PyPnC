import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import numpy as np
import dartpy as dart

from config.valkyrie_config import SimConfig
from pnc.valkyrie_pnc.valkyrie_interface import ValkyrieInterface
from scipy.spatial.transform import Rotation as R
from util import util


class ValkyrieWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, robot):
        super(ValkyrieWorldNode, self).__init__(world)
        self.robot = robot
        self.interface = ValkyrieInterface()

        floating_joint_list = [
            "basePosX", "basePosY", "basePosZ", "baseRotZ", "baseRotY",
            "pelvis"
        ]
        self._n_virtual = 0
        self._floating_id = OrderedDict()
        self._joint_id = OrderedDict()
        self._link_id = OrderedDict()
        for i in range(self.robot.getNumJoints()):
            j = self.robot.getJoint(i)
            if j.getName() in floating_joint_list:
                self._n_virtual += j.getNumDofs()
                self._floating_id[j.getName()] = j
            elif j.getType() != "WeldJoint":
                self._joint_id[j.getName()] = j
            else:
                pass

        for i in range(self.robot.getNumBodyNodes()):
            bn = self.robot.getBodyNode(i)
            self._link_id[bn.getName()] = bn

        self._n_q = self._n_q_dot = self.robot.getNumDofs()
        self._n_a = self._n_q_dot - self._n_virtual
        self._count = 0

    def customPreStep(self):

        sensor_data = OrderedDict()

        sensor_data['base_quat'] = R.from_matrix(
            self._link_id['pelvis'].getTransform(
                dart.dynamics.Frame.World(),
                dart.dynamics.Frame.World()).rotation()).as_quat()
        sensor_data['base_pos'] = self._link_id['pelvis'].getCOM(
            dart.dynamics.Frame.World())
        sensor_data['base_lin_vel'] = self._link_id[
            'pelvis'].getCOMSpatialVelocity(dart.dynamics.Frame.World(),
                                            dart.dynamics.Frame.World())[3:6]
        sensor_data['base_ang_vel'] = self._link_id[
            'pelvis'].getCOMSpatialVelocity(dart.dynamics.Frame.World(),
                                            dart.dynamics.Frame.World())[0:3]
        sensor_data['joint_pos'] = OrderedDict()
        sensor_data['joint_vel'] = OrderedDict()
        for k, v in self._joint_id.items():
            sensor_data['joint_pos'][k] = self._joint_id[k].getPosition(0)
            sensor_data['joint_vel'][k] = self._joint_id[k].getVelocity(0)

        command = self.interface.get_command(sensor_data)
        # util.pretty_print(sensor_data['joint_pos'])
        # print(sensor_data['base_pos'])
        # print(sensor_data['base_quat'])

        forces = np.zeros(self._n_a + 6)
        forces[6:] = np.array([v for v in command['joint_trq'].values()])
        self.robot.setForces(forces)
        self._count += 1


def set_initial_config(robot):
    leftHipPitch = robot.getDof("leftHipPitch").getIndexInSkeleton()
    rightHipPitch = robot.getDof("rightHipPitch").getIndexInSkeleton()
    leftKneePitch = robot.getDof("leftKneePitch").getIndexInSkeleton()
    rightKneePitch = robot.getDof("rightKneePitch").getIndexInSkeleton()
    leftAnklePitch = robot.getDof("leftAnklePitch").getIndexInSkeleton()
    rightAnklePitch = robot.getDof("rightAnklePitch").getIndexInSkeleton()
    rightShoulderPitch = robot.getDof(
        "rightShoulderPitch").getIndexInSkeleton()
    rightShoulderRoll = robot.getDof("rightShoulderRoll").getIndexInSkeleton()
    rightElbowPitch = robot.getDof("rightElbowPitch").getIndexInSkeleton()
    rightForearmYaw = robot.getDof("rightForearmYaw").getIndexInSkeleton()
    leftShoulderPitch = robot.getDof("leftShoulderPitch").getIndexInSkeleton()
    leftShoulderRoll = robot.getDof("leftShoulderRoll").getIndexInSkeleton()
    leftElbowPitch = robot.getDof("leftElbowPitch").getIndexInSkeleton()
    leftForearmYaw = robot.getDof("leftForearmYaw").getIndexInSkeleton()
    q = robot.getPositions()
    q[2] = 2.5 - 1.365 - 0.11
    q[leftHipPitch] = -0.6
    q[rightHipPitch] = -0.6
    q[leftKneePitch] = 1.2
    q[rightKneePitch] = 1.2
    q[leftAnklePitch] = -0.6
    q[rightAnklePitch] = -0.6
    q[rightShoulderPitch] = 0.2
    q[rightShoulderRoll] = 1.1
    q[rightElbowPitch] = 0.4
    q[rightForearmYaw] = 1.5
    q[leftShoulderPitch] = 0.2
    q[leftShoulderRoll] = -1.1
    q[leftElbowPitch] = -0.4
    q[leftForearmYaw] = 1.5
    robot.setPositions(q)


def main():
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    robot = urdfParser.parseSkeleton(
        cwd + "/../PnC/RobotModel/Robot/Valkyrie/ValkyrieSim_Dart.urdf")
    set_initial_config(robot)
    ground = urdfParser.parseSkeleton(
        cwd + "/../PnC/RobotModel/Ground/ground_terrain.urdf")
    # ground = urdfParser.parseSkeleton(
    # cwd + "/robot_model/ground/plane.urdf")
    world.addSkeleton(robot)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(SimConfig.CONTROLLER_DT)

    node = ValkyrieWorldNode(world, robot)

    # Create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    viewer.setUpViewInWindow(0, 0, 2880, 1800)
    viewer.setCameraHomePosition([6., 3., 3.], [0., 0.2, 0.5], [0., 0., 1.])
    viewer.run()


if __name__ == "__main__":
    main()

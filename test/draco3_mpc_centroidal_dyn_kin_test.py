import unittest

import numpy as np

from pnc.mpc import mpc_casadi
from pnc.mpc.centroidal_dyn_kin.centroidal_dynamics_model import CentroidalDynamics
from pnc.mpc.centroidal_dyn_kin.low_dim_trajectory_generator import *
from pnc.mpc.mpc_quadratic_cost import MPCQuadraticCost
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from collections import OrderedDict

import pinocchio as pin

class TestStaticPoses(unittest.TestCase):
    def setUp(self):
        urdf_filename = "/home/carlos/git/PyPnC/robot_model/draco3/draco3_gripper_mesh_updated.urdf"
        package_dir = "/home/carlos/git/PyPnC/robot_model/draco3/"

        b_fixed_base = False
        b_print_info = False
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_filename, package_dir, pin.JointModelFreeFlyer())
        self.robot = PinocchioRobotSystem(urdf_filename, package_dir, b_fixed_base, b_print_info)

        # initial state
        robot_state = self.get_robot_state()
        self.robot_state = robot_state

        self.robot.update_system(robot_state["base_com_pos"], robot_state["base_com_quat"],
                     robot_state["base_com_lin_vel"], robot_state["base_com_ang_vel"],
                     robot_state["base_joint_pos"], robot_state["base_joint_quat"],
                     robot_state["base_joint_lin_vel"], robot_state["base_joint_ang_vel"],
                     robot_state["joint_pos"], robot_state["joint_vel"])
        self.robot._update_centroidal_quantities()

        # MPC control limits
        nu = 6
        nx = 9
        u_max = 500 * np.ones(nu)
        u_min = -u_max
        u_min[2] = 0.
        u_min[5] = 0.

        self.Q = np.diag(1e1 * np.ones(nx))
        self.Q[2, 2] = 5e4
        self.Q[5, 5] = 1e3
        self.R = np.diag(1e-1 * np.ones(nu))
        self.R[2, 2] = 1e-2
        self.R[5, 5] = 1e-2

    def get_robot_state(self):
        #TODO get from robot states
        robot_state = OrderedDict()
        robot_state["base_com_pos"] = np.array([0., 0., 0.7])
        robot_state["base_com_quat"] = np.array([0., 0., 0.7, 0.7])
        robot_state["base_com_lin_vel"] = np.zeros((3,))
        robot_state["base_com_ang_vel"] = np.zeros((3,))
        robot_state["base_joint_pos"] = np.zeros((3,))
        robot_state["base_joint_quat"] = np.array([0., 0., 0.7, 0.7])
        robot_state["base_joint_lin_vel"] = np.zeros((3,))
        robot_state["base_joint_ang_vel"] = np.zeros((3,))

        robot_state["joint_pos"] = OrderedDict()
        robot_state["joint_vel"] = OrderedDict()
        for i, j in self.robot.joint_id.items():
            robot_state["joint_pos"][i] = 0.
            robot_state["joint_vel"][i] = 0.

        return robot_state

    def test_standing(self):
        task = Task.STAND
        centroidal_model = CentroidalDynamics(self.robot)

        # Initial state
        g = 9.81
        mass = self.robot.total_mass
        com0 = self.robot_state["base_com_pos"]
        vcom0 = self.robot_state["base_com_lin_vel"]
        L0 = np.array([0., 0., 0.])
        x0 = np.array([com0[0], com0[1], com0[2], vcom0[0], vcom0[1], vcom0[2], L0[0], L0[1], L0[2]])
        u_guess = np.array([0., 0., mass * g / 2., 0., 0., mass * g / 2.])

        Q = np.diag(1e1 * np.ones(centroidal_model.ns))
        Q[2, 2] = 5e4
        Q[5, 5] = 1e3
        R = np.diag(1e-1 * np.ones(centroidal_model.na))
        R[2, 2] = 1e-2
        R[5, 5] = 1e-2
        P = 10. * Q

        mpc_cost = MPCQuadraticCost(centroidal_model, Q, R, P)

        # Generate trajectory to track
        dt = centroidal_model.dt
        N_horizon = 80      # MPC horizon
        N_steps = 1000      # simulation steps
        traj_freq = np.pi
        traj_amplitude = 0.05
        x_des_traj, u_guess_traj = get_desired_mpc_trajectory(task, x0, u_guess, centroidal_model, N_horizon, w=traj_freq,
                                                              A=traj_amplitude)
        x_des_traj_all, u_guess_traj_all = get_desired_mpc_trajectory(task, x0, u_guess, centroidal_model, N_steps + 1,
                                                                      w=traj_freq, A=traj_amplitude)
        mpc_controller = mpc_casadi.MPCCasadi(centroidal_model, mpc_cost, centroidal_model.u_max, centroidal_model.u_min,
                                              x_des_traj, u_guess_traj, N_horizon)

        #
        # Simulate
        #

        # placeholders for MPC outputs
        x_traj = np.zeros((centroidal_model.ns, N_steps + 1))  # trajectory from entire simulation
        u_traj = np.zeros((centroidal_model.na, N_steps))

        # generate noise trajectory
        w_traj = 0.0 * np.random.normal(size=(centroidal_model.ns, N_steps))

        # closed-loop simulation
        x_traj[:, 0] = x0

        n_mpc = 0  # counter for times the MPC has been run
        mpc_hold = mpc_controller.mpc_hold
        for n in range(N_steps):
            s = x_traj[:, n]  # get state at beginning of this MPC loop
            # u_guess_n = u_guess_traj[:, n]        # get state at beginning of this MPC loop

            # run MPC
            if n % mpc_hold == 0:
                if (n_mpc != 0) & (task == Task.SQUAT):
                    n_offset = (mpc_hold * (n_mpc - 1) + N_horizon)

                    # change/update desired trajectory accordingly
                    x_des_traj[:, :-mpc_hold] = x_des_traj[:, mpc_hold:]  # shift series to the left
                    x_des_traj[:, -mpc_hold:] = np.zeros((centroidal_model.ns, mpc_hold))  # add zeros on new spaces
                    for n_new in range(mpc_hold):
                        idx_offset = N_horizon - mpc_hold + n_new
                        x_des_traj[2, idx_offset] = x0[2] + traj_amplitude * np.sin(traj_freq * (n_offset + n_new) * dt)
                        x_des_traj[5, idx_offset] = x0[5] + traj_amplitude * np.cos(traj_freq * (n_offset + n_new) * dt)

                    # ocp_solver = OCPSolver.OCPsolver(model, stage_cost, terminal_cost, N_horizon, u_max, u_min,
                    #                                  x_des_traj, u_guess_traj)

                # sys.stdout = open(output_file, 'w')
                a = mpc_controller.solve(s)
                # sys.stdout.close()

                # cpu_solve_times[n], number_of_iterations[n] = get_solve_info(output_file)
                n_mpc = n_mpc + 1
            else:
                a = u_traj[:, n - 1]

            u_traj[:, n] = a.flatten()
            x_traj[:, [n + 1]] = centroidal_model.simulate(s, a.flatten(), w0=w_traj[:, n])

        epsilon = 1.e-6
        final_com_x = x_traj[0, -1]
        final_com_y = x_traj[1, -1]
        final_com_z = x_traj[2, -1]
        final_com_v = np.linalg.norm(x_traj[3:, -1])
        self.assertEqual(np.abs(final_com_x - com0[0]) < epsilon, True) # final x-com close to initial
        self.assertEqual(np.abs(final_com_y - com0[1]) < epsilon, True) # final y-com close to initial
        self.assertEqual(np.abs(final_com_z - com0[2]) < epsilon, True) # final z-com close to initial
        self.assertEqual(np.abs(final_com_v) < epsilon, True)           # final com velocity close to zero


if __name__ == '__main__':
    unittest.main()

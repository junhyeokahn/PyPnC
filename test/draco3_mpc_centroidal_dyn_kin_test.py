import unittest

import numpy as np

from pnc.mpc import mpc_casadi
from pnc.mpc.centroidal_dyn_kin.centroidal_dynamics_model import CentroidalDynamics
from pnc.mpc.centroidal_dyn_kin.cmm_joint_dynamics_model import CMMJointDynamics
from pnc.mpc.centroidal_dyn_kin.low_dim_trajectory_generator import *
from pnc.mpc.mpc_quadratic_cost import MPCQuadraticCost
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from collections import OrderedDict
from util.util import quat_to_euler

from pinocchio.visualize import MeshcatVisualizer

import pinocchio as pin
import matplotlib.pyplot as plt
import sys

b_show_plots = False
b_visualize = False


class TestStaticPoses(unittest.TestCase):
    def setUp(self):
        urdf_filename = "/home/carlos/git/PyPnC/robot_model/draco3/draco3_gripper_mesh_updated.urdf"
        package_dir = "/home/carlos/git/PyPnC/robot_model/draco3/"

        b_fixed_base = False
        b_print_info = False
        if b_visualize:
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                urdf_filename, package_dir, pin.JointModelFreeFlyer())
            viz = MeshcatVisualizer(model, collision_model, visual_model)
            try:
                viz.initViewer(open=True)
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            viz.loadViewerModel()

        self.robot = PinocchioRobotSystem(urdf_filename, package_dir, b_fixed_base, b_print_info)

        # initial state
        non_zero_joints = OrderedDict()
        self.set_non_zero_joints(non_zero_joints)
        robot_state = self.initialize_robot_state(non_zero_joints)
        self.robot_state = robot_state

        self.robot.update_system(robot_state["base_com_pos"], robot_state["base_com_quat"],
                                 robot_state["base_com_lin_vel"], robot_state["base_com_ang_vel"],
                                 robot_state["base_joint_pos"], robot_state["base_joint_quat"],
                                 robot_state["base_joint_lin_vel"], robot_state["base_joint_ang_vel"],
                                 robot_state["joint_pos"], robot_state["joint_vel"], b_cent=True)

        if b_visualize:
            viz.display(self.robot.get_q())

    def setup_centroidal_mpc(self):
        # MPC control limits
        nu = 6
        nx = 9

        Q = np.diag(1e1 * np.ones(nx))
        Q[0, 0] = 5e4
        Q[1, 1] = 5e4
        Q[2, 2] = 5e4
        Q[3, 3] = 1e3
        Q[4, 4] = 1e3
        Q[5, 5] = 1e3
        R = np.diag(1e-1 * np.ones(nu))
        R[2, 2] = 1e-2
        R[5, 5] = 1e-2
        P = 10. * Q

        lfoot_pos = self.robot.get_link_iso("l_foot_contact")[0:-1, -1]
        rfoot_pos = self.robot.get_link_iso("r_foot_contact")[0:-1, -1]
        self.model = CentroidalDynamics(self.robot, lfoot_pos, rfoot_pos)

        # Initial state
        g = 9.81
        mass = self.robot.total_mass
        com0 = self.robot.get_com_pos()
        vcom0 = self.robot_state["base_com_lin_vel"]
        L0 = np.array([0., 0., 0.])
        self.x0 = np.array([com0[0], com0[1], com0[2], vcom0[0], vcom0[1], vcom0[2], L0[0], L0[1], L0[2]])
        self.u_guess = np.array([0., 0., mass * g / 2., 0., 0., mass * g / 2.])

        self.mpc_cost = MPCQuadraticCost(self.model, Q, R, P)

    def setup_cmm_mpc(self):
        # MPC control limits
        nc = 2
        nu = 6 * nc + self.robot.n_a
        nx = 12 + self.robot.n_a

        Q = np.diag(1e1 * np.ones(nx))
        # Q[0:6, 0:6] = 1e2 * np.identity(6)     # centroidal momentum
        # Q[2, 2] = 1e2
        Q[6:9, 6:9] = 1e3 * np.identity(3)     # base position
        Q[8, 8] = 5e4
        Q[9:12, 9:12] = 1e3 * np.identity(3)   # base orientation
        # Q[12:, 12:] = 1e2 * np.identity(self.robot.n_a)   # joints
        R = np.diag(1e-1 * np.ones(nu))
        R[2, 2] = 1e-1
        R[8, 8] = 1e-1
        R[12+7:12+17, 12+7:12+17] = 1e2 * np.identity(10)       # penalize differently upper body motions
        R[12+25:12+34, 12+25:12+34] = 1e2 * np.identity(9)      # penalize differently upper body motions
        P = 10. * Q

        com0 = self.robot.get_com_pos()
        lfoot_pos = self.robot.get_link_iso("l_foot_contact")[0:-1, -1]
        rfoot_pos = self.robot.get_link_iso("r_foot_contact")[0:-1, -1]
        lfoot_com = lfoot_pos - com0
        rfoot_com = rfoot_pos - com0
        A_cmm = self.robot.Ag
        self.model = CMMJointDynamics(self.robot, lfoot_com, rfoot_com, A_cmm)

        # Initial state
        g = 9.81
        mass = self.robot.total_mass
        vcom0 = self.robot_state["base_com_lin_vel"]
        wcom0 = self.robot_state["base_com_ang_vel"]
        h_ang_init = self.robot.hg[:3]
        h_lin_init = self.robot.hg[3:]
        q_base_pos = self.robot.get_q()[:3]
        q_base_ori = quat_to_euler(self.robot.get_q()[3:7])
        self.x0 = np.concatenate((h_lin_init, h_ang_init, q_base_pos, q_base_ori, self.robot.get_q()[7:]))
        self.u_guess = np.zeros(self.model.na)
        self.u_guess[2] = mass * g / 2.
        self.u_guess[8] = mass * g / 2.

        self.mpc_cost = MPCQuadraticCost(self.model, Q, R, P)

    def set_non_zero_joints(self, non_zero_joint_dict):
        # shoulder_z
        non_zero_joint_dict["l_shoulder_aa"] = np.pi / 6
        non_zero_joint_dict["r_shoulder_aa"] = -np.pi / 6
        # elbow_y
        non_zero_joint_dict["l_elbow_fe"] = -np.pi / 2
        non_zero_joint_dict["r_elbow_fe"] = -np.pi / 2
        # hip_y
        non_zero_joint_dict["l_hip_fe"] = -np.pi / 4
        non_zero_joint_dict["r_hip_fe"] = -np.pi / 4
        # knee
        non_zero_joint_dict["l_knee_fe_jp"] = np.pi / 4
        non_zero_joint_dict["r_knee_fe_jp"] = np.pi / 4
        non_zero_joint_dict["l_knee_fe_jd"] = np.pi / 4
        non_zero_joint_dict["r_knee_fe_jd"] = np.pi / 4
        # ankle
        non_zero_joint_dict["l_ankle_fe"] = -np.pi / 4
        non_zero_joint_dict["r_ankle_fe"] = -np.pi / 4

    def initialize_robot_state(self, non_zero_joint_list):
        robot_state = OrderedDict()
        robot_state["base_com_pos"] = np.array([0., 0., 88])
        robot_state["base_com_quat"] = np.array([0., 0., 0., 1.])
        robot_state["base_com_lin_vel"] = np.zeros((3,))
        robot_state["base_com_ang_vel"] = np.zeros((3,))
        robot_state["base_joint_pos"] = np.array([0., 0., 0.74])
        robot_state["base_joint_quat"] = np.array([0., 0., 0., 1.])
        robot_state["base_joint_lin_vel"] = np.zeros((3,))
        robot_state["base_joint_ang_vel"] = np.zeros((3,))

        robot_state["joint_pos"] = OrderedDict()
        robot_state["joint_vel"] = OrderedDict()
        for i, j in self.robot.joint_id.items():
            if i in non_zero_joint_list:
                robot_state["joint_pos"][i] = non_zero_joint_list[i]
            else:
                robot_state["joint_pos"][i] = 0.
            robot_state["joint_vel"][i] = 0.

        return robot_state

    def plot_trajectories_tuples(self, ts, s_traj, pos_labels, s_des_traj, s_offset, plot_sine):
        plt.figure(figsize=(6, 6))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts, s_traj[i + s_offset, :].T, '-', alpha=0.7)
            plt.ylabel(pos_labels[i])
            plt.grid()
            if plot_sine:
                plt.plot(ts, s_des_traj[i + s_offset, :].T, 'r--')
        plt.xlabel(r'time [s]')

    @unittest.skip("Skipping standing test with CD model. Was passing and has not been modified since then")
    def test_standing(self):
        task = Task.STAND
        self.setup_centroidal_mpc()

        # Generate trajectory to track
        dt = self.model.dt
        N_horizon = 80  # MPC horizon
        N_steps = 500  # simulation steps
        x_des_traj, u_guess_traj = get_desired_mpc_trajectory(task, self.x0, self.u_guess, self.model,
                                                              N_horizon)
        x_des_traj_all, u_guess_traj_all = get_desired_mpc_trajectory(task, self.x0, self.u_guess,
                                                                      self.model, N_steps + 1)
        mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost, self.model.u_max,
                                              self.model.u_min,
                                              x_des_traj, u_guess_traj, N_horizon)

        # placeholders for MPC outputs
        x_traj = np.zeros((self.model.ns, N_steps + 1))  # trajectory from entire simulation
        u_traj = np.zeros((self.model.na, N_steps))

        # generate noise trajectory
        w_traj = 0.0 * np.random.normal(size=(self.model.ns, N_steps))

        # closed-loop simulation
        x_traj[:, 0] = self.x0

        n_mpc = 0  # counter for times the MPC has been run
        mpc_hold = mpc_controller.mpc_hold
        for n in range(N_steps):
            s = x_traj[:, n]  # get state at beginning of this MPC loop

            # run MPC
            if n % mpc_hold == 0:
                a = mpc_controller.solve(s)
                n_mpc = n_mpc + 1
            else:
                a = u_traj[:, n - 1]

            u_traj[:, n] = a.flatten()
            x_traj[:, [n + 1]] = self.model.simulate(s, a.flatten(), w0=w_traj[:, n])

        epsilon = 1.e-3
        final_com_x = x_traj[0, -1]
        final_com_y = x_traj[1, -1]
        final_com_z = x_traj[2, -1]
        final_com_v = np.linalg.norm(x_traj[3:6, -1])
        if b_show_plots:
            ts = dt * np.arange(0, N_steps + 1)
            pos_labels = ["c_x", "c_y", "c_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all[:, :-1], 0, True)

            pos_labels = ["v_x", "v_y", "v_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["L_x", "L_y", "L_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 6, False)

            pos_labels = ["lf_F_x", "lf_F_y", "lf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 0, False)

            pos_labels = ["rf_F_x", "rf_F_y", "rf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 3, False)
            plt.draw()
            plt.show()
        self.assertAlmostEqual(final_com_x, self.x0[0], delta=epsilon)  # final x-com close to initial
        self.assertAlmostEqual(final_com_y, self.x0[1], delta=epsilon)  # final y-com close to initial
        self.assertAlmostEqual(final_com_z, self.x0[2], delta=epsilon)  # final z-com close to initial
        self.assertAlmostEqual(final_com_v, 0., delta=epsilon)          # final com velocity close to zero

    @unittest.skip("Skipping squatting test with CD model. Was passing and has not been modified since then")
    def test_squatting(self):
        task = Task.SQUAT
        self.setup_centroidal_mpc()

        # Generate trajectory to track
        dt = self.model.dt
        N_horizon = 80  # MPC horizon
        N_steps = 500  # simulation steps
        traj_freq = np.pi
        traj_amplitude = 0.05
        x_des_traj, u_guess_traj = get_desired_mpc_trajectory(task, self.x0, self.u_guess, self.model,
                                                              N_horizon, w=traj_freq,
                                                              A=traj_amplitude)
        x_des_traj_all, u_guess_traj_all = get_desired_mpc_trajectory(task, self.x0, self.u_guess,
                                                                      self.model, N_steps + 1,
                                                                      w=traj_freq, A=traj_amplitude)
        mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost, self.model.u_max,
                                              self.model.u_min,
                                              x_des_traj, u_guess_traj, N_horizon)

        #
        # Simulate
        #

        # placeholders for MPC outputs
        x_traj = np.zeros((self.model.ns, N_steps + 1))  # trajectory from entire simulation
        u_traj = np.zeros((self.model.na, N_steps))

        # generate noise trajectory
        w_traj = 0.0 * np.random.normal(size=(self.model.ns, N_steps))

        # closed-loop simulation
        x_traj[:, 0] = self.x0

        n_mpc = 0  # counter for times the MPC has been run
        mpc_hold = mpc_controller.mpc_hold
        for n in range(N_steps):
            s = x_traj[:, n]  # get state at beginning of this MPC loop
            # u_guess_n = u_guess_traj[:, n]        # get state at beginning of this MPC loop

            # run MPC
            if n % mpc_hold == 0:
                if n_mpc != 0:
                    n_offset = (mpc_hold * (n_mpc - 1) + N_horizon)

                    # change/update desired trajectory accordingly
                    x_des_traj[:, :-mpc_hold] = x_des_traj[:, mpc_hold:]  # shift series to the left
                    x_des_traj[:, -mpc_hold:] = np.zeros(
                        (self.model.ns, mpc_hold))  # add zeros on new spaces
                    for n_new in range(mpc_hold):
                        idx_offset = N_horizon - mpc_hold + n_new
                        x_des_traj[0, idx_offset] = self.x0[0]
                        x_des_traj[1, idx_offset] = self.x0[1]
                        x_des_traj[2, idx_offset] = self.x0[2] + traj_amplitude * np.sin(
                            traj_freq * (n_offset + n_new) * dt)
                        x_des_traj[5, idx_offset] = self.x0[5] + traj_amplitude * np.cos(
                            traj_freq * (n_offset + n_new) * dt)

                    mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost,
                                                          self.model.u_max,
                                                          self.model.u_min, x_des_traj, u_guess_traj)

                a = mpc_controller.solve(s)
                n_mpc = n_mpc + 1
            else:
                a = u_traj[:, n - 1]

            u_traj[:, n] = a.flatten()
            x_traj[:, [n + 1]] = self.model.simulate(s, a.flatten(), w0=w_traj[:, n])

        epsilon = 1.e-3
        final_com_x = x_traj[0, -1]
        final_com_y = x_traj[1, -1]
        final_com_z = x_traj[2, -1]
        final_com_v = np.linalg.norm(x_traj[3:5, -1])
        if b_show_plots:
            ts = dt * np.arange(0, N_steps + 1)
            pos_labels = ["c_x", "c_y", "c_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all[:, :-1], 0, True)

            pos_labels = ["v_x", "v_y", "v_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["L_x", "L_y", "L_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 6, False)

            pos_labels = ["lf_F_x", "lf_F_y", "lf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 0, False)

            pos_labels = ["rf_F_x", "rf_F_y", "rf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 3, False)
            plt.draw()
            plt.show()
        self.assertAlmostEqual(final_com_x, x_des_traj_all[0, N_steps + 1], delta=epsilon)  # final x-com close to desired
        self.assertAlmostEqual(final_com_y, x_des_traj_all[1, N_steps + 1], delta=epsilon)  # final y-com close to desired
        self.assertAlmostEqual(final_com_z, x_des_traj_all[2, N_steps + 1], delta=2. * epsilon)  # final z-com close to desired
        self.assertAlmostEqual(final_com_v, 0., delta=epsilon)  # final x,y-com velocity close to zero

    @unittest.skip("Skipping standing test with CMM model. Was passing and has not been modified since then")
    def test_standing_cmm(self):
        task = Task.STAND
        self.setup_cmm_mpc()

        # Generate trajectory to track
        dt = self.model.dt
        N_horizon = 80  # MPC horizon
        N_steps = 200  # simulation steps
        x_des_traj, u_guess_traj = get_desired_mpc_trajectory(task, self.x0, self.u_guess, self.model, N_horizon)
        x_des_traj_all, u_guess_traj_all = get_desired_mpc_trajectory(task, self.x0, self.u_guess,
                                                                      self.model, N_steps + 1)
        mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost, self.model.u_max,
                                              self.model.u_min,
                                              x_des_traj, u_guess_traj, N_horizon)

        # placeholders for MPC outputs
        x_traj = np.zeros((self.model.ns, N_steps + 1))  # trajectory from entire simulation
        u_traj = np.zeros((self.model.na, N_steps))

        # generate noise trajectory
        w_traj = 0.0 * np.random.normal(size=(self.model.ns, N_steps))

        # closed-loop simulation
        x_traj[:, 0] = self.x0

        n_mpc = 0  # counter for times the MPC has been run
        mpc_hold = mpc_controller.mpc_hold
        for n in range(N_steps):
            s = x_traj[:, n]  # get state at beginning of this MPC loop

            # run MPC
            if n % mpc_hold == 0:
                a = mpc_controller.solve(s)
                n_mpc = n_mpc + 1
            else:
                a = u_traj[:, n - 1]

            u_traj[:, n] = a.flatten()
            x_traj[:, [n + 1]] = self.model.simulate(s, a.flatten(), w0=w_traj[:, n])

        if b_show_plots:
            ts = dt * np.arange(0, N_steps + 1)
            pos_labels = ["l_x", "l_y", "l_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 0, False)

            pos_labels = ["k_x", "k_y", "k_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["base_x", "base_y", "base_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 6, False)

            pos_labels = ["lf_F_x", "lf_F_y", "lf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 0, False)

            pos_labels = ["lf_M_x", "lf_M_y", "lf_M_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["rf_F_x", "rf_F_y", "rf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 6, False)

            pos_labels = ["rf_M_x", "rf_M_y", "rf_M_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 9, False)
            plt.figure(figsize=(6, 6))
            plt.plot(ts[:-1], u_traj[12:, :].T, '-', alpha=0.7)
            plt.grid()
            plt.ylabel("joint_vel")
            plt.draw()
            plt.show()
        epsilon = 1.e-3
        final_base_x = x_traj[6, -1]
        final_base_y = x_traj[7, -1]
        final_base_z = x_traj[8, -1]
        final_com_v = np.linalg.norm(x_traj[:6, -1])
        self.assertAlmostEqual(final_base_x, x_des_traj_all[6, N_steps + 1], delta=epsilon)  # final x-com close to desired
        self.assertAlmostEqual(final_base_y, x_des_traj_all[7, N_steps + 1], delta=epsilon)  # final y-com close to desired
        self.assertAlmostEqual(final_base_z, x_des_traj_all[8, N_steps + 1], delta=epsilon)  # final z-com close to desired
        self.assertAlmostEqual(final_com_v, 0., delta=epsilon)  # final x,y-com velocity close to zero

    @unittest.skip("Skipping squatting test with CMM model. Was passing and has not been modified since then")
    def test_squatting_cmm(self):
        task = Task.SQUAT
        self.setup_cmm_mpc()

        # Generate trajectory to track
        dt = self.model.dt
        N_horizon = 80  # MPC horizon
        N_steps = 100  # simulation steps
        traj_freq = np.pi
        traj_amplitude = 0.05
        x_des_traj, u_guess_traj = get_desired_mpc_trajectory(task, self.x0, self.u_guess, self.model,
                                                              N_horizon, w=traj_freq, A=traj_amplitude,
                                                              pos_idx=8, vel_idx=2)
        x_des_traj_all, u_guess_traj_all = get_desired_mpc_trajectory(task, self.x0, self.u_guess,
                                                              self.model, N_steps, w=traj_freq,
                                                              A=traj_amplitude, pos_idx=8, vel_idx=2)
        x_des_traj[2, :] = self.robot.total_mass * x_des_traj[2, :]
        x_des_traj_all[2, :] = self.robot.total_mass * x_des_traj_all[2, :]
        mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost, self.model.u_max,
                                              self.model.u_min,
                                              x_des_traj, u_guess_traj, N_horizon)

        # placeholders for MPC outputs
        x_traj = np.zeros((self.model.ns, N_steps + 1))  # trajectory from entire simulation
        u_traj = np.zeros((self.model.na, N_steps))

        # generate noise trajectory
        w_traj = 0.0 * np.random.normal(size=(self.model.ns, N_steps))

        # closed-loop simulation
        x_traj[:, 0] = self.x0

        n_mpc = 0  # counter for times the MPC has been run
        mpc_hold = mpc_controller.mpc_hold
        for n in range(N_steps):
            s = x_traj[:, n]  # get state at beginning of this MPC loop

            # run MPC
            if n % mpc_hold == 0:
                if n_mpc != 0:
                    n_offset = (mpc_hold * (n_mpc - 1) + N_horizon)

                    # change/update desired trajectory accordingly
                    x_des_traj[:, :-mpc_hold] = x_des_traj[:, mpc_hold:]  # shift series to the left
                    for n_new in range(mpc_hold+1):
                        idx_offset = N_horizon - mpc_hold + n_new
                        x_des_traj[8, idx_offset] = self.x0[8] + traj_amplitude * np.sin(
                            traj_freq * (n_offset + n_new) * dt)
                        x_des_traj[2, idx_offset] = self.x0[2] + self.robot.total_mass * traj_amplitude * np.cos(
                            traj_freq * (n_offset + n_new) * dt)

                        mpc_controller = mpc_casadi.MPCCasadi(self.model, self.mpc_cost,
                                                              self.model.u_max,
                                                              self.model.u_min, x_des_traj, u_guess_traj)

                a = mpc_controller.solve(s)
                n_mpc = n_mpc + 1
            else:
                a = u_traj[:, n - 1]

            u_traj[:, n] = a.flatten()
            x_traj[:, [n + 1]] = self.model.simulate(s, a.flatten(), w0=w_traj[:, n])

        if b_show_plots:
            ts = dt * np.arange(0, N_steps + 1)
            pos_labels = ["l_x", "l_y", "l_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 0, True)

            pos_labels = ["k_x", "k_y", "k_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["base_x", "base_y", "base_z"]
            self.plot_trajectories_tuples(ts, x_traj, pos_labels, x_des_traj_all, 6, True)

            pos_labels = ["lf_F_x", "lf_F_y", "lf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 0, False)

            pos_labels = ["lf_M_x", "lf_M_y", "lf_M_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 3, False)

            pos_labels = ["rf_F_x", "rf_F_y", "rf_F_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 6, False)

            pos_labels = ["rf_M_x", "rf_M_y", "rf_M_z"]
            self.plot_trajectories_tuples(ts[:-1], u_traj, pos_labels, x_des_traj_all, 9, False)
            plt.figure(figsize=(6, 6))
            plt.plot(ts[:-1], u_traj[12:, :].T, '-', alpha=0.7)
            plt.grid()
            plt.draw()
            plt.show()
        epsilon = 1.e-3
        final_base_x = x_traj[6, -1]
        final_base_y = x_traj[7, -1]
        final_base_z = x_traj[8, -1]
        final_h_com_lin = np.linalg.norm(x_traj[:3, -1] - x_des_traj_all[:3, -1]) / self.robot.total_mass
        final_h_com_ang = np.linalg.norm(x_traj[3:6, -1] - x_des_traj_all[3:6, -1]) / self.robot.total_mass
        final_joint_pos = np.linalg.norm(x_traj[12:, -1] - x_des_traj_all[12:, -1])
        final_upperbody_ljoints_vel = np.linalg.norm(u_traj[12 + 7:12 + 17, :])
        final_upperbody_rjoints_vel = np.linalg.norm(u_traj[12 + 25:12 + 34, :])
        self.assertAlmostEqual(final_base_x, x_des_traj_all[6, N_steps], delta=epsilon)  # final x-base close to desired
        self.assertAlmostEqual(final_base_y, x_des_traj_all[7, N_steps], delta=epsilon)  # final y-base close to desired
        self.assertAlmostEqual(final_base_z, x_des_traj_all[8, N_steps], delta=epsilon)  # final z-base close to desired
        self.assertAlmostEqual(final_h_com_lin, 0., delta=0.1)  # final linear momentum close to zero
        self.assertAlmostEqual(final_h_com_ang, 0., delta=0.01)  # final angular momentum close to zero
        self.assertAlmostEqual(final_joint_pos, 0., delta=0.1)  # final angular momentum close to zero
        self.assertAlmostEqual(final_upperbody_ljoints_vel, 0., delta=0.5)  # final upper body velocities close to zero
        self.assertAlmostEqual(final_upperbody_rjoints_vel, 0., delta=0.5)  # final upper body velocities close to zero

if __name__ == '__main__':
    unittest.main()

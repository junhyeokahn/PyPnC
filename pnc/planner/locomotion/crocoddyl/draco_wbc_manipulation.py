import os
import sys
import crocoddyl
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
import numpy as np
import pinocchio as pin

import plot.meshcat_utils as vis_tools
import knot_generator as knot
import util.util

cwd = os.getcwd()
sys.path.append(cwd)

B_SHOW_JOINT_PLOTS = False
B_SHOW_GRF_PLOTS = True

def createDoubleSupportActionModel(lh_target=None, rh_target=None, N_horizon=1):
    # Creating a double-support contact (feet support)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    lf_contact = crocoddyl.ContactModel6D(
        state,
        lf_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    rf_contact = crocoddyl.ContactModel6D(
        state,
        rf_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    contacts.addContact("lf_contact", lf_contact)
    contacts.addContact("rf_contact", rf_contact)
    contact_data = contacts.createData(rob_data)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Adding the hand-placement cost
    if lh_target is not None:
        w_lhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        lh_Mref = pin.SE3(np.eye(3), lh_target)
        activation_lhand = crocoddyl.ActivationModelWeightedQuad(w_lhand**2)
        lh_cost = crocoddyl.CostModelResidual(
            state,
            activation_lhand,
            crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
        )
        costs.addCost("lh_goal", lh_cost, 1e2)

    if rh_target is not None:
        w_rhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        rh_Mref = pin.SE3(np.eye(3), rh_target)
        activation_rhand = crocoddyl.ActivationModelWeightedQuad(w_rhand**2)
        rh_cost = crocoddyl.CostModelResidual(
            state,
            activation_rhand,
            crocoddyl.ResidualModelFramePlacement(state, rh_id, rh_Mref, actuation.nu),
        )
        costs.addCost("rh_goal", rh_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-4)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1.0)

    # Adding the friction cone penalization
    nsurf, mu = np.array([0, 0, 1]), 0.7
    surf_rotation = np.identity(3)
    cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, False)
    activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    lf_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, lf_id, cone, actuation.nu),
    )
    rf_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, rf_id, cone, actuation.nu),
    )
    costs.addCost("lf_friction", lf_friction, 1e1)
    costs.addCost("rf_friction", rf_friction, 1e1)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
    )
    return dmodel


def createSingleSupportHandActionModel(base_target, lfoot_target=None, rfoot_target=None,
               lhand_target=None, lh_contact=False, rhand_target=None, rh_contact=False):
    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet /hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    floor_rotation, mu = np.eye(3), 0.7
    floor_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, False)
    floor_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(floor_cone.lb, floor_cone.ub)
    )

    # if foot is not moving (e.g., tracking some trajectory), set it as in contact
    if rfoot_target is None:
        rf_contact = crocoddyl.ContactModel6D(
            state,
            rf_id,
            pin.SE3.Identity(),
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("rf_contact", rf_contact)

        rf_friction = crocoddyl.CostModelResidual(
            state,
            floor_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, rf_id, floor_cone, actuation.nu),
        )
        costs.addCost("rf_friction", rf_friction, 1e1)

    # if foot is not moving (e.g., tracking some trajectory), set it as in contact
    if lfoot_target is None:
        lf_contact = crocoddyl.ContactModel6D(
            state,
            lf_id,
            pin.SE3.Identity(),
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("lf_contact", lf_contact)

        lf_friction = crocoddyl.CostModelResidual(
            state,
            floor_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, lf_id, floor_cone, actuation.nu),
        )
        costs.addCost("lf_friction", lf_friction, 1e1)

    # hand is in contact if directly specified
    if lh_contact is True:
        lh_contact = crocoddyl.ContactModel6D(
            state,
            lh_id,
            pin.SE3.Identity(),
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("lh_contact", lh_contact)

        # Adding the friction cone penalization
        nsurf = np.array([0, 0, 1])
        surf_rotation = util.util.euler_to_rot([0., -np.pi / 2., 0.])
        wall_cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, False)
        wall_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wall_cone.lb, wall_cone.ub)
        )
        lh_friction = crocoddyl.CostModelResidual(
            state,
            wall_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, lh_id, wall_cone, actuation.nu),
        )
        costs.addCost("lh_friction", lh_friction, 1e1)

    # hand is in contact if directly specified
    if rh_contact is True:
        rh_contact = crocoddyl.ContactModel6D(
            state,
            rh_id,
            pin.SE3.Identity(),
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("rh_contact", rh_contact)

        # Adding the friction cone penalization
        nsurf = np.array([0, 0, 1])
        surf_rotation = util.util.euler_to_rot([0., -np.pi / 2., 0.])
        wall_cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, False)
        wall_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wall_cone.lb, wall_cone.ub)
        )
        rh_friction = crocoddyl.CostModelResidual(
            state,
            wall_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, rh_id, wall_cone, actuation.nu),
        )
        costs.addCost("rh_friction", rh_friction, 1e1)


    # Add the base-placement cost
    w_base = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
    base_Mref = pin.SE3(np.eye(3), base_target)
    activation_base = crocoddyl.ActivationModelWeightedQuad(w_base**2)
    base_cost = crocoddyl.CostModelResidual(
        state,
        activation_base,
        crocoddyl.ResidualModelFramePlacement(state, base_id, base_Mref, actuation.nu),
    )
    costs.addCost("base_goal", base_cost, 1e2)

    # Add the foot-placement cost
    if lfoot_target is not None:
        w_lfoot = np.array([1.] * 3 + [0.000001] * 3)        # (lin, ang)
        lf_Mref = pin.SE3(np.eye(3), lfoot_target)
        activation_lfoot = crocoddyl.ActivationModelWeightedQuad(w_lfoot**2)
        lf_cost = crocoddyl.CostModelResidual(
            state,
            activation_lfoot,
            crocoddyl.ResidualModelFramePlacement(state, lf_id, lf_Mref, actuation.nu),
        )
        costs.addCost("lf_goal", lf_cost, 1e2)

    if rfoot_target is not None:
        w_rfoot = np.array([1.] * 3 + [0.000001] * 3)        # (lin, ang)
        rf_Mref = pin.SE3(np.eye(3), rfoot_target)
        activation_rfoot = crocoddyl.ActivationModelWeightedQuad(w_rfoot**2)
        rf_cost = crocoddyl.CostModelResidual(
            state,
            activation_rfoot,
            crocoddyl.ResidualModelFramePlacement(state, rf_id, rf_Mref, actuation.nu),
        )
        costs.addCost("rf_goal", rf_cost, 1e2)

    # Adding the hand-placement cost
    if lhand_target is not None:
        w_lhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        lh_Mref = pin.SE3(np.eye(3), lhand_target)
        activation_lhand = crocoddyl.ActivationModelWeightedQuad(w_lhand**2)
        lh_cost = crocoddyl.CostModelResidual(
            state,
            activation_lhand,
            crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
        )
        costs.addCost("lh_goal", lh_cost, 1e2)

    if rhand_target is not None:
        w_rhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        rh_Mref = pin.SE3(np.eye(3), rhand_target)
        activation_rhand = crocoddyl.ActivationModelWeightedQuad(w_rhand**2)
        rh_cost = crocoddyl.CostModelResidual(
            state,
            activation_rhand,
            crocoddyl.ResidualModelFramePlacement(state, rh_id, rh_Mref, actuation.nu),
        )
        costs.addCost("rh_goal", rh_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-8)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1.0)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
    )
    return dmodel


def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]


def get_default_initial_pose():
    q0 = np.zeros(35,)
    hip_yaw_angle = 5
    q0[0] = 0.                          # l_hip_ie
    q0[1] = np.radians(hip_yaw_angle)   # l_hip_aa
    q0[2] = -np.pi / 4                  # l_hip_fe
    q0[3] = np.pi / 4                   # l_knee_fe_jp
    q0[4] = np.pi / 4                   # l_knee_fe_jd
    q0[5] = -np.pi / 4                  # l_ankle_fe
    q0[6] = np.radians(-hip_yaw_angle)  # l_ankle_ie
    q0[7] = 0.                          # l_shoulder_fe
    q0[8] = np.pi / 6                   # l_shoulder_aa
    q0[9] = 0.                          # l_shoulder_ie
    q0[10] = -np.pi / 2                 # l_elbow_fe
    q0[11] = 0.                         # l_wrist_ps
    q0[12] = 0.                         # l_wrist_pitch
    q0[13] = 0.                         # left_ezgripper_knuckle_palm_L1_1
    q0[14] = 0.                         # left_ezgripper_knuckle_L1_L2_1
    q0[15] = 0.                         # left_ezgripper_knuckle_palm_L1_2
    q0[16] = 0.                         # left_ezgripper_knuckle_L1_L2_2
    q0[17] = 0.                         # neck pitch
    q0[18] = 0.                         # r_hip_ie
    q0[19] = np.radians(-hip_yaw_angle) # r_hip_aa
    q0[20] = -np.pi / 4                 # r_hip_fe
    q0[21] = np.pi / 4                  # r_knee_fe_jp
    q0[22] = np.pi / 4                  # r_knee_fe_jd
    q0[23] = -np.pi / 4                 # r_ankle_fe
    q0[24] = np.radians(hip_yaw_angle)  # r_ankle_ie
    q0[25] = 0.                         # r_shoulder_fe
    q0[26] = -0.9                       # r_shoulder_aa
    q0[27] = 0.                         # r_shoulder_ie
    q0[28] = -np.pi / 2                 # r_elbow_fe
    q0[29] = 0.                         # r_wrist_ps
    q0[30] = 0.                         # r_wrist_pitch
    q0[31] = 0.                         # right_ezgripper_knuckle_palm_L1_1
    q0[32] = 0.                         # right_ezgripper_knuckle_L1_L2_1
    q0[33] = 0.                         # right_ezgripper_knuckle_palm_L1_2
    q0[34] = 0.                         # right_ezgripper_knuckle_L1_L2_2

    floating_base = np.array([0., 0., 0.74, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


# Load robot
urdf_file = cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf"
package_dir = cwd + "/robot_model/draco3"
rob_model, col_model, vis_model = pin.buildModelsFromUrdf(urdf_file,
                                      package_dir, pin.JointModelFreeFlyer())
rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)
# robot = PinocchioRobotSystem(urdf_file, package_dir, False, False)
q0 = get_default_initial_pose()
v0 = np.zeros(rob_model.nv)
x0 = np.concatenate([q0, v0])

# Declaring the foot and hand names
rf_name = "r_foot_contact"
lf_name = "l_foot_contact"
lh_name = "l_hand_contact"
rh_name = "r_hand_contact"
base_name = "torso_link"

# Getting the frame ids
rf_id = rob_model.getFrameId(rf_name)
lf_id = rob_model.getFrameId(lf_name)
lh_id = rob_model.getFrameId(lh_name)
rh_id = rob_model.getFrameId(rh_name)
base_id = rob_model.getFrameId(base_name)

# Define the robot's state and actuation
state = crocoddyl.StateMultibody(rob_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Update Pinocchio model
pin.forwardKinematics(rob_model, rob_data, q0)
pin.updateFramePlacements(rob_model, rob_data)

#
# Create targets
#

# Approach left hand to door frame
lhand_ini_pos = rob_data.oMf[lh_id].translation
lhand_end_pos = np.array([0.45, 0.35, 1.2])          # door frame location
lhand_waypoints = 2
lh_targets = knot.linear_knots(lhand_ini_pos, lhand_end_pos, lhand_waypoints)

# TODO add few steps of 3 contact for better stability?

# Switch to left hand - right foot contacts and move base through door
base_ini_pos = rob_data.oMf[base_id].translation + np.array([0., 0., 0.08])
base_pre_mid_pos = base_ini_pos + np.array([0.06, 0.0, 0.02])
base_post_mid_pos = base_ini_pos + np.array([0.18, 0.0, 0.02])
base_end_pos = base_ini_pos + np.array([0.25, 0.0, 0.0])    # base after passing door
base_waypoints = 6
base_into_targets = knot.linear_connection(list((base_ini_pos, base_post_mid_pos)),
                                           list((base_pre_mid_pos, base_end_pos)),
                                           [base_waypoints/2, base_waypoints/2])
# base_into_targets = knot.linear_knots(base_ini_pos, base_end_pos, base_waypoints)

# swing foot trajectory
step_length = 0.45
swing_height = 0.45
lf_ini_pos = rob_data.oMf[lf_id].translation
lf_mid_pre_door_pos = lf_ini_pos + np.array([0.25*step_length, 0.0, swing_height])    # foot on top of door
lf_mid_post_door_pos = lf_ini_pos + np.array([0.65*step_length, 0.0, swing_height])    # foot on top of door
lf_end_pos = lf_ini_pos + np.array([step_length, 0.0, 0.0])    # foot after passing door
lf_targets = knot.linear_connection(list((lf_ini_pos, lf_mid_post_door_pos)),
                                    list((lf_mid_pre_door_pos, lf_end_pos)),
                                    [base_waypoints/2, base_waypoints/2])
# lf_targets = knot.swing_knots(lf_ini_pos, lf_end_pos, base_waypoints, swing_height)

# Move left hand away from door frame (get it out of the way)
# Switch contact to left/right feet

#  Use right hand to complete step through the door
# Switch to right hand - left foot contacts and move foot through door
# Approach left hand to door frame
rhand_ini_pos = rob_data.oMf[rh_id].translation + np.array([base_end_pos[0], 0., 0.])
rhand_end_pos = np.array([0.45, -0.35, 1.2])          # door frame location
rhand_waypoints = 2
rh_targets = knot.linear_knots(rhand_ini_pos, rhand_end_pos, rhand_waypoints)

# Approach right hand to body
# rhand_ini_pos = rob_data.oMf[rh_id].translation
# rhand_end_pos = np.array([0.3, -0.20, 0.8])         # door frame location
# rh_targets = knot.linear_knots(rhand_ini_pos, rhand_end_pos, duration)


# Swing right foot to square up
# Switch to left hand - right foot contacts and move base through door
base_pre_squareup_pos = base_end_pos + np.array([0.02, 0.02, 0.0])
base_squareup_pos = base_end_pos + np.array([0.05, 0.01, 0.0])
base_post_squareup_pos = base_end_pos + np.array([0.10, 0.0, 0.0])    # final base position
base_waypoints = 6
base_outof_targets = knot.linear_connection(list((base_end_pos, base_squareup_pos)),
                               list((base_pre_squareup_pos, base_post_squareup_pos)),
                               [base_waypoints/2, base_waypoints/2])

rf_ini_pos = rob_data.oMf[rf_id].translation
rf_mid_pre_door_pos = rf_ini_pos + np.array([0.25*step_length, 0.0, swing_height])    # foot on top of door
rf_mid_post_door_pos = rf_ini_pos + np.array([0.65*step_length, 0.0, swing_height])    # foot on top of door
rf_end_pos = rf_ini_pos + np.array([step_length, 0.0, 0.0])    # foot after passing door
rf_targets = knot.linear_connection(list((rf_ini_pos, rf_mid_post_door_pos)),
                                    list((rf_mid_pre_door_pos, rf_end_pos)),
                                    [base_waypoints/2, base_waypoints/2])

#
# Solve the problem
#
DT = 2e-2

# Connecting the sequences of models
NUM_OF_CONTACT_CONFIGURATIONS = 4

# for left_t, right_t in zip(lh_targets, rh_targets):
#     dmodel = createDoubleSupportActionModel(left_t, right_t)
#     model_seqs += createSequence([dmodel], DT, N)

# Defining the problem and the solver
fddp = [None] * NUM_OF_CONTACT_CONFIGURATIONS
for i in range(NUM_OF_CONTACT_CONFIGURATIONS):
    model_seqs = []
    if i == 0:
        # Reach door with left hand
        N_lhand_to_door = 20  # knots for left hand reaching
        for lhand_t in lh_targets:
            dmodel = createDoubleSupportActionModel(lh_target=lhand_t)
            model_seqs += createSequence([dmodel], DT, N_lhand_to_door)

    elif i == 1:
        DT = 0.015
        # Using left-hand and right-foot supports, pass left-leg through door
        N_base_through_door = 30  # knots per waypoint to pass through door
        for base_t, lfoot_t in zip(base_into_targets, lf_targets):
            dmodel = createSingleSupportHandActionModel(base_t,
                            lfoot_target=lfoot_t, lhand_target=lhand_end_pos, lh_contact=True)
            model_seqs += createSequence([dmodel], DT, N_base_through_door)

    elif i == 2:
        DT = 0.02
        # Reach door with right hand
        N_rhand_to_door = 20  # knots for left hand reaching
        for rhand_t in rh_targets:
            dmodel = createDoubleSupportActionModel(rh_target=rhand_t)
            model_seqs += createSequence([dmodel], DT, N_rhand_to_door)

    elif i == 3:
        DT = 0.015
        # Using left-hand and right-foot supports, pass right-leg through door
        N_base_square_up = 30  # knots per waypoint to pass through door
        for base_t, rfoot_t in zip(base_outof_targets, rf_targets):
            dmodel = createSingleSupportHandActionModel(base_t, rfoot_target=rfoot_t,
                                lhand_target=lhand_end_pos, lh_contact=True,
                                rhand_target=rhand_end_pos, rh_contact=True)
            model_seqs += createSequence([dmodel], DT, N_base_square_up)


    problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
    fddp[i] = crocoddyl.SolverFDDP(problem)

    # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
    fddp[i].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    # Solver settings
    max_iter = 150
    fddp[i].th_stop = 1e-4

    # Set initial guess
    xs = [x0] * (fddp[i].problem.T + 1)
    us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
    print("Problem solved:", fddp[i].solve(xs, us, max_iter))
    print("Number of iterations:", fddp[i].iter)
    print("Total cost:", fddp[i].cost)
    print("Gradient norm:", fddp[i].stoppingCriteria())

    # Set final state as initial state of next phase
    x0 = fddp[i].xs[-1]

# Creating display
save_freq = 1
display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                              rob_data, vis_data, ctrl_freq=1/DT, save_freq=save_freq)
display.display_targets("lhand_target", lh_targets, [0.5, 0, 0])
display.display_targets("lfoot_target", lf_targets, [0, 0, 1])
display.display_targets("base_pass_target", base_into_targets, [0, 1, 0])
display.display_targets("rhand_target", rh_targets, [0, 0, 0.5])
display.display_targets("base_square_target", base_outof_targets, [0, 1, 0])
display.display_targets("rfoot_target", rf_targets, [1, 0, 0])
display.add_arrow("forces/l_ankle_ie", color=[1, 0, 0])
display.add_arrow("forces/r_ankle_ie", color=[0, 0, 1])
display.add_arrow("forces/l_wrist_pitch", color=[0, 1, 0])
display.add_arrow("forces/r_wrist_pitch", color=[0, 1, 0])
# display.displayForcesFromCrocoddylSolver(fddp)
display.displayFromCrocoddylSolver(fddp)

fig_idx = 1
if B_SHOW_JOINT_PLOTS:
    for it in fddp:
        log = it.getCallbacks()[0]
        crocoddyl.plotOCSolution(log.xs, log.us, figIndex=fig_idx, show=False)
        fig_idx += 1
        crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads,
                                  log.stops, log.steps, figIndex=fig_idx, show=False)
        fig_idx +=1

if B_SHOW_GRF_PLOTS:
    # Note: contact_links are l_ankle_ie, r_ankle_ie, l_wrist_pitch, r_wrist_pitch
    sim_steps_list = [len(fddp[i].xs) for i in range(len(fddp))]
    sim_steps = np.sum(sim_steps_list) + 1
    sim_time = np.zeros((sim_steps,))
    rf_lfoot, rf_rfoot, rf_lwrist, rf_rwrist = np.zeros((3, sim_steps)), \
        np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
    time_idx = 0
    for it in fddp:
        rf_list = vis_tools.get_force_trajectory_from_solver(it)
        for rf_t in rf_list:
            for contact in rf_t:
                # determine contact link
                cur_link = int(contact['key'])
                if rob_model.names[cur_link] == "l_ankle_ie":
                    rf_lfoot[:, time_idx] = contact['f'].linear
                elif rob_model.names[cur_link] == "r_ankle_ie":
                    rf_rfoot[:, time_idx] = contact['f'].linear
                elif rob_model.names[cur_link] == "l_wrist_pitch":
                    rf_lwrist[:, time_idx] = contact['f'].linear
                elif rob_model.names[cur_link] == "r_wrist_pitch":
                    rf_rwrist[:, time_idx] = contact['f'].linear
                else:
                    print("ERROR: Non-specified contact")
            dt = it.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
            sim_time[time_idx+1] = sim_time[time_idx] + dt
            time_idx += 1

    plot_vector_traj(sim_time, rf_lfoot.T, 'RF LFoot', Fxyz_labels)
    plot_vector_traj(sim_time, rf_rfoot.T, 'RF RFoot', Fxyz_labels)
    plot_vector_traj(sim_time, rf_lwrist.T, 'RF LWrist', Fxyz_labels)
    plot_vector_traj(sim_time, rf_rwrist.T, 'RF RWrist', Fxyz_labels)
plt.show()

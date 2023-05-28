import os
import sys
import crocoddyl
import numpy as np
import pinocchio as pin

import plot.meshcat_utils as vis_tools
import knot_generator as knot
import util.util
import copy

cwd = os.getcwd()
sys.path.append(cwd)

B_SHOW_PLOTS = False

def createDoubleSupportActionModel(lh_target, rh_target=None, N_horizon=1):
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

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Adding the hand-placement cost
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


def createSingleSupportHandActionModel(base_target, lfoot_target, lhand_target):
    # Creating a double-support contact (feet support)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    rf_contact = crocoddyl.ContactModel6D(
        state,
        rf_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    lh_contact = crocoddyl.ContactModel6D(
        state,
        lh_id,
        pin.SE3.Identity(),
        actuation.nu,
        np.array([0, 0]),
    )
    contacts.addContact("rf_contact", rf_contact)
    contacts.addContact("lh_contact", lh_contact)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Add the foot-placement cost
    w_lfoot = np.array([1.] * 3 + [0.000001] * 3)        # (lin, ang)
    lf_Mref = pin.SE3(np.eye(3), lfoot_target)
    activation_lfoot = crocoddyl.ActivationModelWeightedQuad(w_lfoot**2)
    lf_cost = crocoddyl.CostModelResidual(
        state,
        activation_lfoot,
        crocoddyl.ResidualModelFramePlacement(state, lf_id, lf_Mref, actuation.nu),
    )
    costs.addCost("lf_goal", lf_cost, 1e2)

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

    # Adding the hand-placement cost
    w_lhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
    lh_Mref = pin.SE3(np.eye(3), lhand_target)
    activation_lhand = crocoddyl.ActivationModelWeightedQuad(w_lhand**2)
    lh_cost = crocoddyl.CostModelResidual(
        state,
        activation_lhand,
        crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
    )
    costs.addCost("lh_goal", lh_cost, 1e2)

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

    # Adding the friction cone penalization
    nsurf, mu = np.array([0, 0, 1]), 0.7
    surf_rotation = util.util.euler_to_rot([0., -np.pi/2., 0.])
    wall_cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, False)
    wall_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(wall_cone.lb, wall_cone.ub)
    )
    lh_friction = crocoddyl.CostModelResidual(
        state,
        wall_activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, lh_id, wall_cone, actuation.nu),
    )
    floor_rotation = np.eye(3)
    floor_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, False)
    floor_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(floor_cone.lb, floor_cone.ub)
    )
    rf_friction = crocoddyl.CostModelResidual(
        state,
        floor_activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, rf_id, floor_cone, actuation.nu),
    )
    costs.addCost("lh_friction", lh_friction, 1e1)
    costs.addCost("rf_friction", rf_friction, 1e1)

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
lhand_end_pos = np.array([0.4, 0.35, 1.2])          # door frame location
lhand_waypoints = 2
lh_targets = knot.linear_knots(lhand_ini_pos, lhand_end_pos, lhand_waypoints)

# TODO add few steps of 3 contact for better stability?

# Switch to left hand - right foot contacts and move base through door
base_ini_pos = rob_data.oMf[base_id].translation + np.array([0., 0., 0.08])
base_end_pos = base_ini_pos + np.array([0.15, 0.0, 0.0])    # base after passing door
lf_ini_pos = rob_data.oMf[lf_id].translation
lf_end_pos = lf_ini_pos + np.array([0.3, 0.0, 0.0])    # base after passing door
base_waypoints = 7
base_targets = knot.linear_knots(base_ini_pos, base_end_pos, base_waypoints)
lf_targets = knot.swing_knots(lf_ini_pos, lf_end_pos, base_waypoints, swing_height=0.2)

# Move left hand away from door frame (get it out of the way)
# Switch contact to left/right feet

#  Use right hand to complete step through the door
# Switch to right hand - left foot contacts and move foot through door

# Approach right hand to body
# rhand_ini_pos = rob_data.oMf[rh_id].translation
# rhand_end_pos = np.array([0.3, -0.20, 0.8])         # door frame location
# rh_targets = knot.linear_knots(rhand_ini_pos, rhand_end_pos, duration)


#
# Solve the problem
#
DT = 2e-2

# Connecting the sequences of models
model_seqs = []
N_lhand_to_door = 20                                # knots for left hand reaching
for lhand_t in lh_targets:
    dmodel = createDoubleSupportActionModel(lhand_t)
    model_seqs += createSequence([dmodel], DT, N_lhand_to_door)


N_base_through_door = 30                           # knots to pass through door
for base_t, lfoot_t in zip(base_targets, lf_targets):
    dmodel = createSingleSupportHandActionModel(base_t, lfoot_t, lhand_end_pos)
    model_seqs += createSequence([dmodel], DT, N_base_through_door)

# for left_t, right_t in zip(lh_targets, rh_targets):
#     dmodel = createDoubleSupportActionModel(left_t, right_t)
#     model_seqs += createSequence([dmodel], DT, N)

# Defining the problem and the solver
problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
fddp = crocoddyl.SolverFDDP(problem)

# Creating display
save_freq = 1
display = vis_tools.MeshcatPinocchioAnimation(rob_model,
col_model, vis_model, rob_data, vis_data, ctrl_freq=1/DT, save_freq=save_freq)

# Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Solver settings
max_iter = 500
fddp.th_stop = 1e-7

xs = [x0] * (fddp.problem.T + 1)
us = fddp.problem.quasiStatic([x0] * fddp.problem.T)
print("Problem solved:", fddp.solve(xs, us, max_iter))
print("Number of iterations:", fddp.iter)
print("Total cost:", fddp.cost)
print("Gradient norm:", fddp.stoppingCriteria())

display.display_targets("lhand_traget", lh_targets)
display.display_targets("lfoot_traget", lf_targets, [0, 0, 1])
display.display_targets("base_traget", base_targets, [0, 1, 0])
display.displayFromCrocoddylSolver(fddp)

if B_SHOW_PLOTS:
    log = fddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)


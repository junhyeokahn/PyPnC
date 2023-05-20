import os
import sys
import crocoddyl
import numpy as np
import pinocchio as pin

import plot.meshcat_utils as vis_tools

cwd = os.getcwd()
sys.path.append(cwd)

B_SHOW_PLOTS = False

def createActionModel(lh_target, rh_target):
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
x0 = np.concatenate([q0, np.zeros(rob_model.nv)])

# Declaring the foot and hand names
rf_name = "r_foot_contact"
lf_name = "l_foot_contact"
lh_name = "l_hand_contact"
rh_name = "r_hand_contact"

# Getting the frame ids
rf_id = rob_model.getFrameId(rf_name)
lf_id = rob_model.getFrameId(lf_name)
lh_id = rob_model.getFrameId(lh_name)
rh_id = rob_model.getFrameId(rh_name)

# Define the robot's state and actuation
state = crocoddyl.StateMultibody(rob_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Create left and right hand targets
lh_targets = []
lh_targets += [np.array([0.3, 0.3, 0.8])]
lh_targets += [np.array([0.35, 0.35, 1.0])]
lh_targets += [np.array([0.35, 0.35, 1.2])]
# lh_targets += [np.array([0.4, 0.4, 0.8])]

rh_targets = []
rh_targets += [np.array([0.3, -0.3, 0.8])]
rh_targets += [np.array([0.35, -0.35, 1.0])]
rh_targets += [np.array([0.35, -0.35, 1.2])]

#
# Solve the problem
#
DT, N = 1e-2, 100

# Creating a running model for each target
seqs = []
for left_t, right_t in zip(lh_targets, rh_targets):
    dmodel = createActionModel(left_t, right_t)
    seqs += createSequence([dmodel], DT, N)

# Defining the problem and the solver
problem = crocoddyl.ShootingProblem(x0, sum(seqs, [])[:-1], seqs[-1][-1])
fddp = crocoddyl.SolverFDDP(problem)

# Creating display
save_freq = 1
display = vis_tools.MeshcatPinocchioAnimation(rob_model,
col_model, vis_model, rob_data, vis_data, ctrl_freq=1/DT, save_freq=save_freq)

# Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])


print("Problem solved:", fddp.solve())
print("Number of iterations:", fddp.iter)
print("Total cost:", fddp.cost)
print("Gradient norm:", fddp.stoppingCriteria())

display.display_targets("lhand_traget", lh_targets)
display.display_targets("rhand_traget", rh_targets)
display.displayFromCrocoddylSolver(fddp)

if B_SHOW_PLOTS:
    log = fddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)


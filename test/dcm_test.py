import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np

from pnc.planner.locomotion.footstep import Footstep
from pnc.planner.locomotion.dcm_planner import DCMPlanner

step_one = Footstep()
step_one.pos = np.array([0.274672, 0.134096, -4.51381e-05])
step_one.quat = np.array([1.89965e-05, 9.56242e-05, -0.000503418, 1])
step_one.side = Footstep.LEFT_SIDE

step_two = Footstep()
step_two.pos = np.array([0.5244, -0.136156, -0.000103187])
step_two.quat = np.array([1.89965e-05, 9.56242e-05, -0.000503418, 1])
step_two.side = Footstep.RIGHT_SIDE

step_third = Footstep()
step_third.pos = np.array([0.524672, 0.133844, -9.2955e-05])
step_third.quat = np.array([1.89965e-05, 9.56242e-05, -0.000503418, 1])
step_third.side = Footstep.LEFT_SIDE

lf_stance = Footstep()
lf_stance.pos = np.array([0.0242475, 0.133069, 5.2579e-09])
lf_stance.quat = np.array([-5.63289e-08, 7.08106e-06, 0.00179564, 0.999998])
lf_stance.side = Footstep.LEFT_SIDE

rf_stance = Footstep()
rf_stance.pos = np.array([0.0248257, -0.134374, -4.11672e-06])
rf_stance.quat = np.array([3.7548e-05, 0.000184163, -0.00280248, 0.999996])
rf_stance.side = Footstep.RIGHT_SIDE

init_dcm = np.array([0.024530, -0.000009, 1.014336])
init_dcm_vel = np.array([-0.000496, -0.000001, 0.002176])

footstep_list = [step_one, step_two, step_third]

dcm_planner = DCMPlanner()
dcm_planner.t_transfer = 0.
dcm_planner.t_ds = 0.45
dcm_planner.t_ss = 0.75
dcm_planner.percentage_settle = 0.9
dcm_planner.alpha_ds = 0.5
dcm_planner.dt = 0.001
dcm_planner.z_vrp = 1.015
dcm_planner.b = 0.321661
dcm_planner.robot_mass = 135.9

dcm_planner.initialize(footstep_list, lf_stance, rf_stance, init_dcm,
                       init_dcm_vel)

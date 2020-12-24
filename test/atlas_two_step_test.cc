#include <ifopt/ipopt_solver.h>
#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/nlp_formulation.h>
#include <towr_plus/terrain/examples/height_map_examples.h>

#include <cmath>
#include <iostream>

using namespace towr_plus;

// A minimal example how to build a trajectory optimization problem using TOWR.
int main() {
  NlpFormulation formulation;

  // terrain
  formulation.terrain_ = std::make_shared<FlatGround>(0.0);

  // Kinematic limits and dynamic parameters of the hopper
  formulation.model_ = RobotModel(RobotModel::Atlas);

  // set the initial position of the hopper
  double nominal_height = 0.766;
  formulation.initial_base_.lin.at(kPos).z() = nominal_height;
  Eigen::Vector3d lfoot, rfoot;
  lfoot << 0.003, -0.111, 0;
  rfoot << 0.003, 0.111, 0;
  formulation.initial_ee_W_.resize(2);
  formulation.initial_ee_W_.at(L) = lfoot;
  formulation.initial_ee_W_.at(R) = rfoot;

  // define the desired goal state of the hopper
  formulation.final_base_.lin.at(towr_plus::kPos) << 0.3, 0.0, nominal_height;

  formulation.params_.ee_phase_durations_.resize(2);
  formulation.params_.ee_phase_durations_.at(L) = {0.45, 0.75, 1.65, 0.75,
                                                   0.45};
  formulation.params_.ee_phase_durations_.at(R) = {1.65, 0.75, 1.65};
  formulation.params_.ee_in_contact_at_start_.resize(2);
  formulation.params_.ee_in_contact_at_start_.at(L) = true;
  formulation.params_.ee_in_contact_at_start_.at(R) = true;

  // Optimize gait timings
  // formulation.params_.OptimizePhaseDurations();

  // Initialize the nonlinear-programming problem with the variables,
  // constraints and costs.
  ifopt::Problem nlp;
  SplineHolder solution;
  for (auto c : formulation.GetVariableSets(solution))
    nlp.AddVariableSet(c);
  for (auto c : formulation.GetConstraints(solution))
    nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    nlp.AddCostSet(c);

  // You can add your own elements to the nlp as well, simply by calling:
  // nlp.AddVariablesSet(your_custom_variables);
  // nlp.AddConstraintSet(your_custom_constraints);

  // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
  // solver->SetOption("derivative_test", "first-order");
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation",
                    "exact"); // "finite difference-values"
  solver->SetOption("max_cpu_time", 500.0);
  solver->Solve(nlp);

  // Can directly view the optimization variables through:
  // Eigen::VectorXd x = nlp.GetVariableValues()
  // However, it's more convenient to access the splines constructed from
  // these variables and query their values at specific times:
  using namespace std;
  cout.precision(2);
  nlp.PrintCurrent(); // view variable-set, constraint violations,
                      // indices,...

  // Print Solution
  cout << fixed;
  cout << "\n====================\nAtlas "
          "trajectory:\n====================\n";

  double t = 0.0;
  while (t <= solution.base_linear_->GetTotalTime() + 1e-5) {
    cout << "t=" << t << "\n";
    cout << "Base linear position x,y,z:   \t";
    cout << solution.base_linear_->GetPoint(t).p().transpose() << "\t[m]"
         << endl;
    cout << "Base linear velocity x,y,z:   \t";
    cout << solution.base_linear_->GetPoint(t).v().transpose() << "\t[m]"
         << endl;

    cout << "Base Euler roll, pitch, yaw:   \t";
    Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
    cout << (rad).transpose() << "\t[deg]" << endl;

    cout << "Base Euler dot roll, pitch, yaw:   \t";
    rad = solution.base_angular_->GetPoint(t).v();
    cout << (rad).transpose() << "\t[deg]" << endl;

    cout << "Left Foot position x,y,z:   \t";
    cout << solution.ee_motion_.at(L)->GetPoint(t).p().transpose() << "\t[m]"
         << endl;

    cout << "Left Foot velocity x,y,z:   \t";
    cout << solution.ee_motion_.at(L)->GetPoint(t).v().transpose() << "\t[m]"
         << endl;

    cout << "Right Foot position x,y,z:   \t";
    cout << solution.ee_motion_.at(R)->GetPoint(t).p().transpose() << "\t[m]"
         << endl;

    cout << "Right Foot velocity x,y,z:   \t";
    cout << solution.ee_motion_.at(R)->GetPoint(t).v().transpose() << "\t[m]"
         << endl;

    cout << "Left Foot Contact force x,y,z:   \t";
    cout << solution.ee_force_.at(L)->GetPoint(t).p().transpose() << "\t[N]"
         << endl;

    cout << "Left Foot Contact force dot x,y,z:   \t";
    cout << solution.ee_force_.at(L)->GetPoint(t).v().transpose() << "\t[N]"
         << endl;

    cout << "Right Foot Contact force x,y,z:   \t";
    cout << solution.ee_force_.at(R)->GetPoint(t).p().transpose() << "\t[N]"
         << endl;

    cout << "Right Foot Contact force dot x,y,z:   \t";
    cout << solution.ee_force_.at(R)->GetPoint(t).v().transpose() << "\t[N]"
         << endl;

    bool contact = solution.phase_durations_.at(L)->IsContactPhase(t);
    std::string foot_in_contact = contact ? "yes" : "no";
    cout << "Left Foot in contact:   \t" + foot_in_contact << endl;

    contact = solution.phase_durations_.at(R)->IsContactPhase(t);
    foot_in_contact = contact ? "yes" : "no";
    cout << "Right Foot in contact:   \t" + foot_in_contact << endl;

    cout << endl;

    t += 0.1;
  }
}

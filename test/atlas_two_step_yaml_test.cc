#include <cmath>
#include <iostream>

#include <ifopt/ipopt_solver.h>

#include <configuration.h>
#include <towr_plus/locomotion_solution.h>
#include <towr_plus/locomotion_task.h>
#include <towr_plus/models/robot_model.h>
#include <towr_plus/nlp_formulation.h>

int main() {
  YAML::Node cfg =
      YAML::LoadFile(THIS_COM "config/towr_plus/atlas_two_step.yaml");

  // Locomotion Task
  LocomotionTask task = LocomotionTask("atlas_two_step_yaml_test");
  task.from_yaml(cfg["locomotion_task"]);

  // Locomotion Solution
  LocomotionSolution sol = LocomotionSolution("atlas_two_step_yaml_test");
  sol.initialize(cfg["locomotion_param"]);

  // Construct NLP from locomotion task
  NlpFormulation formulation;
  formulation.model_ = RobotModel(RobotModel::Atlas);
  formulation.params_.from_yaml(cfg["locomotion_param"]);
  formulation.from_locomotion_task(task);

  // Solve
  ifopt::Problem nlp;
  SplineHolder solution;
  for (auto c : formulation.GetVariableSets(solution))
    nlp.AddVariableSet(c);
  for (auto c : formulation.GetConstraints(solution))
    nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    nlp.AddCostSet(c);

  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 500.0);
  solver->Solve(nlp);

  Eigen::VectorXd vars = nlp.GetVariableValues();
  sol.from_one_hot_vector(vars);

  using namespace std;
  cout.precision(2);
  nlp.PrintCurrent();

  // Pring Solution Trajectories
  // cout << fixed;
  // cout << "\n====================\nAtlas "
  //"trajectory:\n====================\n";

  // double t = 0.0;
  // while (t <= solution.base_linear_->GetTotalTime() + 1e-5) {
  // cout << "t=" << t << "\n";
  // cout << "Base linear position x,y,z:   \t";
  // cout << solution.base_linear_->GetPoint(t).p().transpose() << "\t[m]"
  //<< endl;

  // cout << "Base Euler roll, pitch, yaw:   \t";
  // Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
  // cout << (rad / M_PI * 180).transpose() << "\t[deg]" << endl;

  // cout << "Left Foot position x,y,z:   \t";
  // cout << solution.ee_motion_.at(L)->GetPoint(t).p().transpose() << "\t[m]"
  //<< endl;

  // cout << "Right Foot position x,y,z:   \t";
  // cout << solution.ee_motion_.at(R)->GetPoint(t).p().transpose() << "\t[m]"
  //<< endl;

  // cout << "Left Foot Contact force x,y,z:   \t";
  // cout << solution.ee_force_.at(L)->GetPoint(t).p().transpose() << "\t[N]"
  //<< endl;

  // cout << "Right Foot Contact force x,y,z:   \t";
  // cout << solution.ee_force_.at(R)->GetPoint(t).p().transpose() << "\t[N]"
  //<< endl;

  // bool contact = solution.phase_durations_.at(L)->IsContactPhase(t);
  // std::string foot_in_contact = contact ? "yes" : "no";
  // cout << "Left Foot in contact:   \t" + foot_in_contact << endl;

  // contact = solution.phase_durations_.at(R)->IsContactPhase(t);
  // foot_in_contact = contact ? "yes" : "no";
  // cout << "Right Foot in contact:   \t" + foot_in_contact << endl;

  // cout << endl;

  // t += 0.2;
  //}

  return 0;
}

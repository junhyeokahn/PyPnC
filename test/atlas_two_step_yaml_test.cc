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
  sol.print_solution();
  sol.to_yaml();

  return 0;
}

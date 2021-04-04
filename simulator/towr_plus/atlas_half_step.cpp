#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <ifopt/ipopt_solver.h>

#include <configuration.h>
#include <towr_plus/locomotion_solution.h>
#include <towr_plus/locomotion_task.h>
#include <towr_plus/models/robot_model.h>
#include <towr_plus/nlp_formulation.h>
#include <util/util.hpp>

int main() {
  YAML::Node cfg =
      YAML::LoadFile(THIS_COM "config/towr_plus/atlas_half_step.yaml");
  Clock clock = Clock();
  double time_solving(0.);

  // Locomotion Task
  LocomotionTask task = LocomotionTask("atlas_half_step");
  task.from_yaml(cfg["locomotion_task"]);

  // Locomotion Solution
  LocomotionSolution sol =
      LocomotionSolution("atlas_half_step", cfg["locomotion_param"]);

  // Construct NLP from locomotion task
  NlpFormulation formulation;
  formulation.model_ = RobotModel(RobotModel::Atlas);
  formulation.params_.from_yaml(cfg["locomotion_param"]);
  formulation.from_locomotion_task(task);
  formulation.initialize_from_dcm_planner("dubins");

  formulation.params_.constraints_.erase(
      std::remove(formulation.params_.constraints_.begin(),
                  formulation.params_.constraints_.end(),
                  Parameters::EndeffectorRom),
      formulation.params_.constraints_.end());

  // Solve
  ifopt::Problem nlp;
  SplineHolder solution;
  for (auto c : formulation.GetVariableSets(solution)) {
    nlp.AddVariableSet(c);
  }
  for (auto c : formulation.GetConstraints(solution)) {
    nlp.AddConstraintSet(c);
  }
  for (auto c : formulation.GetCosts()) {
    nlp.AddCostSet(c);
  }

  Eigen::VectorXd initial_vars = nlp.GetVariableValues();
  try {
    YAML::Node one_hot_vec;
    one_hot_vec["sol"] = initial_vars;
    std::cout << one_hot_vec << std::endl;
    std::string file_path = THIS_COM + std::string("data/one_hot.yaml");
    std::ofstream file_out(file_path);
    file_out << one_hot_vec;
  } catch (std::runtime_error &e) {
  }
  sol.from_one_hot_vector(initial_vars);
  sol.to_yaml();
  nlp.PrintCurrent();
  exit(0);

  auto solver = std::make_shared<ifopt::IpoptSolver>();
  // solver->SetOption("derivative_test", "first-order");
  // solver->SetOption("derivative_test_tol", 1e-3);
  // nlp.PrintCurrent();
  // exit(0);
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 1000.0);
  clock.start();
  solver->Solve(nlp);
  time_solving = clock.stop();

  nlp.PrintCurrent();

  Eigen::VectorXd vars = nlp.GetVariableValues();
  sol.from_one_hot_vector(vars);
  // sol.print_solution();
  sol.to_yaml();
  printf("Takes %f seconds\n", 1e-3 * time_solving);

  return 0;
}

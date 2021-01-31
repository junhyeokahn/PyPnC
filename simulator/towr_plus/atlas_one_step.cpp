#include <cmath>
#include <iostream>

#include <configuration.h>

#include <ifopt/ipopt_solver.h>

#include <towr_plus/locomotion_solution.h>
#include <towr_plus/locomotion_task.h>
#include <towr_plus/models/robot_model.h>
#include <towr_plus/nlp_formulation.h>

int main() {
  YAML::Node cfg =
      YAML::LoadFile(THIS_COM "config/towr_plus/atlas_one_step.yaml");
  double max_cpu_time;
  try {
    readParameter(cfg, "max_cpu_time", max_cpu_time);
  } catch (std::runtime_error &e) {
    std::cout << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
  }
  Clock clock = Clock();
  double time_solving(0.);

  // Locomotion Task
  LocomotionTask task = LocomotionTask("atlas_one_step");
  task.from_yaml(cfg["locomotion_task"]);

  // Locomotion Solution
  LocomotionSolution sol =
      LocomotionSolution("atlas_one_step", cfg["locomotion_param"]);

  // Construct NLP from locomotion task
  NlpFormulation formulation;
  formulation.model_ = RobotModel(RobotModel::Atlas);
  formulation.params_.from_yaml(cfg["locomotion_param"]);
  formulation.from_locomotion_task(task);
  formulation.initialize_from_dcm_planner("dubins");

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

  // Eigen::VectorXd initial_vars = nlp.GetVariableValues();
  // sol.from_one_hot_vector(initial_vars);
  // sol.to_yaml();
  // nlp.PrintCurrent();
  // exit(0);

  auto solver = std::make_shared<ifopt::IpoptSolver>();
  // solver->SetOption("derivative_test", "first-order");
  // solver->SetOption("derivative_test_tol", 1e-3);
  // solver->SetOption("derivative_test_print_all", "yes");
  // nlp.PrintCurrent();
  // exit(0);
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", max_cpu_time);
  clock.start();
  solver->Solve(nlp);
  time_solving = clock.stop();

  nlp.PrintCurrent();

  Eigen::VectorXd vars = nlp.GetVariableValues();
  sol.from_one_hot_vector(vars);
  sol.to_yaml();
  printf("Takes %f seconds\n", 1e-3 * time_solving);

  return 0;
}

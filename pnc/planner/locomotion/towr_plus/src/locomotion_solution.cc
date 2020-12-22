#include <towr_plus/locomotion_solution.h>

LocomotionSolution::LocomotionSolution(const std::string &name) {
  name_ = name;
}

void LocomotionTask::print_info() {
  std::cout << "Locomotion Solution for " << name_ << std::endl;
}

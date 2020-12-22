#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <towr_plus/models/endeffector_mappings.h>

using namespace towr_plus;

class LocomotionSolution {
public:
  LocomotionSolution(const std::string &name);
  virtual ~LocomotionSolution();

  // Methods
  void to_yaml();
  // void from_one_hot_vector(const Eigen::VectorXd &one_hot_vec);
  void print_info();

private:
  std::string name_;
};

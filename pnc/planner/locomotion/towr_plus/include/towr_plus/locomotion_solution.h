#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <towr_plus/models/endeffector_mappings.h>
#include <util/util.hpp>

using namespace towr_plus;

// Usage: initialize --> from_one_hot_vector
class LocomotionSolution {
public:
  LocomotionSolution(const std::string &name);
  virtual ~LocomotionSolution();

  // ===========================================================================
  // Methods
  // ===========================================================================
  void from_one_hot_vector(
      const Eigen::VectorXd &one_hot_vec); // Initialize nodes and splines from
                                           // one hot vector
  void initialize(const YAML::Node &node); // Initialize variables
  void print_info();                       // Print solution information
  void to_yaml();                          // Save solution to yaml

private:
  std::string name_;

  double duration_base_polynomial_;
  int force_polynomials_per_stance_phase_;
  int ee_polynomials_per_swing_phase_; // Assume this is always 2
  std::vector<std::vector<double>> ee_phase_durations_;

  Eigen::VectorXd one_hot_vector_;
  int parsing_idx_;

  int n_base_nodes_;
  int n_base_vars_;
  Eigen::MatrixXd base_lin_nodes_;
  Eigen::MatrixXd base_ang_nodes_;

  std::vector<int> n_ee_motion_nodes_;
  std::vector<int> n_ee_motion_vars_;
  std::vector<Eigen::MatrixXd> ee_motion_lin_nodes_;
  std::vector<Eigen::MatrixXd> ee_motion_ang_nodes_;

  std::vector<int> n_ee_wrench_nodes_;
  std::vector<int> n_ee_wrench_vars_;
  std::vector<Eigen::MatrixXd> ee_wrench_lin_nodes_;
  std::vector<Eigen::MatrixXd> ee_wrench_ang_nodes_;

  std::vector<std::vector<double>> ee_schedules_;

  std::vector<double> _get_base_poly_duration();
  double _get_total_time();
  void _set_base_nodes();
  void _set_ee_motion_nodes();
  void _set_ee_wrench_nodes();
  void _set_ee_schedule_variables();
  Eigen::MatrixXd _transpose(Eigen::MatrixXd mat, std::vector<int> order,
                             std::string col_or_row);
};

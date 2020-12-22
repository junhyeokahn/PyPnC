#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/models/robot_model.h>
#include <towr_plus/terrain/height_map.h>
#include <util/util.hpp>

using namespace towr_plus;

class LocomotionTask {
public:
  LocomotionTask(const std::string &name);
  virtual ~LocomotionTask();

  // Initial Configuration
  Eigen::VectorXd initial_base_lin;
  Eigen::VectorXd initial_base_ang;
  std::vector<Eigen::Vector3d> initial_ee_motion_lin;
  std::vector<Eigen::Vector3d> initial_ee_motion_ang;

  // Contact Configuration
  std::vector<std::vector<double>> ee_phase_durations;
  std::vector<bool> ee_in_contact_at_start;

  // Goal Configuration
  Eigen::VectorXd final_base_lin;
  Eigen::VectorXd final_base_ang;

  // Terrain
  std::shared_ptr<HeightMap> terrain;

  // Robot Model
  RobotModel robot_model;

  // Methods
  void from_yaml(const YAML::Node &node);
  void from_one_hot_vector(const Eigen::VectorXd &one_hot_vec);

private:
  std::string name_;
  int num_leg_;
};

/******************************************************************************
Written by Junhyeok Ahn (junhyeokahn91@gmail.com) for towr+
******************************************************************************/

#pragma once

#include "dynamic_model.h"
#include "util/util.hpp"

class MLPModel;

namespace towr_plus {

class CompositeRigidBodyInertia {
public:
  CompositeRigidBodyInertia(const YAML::Node &mlp_model_node,
                            const YAML::Node &data_stat_node);

  virtual ~CompositeRigidBodyInertia();

  Eigen::MatrixXd ComputeInertia(const Eigen::Vector3d &base_pos,
                                 const Eigen::Vector3d &lf_pos,
                                 const Eigen::Vector3d &rf_pos);

private:
  MLPModel *mlp_model_;
  Eigen::VectorXd input_mean_;
  Eigen::VectorXd input_std_;
  Eigen::VectorXd output_mean_;
  Eigen::VectorXd output_std_;

  Eigen::MatrixXd _inertia_from_one_hot_vector(const Eigen::VectorXd &vec);
};

} /* namespace towr_plus */

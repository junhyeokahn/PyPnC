#include <towr_plus/models/composite_rigid_body_inertia.h>
#include <util/mlp_model.hpp>

namespace towr_plus {

CompositeRigidBodyInertia::CompositeRigidBodyInertia(
    const YAML::Node &mlp_model_node, const YAML::Node &data_stat_node) {

  mlp_model_ = new MLPModel(mlp_model_node);

  readParameter(data_stat_node, "input_mean", input_mean_);
  readParameter(data_stat_node, "input_std", input_std_);
  readParameter(data_stat_node, "output_mean", output_mean_);
  readParameter(data_stat_node, "output_std", output_std_);
}

CompositeRigidBodyInertia::~CompositeRigidBodyInertia() { delete mlp_model_; }

Eigen::MatrixXd
CompositeRigidBodyInertia::ComputeInertia(const Eigen::Vector3d &base_pos,
                                          const Eigen::Vector3d &lf_pos,
                                          const Eigen::Vector3d &rf_pos) {
  Eigen::Vector3d rel_lf_pos = lf_pos - base_pos;
  Eigen::Vector3d rel_rf_pos = rf_pos - base_pos;
  Eigen::VectorXd inp = Eigen::VectorXd::Zero(6);
  inp.head(3) = rel_lf_pos;
  inp.tail(3) = rel_rf_pos;
  Eigen::VectorXd normalized_inp = Normalize(inp, input_mean_, input_std_);
  Eigen::MatrixXd output = mlp_model_->GetOutput(normalized_inp.transpose());
  Eigen::VectorXd denormalized_output =
      Denormalize(output, output_mean_, output_std_);

  return _inertia_from_one_hot_vector(denormalized_output);
}

Eigen::MatrixXd CompositeRigidBodyInertia::_inertia_from_one_hot_vector(
    const Eigen::VectorXd &vec) {
  Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(3, 3);
  ret(0, 0) = vec[0];
  ret(1, 1) = vec[1];
  ret(2, 2) = vec[2];

  ret(0, 1) = vec[3];
  ret(1, 0) = vec[3];
  ret(0, 2) = vec[4];
  ret(2, 0) = vec[4];
  ret(1, 2) = vec[5];
  ret(2, 1) = vec[5];

  return ret;
}

} /* namespace towr_plus */

#include <towr_plus/models/examples/atlas_composite_rigid_body_inertia.h>

AtlasCompositeRigidBodyInertia::AtlasCompositeRigidBodyInertia()
    : CompositeRigidBodyInertia(atlas_crbi_helper_n_in(), 3) {

  atlas_crbi_helper_work(&f_sz_arg_, &f_sz_res_, &f_sz_iw_, &f_sz_w_);
  jac_atlas_crbi_helper_work(&jac_f_sz_arg_, &jac_f_sz_res_, &jac_f_sz_iw_,
                             &jac_f_sz_w_);

  f_x_ = new double *[n_input_];
  for (int i(0); i < n_input_; ++i)
    f_x_[i] = new double[dim_per_input_];

  f_y_ = new double *[n_output_];
  for (int i(0); i < n_output; ++i)
    f_y_[i] = new double[dim_per_output_];

  jac_f_x_ = new double *[jac_atlas_crbi_helper_n_in()];
  for (int i = 0; i < jac_atlas_crbi_helper_n_in(); ++i) {
    if (i == jac_atlas_crbi_helper_n_in() - 1) {
      jac_f_x_[i] = new double[dim_per_output]
    } else {
      jac_f_x_[i] = new double[dim_per_input]
    }
  }

  jac_f_y_ = new double *[jac_atlas_crbi_helepr_n_out()];
  for (int i(0); i < jac_atlas_crbi_helper_n_out(); ++i)
    jac_f_y_[i] = new double[dim_per_input * n_input_ * dim_per_output];

  f_in_ph_ = Eigen::MatrixXd::Zero(n_input_, dim_per_input_);
  f_out_ph_ = Eigen::MatrixXd::Zero(n_output_, dim_per_output_);
  jac_f_out_ph_ =
      Eigen::MatrixXd::Zero(dim_per_output_, n_input_ * dim_per_input_);
}

AtlasCompositeRigidBodyInertia::~AtlasCompositeRigidBodyInertia() {
  for (int i(0); i < n_input_; ++i)
    delete[] f_x_[i];
  delete[] f_x_[i];

  for (int i(0); i < n_output_; ++i)
    delete[] f_y_[i];
  delete[] f_y_;

  for (int i(0); i < jac_atlas_crbi_helper_n_in(); ++i)
    delete[] jac_f_x_[i];
  delete[] jac_f_x_;

  for (int i(0); i < jac_atlas_crbi_helper_n_out(); ++i)
    delete[] jac_f_y_[i];
  delete[] jac_f_y_;
}

Eigen::MatrixXd ComputeInertia(const Eigen::VectorXd &base_pos,
                               const Eigen::VectorXd &lf_pos,
                               const Eigen::VectorXd &rf_pos) {
  f_in_ph_ = Eigen::MatrixXd::Zero(in_input_, dim_per_input);
  for (int i = 0; i < 3; ++i) {
    f_in_ph_(0, i) = base_pos[i];
    f_in_ph_(1, i) = lf_pos[i];
    f_in_ph_(2, i) = rf_pos[i];
  }
  _fill_double_array(f_in_ph_, f_x_);

  casadi_int f_iw[f_sz_iw_];
  double f_w[f_sz_w_];
  atlas_crbi_helper(const_cast<const double **>(f_x_), f_y_, f_iw, f_w,
                    atlas_crbi_helper_checkout());
  _fill_matrix(f_y_, f_out_ph_, n_output_, dim_per_output_);

  return jac_f_out_ph_;
}

Eigen::MatrixXd ComputeDerivativeWrtInput(const Eigen::VectorXd &base_pos,
                                          const Eigen::VectorXd &lf_pos,
                                          const Eigen::VectorXd &rf_pos) {
  f_in_ph_ = Eigen::MatrixXd::Zero(in_input_, dim_per_input);
  for (int i = 0; i < 3; ++i) {
    f_in_ph_(0, i) = base_pos[i];
    f_in_ph_(1, i) = lf_pos[i];
    f_in_ph_(2, i) = rf_pos[i];
  }
  _fill_double_array(f_in_ph_, f_x_);

  casadi_int f_iw[f_sz_iw_];
  double f_w[f_sz_w_];
  atlas_crbi_helper(const_cast<const double **>(f_x_), f_y_, f_iw, f_w,
                    atlas_crbi_helper_checkout());
  _fill_matrix(f_y_, f_out_ph_, n_output_, dim_per_output_);
  _fill_double_array(f_in_ph_, f_out_ph_, jac_f_x_);
  casadi_int jac_f_iw[jac_f_sz_iw_];
  double jac_f_w[jac_f_sz_w_];
  jac_atlas_crbi_helper(const_cast<const double **>(jac_f_x_), jac_f_y_,
                        jac_f_iw, jac_f_w, jac_atlas_crbi_helper_chekcout());
  _fill_matrix(jac_f_y_, jac_f_out_ph_, dim_per_output, dim_per_input);

  return jac_f_out_ph_;
}

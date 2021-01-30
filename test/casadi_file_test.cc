#include <iostream>
#include <towr_plus/models/examples/atlas_crbi_helper.h>

#include <Eigen/Dense>

// Use this for filling function args
void _fill_double_array(const Eigen::MatrixXd &mat, double **x) {
  for (int row = 0; row < mat.rows(); ++row) {
    for (int col = 0; col < mat.cols(); ++col) {
      x[col][row] = mat(col, row);
    }
  }
}

// Use this for filling jacobian function args
void _fill_double_array(const Eigen::MatrixXd &mat1,
                        const Eigen::MatrixXd &mat2, double **x) {
  for (int row = 0; row < mat1.rows(); ++row) {
    for (int col = 0; col < mat1.cols(); ++col) {
      x[row][col] = mat1(row, col);
    }
  }
  for (int row = 0; row < mat2.rows(); ++row) {
    for (int col = 0; col < mat2.cols(); ++col) {
      x[mat1.rows() + row][col] = mat2(row, col);
    }
  }
}

// Use this for passing output
// Assume x[1][mat.cols()*mat.rows()]
void _fill_matrix(double **x, Eigen::MatrixXd &mat, int block_row,
                  int block_col) {
  int n_var(mat.cols() / block_col);
  int idx(0);
  for (int block_id = 0; block_id < n_var; ++block_id) {
    for (int col = 0; col < block_col; ++col) {
      for (int row = 0; row < block_row; ++row) {
        mat(row, block_id * block_col + col) = x[0][idx];
        idx += 1;
      }
    }
  }
}

int main(int argc, char *argv[]) {

  int NUM_INPUT = atlas_crbi_helper_n_in();
  int DIM_PER_INPUT = 3;
  int DIM_OUTPUT = 6;

  casadi_int f_sz_arg, f_sz_res, f_sz_iw, f_sz_w;
  casadi_int jac_f_sz_arg, jac_f_sz_res, jac_f_sz_iw, jac_f_sz_w;

  atlas_crbi_helper_work(&f_sz_arg, &f_sz_res, &f_sz_iw, &f_sz_w);
  jac_atlas_crbi_helper_work(&jac_f_sz_arg, &jac_f_sz_res, &jac_f_sz_iw,
                             &jac_f_sz_w);

  casadi_int f_iw[f_sz_iw];
  double f_w[f_sz_w];
  casadi_int jac_f_iw[jac_f_sz_iw];
  double jac_f_w[jac_f_sz_w];

  double **f_x = new double *[atlas_crbi_helper_n_in()];
  for (int i(0); i < atlas_crbi_helper_n_in(); ++i)
    f_x[i] = new double[DIM_PER_INPUT];

  double **f_y = new double *[atlas_crbi_helper_n_out()];
  for (int i(0); i < atlas_crbi_helper_n_out(); ++i)
    f_y[i] = new double[DIM_OUTPUT];

  double **jac_f_x = new double *[jac_atlas_crbi_helper_n_in()];
  for (int i(0); i < jac_atlas_crbi_helper_n_in(); ++i) {
    if (i == jac_atlas_crbi_helper_n_in() - 1) {
      jac_f_x[i] = new double[DIM_OUTPUT];
    } else {
      jac_f_x[i] = new double[DIM_PER_INPUT];
    }
  }

  Eigen::MatrixXd f_in_ph = Eigen::MatrixXd::Ones(NUM_INPUT, DIM_PER_INPUT);
  Eigen::MatrixXd f_out_ph = Eigen::MatrixXd::Zero(1, DIM_OUTPUT);
  Eigen::MatrixXd jac_f_out_ph =
      Eigen::MatrixXd::Zero(DIM_OUTPUT, NUM_INPUT * DIM_PER_INPUT);

  double **jac_f_y = new double *[jac_atlas_crbi_helper_n_out()];
  for (int i(0); i < jac_atlas_crbi_helper_n_out(); ++i)
    jac_f_y[i] = new double[DIM_PER_INPUT * NUM_INPUT * DIM_OUTPUT];

  // ==============================
  // Function F
  // ==============================
  std::cout << "f # in : " << atlas_crbi_helper_n_in() << std::endl;
  std::cout << "f # out : " << atlas_crbi_helper_n_out() << std::endl;
  _fill_double_array(f_in_ph, f_x);
  std::cout << atlas_crbi_helper(const_cast<const double **>(f_x), f_y, f_iw,
                                 f_w, atlas_crbi_helper_checkout())
            << std::endl;
  _fill_matrix(f_y, f_out_ph, 1, DIM_OUTPUT);
  std::cout << "f input" << std::endl;
  std::cout << f_in_ph << std::endl;
  std::cout << "f output" << std::endl;
  std::cout << f_out_ph << std::endl;

  // ==============================
  // Jacobian
  // ==============================
  std::cout << "jac f # in : " << jac_atlas_crbi_helper_n_in() << std::endl;
  std::cout << "jac f # out : " << jac_atlas_crbi_helper_n_out() << std::endl;
  _fill_double_array(f_in_ph, f_out_ph, jac_f_x);
  std::cout << jac_atlas_crbi_helper(const_cast<const double **>(jac_f_x),
                                     jac_f_y, jac_f_iw, jac_f_w,
                                     jac_atlas_crbi_helper_checkout())
            << std::endl;
  _fill_matrix(jac_f_y, jac_f_out_ph, DIM_OUTPUT, DIM_PER_INPUT);
  std::cout << "raw" << std::endl;
  for (int i = 0; i < 54; ++i) {
    std::cout << jac_f_y[0][i] << std::endl;
  }
  std::cout << "f jac output" << std::endl;
  std::cout << jac_f_out_ph << std::endl;

  return 0;
}

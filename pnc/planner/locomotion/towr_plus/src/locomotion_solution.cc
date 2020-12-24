#include <numeric> // std::accumulate
#include <vector>

#include <towr_plus/locomotion_solution.h>
#include <towr_plus/models/endeffector_mappings.h>

LocomotionSolution::LocomotionSolution(const std::string &name) {
  name_ = name;
  int num_leg(2);
  ee_phase_durations_.resize(num_leg);
  n_ee_motion_nodes_.resize(num_leg);
  n_ee_motion_vars_.resize(num_leg);
  ee_motion_lin_nodes_.resize(num_leg);
  ee_motion_ang_nodes_.resize(num_leg);
  n_ee_wrench_nodes_.resize(num_leg);
  n_ee_wrench_vars_.resize(num_leg);
  ee_wrench_lin_nodes_.resize(num_leg);
  ee_wrench_ang_nodes_.resize(num_leg);
  ee_schedules_.resize(2);
}

LocomotionSolution::~LocomotionSolution() {}

void LocomotionSolution::print_info() {
  std::cout << "Locomotion Solution for " << name_ << std::endl;
}

void LocomotionSolution::initialize(const YAML::Node &node) {
  Eigen::VectorXd tmp_vec;
  bool tmp_bool;
  try {
    readParameter(node, "duration_base_polynomial", duration_base_polynomial_);
    readParameter(node, "force_polynomials_per_stance_phase",
                  force_polynomials_per_stance_phase_);
    readParameter(node, "ee_polynomials_per_swing_phase",
                  ee_polynomials_per_swing_phase_);
    assert(ee_polynomials_per_swing_phase == 2); // Assume this is always 2
    for (auto ee : {L, R}) {
      readParameter(node["ee_phase_durations"], std::to_string(ee), tmp_vec);
      for (int i = 0; i < tmp_vec.size(); ++i)
        ee_phase_durations_.at(ee).push_back(tmp_vec(i));
    }

  } catch (std::runtime_error &e) {
    std::cout << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
  }
}

void LocomotionSolution::from_one_hot_vector(
    const Eigen::VectorXd &one_hot_vec) {
  one_hot_vector_ = one_hot_vec;

  parsing_idx_ = 0;
  _set_base_nodes();
  _set_ee_motion_nodes();
  _set_ee_wrench_nodes();
  _set_ee_schedule_variables();

  // using namespace std;
  // cout.precision(2);
  // cout << fixed;
  // std::cout << one_hot_vector_ << std::endl;
}

void LocomotionSolution::_set_base_nodes() {

  n_base_nodes_ = _get_base_poly_duration().size() + 1;
  n_base_vars_ = n_base_nodes_ * 6.;

  Eigen::VectorXd base_lin_vec =
      one_hot_vector_.segment(parsing_idx_, n_base_vars_);
  parsing_idx_ += n_base_vars_;
  Eigen::VectorXd base_ang_vec =
      one_hot_vector_.segment(parsing_idx_, n_base_vars_);
  parsing_idx_ += n_base_vars_;

  base_lin_nodes_ = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      base_lin_vec.data(), n_base_nodes_, 6);
  base_ang_nodes_ = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      base_ang_vec.data(), n_base_nodes_, 6);
}

void LocomotionSolution::_set_ee_motion_nodes() {

  std::vector<Eigen::VectorXd> motion_lin_vec(2);
  // std::vector<Eigen::VectorXd> motion_ang_vec(2);

  // Assume ee phase starts and ends with contact
  for (auto ee : {L, R}) {
    n_ee_motion_nodes_.at(ee) =
        2 + (ee_polynomials_per_swing_phase_ + 1) *
                ((ee_phase_durations_.at(ee).size() - 2) + 1) / 2;
    n_ee_motion_vars_.at(ee) = 3 * (ee_phase_durations_.at(ee).size() + 1) / 2 +
                               5 * (ee_phase_durations_.at(ee).size() - 1) / 2;
    ee_motion_lin_nodes_.at(ee) =
        Eigen::MatrixXd::Zero(n_ee_motion_nodes_.at(ee), 6);
    // ee_motion_ang_nodes_.at(ee) =
    // Eigen::MatrixXd::Zero(n_ee_motion_nodes_.at(ee), 6);

    motion_lin_vec.at(ee) =
        one_hot_vector_.segment(parsing_idx_, n_ee_motion_vars_.at(ee));
    parsing_idx_ += n_ee_motion_vars_.at(ee);
    // motion_ang_vec.at(ee) = ;

    int node_idx(0);
    int variable_idx(0);
    for (int ph = 0; ph < ee_phase_durations_.at(ee).size(); ++ph) {
      // std::cout << "phase: " << ph << std::endl;
      if (ph % 2 == 0) {
        // Contact Phase: Use 3 variables (X, Y, Z) to fill 2 nodes
        // printf("filling node: %i, %i, with variable %i, %i, %i\n", node_idx,
        // node_idx + 1, variable_idx, variable_idx + 1, variable_idx + 2);
        for (auto dim : {0, 1, 2}) {
          ee_motion_lin_nodes_.at(ee)(node_idx, dim) =
              motion_lin_vec.at(ee)(variable_idx + dim);
          ee_motion_lin_nodes_.at(ee)(node_idx + 1, dim) =
              motion_lin_vec.at(ee)(variable_idx + dim);
        }
        node_idx += 2;
        variable_idx += 3;
      } else {
        // Swing Phase: Use 5 variables (X, DX, Y, DY, Z, DZ) to fill 1 node
        // printf("filling node: %i, with variable %i, %i, %i, %i, %i\n",
        // node_idx, variable_idx, variable_idx + 1, variable_idx + 2,
        // variable_idx + 3, variable_idx + 4);
        for (auto dim : {0, 1, 2, 3, 4, 5}) {
          if (dim == 5) {
            ee_motion_lin_nodes_.at(ee)(node_idx, dim) = 0.;
          } else {
            ee_motion_lin_nodes_.at(ee)(node_idx, dim) =
                motion_lin_vec.at(ee)(variable_idx + dim);
          }
        }
        node_idx += 1;
        variable_idx += 5;
      }
    }
  }

  // using namespace std;
  // cout.precision(2);
  // cout << fixed;
  // std::cout << "L" << std::endl;
  // std::cout << ee_motion_lin_nodes_.at(L) << std::endl;
  // std::cout << "R" << std::endl;
  // std::cout << ee_motion_lin_nodes_.at(R) << std::endl;
  // exit(0);
}

void LocomotionSolution::_set_ee_wrench_nodes() {
  std::vector<Eigen::VectorXd> wrench_lin_vec(2);
  // std::vector< Eigen::VectorXd > wrench_ang_vec(2);

  // Assume ee phase starts and ends with contact
  for (auto ee : {L, R}) {
    n_ee_wrench_nodes_.at(ee) = (force_polynomials_per_stance_phase_ + 1) *
                                (ee_phase_durations_.at(ee).size() + 1) / 2;
    n_ee_wrench_vars_.at(ee) =
        6 * (2 * force_polynomials_per_stance_phase_ +
             (((ee_phase_durations_.at(ee).size() + 1) / 2) - 2) *
                 (force_polynomials_per_stance_phase_ - 1));
    ee_wrench_lin_nodes_.at(ee) =
        Eigen::MatrixXd::Zero(n_ee_wrench_nodes_.at(ee), 6);
    // ee_wrench_ang_nodes_.at(ee) =
    // Eigen::MatrixXd::Zero(n_ee_wrench_nodes_.at(ee), 6);

    wrench_lin_vec.at(ee) =
        one_hot_vector_.segment(parsing_idx_, n_ee_wrench_vars_.at(ee));
    parsing_idx_ += n_ee_wrench_vars_.at(ee);
    // wrench_ang_vec.at(ee) = ;

    int node_idx(0);
    int variable_idx(0);
    for (int ph = 0; ph < ee_phase_durations_.at(ee).size(); ++ph) {
      if (ph % 2 == 0) {
        // Contact Phase
        if (ph == 0) {
          // Initial Contact Phase
          ee_wrench_lin_nodes_.at(ee).block(
              node_idx, 0, force_polynomials_per_stance_phase_, 6) =
              Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(
                  wrench_lin_vec.at(ee)
                      .segment(variable_idx,
                               force_polynomials_per_stance_phase_ * 6)
                      .data(),
                  force_polynomials_per_stance_phase_, 6);

          node_idx += (force_polynomials_per_stance_phase_ + 1);
          variable_idx += force_polynomials_per_stance_phase_ * 6;

        } else if (ph == ee_phase_durations_.at(ee).size() - 1) {
          // Final Contact Phase
          ee_wrench_lin_nodes_.at(ee).block(
              node_idx + 1, 0, force_polynomials_per_stance_phase_, 6) =
              Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(
                  wrench_lin_vec.at(ee)
                      .segment(variable_idx,
                               force_polynomials_per_stance_phase_ * 6)
                      .data(),
                  force_polynomials_per_stance_phase_, 6);

        } else {
          // Intermediate Contact Phase
          ee_wrench_lin_nodes_.at(ee).block(
              node_idx + 1, 0, force_polynomials_per_stance_phase_ - 1, 6) =
              Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(
                  wrench_lin_vec.at(ee)
                      .segment(variable_idx,
                               force_polynomials_per_stance_phase_ - 1 * 6)
                      .data(),
                  force_polynomials_per_stance_phase_ - 1, 6);
          node_idx += (force_polynomials_per_stance_phase_ + 1);
          variable_idx += (force_polynomials_per_stance_phase_ - 1) * 6;
        }
      } else {
        // Swing Phase: Do Nothing
      }
    }
    // Rearrange to make X, Y, Z, DX, DY, DZ
    ee_wrench_lin_nodes_.at(ee) =
        _transpose(ee_wrench_lin_nodes_.at(ee), {0, 2, 4, 1, 3, 5}, "col");
  }

  // using namespace std;
  // cout.precision(2);
  // cout << fixed;
  // std::cout << "L" << std::endl;
  // std::cout << wrench_lin_vec.at(L) << std::endl;
  // std::cout << ee_wrench_lin_nodes_.at(L) << std::endl;
  // std::cout << "R" << std::endl;
  // std::cout << wrench_lin_vec.at(R) << std::endl;
  // std::cout << ee_wrench_lin_nodes_.at(R) << std::endl;
  // exit(0);
}

void LocomotionSolution::_set_ee_schedule_variables() {
  std::vector<Eigen::VectorXd> schedule_vector(2);
  for (auto ee : {L, R}) {
    ee_schedules_.at(ee).clear();
    schedule_vector.at(ee) = one_hot_vector_.segment(
        parsing_idx_, ee_phase_durations_.at(ee).size() - 1);
    parsing_idx_ += (ee_phase_durations_.at(ee).size() - 1);
    double sum(0.);
    for (int i = 0; i < ee_phase_durations_.at(ee).size(); ++i) {
      if (i == ee_phase_durations_.at(ee).size() - 1) {
        ee_schedules_.at(ee).push_back(_get_total_time() - sum);
      } else {
        ee_schedules_.at(ee).push_back(schedule_vector.at(ee)(i));
        sum += schedule_vector.at(ee)(i);
      }
    }
  }

  // std::cout << "total_time: " << _get_total_time() << std::endl;
  // std::cout << "L" << std::endl;
  // double sum(0);
  // for (int i = 0; i < ee_schedules_.at(L).size(); ++i) {
  // std::cout << ee_schedules_.at(L)[i] << std::endl;
  // sum += ee_schedules_.at(L)[i];
  //}
  // std::cout << "sum: " << sum << std::endl;

  // sum = 0;
  // std::cout << "R" << std::endl;
  // for (int i = 0; i < ee_schedules_.at(R).size(); ++i) {
  // std::cout << ee_schedules_.at(R)[i] << std::endl;
  // sum += ee_schedules_.at(R)[i];
  //}
  // std::cout << "sum: " << sum << std::endl;
  // exit(0);
}

std::vector<double> LocomotionSolution::_get_base_poly_duration() {

  std::vector<double> base_spline_timings_;
  double dt = duration_base_polynomial_;
  double t_left = _get_total_time();

  double eps = 1e-10; // since repeated subtraction causes inaccuracies
  while (t_left > eps) {
    double duration = t_left > dt ? dt : t_left;
    base_spline_timings_.push_back(duration);

    t_left -= dt;
  }

  return base_spline_timings_;
}

double LocomotionSolution::_get_total_time() {
  std::vector<double> T_feet;

  for (const auto &v : ee_phase_durations_)
    T_feet.push_back(std::accumulate(v.begin(), v.end(), 0.0));

  // safety check that all feet durations sum to same value
  double T =
      T_feet.empty() ? 0.0 : T_feet.front(); // take first foot as reference
  for (double Tf : T_feet)
    assert(fabs(Tf - T) < 1e-6);

  return T;
}

Eigen::MatrixXd LocomotionSolution::_transpose(Eigen::MatrixXd mat,
                                               std::vector<int> order,
                                               std::string col_or_row) {
  int n_row = mat.rows();
  int n_col = mat.cols();
  Eigen::MatrixXd ret(n_row, n_col);
  if (col_or_row == "col") {
    assert(n_col == order.size());
    for (int i = 0; i < n_col; ++i) {
      ret.col(i) = mat.col(order[i]);
    }
  } else if (col_or_row == "row") {
    assert(n_row == order.size());
    for (int i = 0; i < n_row; ++i) {
      ret.row(i) = mat.row(order[i]);
    }
  } else {
    std::cout << "Wrong Option" << std::endl;
    exit(0);
  }
  return ret;
};

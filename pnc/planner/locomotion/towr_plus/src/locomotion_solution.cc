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

  one_hot_ee_motion_lin_.resize(2);
  one_hot_ee_motion_ang_.resize(2);
  one_hot_ee_wrench_lin_.resize(2);
  one_hot_ee_wrench_ang_.resize(2);
  one_hot_ee_contact_schedule_.resize(2);
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

void LocomotionSolution::print_solution(double dt) {

  using namespace std;
  cout.precision(2);
  cout << fixed;
  cout << "\n====================\n Solution "
          "trajectory:\n====================\n";

  double t = 0.0;
  while (t <= spline_holder_.base_linear_->GetTotalTime() + 1e-5) {
    cout << "t=" << t << "\n";
    cout << "Base linear position x,y,z:   \t";
    cout << spline_holder_.base_linear_->GetPoint(t).p().transpose() << "\t[m]"
         << endl;
    cout << "Base linear velocity x,y,z:   \t";
    cout << spline_holder_.base_linear_->GetPoint(t).v().transpose() << "\t[m]"
         << endl;

    cout << "Base Euler roll, pitch, yaw:   \t";
    Eigen::Vector3d rad = spline_holder_.base_angular_->GetPoint(t).p();
    cout << (rad).transpose() << "\t[deg]" << endl;

    cout << "Base Euler dot roll, pitch, yaw:   \t";
    rad = spline_holder_.base_angular_->GetPoint(t).v();
    cout << (rad).transpose() << "\t[deg]" << endl;

    cout << "Left Foot position x,y,z:   \t";
    cout << spline_holder_.ee_motion_.at(L)->GetPoint(t).p().transpose()
         << "\t[m]" << endl;

    cout << "Left Foot velocity x,y,z:   \t";
    cout << spline_holder_.ee_motion_.at(L)->GetPoint(t).v().transpose()
         << "\t[m]" << endl;

    cout << "Right Foot position x,y,z:   \t";
    cout << spline_holder_.ee_motion_.at(R)->GetPoint(t).p().transpose()
         << "\t[m]" << endl;

    cout << "Right Foot velocity x,y,z:   \t";
    cout << spline_holder_.ee_motion_.at(R)->GetPoint(t).v().transpose()
         << "\t[m]" << endl;

    cout << "Left Foot Contact force x,y,z:   \t";
    cout << spline_holder_.ee_force_.at(L)->GetPoint(t).p().transpose()
         << "\t[N]" << endl;

    cout << "Left Foot Contact force dot x,y,z:   \t";
    cout << spline_holder_.ee_force_.at(L)->GetPoint(t).v().transpose()
         << "\t[N]" << endl;

    cout << "Right Foot Contact force x,y,z:   \t";
    cout << spline_holder_.ee_force_.at(R)->GetPoint(t).p().transpose()
         << "\t[N]" << endl;

    cout << "Right Foot Contact force dot x,y,z:   \t";
    cout << spline_holder_.ee_force_.at(R)->GetPoint(t).v().transpose()
         << "\t[N]" << endl;

    bool contact = spline_holder_.phase_durations_.at(L)->IsContactPhase(t);
    std::string foot_in_contact = contact ? "yes" : "no";
    cout << "Left Foot in contact:   \t" + foot_in_contact << endl;

    contact = spline_holder_.phase_durations_.at(R)->IsContactPhase(t);
    foot_in_contact = contact ? "yes" : "no";
    cout << "Right Foot in contact:   \t" + foot_in_contact << endl;

    cout << endl;

    t += dt;
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
  _set_splines();
}

void LocomotionSolution::_set_splines() {
  // Base Lin
  std::shared_ptr<NodesVariablesAll> base_lin =
      std::make_shared<NodesVariablesAll>(n_base_nodes_, k3D,
                                          id::base_lin_nodes);
  base_lin->SetVariables(one_hot_base_lin_);

  // Base Ang
  std::shared_ptr<NodesVariablesAll> base_ang =
      std::make_shared<NodesVariablesAll>(n_base_nodes_, k3D,
                                          id::base_ang_nodes);
  base_ang->SetVariables(one_hot_base_ang_);

  // EE Motion & Wrench & Contact Schedule
  std::vector<std::shared_ptr<NodesVariablesPhaseBased>> ee_motion_lin(2);
  std::vector<std::shared_ptr<NodesVariablesPhaseBased>> ee_wrench_lin(2);
  std::vector<std::shared_ptr<PhaseDurations>> ee_phase_dur(2);
  for (auto ee : {L, R}) {
    ee_motion_lin.at(ee) = std::make_shared<NodesVariablesEEMotion>(
        ee_phase_durations_.at(ee).size(), true, id::EEMotionLinNodes(ee),
        ee_polynomials_per_swing_phase_);
    ee_motion_lin.at(ee)->SetVariables(one_hot_ee_motion_lin_.at(ee));

    ee_wrench_lin.at(ee) = std::make_shared<NodesVariablesEEForce>(
        ee_phase_durations_.at(ee).size(), true, id::EEWrenchLinNodes(ee),
        force_polynomials_per_stance_phase_);
    ee_wrench_lin.at(ee)->SetVariables(one_hot_ee_wrench_lin_.at(ee));

    ee_phase_dur.at(ee) = std::make_shared<PhaseDurations>(
        ee, ee_phase_durations_.at(ee), true, -100.,
        100.); // bound_phase_duration isn't important here
    ee_phase_dur.at(ee)->SetVariables(one_hot_ee_contact_schedule_.at(ee));
  }

  // Construct Splineholder
  spline_holder_ =
      SplineHolder(base_lin, base_ang, _get_base_poly_duration(), ee_motion_lin,
                   ee_wrench_lin, ee_phase_dur, true);
}

void LocomotionSolution::_set_base_nodes() {

  n_base_nodes_ = _get_base_poly_duration().size() + 1;
  n_base_vars_ = n_base_nodes_ * 6.;

  one_hot_base_lin_ = one_hot_vector_.segment(parsing_idx_, n_base_vars_);
  parsing_idx_ += n_base_vars_;
  one_hot_base_ang_ = one_hot_vector_.segment(parsing_idx_, n_base_vars_);
  parsing_idx_ += n_base_vars_;

  base_lin_nodes_ = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      one_hot_base_lin_.data(), n_base_nodes_, 6);
  base_ang_nodes_ = Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      one_hot_base_ang_.data(), n_base_nodes_, 6);
}

void LocomotionSolution::_set_ee_motion_nodes() {

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

    one_hot_ee_motion_lin_.at(ee) =
        one_hot_vector_.segment(parsing_idx_, n_ee_motion_vars_.at(ee));
    parsing_idx_ += n_ee_motion_vars_.at(ee);
    // one_hot_ee_motion_ang_.at(ee) = ;

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
              one_hot_ee_motion_lin_.at(ee)(variable_idx + dim);
          ee_motion_lin_nodes_.at(ee)(node_idx + 1, dim) =
              one_hot_ee_motion_lin_.at(ee)(variable_idx + dim);
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
                one_hot_ee_motion_lin_.at(ee)(variable_idx + dim);
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

    one_hot_ee_wrench_lin_.at(ee) =
        one_hot_vector_.segment(parsing_idx_, n_ee_wrench_vars_.at(ee));
    parsing_idx_ += n_ee_wrench_vars_.at(ee);
    // one_hot_ee_wrench_ang_.at(ee) = ;

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
                  one_hot_ee_wrench_lin_.at(ee)
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
                  one_hot_ee_wrench_lin_.at(ee)
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
                  one_hot_ee_wrench_lin_.at(ee)
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
  // std::cout << one_hot_ee_wrench_lin_.at(L) << std::endl;
  // std::cout << ee_wrench_lin_nodes_.at(L) << std::endl;
  // std::cout << "R" << std::endl;
  // std::cout << one_hot_ee_wrench_lin_.at(R) << std::endl;
  // std::cout << ee_wrench_lin_nodes_.at(R) << std::endl;
  // exit(0);
}

void LocomotionSolution::_set_ee_schedule_variables() {
  for (auto ee : {L, R}) {
    ee_schedules_.at(ee).clear();
    one_hot_ee_contact_schedule_.at(ee) = one_hot_vector_.segment(
        parsing_idx_, ee_phase_durations_.at(ee).size() - 1);
    parsing_idx_ += (ee_phase_durations_.at(ee).size() - 1);
    double sum(0.);
    for (int i = 0; i < ee_phase_durations_.at(ee).size(); ++i) {
      if (i == ee_phase_durations_.at(ee).size() - 1) {
        ee_schedules_.at(ee).push_back(_get_total_time() - sum);
      } else {
        ee_schedules_.at(ee).push_back(one_hot_ee_contact_schedule_.at(ee)(i));
        sum += one_hot_ee_contact_schedule_.at(ee)(i);
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

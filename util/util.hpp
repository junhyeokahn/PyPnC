#pragma once

#include "yaml/include/myYaml/yaml.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <math.h>

// =============================================================================
// YAML Readers
// =============================================================================
template <typename YamlType>
YamlType readParameter(const YAML::Node &node, const std::string &name) {
  try {
    return node[name.c_str()].as<YamlType>();
  } catch (...) {
    throw std::runtime_error(name);
  }
};

template <typename YamlType>
void readParameter(const YAML::Node &node, const std::string &name,
                   YamlType &parameter) {
  try {
    parameter = readParameter<YamlType>(node, name);
  } catch (...) {
    throw std::runtime_error(name);
  }
};

// =============================================================================
// Rotations
// =============================================================================
// Convention here for euler angle is to rotate in Z-Y-X order and represent
// euler angles in (x,y,z) order
Eigen::Matrix3d euler_xyz_to_rot(const Eigen::Vector3d &euler_xyz);
Eigen::Quaternion<double> euler_xyz_to_quat(const Eigen::Vector3d &euler_xyz);
Eigen::Vector3d quat_to_euler_xyz(const Eigen::Quaternion<double> &quat);
Eigen::MatrixXd adjoint(const Eigen::MatrixXd &R, const Eigen::Vector3d &p);
Eigen::MatrixXd adjoint(const Eigen::Isometry3d &iso);
Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d &omg);

// =============================================================================
// Pretty Print
// =============================================================================
void pretty_print(Eigen::Isometry3d const &iso, std::ostream &os,
                  std::string const &title, std::string const &prefix = "",
                  bool nonl = false);
void pretty_print(Eigen::VectorXd const &vv, std::ostream &os,
                  std::string const &title, std::string const &prefix = "",
                  bool nonl = false);
void pretty_print(Eigen::MatrixXd const &mm, std::ostream &os,
                  std::string const &title, std::string const &prefix = "",
                  bool vecmode = false, bool nonl = false);
void pretty_print(Eigen::Quaternion<double> const &qq, std::ostream &os,
                  std::string const &title, std::string const &prefix = "",
                  bool nonl = false);
void pretty_print(Eigen::Vector3d const &vv, std::ostream &os,
                  std::string const &title, std::string const &prefix = "",
                  bool nonl = false);
void pretty_print(const std::vector<double> &_vec, const char *title);
void pretty_print(const std::vector<int> &_vec, const char *title);
std::string pretty_string(Eigen::VectorXd const &vv);
std::string pretty_string(Eigen::MatrixXd const &mm, std::string const &prefix);
std::string pretty_string(double vv);

// =============================================================================
// Hermite Curves
// =============================================================================
class HermiteCurve {
public:
  HermiteCurve();
  HermiteCurve(const double &start_pos, const double &start_vel,
               const double &end_pos, const double &end_vel);
  ~HermiteCurve();
  double evaluate(const double &s_in);
  double evaluateFirstDerivative(const double &s_in);
  double evaluateSecondDerivative(const double &s_in);

private:
  double p1;
  double v1;
  double p2;
  double v2;

  double s_;

  // by default clamps within 0 and 1.
  double clamp(const double &s_in, double lo = 0.0, double hi = 1.0);
};

class HermiteCurveVec {
public:
  HermiteCurveVec();
  HermiteCurveVec(const Eigen::VectorXd &start_pos,
                  const Eigen::VectorXd &start_vel,
                  const Eigen::VectorXd &end_pos,
                  const Eigen::VectorXd &end_vel);

  void initialize(const Eigen::VectorXd &start_pos,
                  const Eigen::VectorXd &start_vel,
                  const Eigen::VectorXd &end_pos,
                  const Eigen::VectorXd &end_vel);

  ~HermiteCurveVec();
  Eigen::VectorXd evaluate(const double &s_in);
  Eigen::VectorXd evaluateFirstDerivative(const double &s_in);
  Eigen::VectorXd evaluateSecondDerivative(const double &s_in);

private:
  Eigen::VectorXd p1;
  Eigen::VectorXd v1;
  Eigen::VectorXd p2;
  Eigen::VectorXd v2;

  std::vector<HermiteCurve> curves;
  Eigen::VectorXd output;
};

class HermiteQuaternionCurve {
public:
  HermiteQuaternionCurve();
  HermiteQuaternionCurve(const Eigen::Quaterniond &quat_start,
                         const Eigen::Vector3d &angular_velocity_start,
                         const Eigen::Quaterniond &quat_end,
                         const Eigen::Vector3d &angular_velocity_end);
  ~HermiteQuaternionCurve();

  void initialize(const Eigen::Quaterniond &quat_start,
                  const Eigen::Vector3d &angular_velocity_start,
                  const Eigen::Quaterniond &quat_end,
                  const Eigen::Vector3d &angular_velocity_end);

  // All values are expressed in "world frame"
  void evaluate(const double &s_in, Eigen::Quaterniond &quat_out);
  void getAngularVelocity(const double &s_in, Eigen::Vector3d &ang_vel_out);
  void getAngularAcceleration(const double &s_in, Eigen::Vector3d &ang_acc_out);

private:
  Eigen::Quaterniond qa;   // Starting quaternion
  Eigen::Vector3d omega_a; // Starting Angular Velocity
  Eigen::Quaterniond qb;   // Ending quaternion
  Eigen::Vector3d omega_b; // Ending Angular velocity

  Eigen::AngleAxisd omega_a_aa; // axis angle representation of omega_a
  Eigen::AngleAxisd omega_b_aa; // axis angle representation of omega_b

  void initialize_data_structures();

  void computeBasis(const double &s_in); // computes the basis functions
  void computeOmegas();

  Eigen::Quaterniond q0; // quat0
  Eigen::Quaterniond q1; // quat1
  Eigen::Quaterniond q2; // quat1
  Eigen::Quaterniond q3; // quat1

  double b1; // basis 1
  double b2; // basis 2
  double b3; // basis 3

  double bdot1; // 1st derivative of basis 1
  double bdot2; // 1st derivative of basis 2
  double bdot3; // 1st derivative of basis 3

  double bddot1; // 2nd derivative of basis 1
  double bddot2; // 2nd derivative of basis 2
  double bddot3; // 2nd derivative of basis 3

  Eigen::Vector3d omega_1;
  Eigen::Vector3d omega_2;
  Eigen::Vector3d omega_3;

  Eigen::AngleAxisd omega_1aa;
  Eigen::AngleAxisd omega_2aa;
  Eigen::AngleAxisd omega_3aa;

  // Allocate memory for quaternion operations
  Eigen::Quaterniond qtmp1;
  Eigen::Quaterniond qtmp2;
  Eigen::Quaterniond qtmp3;

  // progression variable
  double s_;
  // by default clamps within 0 and 1.
  double clamp(const double &s_in, double lo = 0.0, double hi = 1.0);

  void printQuat(const Eigen::Quaterniond &quat);
};

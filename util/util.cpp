#include <util/util.hpp>

// =============================================================================
// Rotations
// =============================================================================
Eigen::Matrix3d euler_xyz_to_rot(Eigen::Vector3d euler_xyz) {
  Eigen::Matrix3d ret;
  double x = euler_xyz(0);
  double y = euler_xyz(1);
  double z = euler_xyz(2);

  ret << cos(y) * cos(z), cos(z) * sin(x) * sin(y) - cos(x) * sin(z),
      sin(x) * sin(z) + cos(x) * cos(z) * sin(y), cos(y) * sin(z),
      cos(x) * cos(z) + sin(x) * sin(y) * sin(z),
      cos(x) * sin(y) * sin(z) - cos(z) * sin(x), -sin(y), cos(y) * sin(x),
      cos(x) * cos(y);
  return ret;

  // Can be also achieved in this formulation
  // double x = euler_xyz(0);
  // double y = euler_xyz(1);
  // double z = euler_xyz(2);
  // Eigen::Matrix3d ret;
  // ret = Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ())* Eigen::AngleAxisd(y,
  // Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX());
  // return ret;
};

Eigen::Quaternion<double> euler_xyz_to_quat(Eigen::Vector3d euler_xyz) {
  Eigen::Quaternion<double> ret(euler_xyz_to_rot(euler_xyz));
  return ret;
};

// =============================================================================
// Pretty Prints
// =============================================================================
void pretty_print(Eigen::VectorXd const &vv, std::ostream &os,
                  std::string const &title, std::string const &prefix,
                  bool nonl) {
  pretty_print((Eigen::MatrixXd const &)vv, os, title, prefix, true, nonl);
}

void pretty_print(const std::vector<double> &_vec, const char *title) {
  std::printf("%s: ", title);
  for (int i(0); i < _vec.size(); ++i) {
    std::printf("% 6.4f, \t", _vec[i]);
  }
  std::printf("\n");
}

void pretty_print(const std::vector<int> &_vec, const char *title) {
  std::printf("%s: ", title);
  for (int i(0); i < _vec.size(); ++i) {
    std::printf("%d, \t", _vec[i]);
  }
  std::printf("\n");
}

void pretty_print(Eigen::MatrixXd const &mm, std::ostream &os,
                  std::string const &title, std::string const &prefix,
                  bool vecmode, bool nonl) {
  char const *nlornot("\n");
  if (nonl) {
    nlornot = "";
  }
  if (!title.empty()) {
    os << title << nlornot;
  }
  if ((mm.rows() <= 0) || (mm.cols() <= 0)) {
    os << prefix << " (empty)" << nlornot;
  } else {
    // if (mm.cols() == 1) {
    //   vecmode = true;
    // }

    if (vecmode) {
      if (!prefix.empty())
        os << prefix;
      for (int ir(0); ir < mm.rows(); ++ir) {
        os << pretty_string(mm.coeff(ir, 0));
      }
      os << nlornot;

    } else {
      for (int ir(0); ir < mm.rows(); ++ir) {
        if (!prefix.empty())
          os << prefix;
        for (int ic(0); ic < mm.cols(); ++ic) {
          os << pretty_string(mm.coeff(ir, ic));
        }
        os << nlornot;
      }
    }
  }
}

void pretty_print(Eigen::Vector3d const &vv, std::ostream &os,
                  std::string const &title, std::string const &prefix,
                  bool nonl) {
  pretty_print((Eigen::MatrixXd const &)vv, os, title, prefix, true, nonl);
}

void pretty_print(Eigen::Quaternion<double> const &qq, std::ostream &os,
                  std::string const &title, std::string const &prefix,
                  bool nonl) {
  pretty_print(qq.coeffs(), os, title, prefix, true, nonl);
}

std::string pretty_string(Eigen::VectorXd const &vv) {
  std::ostringstream os;
  pretty_print(vv, os, "", "", true);
  return os.str();
}

std::string pretty_string(Eigen::MatrixXd const &mm,
                          std::string const &prefix) {
  std::ostringstream os;
  pretty_print(mm, os, "", prefix);
  return os.str();
}

std::string pretty_string(double vv) {
  static int const buflen(32);
  static char buf[buflen];
  memset(buf, 0, sizeof(buf));
  snprintf(buf, buflen - 1, "% 6.6f  ", vv);
  std::string str(buf);
  return str;
}

// =============================================================================
// Hermite Curves
// =============================================================================

HermiteCurve::HermiteCurve() {
  p1 = 0;
  v1 = 0;
  p2 = 0;
  v2 = 0;
  s_ = 0;
  // std::cout << "[Hermite Curve] constructed" << std::endl;
}

HermiteCurve::HermiteCurve(const double &start_pos, const double &start_vel,
                           const double &end_pos, const double &end_vel)
    : p1(start_pos), v1(start_vel), p2(end_pos), v2(end_vel) {
  s_ = 0;
  // std::cout << "[Hermite Curve] constructed with values" << std::endl;
}

// Destructor
HermiteCurve::~HermiteCurve() {}

// Cubic Hermite Spline:
// From https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Unit_interval_(0,_1)
// p(s) = (2s^3 - 3s^2 + 1)*p1 + (-2*s^3 + 3*s^2)*p2 + (s^3 - 2s^2 + s)*v1 +
// (s^3 - s^2)*v2 where 0 <= s <= 1.
double HermiteCurve::evaluate(const double &s_in) {
  s_ = this->clamp(s_in);
  return p1 * (2 * std::pow(s_, 3) - 3 * std::pow(s_, 2) + 1) +
         p2 * (-2 * std::pow(s_, 3) + 3 * std::pow(s_, 2)) +
         v1 * (std::pow(s_, 3) - 2 * std::pow(s_, 2) + s_) +
         v2 * (std::pow(s_, 3) - std::pow(s_, 2));
}

double HermiteCurve::evaluateFirstDerivative(const double &s_in) {
  s_ = this->clamp(s_in);
  return p1 * (6 * std::pow(s_, 2) - 6 * s_) +
         p2 * (-6 * std::pow(s_, 2) + 6 * s_) +
         v1 * (3 * std::pow(s_, 2) - 4 * s_ + 1) +
         v2 * (3 * std::pow(s_, 2) - 2 * s_);
}

double HermiteCurve::evaluateSecondDerivative(const double &s_in) {
  s_ = this->clamp(s_in);
  return p1 * (12 * s_ - 6) + p2 * (-12 * s_ + 6) + v1 * (6 * s_ - 4) +
         v2 * (6 * s_ - 2);
}

double HermiteCurve::clamp(const double &s_in, double lo, double hi) {
  if (s_in < lo) {
    return lo;
  } else if (s_in > hi) {
    return hi;
  } else {
    return s_in;
  }
}

HermiteCurveVec::HermiteCurveVec() {}

HermiteCurveVec::HermiteCurveVec(const Eigen::VectorXd &start_pos,
                                 const Eigen::VectorXd &start_vel,
                                 const Eigen::VectorXd &end_pos,
                                 const Eigen::VectorXd &end_vel)
    : p1(start_pos), v1(start_vel), p2(end_pos), v2(end_vel) {
  initialize(start_pos, start_vel, end_pos, end_vel);
}

void HermiteCurveVec::initialize(const Eigen::VectorXd &start_pos,
                                 const Eigen::VectorXd &start_vel,
                                 const Eigen::VectorXd &end_pos,
                                 const Eigen::VectorXd &end_vel) {
  // Clear and 	create N hermite curves with the specified boundary conditions
  curves.clear();
  p1 = start_pos;
  v1 = start_vel;
  p2 = end_pos;
  v2 = end_vel;

  for (int i = 0; i < start_pos.size(); i++) {
    curves.push_back(
        HermiteCurve(start_pos[i], start_vel[i], end_pos[i], end_vel[i]));
  }
  output = Eigen::VectorXd::Zero(start_pos.size());
}

// Destructor
HermiteCurveVec::~HermiteCurveVec() {}

// Evaluation functions
Eigen::VectorXd HermiteCurveVec::evaluate(const double &s_in) {
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].evaluate(s_in);
  }
  return output;
}

Eigen::VectorXd HermiteCurveVec::evaluateFirstDerivative(const double &s_in) {
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].evaluateFirstDerivative(s_in);
  }
  return output;
}

Eigen::VectorXd HermiteCurveVec::evaluateSecondDerivative(const double &s_in) {
  for (int i = 0; i < p1.size(); i++) {
    output[i] = curves[i].evaluateSecondDerivative(s_in);
  }
  return output;
}

HermiteQuaternionCurve::HermiteQuaternionCurve() {}

HermiteQuaternionCurve::HermiteQuaternionCurve(
    const Eigen::Quaterniond &quat_start,
    const Eigen::Vector3d &angular_velocity_start,
    const Eigen::Quaterniond &quat_end,
    const Eigen::Vector3d &angular_velocity_end) {
  initialize(quat_start, angular_velocity_start, quat_end,
             angular_velocity_end);
}

void HermiteQuaternionCurve::initialize(
    const Eigen::Quaterniond &quat_start,
    const Eigen::Vector3d &angular_velocity_start,
    const Eigen::Quaterniond &quat_end,
    const Eigen::Vector3d &angular_velocity_end) {
  qa = quat_start;
  omega_a = angular_velocity_start;

  qb = quat_end;
  omega_b = angular_velocity_end;

  s_ = 0.0;
  initialize_data_structures();
}

HermiteQuaternionCurve::~HermiteQuaternionCurve() {}

void HermiteQuaternionCurve::initialize_data_structures() {
  q0 = qa;

  if (omega_a.norm() < 1e-6) {
    q1 = qa * Eigen::Quaterniond(1, 0, 0, 0);
  } else {
    q1 = qa * Eigen::Quaterniond(Eigen::AngleAxisd(
                  omega_a.norm() / 3.0,
                  omega_a / omega_a.norm())); // q1 = qa*exp(wa/3.0)
  }

  if (omega_b.norm() < 1e-6) {
    q2 = qb * Eigen::Quaterniond(1, 0, 0, 0);
  } else {
    q2 = qb * Eigen::Quaterniond(Eigen::AngleAxisd(
                  omega_b.norm() / 3.0,
                  -omega_b / omega_b.norm())); // q2 = qb*exp(wb/3.0)^-1
  }

  q3 = qb;

  omega_1aa = q1 * q0.inverse();
  omega_2aa = q2 * q1.inverse();
  omega_3aa = q3 * q2.inverse();

  omega_1 = omega_1aa.axis() * omega_1aa.angle();
  omega_2 = omega_2aa.axis() * omega_2aa.angle();
  omega_3 = omega_3aa.axis() * omega_3aa.angle();
}

void HermiteQuaternionCurve::computeBasis(const double &s_in) {
  s_ = this->clamp(s_in);
  b1 = 1 - std::pow((1 - s_), 3);
  b2 = 3 * std::pow(s_, 2) - 2 * std::pow((s_), 3);
  b3 = std::pow(s_, 3);

  bdot1 = 3 * std::pow((1 - s_), 2);
  bdot2 = 6 * s_ - 6 * std::pow((s_), 2);
  bdot3 = 3 * std::pow((s_), 2);

  bddot1 = -6 * (1 - s_);
  bddot2 = 6 - 12 * s_;
  bddot3 = 6 * s_;
}

void HermiteQuaternionCurve::evaluate(const double &s_in,
                                      Eigen::Quaterniond &quat_out) {
  s_ = this->clamp(s_in);
  computeBasis(s_);

  qtmp1 = Eigen::AngleAxisd(omega_1aa.angle() * b1, omega_1aa.axis());
  qtmp2 = Eigen::AngleAxisd(omega_2aa.angle() * b2, omega_2aa.axis());
  qtmp3 = Eigen::AngleAxisd(omega_3aa.angle() * b3, omega_3aa.axis());

  // quat_out = q0*qtmp1*qtmp2*qtmp3; // local frame
  quat_out = qtmp3 * qtmp2 * qtmp1 * q0; // global frame
}

void HermiteQuaternionCurve::getAngularVelocity(const double &s_in,
                                                Eigen::Vector3d &ang_vel_out) {
  s_ = this->clamp(s_in);
  computeBasis(s_);
  ang_vel_out = omega_1 * bdot1 + omega_2 * bdot2 + omega_3 * bdot3;
}

// For world frame
void HermiteQuaternionCurve::getAngularAcceleration(
    const double &s_in, Eigen::Vector3d &ang_acc_out) {
  s_ = this->clamp(s_in);
  computeBasis(s_);
  ang_acc_out = omega_1 * bddot1 + omega_2 * bddot2 + omega_3 * bddot3;
}

double HermiteQuaternionCurve::clamp(const double &s_in, double lo, double hi) {
  if (s_in < lo) {
    return lo;
  } else if (s_in > hi) {
    return hi;
  } else {
    return s_in;
  }
}

void HermiteQuaternionCurve::printQuat(const Eigen::Quaterniond &quat) {
  std::cout << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
            << " " << std::endl;
}

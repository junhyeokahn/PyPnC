#include <util/util.hpp>

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

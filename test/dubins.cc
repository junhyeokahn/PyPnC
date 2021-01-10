#include <dubins.hpp>
#include <iostream>
#include <stdio.h>

int printConfiguration(double q[3], double x, void *user_data) {
  printf("%f, %f, %f, %f\n", q[0], q[1], q[2], x);
  return 0;
}

int main() {
  double q0[] = {0, 0, 0};
  double q1[] = {4, 4, 3.142};
  double turning_radius = 1.0;
  DubinsPath path;
  dubins_shortest_path(&path, q0, q1, turning_radius);
  std::cout << dubins_path_length(&path) << std::endl;
  double q_res[3];
  dubins_path_sample(&path, 0.7, q_res);
  std::cout << q_res[0] << "," << q_res[1] << "," << q_res[2] << std::endl;
  return 0;
}

#include <dubins.hpp>
#include <iostream>
#include <stdio.h>

int printConfiguration(double q[3], double x, void *user_data) {
  printf("%f, %f, %f, %f\n", q[0], q[1], q[2], x);
  return 0;
}

int main() {
  double q0[] = {0, 0, 0};
  double q1[] = {0.3, -0.3, -1.57};
  double turning_radius = 0.25;
  DubinsPath path;
  dubins_shortest_path(&path, q0, q1, turning_radius);
  dubins_path_sample_many(&path, 0.1, printConfiguration, NULL);
  return 0;
}

#include <iostream>
#include <stdio.h>

#include <configuration.h>
#include <util/mlp_model.hpp>
#include <util/util.hpp>

int main(int argc, char *argv[]) {
  std::string file_path = THIS_COM "data/tf_model/atlas_crbi/mlp_model.yaml";
  YAML::Node mlp_model_cfg = YAML::LoadFile(file_path);

  MLPModel mlp_model = MLPModel(mlp_model_cfg);
  mlp_model.PrintInfo();

  Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(1, 6);
  Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(1, 6);
  Eigen::MatrixXd minus_twos = -2. * ones;

  std::cout << "zeros" << std::endl;
  std::cout << mlp_model.GetOutput(zeros) << std::endl;
  std::cout << "ones" << std::endl;
  std::cout << mlp_model.GetOutput(ones) << std::endl;
  std::cout << "-twos" << std::endl;
  std::cout << mlp_model.GetOutput(minus_twos) << std::endl;

  std::cout << "Done" << std::endl;
  return 0;
}

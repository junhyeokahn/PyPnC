#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <util/util.hpp>

enum ActivationFunction { None = 0, Tanh = 1, ReLU = 2, LeakyReLU = 3 };

class Layer {
public:
  Layer(Eigen::MatrixXd weight, Eigen::MatrixXd bias,
        ActivationFunction act_fn);
  virtual ~Layer();
  Eigen::MatrixXd GetOutput(const Eigen::MatrixXd &input);
  int GetNumInput() { return num_input_; }
  int GetNumOutput() { return num_output_; }
  Eigen::MatrixXd GetWeight() { return weight_; }
  Eigen::MatrixXd GetBias() { return bias_; }
  ActivationFunction GetActivationFunction() { return act_fn_; }

private:
  Eigen::MatrixXd weight_;
  Eigen::MatrixXd bias_;
  int num_input_;
  int num_output_;
  ActivationFunction act_fn_;
};

class MLPModel {
public:
  MLPModel(std::vector<Layer> layers);
  MLPModel(const YAML::Node &node);

  virtual ~MLPModel();

  std::vector<Layer> GetLayers() { return layers_; }
  Eigen::MatrixXd GetOutput(const Eigen::MatrixXd &input);
  Eigen::MatrixXd GetOutput(const Eigen::MatrixXd &input, int idx);

  int GetNumInput() { return num_input_; }
  int GetNumOutput() { return num_output_; }
  void PrintInfo();

private:
  void Initialize_(std::vector<Layer> layers);
  int num_input_;
  int num_output_;
  int num_layer_;
  std::vector<Layer> layers_;
};

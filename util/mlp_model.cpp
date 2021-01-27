#include <cassert>
#include <cmath>
#include <random>

#include <util/mlp_model.hpp>

Layer::Layer(Eigen::MatrixXd weight, Eigen::MatrixXd bias,
             ActivationFunction act_fn) {
  weight_ = weight;
  bias_ = bias;
  num_input_ = weight.rows();
  num_output_ = weight.cols();
  act_fn_ = act_fn;
}

Layer::~Layer() {}

Eigen::MatrixXd Layer::GetOutput(const Eigen::MatrixXd &input) {
  int num_data(input.rows());
  Eigen::MatrixXd aug_bias = Eigen::MatrixXd::Zero(num_data, num_output_);
  for (int i = 0; i < num_data; ++i) {
    aug_bias.block(i, 0, 1, num_output_) = bias_;
  }
  Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(num_data, num_output_);
  ret = input * weight_ + aug_bias;
  switch (act_fn_) {
  case ActivationFunction::None:
    return ret;
    break;
  case ActivationFunction::Tanh:
    for (int row(0); row < num_data; ++row) {
      for (int col(0); col < num_output_; ++col) {
        ret(row, col) = std::tanh(ret(row, col));
      }
    }
    return ret;
    break;
  case ActivationFunction::ReLU:
    for (int row(0); row < num_data; ++row) {
      for (int col(0); col < num_output_; ++col) {
        if (ret(row, col) < 0) {
          ret(row, col) = 0.;
        }
      }
    }
    return ret;
    break;
  default:
    break;
  }
  return ret;
}

MLPModel::MLPModel(const YAML::Node &node) {
  int num_layer;
  Eigen::MatrixXd w, b;
  std::vector<Layer> layers;
  layers.clear();
  int act_fn;

  try {
    readParameter(node, "num_layer", num_layer);
    for (int idx_layer = 0; idx_layer < num_layer; ++idx_layer) {
      readParameter(node, "w" + std::to_string(idx_layer), w);
      readParameter(node, "b" + std::to_string(idx_layer), b);
      readParameter(node, "act_fn" + std::to_string(idx_layer), act_fn);
      layers.push_back(Layer(w, b, static_cast<ActivationFunction>(act_fn)));
    }
  } catch (std::runtime_error &e) {
    std::cout << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
  }

  Initialize_(layers);
}

MLPModel::MLPModel(std::vector<Layer> layers) { Initialize_(layers); }

MLPModel::~MLPModel() {}

Eigen::MatrixXd MLPModel::GetOutput(const Eigen::MatrixXd &input, int ith) {
  Eigen::MatrixXd ret = input;
  if (ith > num_layer_) {
    std::cout << "[[Error]] Wrong output layer idx!!" << std::endl;
    exit(0);
  }

  for (int i = 0; i < ith; ++i) {
    ret = (layers_[i]).GetOutput(ret);
  }

  return ret;
}

Eigen::MatrixXd MLPModel::GetOutput(const Eigen::MatrixXd &input) {
  int num_data(input.rows());
  Eigen::MatrixXd ret = input;
  for (int i = 0; i < num_layer_; ++i) {
    ret = (layers_[i]).GetOutput(ret);
  }

  return ret;
}

void MLPModel::Initialize_(std::vector<Layer> layers) {
  layers_ = layers;
  num_layer_ = layers.size();
  num_input_ = layers[0].GetNumInput();
  num_output_ = layers.back().GetNumOutput();
}

void MLPModel::PrintInfo() {
  std::cout << "Num Layer : " << num_layer_ << std::endl;
  for (int i = 0; i < num_layer_; ++i) {
    std::cout << i << " th Layer : (" << layers_[i].GetNumInput() << ", "
              << layers_[i].GetNumOutput() << ") with Activation type::"
              << layers_[i].GetActivationFunction() << std::endl;
    pretty_print(layers_[i].GetWeight(), std::cout, "w");
    pretty_print(layers_[i].GetBias(), std::cout, "b");
  }
}

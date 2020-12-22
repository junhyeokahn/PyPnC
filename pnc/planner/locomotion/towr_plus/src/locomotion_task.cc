#include <towr_plus/locomotion_task.h>
#include <towr_plus/terrain/examples/height_map_examples.h>

LocomotionTask::LocomotionTask(const std::string &name) {
  name_ = name;
  num_leg_ = 2;
  initial_ee_motion_lin.resize(num_leg_);
  initial_ee_motion_ang.resize(num_leg_);
  ee_phase_durations.resize(num_leg_);
  ee_in_contact_at_start.resize(num_leg_);
}

LocomotionTask::~LocomotionTask() {}

void LocomotionTask::from_yaml(const YAML::Node &node) {
  std::string terrain_type, robot_name;
  Eigen::VectorXd tmp_vec;
  bool tmp_bool;
  try {
    readParameter(node, "initial_base_lin", initial_base_lin);
    readParameter(node, "initial_base_ang", initial_base_ang);
    for (auto ee : {L, R}) {
      readParameter(node["initial_ee_motion_lin"], std::to_string(ee), tmp_vec);
      for (auto dim : {X, Y, Z})
        initial_ee_motion_lin.at(ee)(dim) = tmp_vec(dim);
      readParameter(node["initial_ee_motion_ang"], std::to_string(ee), tmp_vec);
      for (auto dim : {X, Y, Z})
        initial_ee_motion_ang.at(ee)(dim) = tmp_vec(dim);
      readParameter(node["ee_phase_durations"], std::to_string(ee), tmp_vec);
      for (int i = 0; i < tmp_vec.size(); ++i)
        ee_phase_durations.at(ee).push_back(tmp_vec(i));
      readParameter(node["ee_in_contact_at_start"], std::to_string(ee),
                    tmp_bool);
      ee_in_contact_at_start.at(ee) = tmp_bool;
    }
    readParameter(node, "final_base_lin", final_base_lin);
    readParameter(node, "final_base_ang", final_base_ang);

    readParameter(node, "terrain_type", terrain_type);
    readParameter(node, "robot_name", robot_name);

  } catch (std::runtime_error &e) {
    std::cout << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
  }

  if (terrain_type == "flat_ground") {
    terrain = std::make_shared<FlatGround>(0.);
  } else {
    std::cout << "Wrong Terrain Type" << std::endl;
    exit(0);
  }

  if (robot_name == "atlas") {
    robot_model = RobotModel(RobotModel::Atlas);
  } else {
    std::cout << "Wrong Robot Model" << std::endl;
    exit(0);
  }
}

void LocomotionTask::from_one_hot_vector(const Eigen::VectorXd &one_hot_vec) {}

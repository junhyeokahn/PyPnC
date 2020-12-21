#include <towr_plus/locomotion_task.h>
#include <towr_plus/terrain/examples/height_map_examples.h>

LocomotionTask::LocomotionTask(const std::string &name) : name_(name) {}

LocomotionTask::~LocomotionTask() {}

void LocomotionTask::set_from_yaml(const YAML::Node &node) {
  std::string terrain_type, robot_name;
  try {
    readParameter(node, "initial_base_lin", initial_base_lin);
    readParameter(node, "initial_base_ang", initial_base_ang);
    readParameter(node, "initial_ee_motion_lin", initial_ee_motion_lin);
    readParameter(node, "initial_ee_motion_ang", initial_ee_motion_ang);

    readParameter(node, "ee_phase_durations", ee_phase_durations);
    readParameter(node, "ee_in_contact_at_start", ee_in_contact_at_start);

    readParameter(node, "final_base_lin", final_base_lin);
    readParameter(node, "final_base_ang", final_base_ang);

    readParameter(node, "terrain_type", terrain_type);
    readParameter(node, "robot_name", robot_name);

  } catch (std::runtime_error &e) {
    std::cout << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
  }

  if (terrain_type.compare("flat_ground")) {
    terrain = std::make_shared<FlatGround>(0.);
  } else {
    std::cout << "Wrong Terrain Type" << std::endl;
    exit(0);
  }

  if (robot_name.compare("atlas")) {
    robot_model = RobotModel(RobotModel::Atlas);
  } else {
    std::cout << "Wrong Robot Model" << std::endl;
    exit(0);
  }
}

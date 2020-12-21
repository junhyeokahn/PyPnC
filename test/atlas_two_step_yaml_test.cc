#include <configuration.h>
#include <towr_plus/locomotion_task.h>
#include <towr_plus/nlp_formulation.h>

int main() {
  // Locomotion Task
  YAML::Node cfg =
      YAML::LoadFile(THIS_COM "config/towr_plus/atlas_two_step.yaml");
  LocomotionTask task = LocomotionTask("atlas_two_step_yaml_test");
  task.set_from_yaml(cfg["locomotion_task"]);

  // Construct NLP from locomotion task
  NlpFormulation formulation;
  formulation.terrain_ = task.terrain;
  formulation.model_ = task.robot_model;

  formulation.initial_base_.ang.at(kPos) = task.initial_base_lin.segment(0, 3);
  formulation.initial_base_.lin.at(kPos) = task.initial_base_lin.segment(3, 3);
  formulation.initial_base_.ang.at(kVel) = task.initial_base_ang.segment(0, 3);
  formulation.initial_base_.lin.at(kVel) = task.initial_base_ang.segment(3, 3);

  formulation.initial_ee_W_ = task.initial_ee_motion_lin;
  // TODO(JH): initial_ee_motion_ang
  formulation.params_.ee_phase_durations_ = task.ee_phase_durations;
  formulation.params_.ee_in_contact_at_start_ = task.ee_in_contact_at_start;

  formulation.final_base_.ang.at(kPos) = task.final_base_lin.segment(0, 3);
  formulation.final_base_.lin.at(kPos) = task.final_base_lin.segment(3, 3);
  formulation.final_base_.ang.at(kVel) = task.final_base_ang.segment(0, 3);
  formulation.final_base_.lin.at(kVel) = task.final_base_ang.segment(3, 3);

  return 0;
}

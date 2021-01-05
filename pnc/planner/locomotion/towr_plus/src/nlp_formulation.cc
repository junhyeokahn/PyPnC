/******************************************************************************
Copyright (c) 2018, Alexander W. Winkler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

/******************************************************************************
Modified by Junhyeok Ahn (junhyeokahn91@gmail.com) for towr+
******************************************************************************/

#include <towr_plus/constraints/base_motion_constraint.h>
#include <towr_plus/constraints/dynamic_constraint.h>
#include <towr_plus/constraints/force_constraint.h>
#include <towr_plus/constraints/range_of_motion_constraint.h>
#include <towr_plus/constraints/spline_acc_constraint.h>
#include <towr_plus/constraints/swing_constraint.h>
#include <towr_plus/constraints/terrain_constraint.h>
#include <towr_plus/constraints/total_duration_constraint.h>
#include <towr_plus/costs/final_node_cost.h>
#include <towr_plus/costs/intermediate_node_cost.h>
#include <towr_plus/costs/node_cost.h>
#include <towr_plus/costs/node_difference_cost.h>
#include <towr_plus/nlp_formulation.h>
#include <towr_plus/variables/nodes_variables_all.h>
#include <towr_plus/variables/phase_durations.h>
#include <towr_plus/variables/variable_names.h>

#include <iostream>

namespace towr_plus {
NlpFormulation::NlpFormulation() {
  using namespace std;
  cout << "\n";
  cout << "********************************************************************"
          "**********\n";
  cout << "                                    TOWR+ \n";
  cout << "                               \u00a9 Junhyeok Ahn \n";
  cout << "********************************************************************"
          "**********";
  cout << "\n\n";
}

NlpFormulation::VariablePtrVec
NlpFormulation::GetVariableSets(SplineHolder &spline_holder) {
  VariablePtrVec vars;

  auto base_motion = MakeBaseVariables();
  vars.insert(vars.end(), base_motion.begin(), base_motion.end());

  auto ee_motion_lin = MakeEEMotionLinVariables();
  vars.insert(vars.end(), ee_motion_lin.begin(), ee_motion_lin.end());

  auto ee_motion_ang = MakeEEMotionAngVariables();
  vars.insert(vars.end(), ee_motion_ang.begin(), ee_motion_ang.end());

  auto ee_wrench_lin = MakeWrenchLinVariables();
  vars.insert(vars.end(), ee_wrench_lin.begin(), ee_wrench_lin.end());

  auto ee_wrench_ang = MakeWrenchAngVariables();
  vars.insert(vars.end(), ee_wrench_ang.begin(), ee_wrench_ang.end());

  auto contact_schedule = MakeContactScheduleVariables();
  // can also just be fixed timings that aren't optimized over, but still
  // added to spline_holder.
  if (params_.IsOptimizeTimings()) {
    vars.insert(vars.end(), contact_schedule.begin(), contact_schedule.end());
  }

  // stores these readily constructed spline
  spline_holder = SplineHolder(base_motion.at(0), // linear
                               base_motion.at(1), // angular
                               params_.GetBasePolyDurations(),
                               ee_motion_lin, // linear
                               ee_motion_ang, // angular
                               ee_wrench_lin, // linear
                               ee_wrench_ang, // angular
                               contact_schedule, params_.IsOptimizeTimings());
  return vars;
}

std::vector<NodesVariables::Ptr> NlpFormulation::MakeBaseVariables() const {
  std::vector<NodesVariables::Ptr> vars;

  int n_nodes = params_.GetBasePolyDurations().size() + 1;

  auto spline_lin =
      std::make_shared<NodesVariablesAll>(n_nodes, k3D, id::base_lin_nodes);

  double x = final_base_.lin.p().x();
  double y = final_base_.lin.p().y();
  double z = terrain_->GetHeight(x, y) -
             model_.kinematic_model_->GetNominalStanceInBase().front().z();
  Vector3d final_pos(x, y, z);

  spline_lin->SetByLinearInterpolation(initial_base_.lin.p(), final_pos,
                                       params_.GetTotalTime());
  spline_lin->AddStartBound(kPos, {X, Y, Z}, initial_base_.lin.p());
  spline_lin->AddStartBound(kVel, {X, Y, Z}, initial_base_.lin.v());
  vars.push_back(spline_lin);

  auto spline_ang =
      std::make_shared<NodesVariablesAll>(n_nodes, k3D, id::base_ang_nodes);
  spline_ang->SetByLinearInterpolation(
      initial_base_.ang.p(), final_base_.ang.p(), params_.GetTotalTime());
  spline_ang->AddStartBound(kPos, {X, Y, Z}, initial_base_.ang.p());
  spline_ang->AddStartBound(kVel, {X, Y, Z}, initial_base_.ang.v());
  vars.push_back(spline_ang);

  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeEEMotionLinVariables() const {
  std::vector<NodesVariablesPhaseBased::Ptr> vars;

  double T = params_.GetTotalTime();
  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto lin_nodes = std::make_shared<NodesVariablesEEMotion>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        id::EEMotionLinNodes(ee), params_.ee_polynomials_per_swing_phase_,
        true);

    // initialize towards final footholds
    Eigen::Vector3d final_base_euler(final_base_.ang.p().x(),
                                     final_base_.ang.p().y(),
                                     final_base_.ang.p().z());
    Eigen::Matrix3d w_R_b =
        EulerConverter::GetRotationMatrixBaseToWorld(final_base_euler);
    Vector3d final_ee_motion_lin =
        final_base_.lin.p() +
        w_R_b * model_.kinematic_model_->GetNominalStanceInBase().at(ee);
    double x = final_ee_motion_lin.x();
    double y = final_ee_motion_lin.y();
    double z = terrain_->GetHeight(x, y);
    lin_nodes->SetByLinearInterpolation(initial_ee_motion_lin_.at(ee),
                                        Vector3d(x, y, z), T);

    lin_nodes->AddStartBound(kPos, {X, Y, Z}, initial_ee_motion_lin_.at(ee));
    vars.push_back(lin_nodes);
  }

  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeEEMotionAngVariables() const {
  std::vector<NodesVariablesPhaseBased::Ptr> vars;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto ang_nodes = std::make_shared<NodesVariablesEEMotion>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        id::EEMotionAngNodes(ee), params_.ee_polynomials_per_swing_phase_,
        false);

    // compute final footholds orientation
    Eigen::Vector3d base_euler_xyz(final_base_.ang.p().x(),
                                   final_base_.ang.p().y(),
                                   final_base_.ang.p().z());
    Eigen::Matrix3d base_rot = euler_xyz_to_rot(base_euler_xyz);
    Vector3d final_ee_motion_lin =
        final_base_.lin.p() +
        base_rot * model_.kinematic_model_->GetNominalStanceInBase().at(ee);
    double x = final_ee_motion_lin(0);
    double y = final_ee_motion_lin(1);
    Eigen::Vector3d foot_x_axis =
        terrain_->GetProjectionToTerrain(x, y, base_rot.col(X), true);
    Eigen::Vector3d foot_z_axis =
        terrain_->GetNormalizedBasis(HeightMap::Normal, x, y);
    Eigen::Vector3d foot_y_axis = foot_z_axis.cross(foot_x_axis);
    Eigen::Matrix3d foot_rot;
    foot_rot.col(0) = foot_x_axis;
    foot_rot.col(1) = foot_y_axis;
    foot_rot.col(2) = foot_z_axis;
    std::cout << "foot_rot_final" << std::endl;
    std::cout << foot_rot << std::endl;
    exit(0);
    Eigen::Vector3d final_foot_euler_angles;
    // Little detail: Eigen returns zyx order
    final_foot_euler_angles << foot_rot.eulerAngles(2, 1, 0)(2),
        foot_rot.eulerAngles(2, 1, 0)(1), foot_rot.eulerAngles(2, 1, 0)(0);

    ang_nodes->SetByLinearInterpolation(initial_ee_motion_ang_.at(ee),
                                        final_foot_euler_angles,
                                        params_.GetTotalTime());
    ang_nodes->AddStartBound(kPos, {X, Y, Z}, initial_ee_motion_ang_.at(ee));
    vars.push_back(ang_nodes);
  }
  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeWrenchLinVariables() const {
  std::vector<NodesVariablesPhaseBased::Ptr> vars;
  double T = params_.GetTotalTime();
  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto lin_nodes = std::make_shared<NodesVariablesEEForce>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        id::EEWrenchLinNodes(ee), params_.force_polynomials_per_stance_phase_);

    double m = model_.dynamic_model_->m();
    double g = model_.dynamic_model_->g();

    Vector3d f_stance(0.0, 0.0, m * g / params_.GetEECount());
    lin_nodes->SetByLinearInterpolation(f_stance, f_stance,
                                        T); // stay constant
    vars.push_back(lin_nodes);
  }
  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeWrenchAngVariables() const {
  std::vector<NodesVariablesPhaseBased::Ptr> vars;
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    auto ang_nodes = std::make_shared<NodesVariablesEEForce>(
        params_.GetPhaseCount(ee), params_.ee_in_contact_at_start_.at(ee),
        id::EEWrenchAngNodes(ee), params_.force_polynomials_per_stance_phase_);
    vars.push_back(ang_nodes);
  }
  return vars;
}

std::vector<PhaseDurations::Ptr>
NlpFormulation::MakeContactScheduleVariables() const {
  std::vector<PhaseDurations::Ptr> vars;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto var =
        std::make_shared<PhaseDurations>(ee, params_.ee_phase_durations_.at(ee),
                                         params_.ee_in_contact_at_start_.at(ee),
                                         params_.bound_phase_duration_.first,
                                         params_.bound_phase_duration_.second);
    vars.push_back(var);
  }

  return vars;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::GetConstraints(const SplineHolder &spline_holder) const {
  ContraintPtrVec constraints;
  for (auto name : params_.constraints_)
    for (auto c : GetConstraint(name, spline_holder))
      constraints.push_back(c);

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::GetConstraint(Parameters::ConstraintName name,
                              const SplineHolder &s) const {
  switch (name) {
  case Parameters::Dynamic:
    return MakeDynamicConstraint(s);
  case Parameters::EndeffectorRom:
    return MakeRangeOfMotionBoxConstraint(s);
  case Parameters::BaseRom:
    return MakeBaseRangeOfMotionConstraint(s);
  case Parameters::TotalTime:
    return MakeTotalTimeConstraint();
  case Parameters::Terrain:
    return MakeTerrainConstraint();
  case Parameters::Force:
    return MakeForceConstraint();
  case Parameters::Swing:
    return MakeSwingConstraint();
  case Parameters::BaseAcc:
    return MakeBaseAccConstraint(s);
  default:
    throw std::runtime_error("constraint not defined!");
  }
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeBaseRangeOfMotionConstraint(const SplineHolder &s) const {
  return {std::make_shared<BaseMotionConstraint>(
      params_.GetTotalTime(), params_.dt_constraint_base_motion_, s)};
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeDynamicConstraint(const SplineHolder &s) const {
  auto constraint = std::make_shared<DynamicConstraint>(
      model_.dynamic_model_, params_.GetTotalTime(),
      params_.dt_constraint_dynamic_, s);
  return {constraint};
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeRangeOfMotionBoxConstraint(const SplineHolder &s) const {
  ContraintPtrVec c;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto rom = std::make_shared<RangeOfMotionConstraint>(
        model_.kinematic_model_, params_.GetTotalTime(),
        params_.dt_constraint_range_of_motion_, ee, s);
    c.push_back(rom);
  }

  return c;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeTotalTimeConstraint() const {
  ContraintPtrVec c;
  double T = params_.GetTotalTime();

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto duration_constraint = std::make_shared<TotalDurationConstraint>(T, ee);
    c.push_back(duration_constraint);
  }

  return c;
}

NlpFormulation::ContraintPtrVec NlpFormulation::MakeTerrainConstraint() const {
  ContraintPtrVec constraints;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto c = std::make_shared<TerrainConstraint>(
        terrain_, id::EEMotionLinNodes(ee), id::EEMotionAngNodes(ee));
    constraints.push_back(c);
  }

  return constraints;
}

NlpFormulation::ContraintPtrVec NlpFormulation::MakeForceConstraint() const {
  ContraintPtrVec constraints;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto c = std::make_shared<ForceConstraint>(
        terrain_, params_.force_limit_in_normal_direction_, ee);
    constraints.push_back(c);
  }

  return constraints;
}

NlpFormulation::ContraintPtrVec NlpFormulation::MakeSwingConstraint() const {
  ContraintPtrVec constraints;

  for (int ee = 0; ee < params_.GetEECount(); ee++) {
    auto swing = std::make_shared<SwingConstraint>(id::EEMotionLinNodes(ee));
    constraints.push_back(swing);
  }

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeBaseAccConstraint(const SplineHolder &s) const {
  ContraintPtrVec constraints;

  constraints.push_back(std::make_shared<SplineAccConstraint>(
      s.base_linear_, id::base_lin_nodes));

  constraints.push_back(std::make_shared<SplineAccConstraint>(
      s.base_angular_, id::base_ang_nodes));

  return constraints;
}

NlpFormulation::ContraintPtrVec NlpFormulation::GetCosts() const {
  ContraintPtrVec costs;
  for (const auto &pair : params_.costs_)
    for (auto c : GetCost(pair.first, pair.second))
      costs.push_back(c);

  return costs;
}

NlpFormulation::CostPtrVec
NlpFormulation::GetCost(const Parameters::CostName &name,
                        const Eigen::VectorXd weight) const {
  switch (name) {
  case Parameters::FinalBaseLinPosCost:
    return MakeFinalBaseLinCost(Dx::kPos, weight);
  case Parameters::FinalBaseLinVelCost:
    return MakeFinalBaseLinCost(Dx::kVel, weight);
  case Parameters::FinalBaseAngPosCost:
    return MakeFinalBaseAngCost(Dx::kPos, weight);
  case Parameters::FinalBaseAngVelCost:
    return MakeFinalBaseAngCost(Dx::kVel, weight);
  case Parameters::FinalEEMotionLinPosCost:
    return MakeFinalEEMotionLinPosCost(weight);
  case Parameters::FinalEEMotionAngPosCost:
    return MakeFinalEEMotionAngPosCost(weight);
  case Parameters::IntermediateBaseLinVelCost:
    return MakeIntermediateBaseLinCost(Dx::kVel, weight);
  case Parameters::IntermediateBaseAngVelCost:
    return MakeIntermediateBaseAngCost(Dx::kVel, weight);
  case Parameters::BaseLinVelDiffCost:
    return MakeBaseLinVelDiffCost(weight);
  case Parameters::BaseAngVelDiffCost:
    return MakeBaseAngVelDiffCost(weight);
  case Parameters::WrenchLinPosCost:
    return MakeWrenchLinCost(Dx::kPos, weight);
  case Parameters::WrenchLinVelCost:
    return MakeWrenchLinCost(Dx::kVel, weight);
  case Parameters::WrenchAngPosCost:
    return MakeWrenchAngCost(Dx::kPos, weight);
  case Parameters::WrenchAngVelCost:
    return MakeWrenchAngCost(Dx::kVel, weight);
  case Parameters::WrenchLinVelDiffCost:
    return MakeWrenchLinVelDiffCost(weight);
  case Parameters::WrenchAngVelDiffCost:
    return MakeWrenchAngVelDiffCost(weight);
  default:
    throw std::runtime_error("cost not defined!");
  }
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeBaseLinVelDiffCost(const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // X, Y, Z
  for (int i = 0; i < 3; ++i) {
    cost.push_back(std::make_shared<NodeDifferenceCost>(
        id::base_lin_nodes, Dx::kVel, i, weight(i)));
  }
  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeBaseAngVelDiffCost(const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // X, Y, Z
  for (int i = 0; i < 3; ++i) {
    cost.push_back(std::make_shared<NodeDifferenceCost>(
        id::base_ang_nodes, Dx::kVel, i, weight(i)));
  }
  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeWrenchLinVelDiffCost(const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // X, Y, Z
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<NodeDifferenceCost>(
          id::EEWrenchLinNodes(ee), Dx::kVel, i, weight(i)));
    }
  }
  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeWrenchAngVelDiffCost(const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // X, Y, Z
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<NodeDifferenceCost>(
          id::EEWrenchAngNodes(ee), Dx::kVel, i, weight(i)));
    }
  }
  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeWrenchLinCost(Dx dx, const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  double mg = model_.dynamic_model_->m() * model_.dynamic_model_->g();
  // For all endeffector
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    if (dx == Dx::kPos) {
      for (int i = 0; i < 3; ++i) {
        cost.push_back(std::make_shared<NodeCost>(
            id::EEWrenchLinNodes(ee), dx, i, weight(i) / std::pow(mg, 2), 0.));
      }
    } else if (dx == Dx::kVel) {
      for (int i = 0; i < 3; ++i) {
        cost.push_back(std::make_shared<NodeCost>(id::EEWrenchLinNodes(ee), dx,
                                                  i, weight(i), 0.));
      }
    } else {
      throw std::runtime_error("[MakeWrenchLinCost] Wrong dx type");
    }
  }
  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeWrenchAngCost(Dx dx, const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  double mg = model_.dynamic_model_->m() * model_.dynamic_model_->g();
  // For all endeffector
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    if (dx == Dx::kPos) {
      for (int i = 0; i < 3; ++i) {
        cost.push_back(std::make_shared<NodeCost>(
            id::EEWrenchAngNodes(ee), dx, i, weight(i) / std::pow(mg, 2), 0.));
      }
    } else if (dx == Dx::kVel) {
      for (int i = 0; i < 3; ++i) {
        cost.push_back(std::make_shared<NodeCost>(id::EEWrenchAngNodes(ee), dx,
                                                  i, weight(i), 0.));
      }
    } else {
      throw std::runtime_error("[MakeWrenchAngCost] Wrong dx type");
    }
  }
  return cost;
}

NlpFormulation::CostPtrVec NlpFormulation::MakeFinalEEMotionLinPosCost(
    const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // For all endeffector
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    double x = final_base_.lin.p().x() +
               model_.kinematic_model_->GetNominalStanceInBase().at(ee)(0);
    double y = final_base_.lin.p().y() +
               model_.kinematic_model_->GetNominalStanceInBase().at(ee)(1);
    double z = terrain_->GetHeight(x, y);
    Eigen::Vector3d final_ee_motion_lin_pos(x, y, z);
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(
          id::EEMotionLinNodes(ee), kPos, i, weight(i),
          final_ee_motion_lin_pos(i)));
    }
  }

  return cost;
}

NlpFormulation::CostPtrVec NlpFormulation::MakeFinalEEMotionAngPosCost(
    const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  // For all endeffector
  for (int ee = 0; ee < params_.GetEECount(); ++ee) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(id::EEMotionAngNodes(ee),
                                                     kPos, i, weight(i),
                                                     final_base_.ang.p()(i)));
    }
  }

  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeFinalBaseLinCost(Dx dx,
                                     const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  if (dx == Dx::kPos) {
    double x = final_base_.lin.p().x();
    double y = final_base_.lin.p().y();
    double z = terrain_->GetHeight(x, y) -
               model_.kinematic_model_->GetNominalStanceInBase().front().z();
    Eigen::Vector3d final_base_lin_pos(x, y, z);
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(
          id::base_lin_nodes, dx, i, weight(i), final_base_lin_pos(i)));
    }
  } else if (dx == Dx::kVel) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(
          id::base_lin_nodes, dx, i, weight(i), final_base_.lin.v()(i)));
    }
  } else {
    throw std::runtime_error("[MakeFinalBaseLinCost] Wrong dx type");
  }

  return cost;
}

NlpFormulation::CostPtrVec NlpFormulation::MakeIntermediateBaseLinCost(
    Dx dx, const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  if (dx == Dx::kPos) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<IntermediateNodeCost>(
          id::base_lin_nodes, dx, i, weight(i),
          0.5 * (initial_base_.lin.p()(i) + final_base_.lin.p()(i))));
    }
  } else if (dx == Dx::kVel) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<IntermediateNodeCost>(
          id::base_lin_nodes, dx, i, weight(i), 0.));
    }
  } else {
    throw std::runtime_error("[MakeIntermediateBaseLinCost] Wrong dx type");
  }

  return cost;
}

NlpFormulation::CostPtrVec
NlpFormulation::MakeFinalBaseAngCost(Dx dx,
                                     const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  if (dx == Dx::kPos) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(
          id::base_ang_nodes, dx, i, weight(i), final_base_.ang.p()(i)));
    }
  } else if (dx == Dx::kVel) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<FinalNodeCost>(
          id::base_ang_nodes, dx, i, weight(i), final_base_.ang.v()(i)));
    }
  } else {
    throw std::runtime_error("[MakeFinalBaseAngCost] Wrong dx type");
  }

  return cost;
}

NlpFormulation::CostPtrVec NlpFormulation::MakeIntermediateBaseAngCost(
    Dx dx, const Eigen::VectorXd &weight) const {
  CostPtrVec cost;
  if (dx == Dx::kPos) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<IntermediateNodeCost>(
          id::base_ang_nodes, dx, i, weight(i),
          0.5 * (initial_base_.ang.p()(i) + final_base_.ang.p()(i))));
    }
  } else if (dx == Dx::kVel) {
    for (int i = 0; i < 3; ++i) {
      cost.push_back(std::make_shared<IntermediateNodeCost>(
          id::base_ang_nodes, dx, i, weight(i), 0.));
    }
  } else {
    throw std::runtime_error("[MakeIntermediateBaseAngCost] Wrong dx type");
  }

  return cost;
}

void NlpFormulation::from_locomotion_task(const LocomotionTask &task) {
  terrain_ = task.terrain;

  initial_base_.lin.at(kPos) = task.initial_base_lin.segment(0, 3);
  initial_base_.lin.at(kVel) = task.initial_base_lin.segment(3, 3);
  initial_base_.ang.at(kPos) = task.initial_base_ang.segment(0, 3);
  initial_base_.ang.at(kVel) = task.initial_base_ang.segment(3, 3);

  initial_ee_motion_lin_ = task.initial_ee_motion_lin;
  initial_ee_motion_ang_ = task.initial_ee_motion_ang;

  final_base_.lin.at(kPos) = task.final_base_lin.segment(0, 3);
  final_base_.lin.at(kVel) = task.final_base_lin.segment(3, 3);
  final_base_.ang.at(kPos) = task.final_base_ang.segment(0, 3);
  final_base_.ang.at(kVel) = task.final_base_ang.segment(3, 3);
}

} /* namespace towr_plus */

#pragma once

#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/models/kinematic_model.h>
#include <towr_plus/models/single_rigid_body_dynamics.h>

namespace towr_plus {

class AtlasKinematicModel : public KinematicModel {
public:
  AtlasKinematicModel() : KinematicModel(2) {
    const double x_nominal_b = -0.008;
    const double y_nominal_b = 0.111;
    const double z_nominal_b = -0.765;

    foot_half_length_ = 0.11;
    foot_half_width_ = 0.065;

    nominal_stance_.at(L) << x_nominal_b, y_nominal_b, z_nominal_b;
    nominal_stance_.at(R) << x_nominal_b, -y_nominal_b, z_nominal_b;

    max_dev_from_nominal_ << 0.18, 0.1, 0.05;
    min_dev_from_nominal_ << -0.18, -0.1, -0.05;
  }
};

// class AtlasDynamicModel : public SingleRigidBodyDynamics {
// public:
// AtlasDynamicModel()
//: SingleRigidBodyDynamics(98.4068, 34., 27.5, 14.4, 0.15, 4.1, -0.06, 2) {
//}
//};

class AtlasDynamicModel : public CompositeRigidBodyDynamics {
public:
  AtlasDynamicModel()
      : CompositeRigidBodyDynamics(
            98.4068, THIS_COM "data/tf_model/atlas_crbi/mlp_model.yaml",
            THIS_COM "data/tf_model/atlas_crbi/data_stat.yaml", 2) {}
};

} /* namespace towr_plus */

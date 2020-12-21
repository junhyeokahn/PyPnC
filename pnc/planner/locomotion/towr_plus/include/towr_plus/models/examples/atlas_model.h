#pragma once

#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/models/kinematic_model.h>
#include <towr_plus/models/single_rigid_body_dynamics.h>

namespace towr_plus {

class AtlasKinematicModel : public KinematicModel {
   public:
    AtlasKinematicModel() : KinematicModel(2) {
        const double x_nominal_b = -0.008;
        const double z_nominal_b = -0.765;
        const double y_nominal_b = 0.111;

        nominal_stance_.at(L) << x_nominal_b, y_nominal_b, z_nominal_b;
        nominal_stance_.at(R) << x_nominal_b, -y_nominal_b, z_nominal_b;

        max_dev_from_nominal_ << 0.45, 0.25, 0.25;
    }
};

class AtlasDynamicModel : public SingleRigidBodyDynamics {
   public:
    /* Atlas Reduced model
     Mass:
     98.4068
     Inertia:
        4.48975 -0.0282483   0.386339
     -0.0282483    4.62886  0.0325983
       0.386339  0.0325983   0.830916
    */
    AtlasDynamicModel()
        : SingleRigidBodyDynamics(100, 4.49, 4.62, 0.83, -0.03, 0.39, 0.03, 2) {
    }
};

} /* namespace towr_plus */

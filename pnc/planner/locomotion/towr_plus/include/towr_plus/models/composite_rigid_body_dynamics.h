/******************************************************************************
Written by Junhyeok Ahn (junhyeokahn91@gmail.com) for towr+
******************************************************************************/

#pragma once

#include "dynamic_model.h"

class CompositeRigidBodyInertia;

namespace towr_plus {

class CompositeRigidBodyDynamics : public DynamicModel {
public:
  CompositeRigidBodyDynamics(double mass, std::string model_path,
                             std::string data_stat_path, int ee_count);

  virtual ~CompositeRigidBodyDynamics();

  BaseAcc GetDynamicViolation() const override;

  Jac GetJacobianWrtBaseLin(const Jac &jac_base_lin_pos,
                            const Jac &jac_acc_base_lin) const override;
  Jac GetJacobianWrtBaseAng(const EulerConverter &base_angular,
                            double t) const override;
  Jac GetJacobianWrtForce(const Jac &jac_force, EE) const override;

  Jac GetJacobianWrtEEPos(const Jac &jac_ee_pos, EE) const override;

private:
  /** Inertia of entire robot around the CoM expressed in a frame anchored
   *  in the base.
   */
  CompositeRigidBodyInertia *crbi_;
};

} /* namespace towr_plus */

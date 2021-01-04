/******************************************************************************
Written by Junhyeok Ahn (junhyeokahn91@gmail.com) for towr+
******************************************************************************/

#pragma once

#include "dynamic_model.h"

namespace towr_plus {

class CompositeRigidBodyDynamics : public DynamicModel {
public:
  CompositeRigidBodyDynamics(double mass, const Eigen::Matrix3d &inertia_b,
                             int ee_count);

  CompositeRigidBodyDynamics(double mass, double Ixx, double Iyy, double Izz,
                             double Ixy, double Ixz, double Iyz, int ee_count);

  virtual ~CompositeRigidBodyDynamics() = default;

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
  Eigen::SparseMatrix<double, Eigen::RowMajor> I_b;
};

} /* namespace towr_plus */

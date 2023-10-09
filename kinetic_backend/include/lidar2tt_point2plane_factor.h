#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>
#include <boost/optional.hpp>

#include "geometry/surfel3.h"


using gtsam::Key;
using gtsam::NoiseModelFactor3;
using gtsam::Pose3;
using gtsam::Point3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;

namespace kinetic_backend {

class LiDAR2TtPoint2PlaneFactor : public NoiseModelFactor3<Pose3, Pose3, Pose3> {

public:
  LiDAR2TtPoint2PlaneFactor(Key key_sensor2tt, Key key_target2tt, Key key_tt2tt0,
                            const Point3& measured, const Surfel3& surfel,
                            const SharedNoiseModel& model, bool fix_sensor2tt=false,
                            bool fix_target2tt=false, bool fix_tt2tt0=false):
    NoiseModelFactor3<Pose3, Pose3, Pose3>(
      model,
      key_sensor2tt,
      key_target2tt,
      key_tt2tt0
    ),
    measured_(measured), surfel_(surfel), fix_sensor2tt_(fix_sensor2tt),
    fix_target2tt_(fix_target2tt), fix_tt2tt0_(fix_tt2tt0) {}

  gtsam::Vector getError(const gtsam::Point3& point_in_target, const Surfel3& surfel_in_target) const {
    return gtsam::Vector1(surfel_in_target.Distance(point_in_target));
  }

  Vector evaluateError(const Pose3& sensor2tt0,
                       const Pose3& target2tt,
                       const Pose3& tt2tt0,
                       boost::optional<Matrix&> H_sensor2tt = boost::none,
                       boost::optional<Matrix&> H_target2tt = boost::none,
                       boost::optional<Matrix&> H_tt2tt0 = boost::none) const override
  {
    // tmp jacobian
    Matrix D_tt02tt_inv, D_tt2target_inv, D_sensor2tt_tt02tt, D_sensor2tt_sensor2tt0;
    Matrix D_sensor2target_tt2target, D_sensor2target_sensor2tt, D_proj_sensor2target;
    Matrix D_pt_sensor2target;

    // pose3 transform 
    Pose3 tt02tt = tt2tt0.inverse(D_tt02tt_inv);
    Pose3 tt2target = target2tt.inverse(D_tt2target_inv);
    Pose3 sensor2tt = tt02tt.compose(sensor2tt0, D_sensor2tt_tt02tt, D_sensor2tt_sensor2tt0);
    Pose3 sensor2target = tt2target.compose(sensor2tt, D_sensor2target_tt2target, D_sensor2target_sensor2tt);
    
    // compute the point to plane error
    Point3 point_target = sensor2target.transformFrom(measured_, D_pt_sensor2target, boost::none);
    // compute numerical derivative of error wrt point in target
    Matrix D_error_point = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Point3>(
      boost::bind(&LiDAR2TtPoint2PlaneFactor::getError, this, _1, surfel_), point_target
    );


    if (H_sensor2tt) {
      if (fix_sensor2tt_) {
        *H_sensor2tt = gtsam::Matrix16::Zero();
      } else {
        *H_sensor2tt = D_error_point * D_pt_sensor2target * D_sensor2target_sensor2tt * D_sensor2tt_sensor2tt0; 
      }
    }

    if (H_target2tt) {
      if (fix_target2tt_) {
        *H_target2tt = gtsam::Matrix16::Zero();
      } else {
        *H_target2tt = D_error_point * D_pt_sensor2target * D_sensor2target_tt2target * D_tt2target_inv;
      }
    }

    if (H_tt2tt0) {
      if (fix_tt2tt0_) {
        *H_tt2tt0 = gtsam::Matrix16::Zero();
      } else {
        *H_tt2tt0 = D_error_point * D_pt_sensor2target * D_sensor2target_sensor2tt * D_sensor2tt_tt02tt * D_tt02tt_inv;
      }
    }

    return getError(point_target, surfel_);
  }

private:
  gtsam::Point3 measured_; 
  Surfel3 surfel_; 
  bool fix_sensor2tt_;
  bool fix_target2tt_;
  bool fix_tt2tt0_;
};

}

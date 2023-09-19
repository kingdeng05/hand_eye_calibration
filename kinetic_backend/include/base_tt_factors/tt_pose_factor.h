#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor2;
using gtsam::Pose3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;

namespace kinetic_backend {

class TtPoseFactor: public NoiseModelFactor2<Pose3, Pose3> {

public:
  TtPoseFactor(Key key_target2tt_i, Key key_target2tt_0,
               const Pose3& tti2tt0_meas, const SharedNoiseModel& model,
               bool fix_target2tt_i=false, bool fix_target2tt_0=false):
    NoiseModelFactor2<Pose3, Pose3>(
      model,
      key_target2tt_i, 
      key_target2tt_0 
    ),
    measured_(tti2tt0_meas.inverse()),
    fix_target2tt_i_(fix_target2tt_i),
    fix_target2tt_0_(fix_target2tt_0) {}

  Vector evaluateError(const Pose3& target2tt_i,
                       const Pose3& target2tt_0,
                       boost::optional<Matrix&> H_target2tt_i = boost::none,
                       boost::optional<Matrix&> H_target2tt_0 = boost::none) const override
  {
    // tmp jacobian
    Matrix D_tti2ti_t02tt0, D_tti2tt0_ti2tt0, D_tti2tt0_tti2ti;
    Matrix D_tt02tt0_tti2tt0, D_log;

    // pose3 transform 
    Pose3 tti2ti = target2tt_0.inverse(D_tti2ti_t02tt0); 
    Pose3 tti2tt0 = target2tt_i.compose(tti2ti, D_tti2tt0_ti2tt0, D_tti2tt0_tti2ti);
    Pose3 tt02tt0 = tti2tt0.compose(measured_, D_tt02tt0_tti2tt0);
    Vector error = gtsam::Pose3::Logmap(tt02tt0, D_log);

    if (H_target2tt_i) {
      if (fix_target2tt_i_) {
        *H_target2tt_i = gtsam::Matrix66::Zero();
      } else {
        *H_target2tt_i = D_log * D_tt02tt0_tti2tt0 * D_tti2tt0_ti2tt0; 
      }
    }

    if (H_target2tt_0) {
      if (fix_target2tt_0_) {
        *H_target2tt_0 = gtsam::Matrix66::Zero();
      } else {
        *H_target2tt_0 = D_log * D_tt02tt0_tti2tt0 * D_tti2tt0_tti2ti * D_tti2ti_t02tt0; 
      }
    }

    return error;
  }

private:
  Pose3 measured_; 
  bool fix_target2tt_i_;
  bool fix_target2tt_0_;
};

}

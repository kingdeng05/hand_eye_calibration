#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor3;
using gtsam::Pose3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;

namespace kinetic_backend {

class HEPoseConstraintFactor: public NoiseModelFactor3<Pose3, Pose3, Pose3> {

public:
  HEPoseConstraintFactor(Key key_eye2hand, Key key_world2target, Key key_target2eye,
                         const Pose3& hand2world, const SharedNoiseModel& model,
                         bool fix_eye2hand=false, bool fix_world2target=false,
                         bool fix_target2eye=false):
    NoiseModelFactor3<Pose3, Pose3, Pose3>(
      model,
      key_eye2hand, 
      key_world2target,
      key_target2eye
    ),
    measured_(hand2world), fix_eye2hand_(fix_eye2hand),
    fix_world2target_(fix_world2target), fix_target2eye_(fix_target2eye) {}

  Vector evaluateError(const Pose3& eye2hand,
                       const Pose3& world2target,
                       const Pose3& target2eye,
                       boost::optional<Matrix&> H_eye2hand = boost::none,
                       boost::optional<Matrix&> H_world2target = boost::none,
                       boost::optional<Matrix&> H_target2eye = boost::none) const override
  {
    // tmp jacobian
    Matrix D_t2h_e2h, D_t2h_t2e, D_w2h_t2h, D_w2h_w2t;
    Matrix D_h2h_w2h, D_log;

    // pose3 transform 
    Pose3 t2h = eye2hand.compose(target2eye, D_t2h_e2h, D_t2h_t2e);
    Pose3 w2h = t2h.compose(world2target, D_w2h_t2h, D_w2h_w2t);
    Pose3 h2h = w2h.compose(measured_, D_h2h_w2h);
    Vector error = gtsam::Pose3::Logmap(h2h, D_log);

    if (H_eye2hand) {
      if (fix_eye2hand_) {
        *H_eye2hand = gtsam::Matrix66::Zero();
      } else {
        *H_eye2hand = D_log * D_h2h_w2h * D_w2h_t2h * D_t2h_e2h; 
      }
    }

    if (H_world2target) {
      if (fix_world2target_) {
        *H_world2target = gtsam::Matrix66::Zero();
      } else {
        *H_world2target = D_log * D_h2h_w2h * D_w2h_w2t;
      }
    }

    if (H_target2eye) {
      if (fix_target2eye_) {
        *H_target2eye = gtsam::Matrix66::Zero();
      } else {
        *H_target2eye = D_log * D_h2h_w2h * D_w2h_t2h * D_t2h_t2e;
      }
    }

    return error;
  }

private:
  Pose3 measured_; 
  bool fix_eye2hand_;
  bool fix_world2target_;
  bool fix_target2eye_;
};

}

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

class HandPoseFactor: public NoiseModelFactor3<Pose3, Pose3, Pose3> {

public:
  HandPoseFactor(Key key_cam2ee, Key key_target2base, Key key_target2cam,
                 const Pose3& ee2base_meas, const SharedNoiseModel& model,
                 bool fix_cam2ee=true, bool fix_target2base=false,
                 bool fix_target2cam=false):
    NoiseModelFactor3<Pose3, Pose3, Pose3>(
      model,
      key_cam2ee, 
      key_target2base,
      key_target2cam 
    ),
    measured_(ee2base_meas), fix_cam2ee_(fix_cam2ee),
    fix_target2base_(fix_target2base), fix_target2cam_(fix_target2cam) {}

  Vector evaluateError(const Pose3& cam2ee,
                       const Pose3& target2base,
                       const Pose3& target2cam,
                       boost::optional<Matrix&> H_cam2ee = boost::none,
                       boost::optional<Matrix&> H_target2base = boost::none,
                       boost::optional<Matrix&> H_target2cam = boost::none) const override
  {
    // tmp jacobian
    Matrix D_t2ee_c2ee, D_t2ee_t2c, D_b2t_t2b, D_b2ee_t2ee;
    Matrix D_b2ee_b2t, D_ee2ee_b2ee, D_log;

    // pose3 transform 
    Pose3 t2ee = cam2ee.compose(target2cam, D_t2ee_c2ee, D_t2ee_t2c);
    Pose3 base2target = target2base.inverse(D_b2t_t2b);
    Pose3 b2ee = t2ee.compose(base2target, D_b2ee_t2ee, D_b2ee_b2t);
    Pose3 ee2ee = b2ee.compose(measured_, D_ee2ee_b2ee);
    Vector error = gtsam::Pose3::Logmap(ee2ee, D_log);

    if (H_cam2ee) {
      if (fix_cam2ee_) {
        *H_cam2ee = gtsam::Matrix66::Zero();
      } else {
        *H_cam2ee = D_log * D_ee2ee_b2ee * D_b2ee_t2ee * D_t2ee_c2ee; 
      }
    }

    if (H_target2base) {
      if (fix_target2base_) {
        *H_target2base = gtsam::Matrix66::Zero();
      } else {
        *H_target2base = D_log * D_ee2ee_b2ee * D_b2ee_b2t * D_b2t_t2b;
      }
    }

    if (H_target2cam) {
      if (fix_target2cam_) {
        *H_target2cam = gtsam::Matrix66::Zero();
      } else {
        *H_target2cam = D_log * D_ee2ee_b2ee * D_b2ee_t2ee * D_t2ee_t2c;
      }
    }

    return error;
  }

private:
  Pose3 measured_; 
  bool fix_cam2ee_;
  bool fix_target2base_;
  bool fix_target2cam_;
};

}

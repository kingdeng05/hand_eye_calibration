#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/PinholeCamera.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor3;
using gtsam::Pose3;
using gtsam::Point3;
using gtsam::Point2;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;
using gtsam::PinholeCamera;

namespace kinetic_backend {

template<class Calibration>
class Cam2TtProjectionFactor: public NoiseModelFactor3<Pose3, Pose3, Pose3> {

public:
  Cam2TtProjectionFactor(Key key_cam2tt, Key key_target2tt, Key key_tt2tt0,
                         const Calibration& intrinsic,
                         const Point3& pt_3d, const Point2& measured,
                         const SharedNoiseModel& model, bool fix_cam2tt=false,
                         bool fix_target2tt=false, bool fix_tt2tt0=false):
    NoiseModelFactor3<Pose3, Pose3, Pose3>(
      model,
      key_cam2tt,
      key_target2tt,
      key_tt2tt0
    ),
    pt_3d_(pt_3d), measured_(measured), intrinsic_(intrinsic), fix_cam2tt_(fix_cam2tt),
    fix_target2tt_(fix_target2tt), fix_tt2tt0_(fix_tt2tt0) {}

  Vector evaluateError(const Pose3& cam2tt0,
                       const Pose3& target2tt,
                       const Pose3& tt2tt0,
                       boost::optional<Matrix&> H_cam2tt = boost::none,
                       boost::optional<Matrix&> H_target2tt = boost::none,
                       boost::optional<Matrix&> H_tt2tt0 = boost::none) const override
  {
    // tmp jacobian
    Matrix D_tt02tt_inv, D_tt2target_inv, D_cam2tt_tt02tt, D_cam2tt_cam2tt0;
    Matrix D_cam2target_tt2target, D_cam2target_cam2tt, D_proj_cam2target;

    // pose3 transform 
    Pose3 tt02tt = tt2tt0.inverse(D_tt02tt_inv);
    Pose3 tt2target = target2tt.inverse(D_tt2target_inv);
    Pose3 cam2tt = tt02tt.compose(cam2tt0, D_cam2tt_tt02tt, D_cam2tt_cam2tt0);
    Pose3 cam2target = tt2target.compose(cam2tt, D_cam2target_tt2target, D_cam2target_cam2tt);
    PinholeCamera<Calibration> camera(cam2target, intrinsic_);
    Point2 proj = camera.project(pt_3d_, D_proj_cam2target, boost::none, boost::none);

    if (H_cam2tt) {
      if (fix_cam2tt_) {
        *H_cam2tt = gtsam::Matrix26::Zero();
      } else {
        *H_cam2tt = D_proj_cam2target * D_cam2target_cam2tt * D_cam2tt_cam2tt0; 
      }
    }

    if (H_target2tt) {
      if (fix_target2tt_) {
        *H_target2tt = gtsam::Matrix26::Zero();
      } else {
        *H_target2tt = D_proj_cam2target * D_cam2target_tt2target * D_tt2target_inv;
      }
    }

    if (H_tt2tt0) {
      if (fix_tt2tt0_) {
        *H_tt2tt0 = gtsam::Matrix26::Zero();
      } else {
        *H_tt2tt0 = D_proj_cam2target * D_cam2target_cam2tt * D_cam2tt_tt02tt * D_tt02tt_inv;
      }
    }

    return proj - measured_;
  }

private:
  Point3 pt_3d_; 
  Point2 measured_; 
  Calibration intrinsic_;
  bool fix_cam2tt_;
  bool fix_target2tt_;
  bool fix_tt2tt0_;
};

}

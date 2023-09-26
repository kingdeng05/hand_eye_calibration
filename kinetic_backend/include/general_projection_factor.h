#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/PinholeCamera.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor2;
using gtsam::Pose3;
using gtsam::Point3;
using gtsam::Point2;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;
using gtsam::PinholeCamera;

namespace kinetic_backend {

template<class Calibration>
class GeneralProjectionFactor: public NoiseModelFactor2<Pose3, Calibration> {

public:
  GeneralProjectionFactor(Key key_world2cam, Key key_intrinsic, const Point3& pt_3d,
                          const Point2& measured, const SharedNoiseModel& model,
                          bool fix_world2cam=false, bool fix_intrinsic=true):
    NoiseModelFactor2<Pose3, Calibration>(
      model,
      key_world2cam,
      key_intrinsic
    ),
    pt_3d_(pt_3d), measured_(measured), fix_world2cam_(fix_world2cam),
    fix_intrinsic_(fix_intrinsic) {}

  Vector evaluateError(const Pose3& world2cam,
                       const Calibration& intrinsic,
                       boost::optional<Matrix&> H_world2cam = boost::none,
                       boost::optional<Matrix&> H_intrinsic = boost::none) const override
  {
    // tmp jacobian
    Matrix D_inv, D_proj_cam2world, D_proj_intr;

    // pose3 transform 
    Pose3 cam2world = world2cam.inverse(D_inv);
    PinholeCamera<Calibration> camera(cam2world, intrinsic);
    Point2 proj = camera.project(pt_3d_, D_proj_cam2world, boost::none, D_proj_intr);

    if (H_world2cam) {
      if (fix_world2cam_) {
        *H_world2cam = gtsam::Matrix26::Zero();
      } else {
        *H_world2cam = D_proj_cam2world * D_inv; 
      }
    }

    if (H_intrinsic) {
      if (fix_intrinsic_) {
        *H_intrinsic = gtsam::Matrix29::Zero();
      } else {
        std::cout << D_proj_intr << std::endl;
        *H_intrinsic = D_proj_intr;
      }
    }

    return proj - measured_;
  }

private:
  Point3 pt_3d_; 
  Point2 measured_; 
  bool fix_world2cam_;
  bool fix_intrinsic_;
};

}

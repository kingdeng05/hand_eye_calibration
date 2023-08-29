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
using gtsam::NoiseModelFactor4;
using gtsam::Pose3;
using gtsam::Point3;
using gtsam::Point2;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;
using gtsam::PinholeCamera;

namespace kinetic_backend {

template<class Calibration>
class RMFactor: public NoiseModelFactor4<Pose3, Pose3, Pose3, Calibration> {

public:
  RMFactor(Key key_hand2eye, Key key_world2hand, Key key_target2world,
           Key key_intrinsic, const Point3& pt_3d, const Point2& measured,
           const SharedNoiseModel& model, bool fix_hand2eye=false, 
           bool fix_world2hand=true, bool fix_target2world=false,
           bool fix_intrinsic=true):
    NoiseModelFactor4<Pose3, Pose3, Pose3, Calibration>(
      model,
      key_hand2eye, 
      key_world2hand,
      key_target2world,
      key_intrinsic
    ),
    pt_3d_(pt_3d), measured_(measured), fix_hand2eye_(fix_hand2eye),
    fix_world2hand_(fix_world2hand), fix_target2world_(fix_target2world),
    fix_intrinsic_(fix_intrinsic) {}

  Vector evaluateError(const Pose3& hand2eye,
                       const Pose3& world2hand,
                       const Pose3& target2world,
                       const Calibration& intrinsic,
                       boost::optional<Matrix&> H_hand2eye = boost::none,
                       boost::optional<Matrix&> H_world2hand = boost::none,
                       boost::optional<Matrix&> H_target2world = boost::none,
                       boost::optional<Matrix&> H_intrinsic = boost::none) const override
  {
    // tmp jacobian
    Matrix D_t2g_w2h, D_t2g_t2w, D_t2e_h2e, D_t2e_t2g;
    Matrix D_e2t_t2e, D_proj_e2t, D_proj_intr;

    // pose3 transform 
    Pose3 t2g = world2hand.compose(target2world, D_t2g_w2h, D_t2g_t2w);
    Pose3 t2e = hand2eye.compose(t2g, D_t2e_h2e, D_t2e_t2g);
    Pose3 e2t = t2e.inverse(D_e2t_t2e);
    PinholeCamera<Calibration> camera(e2t, intrinsic);
    Point2 proj = camera.project(pt_3d_, D_proj_e2t, boost::none, D_proj_intr);

    if (H_hand2eye) {
      if (fix_hand2eye_) {
        *H_hand2eye = gtsam::Matrix26::Zero();
      } else {
        *H_hand2eye = D_proj_e2t * D_e2t_t2e * D_t2e_h2e; 
      }
    }

    if (H_world2hand) {
      if (fix_world2hand_) {
        *H_world2hand = gtsam::Matrix26::Zero();
      } else {
        *H_world2hand = D_proj_e2t * D_e2t_t2e * D_t2e_t2g * D_t2g_w2h;
      }
    }

    if (H_target2world) {
      if (fix_target2world_) {
        *H_target2world = gtsam::Matrix26::Zero();
      } else {
        *H_target2world = D_proj_e2t * D_e2t_t2e * D_t2e_t2g * D_t2g_t2w;
      }
    }

    if (H_intrinsic) {
      if (fix_intrinsic_) {
        *H_intrinsic = gtsam::Matrix25::Zero();
      } else {
        *H_intrinsic = D_proj_intr;
      }
    }

    return proj - measured_;
  }

private:
  Point3 pt_3d_; 
  Point2 measured_; 
  bool fix_hand2eye_;
  bool fix_world2hand_;
  bool fix_target2world_;
  bool fix_intrinsic_;
};

}

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/PinholeCamera.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor6;
using gtsam::Pose3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;
using gtsam::Point3;
using gtsam::Point2;
using gtsam::PinholeCamera;

namespace kinetic_backend {

template<typename Calibration>
class BaseTtProjectionFactor: public NoiseModelFactor6<Pose3, Pose3, Pose3, Pose3, Pose3, Pose3> {

public:
  BaseTtProjectionFactor(Key key_ee2base, Key key_base2track,
                         Key key_track2track0, Key key_track2tt, Key key_tt2tt0,
                         Key key_target2tt, const Point3& pt_3d,
                         const Point2& pt_2d, const Pose3& cam2ee,
                         const Calibration& intrinsic, const SharedNoiseModel& model,
                         bool fix_ee2base=false, bool fix_base2track=false, 
                         bool fix_track2track0=false, bool fix_track2tt=false,
                         bool fix_tt2tt0=false, bool fix_target2tt=false):
    NoiseModelFactor6<Pose3, Pose3, Pose3, Pose3, Pose3, Pose3>(
      model,
      key_ee2base,
      key_base2track,
      key_track2track0,
      key_track2tt,
      key_tt2tt0,
      key_target2tt
    ),
    pt_3d_(pt_3d), pt_2d_(pt_2d), cam2ee_(cam2ee),
    intrinsic_(intrinsic), fix_ee2base_(fix_ee2base),
    fix_base2track_(fix_base2track), fix_track2track0_(fix_track2track0),
    fix_track2tt_(fix_track2tt), fix_tt2tt0_(fix_tt2tt0), fix_target2tt_(fix_target2tt) {}

  Vector evaluateError(const Pose3& ee2base,
                       const Pose3& base2track,
                       const Pose3& track2track0,
                       const Pose3& track2tt,
                       const Pose3& tt2tt0,
                       const Pose3& target2tt,
                       boost::optional<Matrix&> H_ee2base = boost::none,
                       boost::optional<Matrix&> H_base2track = boost::none,
                       boost::optional<Matrix&> H_track2track0 = boost::none,
                       boost::optional<Matrix&> H_track2tt = boost::none,
                       boost::optional<Matrix&> H_tt2tt0 = boost::none,
                       boost::optional<Matrix&> H_target2tt = boost::none) const override
  {
    // tmp jacobian
    Matrix D_t2tt0_tt2tt0, D_t2tt0_t2tt, D_tt02t_t2tt0; 
    Matrix D_tr2t_tt02t, D_tr2t_tr2tt;
    Matrix D_tri2t_tr2t, D_tri2t_tr2tr0;
    Matrix D_b2t_tri2t, D_b2t_b2tr;
    Matrix D_e2t_b2t, D_e2t_e2b;
    Matrix D_c2t_e2t, D_proj_c2t;

    // pose3 transform 
    Pose3 target2tt0 = tt2tt0.compose(target2tt, D_t2tt0_tt2tt0, D_t2tt0_t2tt); 
    Pose3 tt02target = target2tt0.inverse(D_tt02t_t2tt0);
    Pose3 track2target = tt02target.compose(track2tt, D_tr2t_tt02t, D_tr2t_tr2tt);
    Pose3 tracki2target = track2target.compose(track2track0, D_tri2t_tr2t, D_tri2t_tr2tr0); 
    Pose3 base2target = tracki2target.compose(base2track, D_b2t_tri2t, D_b2t_b2tr);
    Pose3 ee2target = base2target.compose(ee2base, D_e2t_b2t, D_e2t_e2b);
    Pose3 cam2target = ee2target.compose(cam2ee_, D_c2t_e2t);

    // projection
    PinholeCamera<Calibration> camera(cam2target, intrinsic_);
    Point2 proj = camera.project(pt_3d_, D_proj_c2t);

    if (H_ee2base) {
      if (fix_ee2base_) {
        *H_ee2base = gtsam::Matrix26::Zero();
      } else {
        *H_ee2base = D_proj_c2t * D_c2t_e2t * D_e2t_e2b; 
      }
    }

    if (H_base2track) {
      if (fix_base2track_) {
        *H_base2track = gtsam::Matrix26::Zero();
      } else {
        *H_base2track = D_proj_c2t * D_c2t_e2t * D_e2t_b2t * D_b2t_b2tr;
      }
    }

    if (H_track2track0) {
      if (fix_track2track0_) {
        *H_track2track0 = gtsam::Matrix26::Zero();
      } else {
        *H_track2track0 = D_proj_c2t * D_c2t_e2t * D_e2t_b2t * D_b2t_tri2t * D_tri2t_tr2tr0;
      }
    }

    if (H_track2tt) {
      if (fix_track2tt_) {
        *H_track2tt = gtsam::Matrix26::Zero();
      } else {
        *H_track2tt = D_proj_c2t * D_c2t_e2t * D_e2t_b2t * D_b2t_tri2t * D_tri2t_tr2t * D_tr2t_tr2tt;
      }
    }

    if (H_tt2tt0) {
      if (fix_tt2tt0_) {
        *H_tt2tt0 = gtsam::Matrix26::Zero();
      } else {
        *H_tt2tt0 = D_proj_c2t * D_c2t_e2t * D_e2t_b2t * D_b2t_tri2t * D_tri2t_tr2t * D_tr2t_tt02t * D_tt02t_t2tt0 * D_t2tt0_tt2tt0;
      }
    }

    if (H_target2tt) {
      if (fix_target2tt_) {
        *H_target2tt = gtsam::Matrix26::Zero();
      } else {
        *H_target2tt = D_proj_c2t * D_c2t_e2t * D_e2t_b2t * D_b2t_tri2t * D_tri2t_tr2t * D_tr2t_tt02t * D_tt02t_t2tt0 * D_t2tt0_t2tt;
      }
    }

    return proj - pt_2d_;
  }

private:
  Point3 pt_3d_; 
  Point2 pt_2d_;
  Pose3 cam2ee_;
  Calibration intrinsic_;
  bool fix_ee2base_;
  bool fix_base2track_;
  bool fix_track2track0_;
  bool fix_track2tt_;
  bool fix_tt2tt0_;
  bool fix_target2tt_;
};

}

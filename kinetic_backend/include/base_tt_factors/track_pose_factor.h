#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor4;
using gtsam::Pose3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;

namespace kinetic_backend {

class TrackPoseFactor: public NoiseModelFactor4<Pose3, Pose3, Pose3, Pose3> {

public:
  TrackPoseFactor(Key key_base2track, Key key_target2base, Key key_target2tt, Key key_track2tt,
                  const Pose3& track2track_meas, const SharedNoiseModel& model,
                  bool fix_base2track=false, bool fix_target2base=false,
                  bool fix_target2tt=false, bool fix_track2tt=false):
    NoiseModelFactor4<Pose3, Pose3, Pose3, Pose3>(
      model,
      key_base2track, 
      key_target2base, 
      key_target2tt,
      key_track2tt
    ),
    measured_(track2track_meas),
    fix_base2track_(fix_base2track),
    fix_target2base_(fix_target2base),
    fix_target2tt_(fix_target2tt),
    fix_track2tt_(fix_track2tt) {}

  Vector evaluateError(const Pose3& base2track,
                       const Pose3& target2base,
                       const Pose3& target2tt,
                       const Pose3& track2tt,
                       boost::optional<Matrix&> H_base2track = boost::none,
                       boost::optional<Matrix&> H_target2base = boost::none,
                       boost::optional<Matrix&> H_target2tt = boost::none,
                       boost::optional<Matrix&> H_track2tt = boost::none) const override
  {
    // tmp jacobian
    Matrix D_b2tr_b2tr, D_t2tr_b2tr, D_t2tr_t2b;
    Matrix D_t2tt_inv, D_tt2tr_t2tr, D_tt2tr_tt2t; 
    Matrix D_tr2tr_tt2tr, D_tr2tr_tr2tt, D_log; 

    // pose3 transform 
    Pose3 b2tr = measured_.compose(base2track, boost::none, D_b2tr_b2tr);
    Pose3 t2tr = b2tr.compose(target2base, D_t2tr_b2tr, D_t2tr_t2b);
    Pose3 tt2t = target2tt.inverse(D_t2tt_inv);
    Pose3 tt2tr = t2tr.compose(tt2t, D_tt2tr_t2tr, D_tt2tr_tt2t);
    Pose3 tr2tr = tt2tr.compose(track2tt, D_tr2tr_tt2tr, D_tr2tr_tr2tt);
    Vector error = gtsam::Pose3::Logmap(tr2tr, D_log);

    if (H_base2track) {
      if (fix_base2track_) {
        *H_base2track = gtsam::Matrix66::Zero();
      } else {
        *H_base2track = D_log * D_tr2tr_tt2tr * D_tt2tr_t2tr * D_t2tr_b2tr * D_b2tr_b2tr; 
      }
    }

    if (H_target2base) {
      if (fix_target2base_) {
        *H_target2base = gtsam::Matrix66::Zero();
      } else {
        *H_target2base = D_log * D_tr2tr_tt2tr * D_tt2tr_t2tr * D_t2tr_t2b; 
      }
    }

    if (H_target2tt) {
      if (fix_target2tt_) {
        *H_target2tt = gtsam::Matrix66::Zero();
      } else {
        *H_target2tt = D_log * D_tr2tr_tt2tr * D_tt2tr_tt2t * D_t2tt_inv;
      }
    }

    if (H_track2tt) {
      if (fix_track2tt_) {
        *H_track2tt = gtsam::Matrix66::Zero();
      } else {
        *H_track2tt = D_log * D_tr2tr_tr2tt;
      }
    }
    return error;
  }

private:
  Pose3 measured_; 
  bool fix_base2track_;
  bool fix_target2base_;
  bool fix_target2tt_;
  bool fix_track2tt_;
};

}

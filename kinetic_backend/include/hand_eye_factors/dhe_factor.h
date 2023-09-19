#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/optional.hpp>

using gtsam::Key;
using gtsam::NoiseModelFactor5;
using gtsam::Pose3;
using gtsam::SharedNoiseModel;
using gtsam::Vector;
using gtsam::Matrix;

namespace kinetic_backend {

class DHEFactor: public NoiseModelFactor5<Pose3, Pose3, Pose3, Pose3, Pose3> {

public:
  DHEFactor(Key X, Key E_i, Key E_j, Key C_i, Key C_j, const SharedNoiseModel& model):
    NoiseModelFactor5<Pose3, Pose3, Pose3, Pose3, Pose3>(model, X, E_i, E_j, C_i, C_j) {}

  Vector evaluateError(const Pose3& x,
                       const Pose3& e_i,
                       const Pose3& e_j,
                       const Pose3& c_i,
                       const Pose3& c_j,
                       boost::optional<Matrix&> H_x = boost::none,
                       boost::optional<Matrix&> H_e_i = boost::none,
                       boost::optional<Matrix&> H_e_j = boost::none,
                       boost::optional<Matrix&> H_c_i = boost::none,
                       boost::optional<Matrix&> H_c_j = boost::none) const override
  {
    Matrix H1_tmp1, H2_tmp1, H3_tmp1, H4_tmp1, H5_tmp1;
    Matrix H1_tmp2, H2_tmp2, H4_tmp2;
    Matrix H1_tmp3, H1_tmp4, H1_tmp5, H1_tmp6, H1_tmp7;
    Matrix D_log;

    // get the X_c2ee^{-1}X_{eei}^{-1}X_{eej} 
    Pose3 x_inv = x.inverse(H1_tmp1);
    Pose3 ei_inv = e_i.inverse(H2_tmp1);
    Pose3 x_inv_ei_inv = x_inv.compose(ei_inv, H1_tmp2, H2_tmp2);
    Pose3 x_inv_ei_inv_ej = x_inv_ei_inv.compose(e_j, H1_tmp3, H3_tmp1);

    // get the X_c2eeX_{cj}X_{ci}^{-1} 
    Pose3 ci_inv = c_i.inverse(H4_tmp1);
    Pose3 x_cj = x.compose(c_j, H1_tmp4, H5_tmp1);
    Pose3 x_cj_ci_inv = x_cj.compose(ci_inv, H1_tmp5, H4_tmp2); 

    // get these two together
    Pose3 error_pose = x_inv_ei_inv_ej.compose(x_cj_ci_inv, H1_tmp6, H1_tmp7);
    Vector error = gtsam::Pose3::Logmap(error_pose, D_log);

    // Compute final Jacobians
    if (H_x) {
        *H_x = D_log * (H1_tmp6 * H1_tmp3 * H1_tmp2 * H1_tmp1 + H1_tmp7 * H1_tmp5 * H1_tmp4);

        // numerical diff
        // auto wrapperX = [this, &e_i, &e_j, &c_i, &c_j](const gtsam::Pose3& x) {
        //     return evaluateError(x, e_i, e_j, c_i, c_j); // Recursion but without computing Jacobians
        // };
        
        // *H_x = gtsam::numericalDerivative11<Vector, gtsam::Pose3>(wrapperX, x);
        // std::cout << "Numerical H_x:\n" << *H_x << std::endl;
    }
    
    if (H_e_i) {
        *H_e_i = gtsam::Matrix6::Zero();
        // *H_e_i = D_log * H1_tmp6 * H1_tmp3 * H2_tmp2 * H2_tmp1;
    }

    if (H_e_j) {
        *H_e_j = gtsam::Matrix6::Zero();
        // *H_e_j = D_log * H1_tmp6 * H3_tmp1;
    }
    
    if (H_c_i) {
        *H_c_i = gtsam::Matrix6::Zero();
        // *H_c_i = D_log * H1_tmp7 * H4_tmp2 * H4_tmp1;
    }
   
    if (H_c_j) {
        *H_c_j = gtsam::Matrix6::Zero();
        // *H_c_j = D_log * H1_tmp7 * H1_tmp5 * H5_tmp1;
    }

    return error;

  }
};

}

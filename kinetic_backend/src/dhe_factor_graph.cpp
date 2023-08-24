#include <iostream>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>


#include <dhe_factor.h>
#include <dhe_factor_graph.h>

using namespace std;
using namespace gtsam;

namespace kinetic_backend {

Vector Rad2Deg(const Vector& vec_deg) {
    return vec_deg / M_PI * 180;
}

void calibrate_dhe_factor_graph(const gtsam::Matrix4& calib_gt,
                                const std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>>& poses_a, 
                                const std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>>& poses_b) {
    // Create a noise model for the measurements
    SharedNoiseModel dhe_noise_model = noiseModel::Diagonal::Sigmas((Vector6() << 0.0017, 0.0017, 0.0017, 0.1, 0.1, 0.1).finished());
    SharedNoiseModel calib_noise_model = noiseModel::Isotropic::Sigma(6, 0.1);
    SharedNoiseModel obs_noise_model = noiseModel::Diagonal::Sigmas(
        (Vector6() << 0.2/M_PI*180, 0.2/M_PI*180, 0.2/M_PI*180, 0.2, 0.3, 0.1).finished());

    // Initialize factor graph
    NonlinearFactorGraph graph;

    Key calib_key = symbol('X', 0);
    Values initial;

    // // Add the first prior factors 
    Pose3 prior_pose(poses_a[0]);
    // graph.add(PriorFactor<Pose3>(symbol('E', 0), prior_pose, obs_noise_model));
    initial.insert(symbol('E', 0), prior_pose);

    prior_pose = Pose3(poses_b[0]);
    // graph.add(PriorFactor<Pose3>(symbol('C', 0), prior_pose, obs_noise_model));
    initial.insert(symbol('C', 0), prior_pose);

    for (size_t idx = 1; idx < poses_a.size(); ++idx) {
        prior_pose = Pose3(poses_a[idx]);
        // graph.add(PriorFactor<Pose3>(symbol('E', idx), prior_pose, obs_noise_model));
        initial.insert(symbol('E', idx), prior_pose);

        prior_pose = Pose3(poses_b[idx]);
        // graph.add(PriorFactor<Pose3>(symbol('C', idx), prior_pose, obs_noise_model));
        initial.insert(symbol('C', idx), prior_pose);

        graph.add(DHEFactor(calib_key, symbol('E', idx-1), symbol('E', idx), symbol('C', idx-1), symbol('C', idx), dhe_noise_model));
    }

    // create a small perturbation
    Rot3 rot = Rot3::RzRyRx(0.01, 0.02, 0.01);
    Point3 t(0.02, 0.04, 0.03);
    Pose3 pert = Pose3(rot, t);

    Pose3 calib_gt_pose(calib_gt);
    calib_gt_pose = calib_gt_pose * pert;
    // graph.add(PriorFactor<Pose3>(calib_key, calib_gt_pose, calib_noise_model));
    initial.insert(calib_key, calib_gt_pose);

    // Optimize the factor graph
    LevenbergMarquardtOptimizer optimizer(graph, initial);
    Values result = optimizer.optimize();
    std::cout << "error change: " << graph.error(initial) << " -> " << graph.error(result) << std::endl;
    Vector vec_diff =  Pose3::Logmap(Pose3(calib_gt).between(result.at<Pose3>(calib_key))).transpose();
    std::cout << "rot diff: " << Rad2Deg(vec_diff.head(3)).transpose() << std::endl;
    std::cout << "t diff: " << vec_diff.tail(3).transpose() << std::endl;

    return;
}

}

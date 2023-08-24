#pragma once

#include <vector>

#include <gtsam/base/Matrix.h>

namespace kinetic_backend {

void calibrate_dhe_factor_graph(const gtsam::Matrix4& calib_gt, 
                                const std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>>& poses_a,
                                const std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>>& poses_b);

}


#include <dhe_factor_graph.h>

int main() {
    gtsam::Matrix4 ret = gtsam::Matrix4::Identity();
    gtsam::Matrix4 A = gtsam::Matrix4::Identity();
    gtsam::Matrix4 B = gtsam::Matrix4::Identity();
    std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>> ret_A{A};
    std::vector<gtsam::Matrix4, Eigen::aligned_allocator<gtsam::Matrix4>> ret_B{B};
    kinetic_backend::calibrate_dhe_factor_graph(ret, ret_A, ret_B);
    return 0;
}
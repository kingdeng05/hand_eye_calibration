#include "geometry/surfel3.h"

#include <cmath>


namespace kinetic_backend {

Surfel3::Surfel3(const gtsam::Point3& c, const gtsam::Unit3& n, double r) : 
  center(c), normal(n), radius(r) {}

double Surfel3::Distance(const gtsam::Point3& p) const {
  return std::abs(normal.point3().dot(p - center));
}

}
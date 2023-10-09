#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Unit3.h>


namespace kinetic_backend {

struct Surfel3 {
  gtsam::Point3 center;
  gtsam::Unit3 normal;
  double radius;

  Surfel3(const gtsam::Point3& c, const gtsam::Unit3& u, double r); 
  double Distance(const gtsam::Point3& p) const;
};

}
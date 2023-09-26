/**
 * @file Cal3Rational.cpp
 * @date Sep 25, 2023
 * @author Fuheng Deng 
 */
#include "geometry/Cal3Rational.h"

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>

using namespace gtsam;

namespace kinetic_backend {

/* ************************************************************************* */
void Cal3Rational::print(const std::string& s_) const {
  Base::print(s_);
}

/* ************************************************************************* */
bool Cal3Rational::equals(const Cal3Rational& K, double tol) const {
  if (std::abs(fx_ - K.fx_) > tol || std::abs(fy_ - K.fy_) > tol || 
      std::abs(u0_ - K.u0_) > tol || std::abs(v0_ - K.v0_) > tol ||
      std::abs(k1_ - K.k1_) > tol || std::abs(k2_ - K.k2_) > tol ||
      std::abs(k3_ - K.k3_) > tol || std::abs(k4_ - K.k4_) > tol ||
      std::abs(k5_ - K.k5_) > tol || std::abs(k6_ - K.k6_) > tol ||
      std::abs(p1_ - K.p1_) > tol || std::abs(p2_ - K.p2_) > tol)
    return false;
  return true;
}

/* ************************************************************************* */
Cal3Rational Cal3Rational::retract(const Vector& d) const {
  return Cal3Rational(vector() + d);
}

/* ************************************************************************* */
Vector Cal3Rational::localCoordinates(const Cal3Rational& T2) const {
  return T2.vector() - vector();
}

}


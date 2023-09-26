/**
 * @file Cal3Rational_Base.cpp
 * @date Sep 25, 2023
 * @author Fuheng Deng 
 */

#include "geometry/Cal3Rational_Base.h"

#include <gtsam/base/Testable.h>
#include <gtsam/geometry/Point3.h>

using namespace gtsam;

namespace kinetic_backend {

/* ************************************************************************* */
Cal3Rational_Base::Cal3Rational_Base(const Vector &v):
    fx_(v[0]), fy_(v[1]), u0_(v[2]), v0_(v[3]), k1_(v[4]), k2_(v[5]),
    k3_(v[6]), k4_(v[7]), k5_(v[8]), k6_(v[9]), p1_(v[10]), p2_(v[11]){}

/* ************************************************************************* */
Matrix3 Cal3Rational_Base::K() const {
    Matrix3 K;
    K << fx_, 0.0, u0_, 0.0, fy_, v0_, 0.0, 0.0, 1.0;
    return K;
}

/* ************************************************************************* */
Vector12 Cal3Rational_Base::vector() const {
  Vector12 v;
  v << fx_, fy_, u0_, v0_, k1_, k2_, k3_, k4_, k5_, k6_, p1_, p2_;
  return v;
}

/* ************************************************************************* */
void Cal3Rational_Base::print(const std::string& s_) const {
  gtsam::print((Matrix)K(), s_ + ".K");
  gtsam::print(Vector(k()), s_ + ".k");
}

/* ************************************************************************* */
bool Cal3Rational_Base::equals(const Cal3Rational_Base& K, double tol) const {
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
static Cal3Rational_Base::Matrix212 D2dcalibration(double x, double y, double xx,
    double yy, double xy, double r2, double r4, double r6, double nom, double deno,
    double pnx, double pny, const Matrix2& DK) {
  Matrix24 DR1;
  DR1 << pnx, 0.0, 1.0, 0.0, 0.0, pny, 0.0, 1.0;
  Matrix28 DR2;
  const double dgdk1 = r2 / deno; 
  const double dgdk2 = r4 / deno; 
  const double dgdk3 = r6 / deno; 
  const double dgdk4 = -r2 * nom / deno; 
  const double dgdk5 = -r4 * nom / deno; 
  const double dgdk6 = -r6 * nom / deno; 
  DR2 << x * dgdk1, x * dgdk2, x * dgdk3, x * dgdk4, x * dgdk5, x * dgdk6, 2 * xy, r2 + 2 * xx, //
         y * dgdk1, y * dgdk2, y * dgdk3, y * dgdk4, y * dgdk5, y * dgdk6, r2 + 2 * yy, 2 * xy;
  Cal3Rational_Base::Matrix212 D;
  D << DR1, DK * DR2;
  return D;
}

/* ************************************************************************* */
static Matrix2 D2dintrinsic(double x, double y, double r2, double r4,
    double nom, double deno, double k1, double k2, double k3, double k4,
    double k5, double k6, double p1, double p2, const Matrix2& DK) {
  const double form = (2 * k1 + 4 * k2 * r2 + 6 * k3 * r4) / deno +
                      (-2 * k4 - 4 * k5 * r2 - 6 * k6 * r4) * nom /
                      (deno * deno);
  const double dgdx = form * x; 
  const double dgdy = form * y;

  // Dx = 2*p1*xy + p2*(rr+2*xx);
  // Dy = 2*p2*xy + p1*(rr+2*yy);
  const double drdx = 2. * x;
  const double drdy = 2. * y;
  const double dDxdx = 2. * p1 * y + p2 * (drdx + 4. * x);
  const double dDxdy = 2. * p1 * x + p2 * drdy;
  const double dDydx = 2. * p2 * y + p1 * drdx;
  const double dDydy = 2. * p2 * x + p1 * (drdy + 4. * y);

  Matrix2 DR;
  const double g = nom / deno;
  DR << g + x * dgdx + dDxdx, x * dgdy + dDxdy, //
        y * dgdx + dDydx, g + y * dgdy + dDydy;

  return DK * DR;
}

/* ************************************************************************* */
Point2 Cal3Rational_Base::uncalibrate(
    const Point2& p,
    OptionalJacobian<2, 12> H1,
    OptionalJacobian<2, 2> H2) const {

  //  rr = x^2 + y^2;
  //  g = (1 + k(1)*rr + k(2)*rr^2 + k(3)*rr^3) / (1 + k(4)*rr + k(5)*rr^2 + k(6)*rr^3);
  //  dp = [2*k(7)*x*y + k(8)*(rr + 2*x^2); 2*k(8)*x*y + k(7)*(rr + 2*y^2)];
  //  pi(:,i) = g * pn(:,i) + dp;
  const double x = p.x(), y = p.y(), xy = x * y, xx = x * x, yy = y * y;
  const double r2 = xx + yy;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  double nom = 1. + k1_ * r2 + k2_ * r4 + k3_ * r6;
  double deno = 1. + k4_ * r2 + k5_ * r4 + k6_ * r6;
  const double g = nom / deno; // scaling factor

  // tangential component
  const double dx = 2. * p1_ * xy + p2_ * (r2 + 2. * xx);
  const double dy = 2. * p2_ * xy + p1_ * (r2 + 2. * yy);

  // Radial and tangential distortion applied
  const double pnx = g * x + dx;
  const double pny = g * y + dy;

  Matrix2 DK;
  if (H1 || H2) DK << fx_, 0.0, 0.0, fy_;

  // Derivative for calibration
  if (H1)
    *H1 = D2dcalibration(x, y, xx, yy, xy, r2, r4, r6, nom, deno, pnx, pny, DK);

  // Derivative for points
  if (H2)
    *H2 = D2dintrinsic(x, y, r2, r4, nom, deno, k1_, k2_, k3_, k4_, k5_, k6_, p1_, p2_, DK);

  // Regular uncalibrate after distortion
  return Point2(fx_ * pnx + u0_, fy_ * pny + v0_);
}

/* ************************************************************************* */
Point2 Cal3Rational_Base::calibrate(const Point2& pi, const double tol) const {
  // Use the following fixed point iteration to invert the radial distortion.
  // pn_{t+1} = (inv(K)*pi - dp(pn_{t})) / g(pn_{t})

  const Point2 invKPi ((1 / fx_) * (pi.x() - u0_),
                       (1 / fy_) * (pi.y() - v0_));

  // initialize by ignoring the distortion at all, might be problematic for pixels around boundary
  Point2 pn = invKPi;

  // iterate until the uncalibrate is close to the actual pixel coordinate
  const int maxIterations = 10;
  int iteration;
  for (iteration = 0; iteration < maxIterations; ++iteration) {
    if (distance2(uncalibrate(pn), pi) <= tol) break;
    const double x = pn.x(), y = pn.y(), xy = x * y, xx = x * x, yy = y * y;
    const double r2 = xx + yy;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double g = (1. + k1_ * r2 + k2_ * r4 + k3_ * r6) / (1. + k4_ * r2 + k5_ * r4 + k6_ * r6); // scaling factor
    const double dx = 2 * p1_ * xy + p2_ * (r2 + 2 * xx);
    const double dy = 2 * p2_ * xy + p1_ * (r2 + 2 * yy);
    pn = (invKPi - Point2(dx, dy)) / g;
  }

  if ( iteration >= maxIterations )
    throw std::runtime_error("Cal3Rational::calibrate fails to converge. need a better initialization");

  return pn;
}

/* ************************************************************************* */
Matrix2 Cal3Rational_Base::D2d_intrinsic(const Point2& p) const {
  const double x = p.x(), y = p.y(), xx = x * x, yy = y * y;
  const double r2 = xx + yy;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double nom = (1 + k1_ * r2 + k2_ * r4 + k3_ * r6);
  const double deno = (1 + k4_ * r2 + k5_ * r4 + k6_ * r6);
  const double g = nom / deno;
  Matrix2 DK;
  DK << fx_, 0.0, 0.0, fy_;
  return D2dintrinsic(x, y, r2, r4, nom, deno, k1_, k2_, k3_, k4_, k5_, k6_, p1_, p2_, DK);
}

/* ************************************************************************* */
Cal3Rational_Base::Matrix212 Cal3Rational_Base::D2d_calibration(const Point2& p) const {
  const double x = p.x(), y = p.y(), xx = x * x, yy = y * y, xy = x * y;
  const double r2 = xx + yy;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double nom = 1 + k1_ * r2 + k2_ * r4 + k3_ * r6;
  const double deno = 1 + k4_ * r2 + k5_ * r4 + k6_ * r6;
  const double g = nom / deno;
  const double dx = 2 * p1_ * xy + p2_ * (r2 + 2 * xx);
  const double dy = 2 * p2_ * xy + p1_ * (r2 + 2 * yy);
  const double pnx = g * x + dx;
  const double pny = g * y + dy;
  Matrix2 DK;
  DK << fx_, 0.0, 0.0, fy_;
  return D2dcalibration(x, y, xx, yy, xy, r2, r4, r6, nom, deno, pnx, pny, DK);
}

}

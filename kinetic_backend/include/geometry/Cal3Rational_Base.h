/**
 * @file Cal3Rational.h
 * @brief Calibration of a camera with rational distortion model 
 * @date Sep 25, 2023
 * @author Fuheng Deng 
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Point2.h>

namespace kinetic_backend {

/**
 * @brief Calibration of a camera with rational distortion
 */
class Cal3Rational_Base {

protected:

  double fx_, fy_, u0_, v0_ ; // focal length and principal point
  double k1_, k2_, k3_, k4_, k5_, k6_; // rational params 
  double p1_, p2_ ; // tangential distortion

public:
  using Matrix212 = Eigen::Matrix<double, 2, 12>;

  /// @name Standard Constructors
  /// @{

  /// Default Constructor with only unit focal length
  Cal3Rational_Base() : fx_(1), fy_(1), u0_(0), v0_(0), k1_(0), k2_(0), k3_(0),
                        k4_(0), k5_(0), k6_(0), p1_(0), p2_(0) {}

  Cal3Rational_Base(double fx, double fy, double u0, double v0,
      double k1, double k2, double p1, double p2, double k3, double k4,
      double k5, double k6) :
  fx_(fx), fy_(fy), u0_(u0), v0_(v0), k1_(k1), k2_(k2), k3_(k3), k4_(k4),
  k5_(k5), k6_(k6), p1_(p1), p2_(p2) {}

  virtual ~Cal3Rational_Base() {}

  /// @}
  /// @name Advanced Constructors
  /// @{

  Cal3Rational_Base(const gtsam::Vector &v) ;

  /// @}
  /// @name Testable
  /// @{

  /// print with optional string
  virtual void print(const std::string& s = "") const ;

  /// assert equality up to a tolerance
  bool equals(const Cal3Rational_Base& K, double tol = 10e-9) const;

  /// @}
  /// @name Standard Interface
  /// @{

  /// focal length x
  inline double fx() const { return fx_;}

  /// focal length x
  inline double fy() const { return fy_;}

  /// image center in x
  inline double px() const { return u0_;}

  /// image center in y
  inline double py() const { return v0_;}

  /// First distortion coefficient
  inline double k1() const { return k1_;}

  /// Second distortion coefficient
  inline double k2() const { return k2_;}

  /// Second distortion coefficient
  inline double k3() const { return k3_;}

  /// Second distortion coefficient
  inline double k4() const { return k4_;}

  /// Second distortion coefficient
  inline double k5() const { return k5_;}

  /// Second distortion coefficient
  inline double k6() const { return k6_;}

  /// First tangential distortion coefficient
  inline double p1() const { return p1_;}

  /// Second tangential distortion coefficient
  inline double p2() const { return p2_;}

  /// return calibration matrix -- not really applicable
  gtsam::Matrix3 K() const;

  /// return distortion parameter vector
  gtsam::Vector8 k() const { return (gtsam::Vector8() << k1_, k2_, p1_, p2_, k3_, k4_, k5_, k6_).finished(); }

  /// Return all parameters as a vector
  gtsam::Vector12 vector() const;

  /**
   * convert intrinsic coordinates xy to (distorted) image coordinates uv
   * @param p point in intrinsic coordinates
   * @param Dcal optional 2*9 Jacobian wrpt Cal3Rational parameters
   * @param Dp optional 2*2 Jacobian wrpt intrinsic coordinates
   * @return point in (distorted) image coordinates
   */
  gtsam::Point2 uncalibrate(const gtsam::Point2& p,
       gtsam::OptionalJacobian<2, 12> Dcal = boost::none,
       gtsam::OptionalJacobian<2, 2> Dp = boost::none) const;

  /// Convert (distorted) image coordinates uv to intrinsic coordinates xy
  gtsam::Point2 calibrate(const gtsam::Point2& p, const double tol=1e-5) const;

  /// Derivative of uncalibrate wrpt intrinsic coordinates
  gtsam::Matrix2 D2d_intrinsic(const gtsam::Point2& p) const;

  /// Derivative of uncalibrate wrpt the calibration parameters
  Matrix212 D2d_calibration(const gtsam::Point2& p) const;

  /// @}
  /// @name Clone
  /// @{

  /// @return a deep copy of this object
  virtual boost::shared_ptr<Cal3Rational_Base> clone() const {
    return boost::shared_ptr<Cal3Rational_Base>(new Cal3Rational_Base(*this));
  }

  /// @}

private:

  /// @name Advanced Interface
  /// @{

  /** Serialization function */
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /*version*/)
  {
    ar & BOOST_SERIALIZATION_NVP(fx_);
    ar & BOOST_SERIALIZATION_NVP(fy_);
    ar & BOOST_SERIALIZATION_NVP(u0_);
    ar & BOOST_SERIALIZATION_NVP(v0_);
    ar & BOOST_SERIALIZATION_NVP(k1_);
    ar & BOOST_SERIALIZATION_NVP(k2_);
    ar & BOOST_SERIALIZATION_NVP(k3_);
    ar & BOOST_SERIALIZATION_NVP(k4_);
    ar & BOOST_SERIALIZATION_NVP(k5_);
    ar & BOOST_SERIALIZATION_NVP(k6_);
    ar & BOOST_SERIALIZATION_NVP(p1_);
    ar & BOOST_SERIALIZATION_NVP(p2_);
  }

  /// @}

};

}


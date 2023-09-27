/**
 * @file Cal3Rational.h
 * @brief Calibration of a camera with rational distortion model, calculations in base class Cal3Rational_Base
 * @date Sep 25, 2023
 * @author Fuheng Deng 
 */

#pragma once

#include "Cal3Rational_Base.h"

namespace kinetic_backend {

/**
 * @brief Calibration of a pinhole camera with rational distortion model.
 * \sa Rational_Base
 * @addtogroup geometry
 * \nosubgrouping
 */
class Cal3Rational : public Cal3Rational_Base {

  typedef Cal3Rational_Base Base;

public:

  enum { dimension = 12 };

  /// @name Standard Constructors
  /// @{

  /// Default Constructor with only unit focal length
  Cal3Rational() : Base() {}

  Cal3Rational(double fx, double fy, double u0, double v0,
      double k1, double k2, double p1, double p2, double k3, 
      double k4, double k5, double k6) :
        Base(fx, fy, u0, v0, k1, k2, p1, p2, k3, k4, k5, k6) {}

  virtual ~Cal3Rational() {}

  /// @}
  /// @name Advanced Constructors
  /// @{

  Cal3Rational(const gtsam::Vector &v) : Base(v) {}

  /// @}
  /// @name Testable
  /// @{

  /// print with optional string
  virtual void print(const std::string& s = "") const ;

  /// assert equality up to a tolerance
  bool equals(const Cal3Rational& K, double tol = 10e-9) const;

  /// @}
  /// @name Manifold
  /// @{

  /// Given delta vector, update calibration
  Cal3Rational retract(const gtsam::Vector& d) const ;

  /// Given a different calibration, calculate update to obtain it
  gtsam::Vector localCoordinates(const Cal3Rational& T2) const ;

  /// Return dimensions of calibration manifold object
  virtual size_t dim() const { return dimension ; }

  /// Return dimensions of calibration manifold object
  static size_t Dim() { return dimension; }

  /// @}
  /// @name Clone
  /// @{

  /// @return a deep copy of this object
  virtual boost::shared_ptr<Base> clone() const {
    return boost::shared_ptr<Base>(new Cal3Rational(*this));
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
    ar & boost::serialization::make_nvp("Cal3Rational",
        boost::serialization::base_object<Cal3Rational_Base>(*this));
  }

  /// @}

};

}

template<>
struct gtsam::traits<kinetic_backend::Cal3Rational> : public gtsam::internal::Manifold<kinetic_backend::Cal3Rational> {};

template<>
struct gtsam::traits<const kinetic_backend::Cal3Rational> : public gtsam::internal::Manifold<kinetic_backend::Cal3Rational> {};



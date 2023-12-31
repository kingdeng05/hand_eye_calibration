#include <boost/shared_ptr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>  // Necessary for py::self

#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/PinholeCamera.h>

#include "general_projection_factor.h"
#include "cam2tt_projection_factor.h"
#include "lidar2tt_point2plane_factor.h"
#include "hand_eye_factors/dhe_factor.h"
#include "hand_eye_factors/rm_factor.h"
#include "hand_eye_factors/he_pose_constraint_factor.h"
#include "base_tt_factors/track_pose_factor.h"
#include "base_tt_factors/hand_pose_factor.h"
#include "base_tt_factors/tt_pose_factor.h"
#include "base_tt_factors/base_tt_projection_factor.h"
#include "geometry/Cal3Rational.h"
#include "geometry/surfel3.h"

namespace py = pybind11;
using namespace gtsam;
using namespace kinetic_backend;

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

PYBIND11_MODULE(py_kinetic_backend, m) {
    m.doc() = "pybind11 wrapper for GTSAM";

    // Key and Symbol
    py::class_<Key>(m, "Key")
        .def(py::init<>());

    // m.def("symbol", py::overload_cast<unsigned char, std::uint64_t>(&symbol), "Create a new key using character and index");
    m.def("symbol", [](py::object c, std::uint64_t j) {
        if (py::isinstance<py::str>(c) && py::len(c) == 1) {
            std::string c_str = py::str(c);
            return symbol((unsigned char)c_str[0], j);
        }
        throw std::runtime_error("Expected a single-character string for argument 1");
    }, "Create a new key using character and index");

    // OptionalJacobian
    py::class_<OptionalJacobian<2, 3>>(m, "OptionalJacobian23");
    py::class_<OptionalJacobian<2, 5>>(m, "OptionalJacobian25");
    py::class_<OptionalJacobian<2, 6>>(m, "OptionalJacobian26");
    py::class_<OptionalJacobian<2, 11>>(m, "OptionalJacobian211");
    py::class_<OptionalJacobian<2, 15>>(m, "OptionalJacobian215");
    py::class_<OptionalJacobian<2, 18>>(m, "OptionalJacobian218");
    py::class_<OptionalJacobian<3, 2>>(m, "OptionalJacobian32");
    py::class_<OptionalJacobian<3, 3>>(m, "OptionalJacobian33");
    py::class_<OptionalJacobian<3, 6>>(m, "OptionalJacobian36");
    py::class_<OptionalJacobian<6, 6>>(m, "OptionalJacobian66");

    // Rot3
    py::class_<Rot3, boost::shared_ptr<Rot3>>(m, "Rot3")
        .def(py::init<>())
        .def(py::init<const Matrix3&>())
        .def(py::init<double, double, double, double>())
        .def(py::self * py::self)
        .def("matrix", &Rot3::matrix)
        .def("quaternion", static_cast<Vector (Rot3::*)() const>(&Rot3::quaternion))
        .def("xyz", &Rot3::xyz)
        .def_static("random", &Rot3::Random)
        .def_static("rodrigues", py::overload_cast<const Vector3&>(&Rot3::Rodrigues))
        .def_static("rzryrx", py::overload_cast<const Vector&>(&Rot3::RzRyRx))
        .def_static("logmap", &Rot3::Logmap, py::arg(), py::arg()=OptionalJacobian<3, 3>())
        .def_static("expmap", &Rot3::Expmap, py::arg(), py::arg()=OptionalJacobian<3, 3>())
        ;

    // Pose3
    py::class_<Pose3, boost::shared_ptr<Pose3>>(m, "Pose3")
        .def(py::init<>())
        .def(py::init<const Rot3&, const Point3&>())
        .def(py::init<const Matrix&>())
        .def(py::self * py::self)
        .def("matrix", &Pose3::matrix)
        .def("rotation", &Pose3::rotation, py::arg()=OptionalJacobian<3, 6>())
        .def("translation", &Pose3::translation, py::arg()=OptionalJacobian<3, 6>())
        .def("compose", py::overload_cast<const Pose3&>(&Pose3::compose, py::const_))
        .def("between", py::overload_cast<const Pose3&>(&Pose3::between, py::const_))
        .def("transform_to", &Pose3::transformTo, py::arg(), 
             py::arg()=OptionalJacobian<3, 6>(), py::arg()=OptionalJacobian<3, 3>())
        .def("transform_from", &Pose3::transformFrom, py::arg(), 
             py::arg()=OptionalJacobian<3, 6>(), py::arg()=OptionalJacobian<3, 3>())
        .def_static("logmap", &Pose3::Logmap, py::arg(), py::arg()=OptionalJacobian<6, 6>())
        .def_static("expmap", &Pose3::Expmap, py::arg(), py::arg()=OptionalJacobian<6, 6>())
        ;

    // Values
    py::class_<Values, boost::shared_ptr<Values>>(m, "Values")
        .def(py::init<>())
        .def("insertPose3", &gtsam::Values::insert<Pose3>)
        .def("insertCal3_S2", &gtsam::Values::insert<Cal3_S2>)
        .def("insertCal3DS2", &gtsam::Values::insert<Cal3DS2>)
        .def("insertCal3Rational", &gtsam::Values::insert<Cal3Rational>)
        .def("atPose3", &Values::at<Pose3>)
        .def("atCal3_S2", &Values::at<Cal3_S2>)
        .def("atCal3DS2", &Values::at<Cal3DS2>)
        .def("atCal3Rational", &Values::at<Cal3Rational>)
        ;

    // NonlinearFactorGraph
    py::class_<NonlinearFactorGraph, boost::shared_ptr<NonlinearFactorGraph>>(m, "NonlinearFactorGraph")
        .def(py::init<>())
        .def("add", static_cast<void (NonlinearFactorGraph::*)(const boost::shared_ptr<NonlinearFactor>&)>(&NonlinearFactorGraph::add))
        .def("error", &NonlinearFactorGraph::error)
        .def("__len__", [](const NonlinearFactorGraph &graph) {
            return graph.size();
        })
        .def("__getitem__", [](const NonlinearFactorGraph &graph, size_t i) {
            if (i >= graph.size()) throw py::index_error();
            return graph[i];
        })
        ;

    // LevenbergMarquardtOptimizer
    py::class_<LevenbergMarquardtOptimizer, boost::shared_ptr<LevenbergMarquardtOptimizer>>(m, "LevenbergMarquardtOptimizer")
        .def(py::init<const NonlinearFactorGraph&, const Values&>())
        .def("optimize", &LevenbergMarquardtOptimizer::optimize);

    // noise models
    py::class_<noiseModel::Base, boost::shared_ptr<noiseModel::Base>>(m, "Base");
    py::class_<noiseModel::mEstimator::Base, boost::shared_ptr<noiseModel::mEstimator::Base>>(m, "mEstimatorBase");
    py::class_<noiseModel::Diagonal, boost::shared_ptr<noiseModel::Diagonal>, noiseModel::Base>(m, "Diagonal")
        .def_static("sigmas", &noiseModel::Diagonal::Sigmas, py::arg(), py::arg()=true)
        ;
    py::class_<noiseModel::Robust, std::shared_ptr<noiseModel::Robust>, noiseModel::Base>(m, "RobustNoiseModel")
        .def_static("create", &noiseModel::Robust::Create, py::arg("robust"), py::arg("noise"))
        ;
    py::enum_<noiseModel::mEstimator::Base::ReweightScheme>(m, "ReweightScheme")
        .value("Scalar", noiseModel::mEstimator::Base::ReweightScheme::Scalar)
        .value("Block", noiseModel::mEstimator::Base::ReweightScheme::Block)
        .export_values()
        ;
    py::class_<noiseModel::mEstimator::Cauchy, noiseModel::mEstimator::Base, std::shared_ptr<noiseModel::mEstimator::Cauchy>>(m, "CauchyNoiseModel")
        .def_static("create", &noiseModel::mEstimator::Cauchy::Create, py::arg("scale"), py::arg("Reweight")=noiseModel::mEstimator::Base::ReweightScheme::Block)
        ;

    // non linear factor
    py::class_<NonlinearFactor, boost::shared_ptr<NonlinearFactor>>(m, "NonlinearFactor")
        .def("error", &NonlinearFactor::error)
        ;

    // camera calibration
    py::class_<PinholeCamera<Cal3_S2>, boost::shared_ptr<PinholeCamera<Cal3_S2>>>(m, "PinholeCameraCal3_S2")
        .def(py::init<const Pose3&, const Cal3_S2&>())
        .def("project", static_cast<Point2 (PinholeCamera<Cal3_S2>::*)(const Point3&, OptionalJacobian<2, 11>, OptionalJacobian<2, 3>) const>(&PinholeCamera<Cal3_S2>::project2),
            py::arg(), py::arg()=OptionalJacobian<2, 11>(),
            py::arg()=OptionalJacobian<2, 3>())
        ;
    py::class_<PinholeCamera<Cal3DS2>, boost::shared_ptr<PinholeCamera<Cal3DS2>>>(m, "PinholeCameraCal3DS2")
        .def(py::init<const Pose3&, const Cal3DS2&>())
        .def("project", static_cast<Point2 (PinholeCamera<Cal3DS2>::*)(const Point3&, OptionalJacobian<2, 15>, OptionalJacobian<2, 3>) const>(&PinholeCamera<Cal3DS2>::project2),
            py::arg(), py::arg()=OptionalJacobian<2, 15>(),
            py::arg()=OptionalJacobian<2, 3>())
        ;
    py::class_<PinholeCamera<Cal3Rational>, boost::shared_ptr<PinholeCamera<Cal3Rational>>>(m, "PinholeCameraCal3Rational")
        .def(py::init<const Pose3&, const Cal3Rational&>())
        .def("project", static_cast<Point2 (PinholeCamera<Cal3Rational>::*)(const Point3&, OptionalJacobian<2, 18>, OptionalJacobian<2, 3>) const>(&PinholeCamera<Cal3Rational>::project2),
            py::arg(), py::arg()=OptionalJacobian<2, 18>(),
            py::arg()=OptionalJacobian<2, 3>())
        ;
    py::class_<Cal3_S2, boost::shared_ptr<Cal3_S2>>(m, "Cal3_S2")
        .def(py::init<const Vector&>())
        .def("vector", &Cal3_S2::vector)
        ;
    py::class_<Cal3DS2, boost::shared_ptr<Cal3DS2>>(m, "Cal3DS2")
        .def(py::init<const Vector&>())
        .def("vector", &Cal3DS2::vector)
        ;
    py::class_<Cal3Rational, boost::shared_ptr<Cal3Rational>>(m, "Cal3Rational")
        .def(py::init<const Vector&>())
        .def("vector", &Cal3Rational::vector)
        ;

    // surfel3
    py::class_<Surfel3, boost::shared_ptr<Surfel3>>(m, "Surfel3")
        .def(py::init<const Point3&, const Unit3&, double>())
        .def_readwrite("center", &Surfel3::center)
        .def_readwrite("normal", &Surfel3::normal)
        .def_readwrite("radius", &Surfel3::radius)
        .def("distance", &Surfel3::Distance)
        ;
    py::class_<Unit3, boost::shared_ptr<Unit3>>(m, "Unit3")
        .def(py::init<const Vector3&>())
        .def("point3", &Unit3::point3, py::arg()=OptionalJacobian<3, 2>())
        ;

    // PriorFactor
    py::class_<PriorFactor<Pose3>, boost::shared_ptr<PriorFactor<Pose3>>, NonlinearFactor>(m, "PriorFactorPose3")
        .def(py::init<>())
        .def(py::init<Key, const Pose3&, const SharedNoiseModel&>());
    py::class_<PriorFactor<Cal3DS2>, boost::shared_ptr<PriorFactor<Cal3DS2>>, NonlinearFactor>(m, "PriorFactorCal3DS2")
        .def(py::init<>())
        .def(py::init<Key, const Cal3DS2&, const SharedNoiseModel&>());

    // Between factor
    py::class_<BetweenFactor<Pose3>, boost::shared_ptr<BetweenFactor<Pose3>>, NonlinearFactor>(m, "BetweenFactorPose3")
        .def(py::init<>())
        .def(py::init<Key, Key, const Pose3&, const SharedNoiseModel&>());

    // Custom factors
    py::class_<DHEFactor, boost::shared_ptr<DHEFactor>, NonlinearFactor>(m, "DHEFactor")
        .def(py::init<Key, Key, Key, Key, Key, const SharedNoiseModel&>(),
             py::arg("X"), py::arg("E_i"), py::arg("E_j"), py::arg("C_i"), py::arg("C_j"), py::arg("model"))
            ;
    py::class_<RMFactor<Cal3_S2>, boost::shared_ptr<RMFactor<Cal3_S2>>, NonlinearFactor>(m, "RMFactorCal3_S2")
        .def(py::init<Key, Key, Key, Key, const Point3, const Point2, const SharedNoiseModel&, bool, bool, bool, bool>(),
             py::arg("hand2eye"), py::arg("world2hand"), py::arg("target2world"), py::arg("intrinsic"), 
             py::arg("point_in_target"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_hand2eye")=false, py::arg("fix_world2hand")=true, py::arg("fix_target2world")=false,
             py::arg("fix_intrinsic")=true)
            ;
    py::class_<RMFactor<Cal3DS2>, boost::shared_ptr<RMFactor<Cal3DS2>>, NonlinearFactor>(m, "RMFactorCal3DS2")
        .def(py::init<Key, Key, Key, Key, const Point3, const Point2, const SharedNoiseModel&, bool, bool, bool, bool>(),
             py::arg("hand2eye"), py::arg("world2hand"), py::arg("target2world"), py::arg("intrinsic"), 
             py::arg("point_in_target"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_hand2eye")=false, py::arg("fix_world2hand")=true, py::arg("fix_target2world")=false,
             py::arg("fix_intrinsic")=true)
            ;

    py::class_<HEPoseConstraintFactor, boost::shared_ptr<HEPoseConstraintFactor>, NonlinearFactor>(m, "HEPoseConstraintFactor")
        .def(py::init<Key, Key, Key, const Pose3, const SharedNoiseModel&, bool, bool, bool>(),
             py::arg("eye2hand"), py::arg("world2target"), py::arg("target2eye"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_eye2hand")=false, py::arg("fix_world2target")=false, py::arg("fix_target2eye")=false)
            ;

    py::class_<GeneralProjectionFactor<Cal3DS2>, boost::shared_ptr<GeneralProjectionFactor<Cal3DS2>>, NonlinearFactor>(m, "GeneralProjectionFactorCal3DS2")
        .def(py::init<Key, Key, const Point3, const Point2, const SharedNoiseModel&, bool, bool>(),
             py::arg("world2cam"), py::arg("intrinsic"), 
             py::arg("point_in_target"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_world2cam")=false, py::arg("fix_intrinsic")=false)
            ;

    py::class_<GeneralProjectionFactor<Cal3Rational>, boost::shared_ptr<GeneralProjectionFactor<Cal3Rational>>, NonlinearFactor>(m, "GeneralProjectionFactorCal3Rational")
        .def(py::init<Key, Key, const Point3, const Point2, const SharedNoiseModel&, bool, bool>(),
             py::arg("world2cam"), py::arg("intrinsic"), 
             py::arg("point_in_target"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_world2cam")=false, py::arg("fix_intrinsic")=false)
            ;

    py::class_<Cam2TtProjectionFactor<Cal3Rational>, boost::shared_ptr<Cam2TtProjectionFactor<Cal3Rational>>, NonlinearFactor>(m, "Cam2TtProjectionFactorCal3Rational")
        .def(py::init<Key, Key, Key, const Cal3Rational&, const Point3, const Point2, const SharedNoiseModel&, bool, bool, bool>(),
             py::arg("cam2tt"), py::arg("target2tt"), 
             py::arg("tt2tt0"), py::arg("intrinsic"), py::arg("pt_3d"),
             py::arg("measurement"), py::arg("model"), py::arg("fix_cam2tt")=false,
             py::arg("fix_target2tt")=false, py::arg("fix_tt2tt0")=false)
            ;

    py::class_<TrackPoseFactor, boost::shared_ptr<TrackPoseFactor>, NonlinearFactor>(m, "TrackPoseFactor")
        .def(py::init<Key, Key, Key, Key, const Pose3, const SharedNoiseModel&, bool, bool, bool, bool>(),
             py::arg("base2track"), py::arg("target2base"), py::arg("target2tt"), py::arg("track2tt"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_base2track")=false, py::arg("fix_target2base")=false, py::arg("fix_target2tt")=false, py::arg("fix_track2tt")=false)
            ;

    py::class_<HandPoseFactor, boost::shared_ptr<HandPoseFactor>, NonlinearFactor>(m, "HandPoseFactor")
        .def(py::init<Key, Key, Key, const Pose3, const SharedNoiseModel&, bool, bool, bool>(),
             py::arg("cam2ee"), py::arg("target2base"), py::arg("target2cam"), py::arg("ee2base_meas"), py::arg("model"),
             py::arg("fix_cam2ee")=true, py::arg("fix_target2base")=false, py::arg("fix_target2cam")=false)
            ;

    py::class_<TtPoseFactor, boost::shared_ptr<TtPoseFactor>, NonlinearFactor>(m, "TtPoseFactor")
        .def(py::init<Key, Key, const Pose3, const SharedNoiseModel&, bool, bool>(),
             py::arg("target2tt_i"), py::arg("target2tt_0"), py::arg("tti2tt0_meas"), py::arg("model"),
             py::arg("fix_target2tt_i")=true, py::arg("fix_target2tt_0")=false)
            ;

    py::class_<BaseTtProjectionFactor<Cal3Rational>, boost::shared_ptr<BaseTtProjectionFactor<Cal3Rational>>, NonlinearFactor>(m, "BaseTtProjectionFactor")
        .def(py::init<Key, Key, Key, Key, Key, Key, const Point3&, const Point2&, const Pose3&, const Cal3Rational&, const SharedNoiseModel&, bool, bool, bool, bool, bool, bool>(),
             py::arg("ee2base"), py::arg("base2track"), py::arg("track2track0"), py::arg("track2tt"),
             py::arg("tt2tt0"), py::arg("target2tt"), py::arg("point_3d"), py::arg("point_2d"), py::arg("cam2ee"),
             py::arg("intrinsic"), py::arg("model"), py::arg("fix_ee2base")=false, py::arg("fix_base2track")=false,
             py::arg("fix_track2track0")=false, py::arg("fix_track2tt")=false, py::arg("fix_tt2tt0")=false,
             py::arg("fix_target2tt")=false)
            ;

    py::class_<LiDAR2TtPoint2PlaneFactor, boost::shared_ptr<LiDAR2TtPoint2PlaneFactor>, NonlinearFactor>(m, "LiDAR2TtPoint2PlaneFactor")
        .def(py::init<Key, Key, Key, const Point3&, const Surfel3&, const SharedNoiseModel&, bool, bool, bool>(),
             py::arg("sensor2tt"), py::arg("target2tt"), 
             py::arg("tt2tt0"), py::arg("measured"), py::arg("surfel"),
             py::arg("model"), py::arg("fix_sensor2tt")=false,
             py::arg("fix_target2tt")=false, py::arg("fix_tt2tt0")=false)
            ;

}

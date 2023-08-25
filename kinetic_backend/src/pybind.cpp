#include <boost/shared_ptr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>

#include "dhe_factor.h"
#include "rm_factor.h"

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
    py::class_<OptionalJacobian<3, 3>>(m, "OptionalJacobian33");
    py::class_<OptionalJacobian<3, 6>>(m, "OptionalJacobian36");
    py::class_<OptionalJacobian<6, 6>>(m, "OptionalJacobian66");

    // Rot3
    py::class_<Rot3, boost::shared_ptr<Rot3>>(m, "Rot3")
        .def(py::init<>())
        .def(py::init<const Matrix3&>())
        .def(py::init<double, double, double, double>())
        .def("matrix", &Rot3::matrix)
        .def_static("Random", &Rot3::Random)
        .def_static("Rodrigues", py::overload_cast<const Vector3&>(&Rot3::Rodrigues))
        .def_static("RzRyRx", py::overload_cast<const Vector3&>(&Rot3::Rodrigues))
        .def_static("Logmap", &Rot3::Logmap, py::arg(), py::arg()=OptionalJacobian<3, 3>())
        .def_static("Expmap", &Rot3::Expmap, py::arg(), py::arg()=OptionalJacobian<3, 3>())
        ;

    // Pose3
    py::class_<Pose3, boost::shared_ptr<Pose3>>(m, "Pose3")
        .def(py::init<>())
        .def(py::init<const Rot3&, const Point3&>())
        .def(py::init<const Matrix&>())
        .def("matrix", &Pose3::matrix)
        .def("rotation", &Pose3::rotation, py::arg()=OptionalJacobian<3, 6>())
        .def("translation", &Pose3::translation, py::arg()=OptionalJacobian<3, 6>())
        .def("compose", py::overload_cast<const Pose3&>(&Pose3::compose, py::const_))
        .def("between", py::overload_cast<const Pose3&>(&Pose3::between, py::const_))
        .def_static("logmap", &Pose3::Logmap, py::arg(), py::arg()=OptionalJacobian<6, 6>())
        .def_static("expmap", &Pose3::Expmap, py::arg(), py::arg()=OptionalJacobian<6, 6>())
        ;

    // Values
    py::class_<Values, boost::shared_ptr<Values>>(m, "Values")
        .def(py::init<>())
        // .def("insert", static_cast<void (gtsam::Values::*)(gtsam::Key, const gtsam::Value&)>(&gtsam::Values::insert))
        .def("insertPose3", &gtsam::Values::insert<Pose3>)
        // .def("atPose3", static_cast<const Pose3& (Values::*)(Key) const>(&Values::at));
        .def("atPose3", &Values::at<Pose3>);

    // NonlinearFactorGraph
    py::class_<NonlinearFactorGraph, boost::shared_ptr<NonlinearFactorGraph>>(m, "NonlinearFactorGraph")
        .def(py::init<>())
        .def("add", static_cast<void (NonlinearFactorGraph::*)(const boost::shared_ptr<NonlinearFactor>&)>(&NonlinearFactorGraph::add))
        .def("error", &NonlinearFactorGraph::error)
        ;

    // LevenbergMarquardtOptimizer
    py::class_<LevenbergMarquardtOptimizer, boost::shared_ptr<LevenbergMarquardtOptimizer>>(m, "LevenbergMarquardtOptimizer")
        .def(py::init<const NonlinearFactorGraph&, const Values&>())
        .def("optimize", &LevenbergMarquardtOptimizer::optimize);

    // PriorFactor
    // py::class_<PriorFactor<Pose3>, boost::shared_ptr<PriorFactor<Pose3>>>(m, "PriorFactor")
    py::class_<PriorFactor<Pose3>, boost::shared_ptr<PriorFactor<Pose3>>>(m, "PriorFactorPose3")
        .def(py::init<>())
        .def(py::init<Key, const Pose3&, const SharedNoiseModel&>());

    // noise models
    py::class_<noiseModel::Base, boost::shared_ptr<noiseModel::Base>>(m, "Base");

    py::class_<noiseModel::Diagonal, boost::shared_ptr<noiseModel::Diagonal>, noiseModel::Base>(m, "Diagonal")
        .def_static("sigmas", &noiseModel::Diagonal::Sigmas, py::arg(), py::arg()=true)
        ;

    py::class_<NonlinearFactor, boost::shared_ptr<NonlinearFactor>>(m, "NonlinearFactor");

    // Custom factors
    py::class_<DHEFactor, boost::shared_ptr<DHEFactor>, NonlinearFactor>(m, "DHEFactor")
        .def(py::init<Key, Key, Key, Key, Key, const SharedNoiseModel&>(),
             py::arg("X"), py::arg("E_i"), py::arg("E_j"), py::arg("C_i"), py::arg("C_j"), py::arg("model"))
            ;
    py::class_<RMFactor, boost::shared_ptr<RMFactor>, NonlinearFactor>(m, "RMFactor")
        .def(py::init<Key, Key, Key, Key, const Point3, const Point2, const SharedNoiseModel&, bool, bool, bool, bool>(),
             py::arg("hand2eye"), py::arg("world2hand"), py::arg("target2world"), py::arg("intrinsic"), 
             py::arg("point_in_target"), py::arg("measurement"), py::arg("model"),
             py::arg("fix_hand2eye")=false, py::arg("fix_world2hand")=true, py::arg("fix_target2world")=false,
             py::arg("fix_intrinsic")=true)
            ;

}

#include <memory>
#include <string>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/Data.h"

namespace py = pybind11;

class PyFM {
 public:
  PyFM(const std::string& method,
       const std::vector<int>& dim,
       const double lr=0.1,
       const std::vector<double>& reg={},
       const double init_stdev=0.1,
       const int num_iter=100,
       const int num_eval_cases=-1,
       const std::string& r_log_str="",
       const int verbosity=0);

  void train(std::shared_ptr<Data> train,
             std::shared_ptr<Data> test=nullptr,
             std::shared_ptr<Data> validation=nullptr);

  Eigen::VectorXd predict(std::shared_ptr<Data> test);

  std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> parameters();

 private:
  const std::string method;
  const std::vector<double> reg;
  const int num_eval_cases;
  const int verbosity;
  DataMetaInfo meta = DataMetaInfo(0);
  std::unique_ptr<std::ofstream> out_rlog;
  std::unique_ptr<RLog> rlog;
  fm_model fm;
  std::unique_ptr<fm_learn> fml;
};


PYBIND11_MODULE(pyfm, m) {
  py::class_<PyFM>(m, "PyFM")
    .def(py::init<const std::string&,
                  const std::vector<int>&,
                  const double,
                  const std::vector<double>&,
                  const double,
                  const int,
                  const int,
                  const std::string&,
                  const int>(),
         py::arg("method"),
         py::arg("dim"),
         py::arg("lr") = 0.1,
         py::arg("reg") = std::vector<double>(),
         py::arg("init_stdev") = 0.1,
         py::arg("num_iter") = 100,
         py::arg("num_eval_cases") = -1,
         py::arg("r_log_str") = "",
         py::arg("verbosity") = 0)
    .def("train",
         &PyFM::train,
         py::arg("train"),
         py::arg("test") = nullptr,
         py::arg("validation") = nullptr)
    .def("predict",
         &PyFM::predict,
         py::arg("test") = nullptr)
    .def("parameters",
         &PyFM::parameters);

  py::class_<Data, std::shared_ptr<Data>>(m, "Data")
    .def(py::init<const Eigen::SparseMatrix<double, Eigen::RowMajor>&,
                  const Eigen::VectorXd&,
                  bool>(),
         py::arg("data"),
         py::arg("target"),
         py::arg("has_xt") = false)
    .def("add_rows",
         &Data::add_rows,
         py::arg("data"),
         py::arg("target"));
}

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
       const std::vector<double>& lr={},
       const std::vector<double>& reg={},
       const double init_stdev=0.1,
       const int num_iter=100,
       const int num_eval_cases=-1,
       const bool do_sampling=true,
       const bool do_multilevel=true,
       const std::string& r_log_str="",
       const int verbosity=0);

  void train(std::shared_ptr<Data> train,
             std::shared_ptr<Data> test=nullptr,
             std::shared_ptr<Data> validation=nullptr);

  Eigen::VectorXd predict(std::shared_ptr<Data> test);

 private:
  const std::string method;
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
                  const std::vector<double>&,
                  const std::vector<double>&,
                  const double,
                  const int,
                  const int,
                  const bool,
                  const bool,
                  const std::string&,
                  const int>(),
         py::arg("method"),
         py::arg("dim"),
         py::arg("lr") = std::vector<double>(),
         py::arg("reg") = std::vector<double>(),
         py::arg("init_stdev") = 0.1,
         py::arg("num_iter") = 100,
         py::arg("num_eval_cases") = -1,
         py::arg("do_sampling") = true,
         py::arg("do_multilevel") = true,
         py::arg("r_log_str") = "",
         py::arg("verbosity") = 0)
    .def("train",
         &PyFM::train,
         py::arg("train"),
         py::arg("test") = nullptr,
         py::arg("validation") = nullptr)
    .def("predict",
         &PyFM::predict,
         py::arg("test") = nullptr);

  py::class_<Data>(m, "Data")
    .def(py::init<uint64, bool, bool>(),
         py::arg("cache_size"),
         py::arg("has_x"),
         py::arg("has_xt"))
    .def("set_data",
         &Data::set_data,
         py::arg("data"),
         py::arg("target"))
    .def("add_rows",
         &Data::add_rows,
         py::arg("data"),
         py::arg("target"));
}

#include <Eigen/Dense>
#include <memory>
#include <string>

#include "src/Data.h"

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

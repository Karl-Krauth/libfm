#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <memory>
#include "../util/util.h"
#include "../util/cmdline.h"
#include "../fm_core/fm_model.h"
#include "src/Data.h"
#include "src/fm_learn.h"
#include "src/fm_learn_sgd.h"
#include "src/fm_learn_sgd_element.h"
#include "src/fm_learn_sgd_element_adapt_reg.h"
#include "src/fm_learn_mcmc_simultaneous.h"
#include "pyfm.h"


PyFM::PyFM(const std::string& method,
           const std::vector<int>& dim,
           const std::vector<double>& lr,
           const std::vector<double>& reg,
           const double init_stdev,
           const int num_iter,
           const int num_eval_case,
           const bool do_sampling,
           const bool do_multilevel,
           const std::string& r_log_str,
           const int verbosity) :
           method{method},
           num_eval_cases{num_eval_cases},
           verbosity{verbosity} {
  // Setup the factorization machine
  this->fm = fm_model();
  this->fm.init_stdev = init_stdev;
  // Set the number of dimensions in the factorization machine.
  assert(dim.size() == 3);
  this->fm.k0 = dim[0] != 0;
  this->fm.k1 = dim[1] != 0;
  this->fm.num_factor = dim[2];

  // Setup the learning method.
  if (method == "sgd") {
    this->fml = std::make_unique<fm_learn_sgd_element>();
    ((fm_learn_sgd*)this->fml.get())->num_iter = num_iter;
  } else if (method == "sgda") {
    this->fml = std::make_unique<fm_learn_sgd_element_adapt_reg>();
    ((fm_learn_sgd*)this->fml.get())->num_iter = num_iter;
  } else if (method == "mcmc") {
    this->fml = std::make_unique<fm_learn_mcmc_simultaneous>();
    this->fm.w.init_normal(this->fm.init_mean, this->fm.init_stdev);
    ((fm_learn_mcmc*)this->fml.get())->num_iter = num_iter;
    ((fm_learn_mcmc*)this->fml.get())->do_sample = do_sampling;
    ((fm_learn_mcmc*)this->fml.get())->do_multilevel = do_multilevel;
  } else {
    throw "Unknown method.";
  }

  this->fml->fm = &(this->fm);
  // Assume we only do regression.
  this->fml->meta = &(this->meta);
  this->fml->task = 0;

  // Init the logging
  this->rlog = nullptr;
  if (!r_log_str.empty()) {
    this->out_rlog = std::make_unique<std::ofstream>(r_log_str.c_str());
    if (!this->out_rlog->is_open())  {
      throw "Unable to open file " + r_log_str;
    }
    std::cout << "logging to " << r_log_str.c_str() << std::endl;
    this->rlog = std::make_unique<RLog>(out_rlog.get());
  }

  this->fml->log = rlog.get();
  if (method == "mcmc") {
    // set the regularization; for als and mcmc this can be individual per group
    assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3) || (reg.size() == (1+this->fml->meta->num_attr_groups*2)));
    if (reg.size() == 0) {
      this->fm.reg0 = 0.0;
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else if (reg.size() == 1) {
      this->fm.reg0 = reg[0];
      this->fm.regw = reg[0];
      this->fm.regv = reg[0];
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else if (reg.size() == 3) {
      this->fm.reg0 = reg[0];
      this->fm.regw = reg[1];
      this->fm.regv = reg[2];
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else {
      this->fm.reg0 = reg[0];
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
      int j = 1;
      for (uint g = 0; g < this->fml->meta->num_attr_groups; g++) {
        ((fm_learn_mcmc*)this->fml.get())->w_lambda(g) = reg[j];
        j++;
      }
      for (uint g = 0; g < this->fml->meta->num_attr_groups; g++) {
        for (int f = 0; f < this->fm.num_factor; f++) {
          ((fm_learn_mcmc*)this->fml.get())->v_lambda(g,f) = reg[j];
        }
        j++;
      }
    }
  } else {
    // set the regularization; for standard SGD, groups are not supported
    assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
    if (reg.size() == 0) {
      this->fm.reg0 = 0.0;
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
    } else if (reg.size() == 1) {
      this->fm.reg0 = reg[0];
      this->fm.regw = reg[0];
      this->fm.regv = reg[0];
    } else {
      this->fm.reg0 = reg[0];
      this->fm.regw = reg[1];
      this->fm.regv = reg[2];
    }
  }
  {
    fm_learn_sgd* fmlsgd = dynamic_cast<fm_learn_sgd*>(this->fml.get());
    if (fmlsgd) {
      // set the learning rates (individual per layer)
      {
        assert((lr.size() == 1) || (lr.size() == 3));
        if (lr.size() == 1) {
          fmlsgd->learn_rate = lr[0];
          fmlsgd->learn_rates.init(lr[0]);
        } else {
          fmlsgd->learn_rate = 0;
          fmlsgd->learn_rates(0) = lr[0];
          fmlsgd->learn_rates(1) = lr[1];
          fmlsgd->learn_rates(2) = lr[2];
        }
      }
    }
  }

  if (rlog != nullptr) {
    rlog->init();
  }
}

void PyFM::train(std::shared_ptr<Data> train,
                 std::shared_ptr<Data> test,
                 std::shared_ptr<Data> validation) {
  assert(train != nullptr);
  this->fm.num_attribute = train->num_feature;
  if (test != nullptr) {
    this->fm.num_attribute = std::max((int)this->fm.num_attribute, test->num_feature);
  }

  if (validation != nullptr) {
    this->fm.num_attribute = std::max((int)this->fm.num_attribute, validation->num_feature);
  }

  this->fm.init();

  if (this->method == "sgda") {
    assert(validation != nullptr);
    ((fm_learn_sgd_element_adapt_reg*)this->fml.get())->validation = validation.get();
  } else if (this->method == "mcmc") {
    this->fml->validation = validation.get();
    if (this->num_eval_cases == -1) {
      if (test == nullptr) {
        ((fm_learn_mcmc*)this->fml.get())->num_eval_cases = 0;
      } else {
        ((fm_learn_mcmc*)this->fml.get())->num_eval_cases = test->num_cases;
      }
    }
  }

  train->debug();
  std::cout << "BBBB" << std::endl;
  this->fml->max_target = train->max_target;
  this->fml->min_target = train->min_target;
  this->meta = DataMetaInfo(this->fm.num_attribute);
  this->fml->init();
  if (this->verbosity > 0) {
    this->fm.debug();
    this->fml->debug();
  }

  // learn
  this->fml->learn(*train, *test);
  std::cout << "GGGG" << std::endl;

  //  Prediction at the end  (not for mcmc and als)
  if (method != "mcmc") {
    std::cout << "Final\t" << "Train=" << this->fml->evaluate(*train) << "\tTest=" <<
      this->fml->evaluate(*test) << std::endl;
  }
}

Eigen::VectorXd PyFM::predict(std::shared_ptr<Data> test) {
  DVector<double> pred;
  pred.setSize(test->num_cases);
  fml->predict(*test, pred);
  Eigen::VectorXd pred_vector(pred.dim);
  for (uint i = 0; i < pred.dim; ++i) {
    pred_vector[i] = pred(i);
  }

  return pred_vector;
}

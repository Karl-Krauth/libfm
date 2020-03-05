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
           reg{reg},
           num_eval_cases{num_eval_cases},
           verbosity{verbosity} {
  // Setup the factorization machine
  this->fm = fm_model();
  this->fm.init_stdev = init_stdev;

  if (dim.size() != 3) {
    throw "Dim needs to be a list of size 3.";
  }
  // Set the number of dimensions in the factorization machine.
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
  {
    fm_learn_sgd* fmlsgd = dynamic_cast<fm_learn_sgd*>(this->fml.get());
    if (fmlsgd) {
      // set the learning rates (individual per layer)
      {
        if ((lr.size() != 1) && (lr.size() != 3)) {
          throw "Learning rate needs to have size 1 or 3 for SGD.";
        }

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
  if (test == nullptr) {
    test = std::make_shared<Data>(Eigen::SparseMatrix<double, Eigen::RowMajor>(0, 0),
                                  Eigen::VectorXd(0), true);
  }

  this->fm.num_attribute = train->num_feature;
  this->fm.num_attribute = std::max((int)this->fm.num_attribute, test->num_feature);

  if (validation != nullptr) {
    this->fm.num_attribute = std::max((int)this->fm.num_attribute, validation->num_feature);
  }

  this->fm.init();

  if (this->method == "sgda") {
    if (validation == nullptr) {
      throw "Need to provide a validation set when doing sgda.";
    }
    ((fm_learn_sgd_element_adapt_reg*)this->fml.get())->validation = validation.get();
  } else if (this->method == "mcmc") {
    this->fml->validation = validation.get();
    if (this->num_eval_cases == -1) {
      ((fm_learn_mcmc*)this->fml.get())->num_eval_cases = 0;
    }
  }

  this->fml->max_target = train->max_target;
  this->fml->min_target = train->min_target;
  this->meta = DataMetaInfo(this->fm.num_attribute);
  this->meta.num_relations = train->relation.dim;
  this->fml->init();
  if (this->verbosity > 0) {
    this->fm.debug();
    this->fml->debug();
  }

  if (method == "mcmc") {
    // set the regularization; for als and mcmc this can be individual per group
    if ((this->reg.size() != 0) && (this->reg.size() != 1) && (this->reg.size() != 3) &&
        (this->reg.size() != (1+this->fml->meta->num_attr_groups*2))) {
      throw "Regularization for mcmc is of incorrect size.";
    }

    if (this->reg.size() == 0) {
      this->fm.reg0 = 0.0;
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else if (this->reg.size() == 1) {
      this->fm.reg0 = this->reg[0];
      this->fm.regw = this->reg[0];
      this->fm.regv = this->reg[0];
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else if (this->reg.size() == 3) {
      this->fm.reg0 = this->reg[0];
      this->fm.regw = this->reg[1];
      this->fm.regv = this->reg[2];
      ((fm_learn_mcmc*)this->fml.get())->w_lambda.init(this->fm.regw);
      ((fm_learn_mcmc*)this->fml.get())->v_lambda.init(this->fm.regv);
    } else {
      this->fm.reg0 = this->reg[0];
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
      int j = 1;
      for (uint g = 0; g < this->fml->meta->num_attr_groups; g++) {
        ((fm_learn_mcmc*)this->fml.get())->w_lambda(g) = this->reg[j];
        j++;
      }
      for (uint g = 0; g < this->fml->meta->num_attr_groups; g++) {
        for (int f = 0; f < this->fm.num_factor; f++) {
          ((fm_learn_mcmc*)this->fml.get())->v_lambda(g,f) = this->reg[j];
        }
        j++;
      }
    }
  } else {
    // set the regularization; for standard SGD, groups are not supported
    if ((this->reg.size() != 0) && (this->reg.size() != 1) && (this->reg.size() != 3)) {
      throw "For SGD regularization list size needs to be 0, 1, or 3.";
    }

    if (this->reg.size() == 0) {
      this->fm.reg0 = 0.0;
      this->fm.regw = 0.0;
      this->fm.regv = 0.0;
    } else if (this->reg.size() == 1) {
      this->fm.reg0 = this->reg[0];
      this->fm.regw = this->reg[0];
      this->fm.regv = this->reg[0];
    } else {
      this->fm.reg0 = this->reg[0];
      this->fm.regw = this->reg[1];
      this->fm.regv = this->reg[2];
    }
  }

  // learn
  this->fml->learn(*train, *test);

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

std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> PyFM::parameters() {
  std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> ret;

  // Copy the global bias.
  std::get<0>(ret) = 0;
  if (this->fm.k0) {
    std::get<0>(ret) = this->fm.w0;
  }

  // Copy the unary interactions.
  if (this->fm.k1) {
    Eigen::VectorXd weights(this->fm.num_attribute);
    for (uint i = 0; i < this->fm.num_attribute; ++i) {
      weights(i) = this->fm.w(i);
    }

    std::get<1>(ret) = weights;
  }

  // Copy the pairwise interactions.
  Eigen::MatrixXd pairwise(this->fm.num_attribute, this->fm.num_factor);
  for (uint i = 0; i < this->fm.num_attribute; ++i) {
    for (int f = 0; f < this->fm.num_factor; ++f) {
      pairwise(i, f) = this->fm.v(f, i);
    }
  }

  std::get<2>(ret) = pairwise;

  return ret;
}

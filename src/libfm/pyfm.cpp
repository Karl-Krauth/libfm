#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include "../util/util.h"
#include "../util/cmdline.h"
#include "../fm_core/fm_model.h"
#include "src/Data.h"
#include "src/fm_learn.h"
#include "src/fm_learn_sgd.h"
#include "src/fm_learn_sgd_element.h"
#include "src/fm_learn_sgd_element_adapt_reg.h"
#include "src/fm_learn_mcmc_simultaneous.h"


fm_learn* train_fm(Data* train, const std::string& method, const std::vector<int>& dim,
                   Data* test=NULL, Data* validation=NULL, const std::vector<double>& lr={},
                   const std::vector<double>& reg={}, const double init_stdev=0.1, const int num_iter=100,
                   int num_eval_cases=-1, const bool do_sampling=true, const bool do_multilevel=true,
                   const std::string& r_log_str="", const int verbosity=0) {
  assert(train != NULL);
  // Setup the factorization machine
  fm_model* fm = new fm_model;
  fm->num_attribute = train->num_feature;
  if (test != NULL) {
    fm->num_attribute = std::max((int)fm->num_attribute, test->num_feature);
  }

  if (validation != NULL) {
    fm->num_attribute = std::max((int)fm->num_attribute, validation->num_feature);
  }

  fm->init_stdev = init_stdev;
  // set the number of dimensions in the factorization
  assert(dim.size() == 3);
  fm->k0 = dim[0] != 0;
  fm->k1 = dim[1] != 0;
  fm->num_factor = dim[2];
  fm->init();

  // Setup the learning method:
  fm_learn* fml;
  if (method == "sgd") {
    fml = new fm_learn_sgd_element();
    ((fm_learn_sgd*)fml)->num_iter = num_iter;
  } else if (method == "sgda") {
    assert(validation != NULL);
    fml = new fm_learn_sgd_element_adapt_reg();
    ((fm_learn_sgd*)fml)->num_iter = num_iter;
    ((fm_learn_sgd_element_adapt_reg*)fml)->validation = validation;
  } else if (method == "mcmc") {
    fm->w.init_normal(fm->init_mean, fm->init_stdev);
    fml = new fm_learn_mcmc_simultaneous();
    fml->validation = validation;
    ((fm_learn_mcmc*)fml)->num_iter = num_iter;
    if (num_eval_cases == -1) {
      if (test == NULL) {
        num_eval_cases = 0;
      } else {
        num_eval_cases = test->num_cases;
      }
    }

    ((fm_learn_mcmc*)fml)->num_eval_cases = num_eval_cases;
    ((fm_learn_mcmc*)fml)->do_sample = do_sampling;
    ((fm_learn_mcmc*)fml)->do_multilevel = do_multilevel;
  } else {
    throw "unknown method";
  }
  fml->fm = fm;
  fml->max_target = train->max_target;
  fml->min_target = train->min_target;
  fml->meta = new DataMetaInfo(fm->num_attribute);
  // Assume we only do regression.
  fml->task = 0;

  // Init the logging
  RLog* rlog = NULL;
  if (!r_log_str.empty()) {
    std::ofstream* out_rlog = NULL;
    out_rlog = new std::ofstream(r_log_str.c_str());
    if (! out_rlog->is_open())  {
      throw "Unable to open file " + r_log_str;
    }
    std::cout << "logging to " << r_log_str.c_str() << std::endl;
    rlog = new RLog(out_rlog);
  }

  fml->log = rlog;
  fml->init();
  if (method == "mcmc") {
    // set the regularization; for als and mcmc this can be individual per group
    assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3) || (reg.size() == (1+fml->meta->num_attr_groups*2)));
    if (reg.size() == 0) {
      fm->reg0 = 0.0;
      fm->regw = 0.0;
      fm->regv = 0.0;
      ((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
      ((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
    } else if (reg.size() == 1) {
      fm->reg0 = reg[0];
      fm->regw = reg[0];
      fm->regv = reg[0];
      ((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
      ((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
    } else if (reg.size() == 3) {
      fm->reg0 = reg[0];
      fm->regw = reg[1];
      fm->regv = reg[2];
      ((fm_learn_mcmc*)fml)->w_lambda.init(fm->regw);
      ((fm_learn_mcmc*)fml)->v_lambda.init(fm->regv);
    } else {
      fm->reg0 = reg[0];
      fm->regw = 0.0;
      fm->regv = 0.0;
      int j = 1;
      for (uint g = 0; g < fml->meta->num_attr_groups; g++) {
        ((fm_learn_mcmc*)fml)->w_lambda(g) = reg[j];
        j++;
      }
      for (uint g = 0; g < fml->meta->num_attr_groups; g++) {
        for (int f = 0; f < fm->num_factor; f++) {
          ((fm_learn_mcmc*)fml)->v_lambda(g,f) = reg[j];
        }
        j++;
      }
    }
  } else {
    // set the regularization; for standard SGD, groups are not supported
    assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
    if (reg.size() == 0) {
      fm->reg0 = 0.0;
      fm->regw = 0.0;
      fm->regv = 0.0;
    } else if (reg.size() == 1) {
      fm->reg0 = reg[0];
      fm->regw = reg[0];
      fm->regv = reg[0];
    } else {
      fm->reg0 = reg[0];
      fm->regw = reg[1];
      fm->regv = reg[2];
    }
  }
  {
    fm_learn_sgd* fmlsgd= dynamic_cast<fm_learn_sgd*>(fml);
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
  if (rlog != NULL) {
    rlog->init();
  }

  if (verbosity > 0) {
    fm->debug();
    fml->debug();
  }

  // learn
  fml->learn(*train, *test);

  //  Prediction at the end  (not for mcmc and als)
  if (method != "mcmc") {
    std::cout << "Final\t" << "Train=" << fml->evaluate(*train) << "\tTest=" << fml->evaluate(*test) << std::endl;
  }

  return fml;
}

Eigen::VectorXd predict_fm(fm_learn* fml, Data& test) {
  DVector<double> pred;
  pred.setSize(test.num_cases);
  fml->predict(test, pred);
  Eigen::VectorXd pred_vector(pred.dim);
  for (uint i = 0; i < pred.dim; ++i) {
    pred_vector[i] = pred(i);
  }

  return pred_vector;
}

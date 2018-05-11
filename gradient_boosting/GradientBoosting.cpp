#include "gradient_boosting/GradientBoosting.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <gradient_boosting/data_transformer/DataTransformer.h>
#include <gradient_boosting/trees/GradientBoostingTreeOblivious.h>
#include <gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h>

namespace gradient_boosting {


using std::unordered_map;
using std::unique_ptr;
using std::string;
using std::vector;

using InternalDataContainer = gradient_boosting::internal_data_container::InternalDataContainer;
using GradientBoostingLossFunction = gradient_boosting::loss_functions::GradientBoostingLossFunction;
using GradientBoostingMSELossFunction = gradient_boosting::loss_functions::GradientBoostingMSELossFunction;
using GradientBoostingConfig = gradient_boosting::config::GradientBoostingConfig;
using DataContainer = utils::data_containers::DataContainer;
using DataTransformer = gradient_boosting::data_transformer::DataTransformer;
using GradientBoostingTreeOblivious = gradient_boosting::trees::GradientBoostingTreeOblivious;

GradientBoosting::GradientBoosting(
    const GradientBoostingConfig& config)
    : config_(config)
    , thread_pool_(config.GetNumberOfThreads())
    , number_of_trees_(config.GetNumberOfTrees())
{
}

vector<size_t> GetNumberedVector (size_t size) {
  vector<size_t> result(size, 0);
  for (size_t index = 0; index < size; ++index) {
    result[index] = index;
  }
  return result;
}

vector<size_t> GetSample(size_t number, size_t sample_size) {
  vector<size_t> res = GetNumberedVector(number);
  std::random_shuffle(res.begin(), res.end());
  res.resize(sample_size);
  return res;
}

void GradientBoosting::TestGradientBoosting(
    const utils::data_containers::DataContainer& data) {
  for (size_t index = 0; index < data.columns(); ++index) {
    std::cout << data[0][index].GetString() << " ";
  }
  std::cout << std::endl;
  DataTransformer trans(config_);
  auto res = trans.FitAndTransform(data);
  for (size_t index = 0; index < res.GetObjectsFeatures().size(); ++index) {
    std::cout << res.GetObjectsFeatures()[0][index] << " ";
  }
  std::cout << std::endl;
  auto train_objects = GetSample(res.GetObjectsFeatures().size(), 60);
  auto features = GetSample(res.GetFeaturesObjects().size(), res.GetFeaturesObjects().size());
  GradientBoostingMSELossFunction loss_function(res.GetFeaturesObjects(),
                                                res.GetObjectsFeatures(),
                                                res.GetTargetValues());
  GradientBoostingTreeOblivious abcd(loss_function, 3);
  std::cout << "Here we are" << std::endl;
  std::cout << res.GetFeaturesObjects().size() << std::endl;
  std::cout << res.GetObjectsFeatures().size() << std::endl;
  abcd.Fit(res.GetFeaturesObjects(), res.GetObjectsFeatures(), res.GetTargetValues(),
           train_objects, features, thread_pool_);
  std::cout << "Here we are[3]" << std::endl;
  //std::cout << res.GetObjectsFeatures().size() << std::endl;
  std::unordered_map<double, size_t> cnt;
  for (size_t index = 0; index < 1 /*res.GetObjectsFeatures().size()*/; ++index) {
    std::cout << abcd.Predict(res.GetObjectsFeatures()[index]) << " " << res.GetTargetValues()[index] << std::endl;
    cnt[abcd.Predict(res.GetObjectsFeatures()[index])]++;
  }
  for (auto el : cnt ) {
    std::cout << el.first << " " << el.second << std::endl;
  }

}

unique_ptr<GradientBoostingLossFunction> GetLossFunction(const GradientBoostingConfig& config,
                                                         const InternalDataContainer& data) {
  if (GradientBoostingConfig::LossFunction::MSE == config.GetLossFunction()) {
    return std::make_unique<GradientBoostingMSELossFunction>(data.GetFeaturesObjects(),
                                                             data.GetObjectsFeatures(),
                                                             data.GetTargetValues());

  }
  assert(false);
}

std::vector<double> Predict(const GradientBoostingTreeOblivious &tree,
                            const InternalDataContainer& data,
                            vector<size_t> objects) {
  std::vector<double> result;
  for (auto index : objects) {
    result.push_back(tree.Predict(data.GetObjectsFeatures()[index]));
  }
  return result;
}

double EvaluateTree(const GradientBoostingTreeOblivious &tree,
                    const GradientBoostingLossFunction& loss_function,
                    const InternalDataContainer& data,
                    vector<size_t> objects) {
  auto predicted_targets = Predict(tree, data, objects);
  double sum = 0.0;
  for (size_t index = 0; index < objects.size(); ++index){
    sum = loss_function.GetLoss(predicted_targets[index], data.GetTargetValues()[index]);
  }
  return sum;
}


std::pair<double, GradientBoostingTreeOblivious> GradientBoosting::GetScoreAndTree(
    const GradientBoostingLossFunction& loss_function,
    const InternalDataContainer& data) {

  size_t min_number_of_object = std::min(1000Ul, data.GetNumberOfObject());
  size_t sample_size = std::max(min_number_of_object,
                                size_t(sqrt(data.GetNumberOfObject())));

  std::vector<size_t> objects = GetSample(data.GetNumberOfObject(),
                                          sample_size);
  GradientBoostingTreeOblivious best_tree(loss_function, config_.GetHeight());
  best_tree.Fit(data.GetFeaturesObjects(),
                data.GetObjectsFeatures(),
                data.GetTargetValues(),
                objects,
                GetNumberedVector(data.GetNumberOfFeatures()),
                thread_pool_);
  std::vector<size_t> objects_to_evaluate = GetNumberedVector(data.GetNumberOfObject());
  double best_res = EvaluateTree(best_tree, loss_function, data, objects_to_evaluate);
  return {best_res, best_tree};
}

void GradientBoosting::Fit(const DataContainer& data) {
  data_transformer_ = std::make_unique<DataTransformer>(config_);
  InternalDataContainer transformed_data = data_transformer_->FitAndTransform(data);
  Fit(transformed_data);
}


void GradientBoosting::Fit(const InternalDataContainer & data) {
  std::vector<double> current_gradient = data.GetTargetValues();
  auto ptr_loss_function = GetLossFunction(config_, data);

  size_t number_of_trees_per_tree = std::max(10Ul, size_t(log(data.GetNumberOfObject() + 1.0)));

  for (size_t index_tree = 0; index_tree < number_of_trees_; ++index_tree) {
    auto best_score = GetScoreAndTree(*ptr_loss_function, data);
    for (size_t t = 1; t < number_of_trees_per_tree; ++t) {
      auto score = GetScoreAndTree(*ptr_loss_function, data);
      if (score.first < best_score.first) {
        best_score.first = score.first;
        best_score.second = score.second;
      }
    }

    forest_.push_back(best_score.second);
  }

}

unordered_map<string, double> GradientBoosting::PredictProba(
    const DataContainer& data) const {
  return {};
};
unordered_map<string, string> GradientBoosting::PredictClassName(
    const utils::data_containers::DataContainer& data) const {
  return {};
};

}  // namespace gradient_boosting

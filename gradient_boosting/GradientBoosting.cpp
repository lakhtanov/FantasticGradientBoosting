#include "gradient_boosting/GradientBoosting.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include "algorithm"
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

GradientBoosting::GradientBoosting(
    const GradientBoostingConfig& config)
    : config_(config)
    , thread_pool_(0)
{
}

vector<size_t> GetSample(size_t number, size_t sample_size) {
  vector<size_t> res(number, 0);
  for (size_t index = 0; index < number; ++index) {
    res[index] = index;
  }
  std::random_shuffle(res.begin(), res.end());
  res.resize(sample_size);
  return res;
}

InternalDataContainer GetSample(const InternalDataContainer& data, size_t sample_size) {
  auto indexes = GetSample(data.GetObjectsFeatures().size(), sample_size);
  vector<vector<size_t>> sample;
  vector<double> target_values;
  for (auto index : indexes) {
    sample.push_back(data.GetObjectsFeatures()[index]);
    target_values.push_back(data.GetTargetValues()[index]);
  }
  return (InternalDataContainer(sample, target_values, data.GetFeaturesNames()));
}

void GradientBoosting::TestGradientBoosting(
    const utils::data_containers::DataContainer& data) {
  for (size_t index = 0; index < data.columns(); ++index) {
    std::cout << data[0][index].GetString() << " ";
  }
  std::cout << std::endl;
  for (const auto& el : data.GetNames()) {
    std::cout << el << "\n";
  }
  gradient_boosting::data_transformer::DataTransformer trans(config_);
  auto res = trans.FitAndTransform(data);
  for (size_t index = 0; index < res.GetObjectsFeatures().size(); ++index) {
    std::cout << res.GetObjectsFeatures()[0][index] << " ";
  }
  std::cout << std::endl;
  for (const auto& el : res.GetFeaturesNames()) {
    std::cout << el << "\n";
  }
  auto sample = GetSample(res, data.rows() * 2 / 3);
  GradientBoostingMSELossFunction func(res.GetFeaturesObjects(),
                                       res.GetObjectsFeatures(),
                                       res.GetTargetValues());
  auto objects = GetSample(res.GetObjectsFeatures().size(), res.GetObjectsFeatures().size());
  auto features = GetSample(res.GetFeaturesObjects().size(), res.GetFeaturesObjects().size());
  gradient_boosting::trees::GradientBoostingTreeOblivious abcd(func, 3);
  std::cout << "Here we are" << std::endl;
  std::cout << res.GetFeaturesObjects().size() << std::endl;
  std::cout << res.GetObjectsFeatures().size() << std::endl;
  abcd.Fit(res.GetFeaturesObjects(), res.GetObjectsFeatures(), res.GetTargetValues(),
           objects, features, thread_pool_);
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
    /*return std::make_unique<GradientBoostingMSELossFunction>(data.GetFeaturesObjects(),
                                                             data.GetObjectsFeatures(),
                                                             data.GetTargetValues());*/
  }
  assert(false);
}

void GradientBoosting::Fit(
    const utils::data_containers::DataContainer& data) {
}
unordered_map<string, double> GradientBoosting::PredictProba(
    const utils::data_containers::DataContainer& data) const {
  return {};
};
unordered_map<string, string> GradientBoosting::PredictClassName(
    const utils::data_containers::DataContainer& data) const {
  return {};
};

}  // namespace gradient_boosting

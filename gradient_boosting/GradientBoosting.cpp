#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "gradient_boosting/GradientBoosting.h"
#include <gradient_boosting/data_transformer/DataTransformer.h>
#include <gradient_boosting/trees/GradientBoostingTreeOblivious.h>
#include <gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h>

namespace gradient_boosting {


using std::unordered_map;
using std::unique_ptr;
using std::string;
using std::vector;
using std::pair;

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
    , learning_rate_(config.GetLearningRate())
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
  std::cout << "GradientBoosting::TestGradientBoosting" << std::endl;
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
  std::cout << res.GetFeaturesObjects().size() << std::endl;
  std::cout << res.GetObjectsFeatures().size() << std::endl;
  abcd.Fit(res.GetFeaturesObjects(), res.GetObjectsFeatures(), res.GetTargetValues(),
           train_objects, features, thread_pool_);
  std::unordered_map<double, size_t> cnt;
  for (size_t index = 0; index < 1 /*res.GetObjectsFeatures().size()*/; ++index) {
    std::cout << "GradientBoosting::TestGradientBoosting " << abcd.Predict(res.GetObjectsFeatures()[index]) << " " << res.GetTargetValues()[index] << std::endl;
    cnt[abcd.Predict(res.GetObjectsFeatures()[index])]++;
  }
  std::cout << "GradientBoosting::TestGradientBoosting::predictions" << std::endl;
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
                            const vector<size_t>& objects) {
  // std::cout << "Predict" << std::endl;
  std::vector<double> result;
  result.reserve(objects.size());
  for (auto object : objects) {
    // std::cout << "Predict for object #" << object << std::endl;
    result.push_back(tree.Predict(data.GetObjectsFeatures()[object]));
  }
  return result;
}

double EvaluateTree(const GradientBoostingTreeOblivious &tree,
                    const GradientBoostingLossFunction& loss_function,
                    const InternalDataContainer& data,
                    const vector<size_t>& objects) {
  auto predicted_targets = Predict(tree, data, objects);
  double sum = 0.0;
  for (size_t index = 0; index < objects.size(); ++index){
    sum = loss_function.GetLoss(predicted_targets[index], data.GetTargetValues()[objects[index]]);
  }
  return sum;
}


pair<double, GradientBoostingTreeOblivious> GradientBoosting::GetScoreAndTree(
    const GradientBoostingLossFunction& loss_function,
    const InternalDataContainer& data,
    const vector<double>& gradient) {

  // size_t min_number_of_object = std::min(1000Ul, data.GetNumberOfObject());
  size_t min_number_of_object = data.GetNumberOfObject();
  // size_t sample_size = std::max(min_number_of_object,
  //                               size_t(sqrt(data.GetNumberOfObject())));
  size_t sample_size = data.GetNumberOfObject();

  std::vector<size_t> objects = GetSample(data.GetNumberOfObject(),
                                          sample_size);
  GradientBoostingTreeOblivious best_tree(loss_function, config_.GetHeight());
  best_tree.Fit(data.GetFeaturesObjects(),
                data.GetObjectsFeatures(),
                gradient,
                objects,
                GetNumberedVector(data.GetNumberOfFeatures()),
                thread_pool_);
  std::vector<size_t> objects_to_evaluate = GetNumberedVector(data.GetNumberOfObject());
  double best_res = EvaluateTree(best_tree, loss_function, data, objects_to_evaluate);
  return {best_res, best_tree};
}

void GradientBoosting::UpdateGradient(vector<double>* gradient,
                                      const GradientBoostingTreeOblivious &tree,
                                      const InternalDataContainer& data) const {
  auto prediction = Predict(tree, data, GetNumberedVector(data.GetNumberOfObject()));
  for(size_t index = 0; index < prediction.size(); ++index) {
    (*gradient)[index] -= learning_rate_ * prediction[index];
  }
}

void GradientBoosting::Fit(const DataContainer& data) {
  data_transformer_ = std::make_unique<DataTransformer>(config_);
  InternalDataContainer transformed_data = data_transformer_->FitAndTransform(data);
  Fit(transformed_data);
}


void GradientBoosting::Fit(const InternalDataContainer & data) {
  std::cout << "GradientBoosting::Fit" << std::endl;

  std::vector<double> gradient = data.GetTargetValues();
  auto ptr_loss_function = GetLossFunction(config_, data);

  // size_t number_of_trees_per_tree = std::max(10Ul, size_t(log(data.GetNumberOfObject() + 1.0)));
  size_t number_of_trees_per_tree = 1;

  for (size_t index_tree = 0; index_tree < number_of_trees_; ++index_tree) {
    std::cout << "GradientBoosting::Fit learning tree #" << index_tree << std::endl;
    vector<pair<double, GradientBoostingTreeOblivious>> trees_to_chose;
    size_t best_index = 0;
    for (size_t t = 0; t < number_of_trees_per_tree; ++t) {
      trees_to_chose.push_back(GetScoreAndTree(*ptr_loss_function, data, gradient));
      if (trees_to_chose[t].first < trees_to_chose[best_index].first) {
        best_index = t;
      }
    }
    UpdateGradient(&gradient, trees_to_chose[best_index].second, data);
    forest_.push_back(trees_to_chose[best_index].second);
  }
}

unordered_map<string, double> GradientBoosting::PredictProba(
    const DataContainer& data) const {
  auto transformed_data = data_transformer_->Transform(data);
  return PredictProba(transformed_data);
};


unordered_map<string, double> GradientBoosting::PredictProba(
    const InternalDataContainer& data) const {
  std::cout << "GradientBoosting::PredictProba" << std::endl;
  vector<double>accumulate(data.GetNumberOfObject());
  for(const auto& tree : forest_) {
    auto target_values = Predict(tree, data, GetNumberedVector(data.GetNumberOfObject()));
    for (size_t index = 0; index < accumulate.size(); ++index) {
      accumulate[index] += target_values[index];
    }
  }
  unordered_map<string, double> result;
  for (size_t index = 0; index < accumulate.size(); ++index) {
    result[data.GetIdNames()[index]] = accumulate[index] / forest_.size();
  }
  
  auto id_names = data.GetIdNames();
  for (auto el : result) {
    auto id = std::find(id_names.begin(), id_names.end(), el.first) - id_names.begin();
    double target = data.GetTargetValues()[id];
    std::cout << el.first << " " << el.second << " " << target << std::endl;
  }

  return result;
};

unordered_map<string, string> GradientBoosting::PredictClassName(
    const utils::data_containers::DataContainer& data) const {
  return {}; // ? do we need this?
};



}  // namespace gradient_boosting

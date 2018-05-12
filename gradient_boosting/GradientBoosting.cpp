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
using GradientBoostingTree = gradient_boosting::trees::GradientBoostingTree;

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

vector<size_t> GetSample(vector<size_t> all_indexes, size_t sample_size) {
  std::random_shuffle(all_indexes.begin(), all_indexes.end());
  all_indexes.resize(sample_size);
  return all_indexes;
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
  auto train_objects = GetNumberedVector(60);
  auto features = GetNumberedVector(30);
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

unique_ptr<GradientBoostingTree> GetTree(const GradientBoostingConfig& config,
                                                  const GradientBoostingLossFunction& loss_function) {
  // TODO(rialeksandrov) Get from config!
  if (GradientBoostingConfig::LossFunction::MSE == config.GetLossFunction()) {
    return std::make_unique<GradientBoostingTreeOblivious >(loss_function,
                                                            config.GetHeight());

  }
  assert(false);
}

std::vector<double> GradientBoosting::Predict(const GradientBoostingTree &tree,
                                              const InternalDataContainer& data,
                                              const vector<size_t>& objects) const {
  std::vector<double> result;
  result.reserve(objects.size());
  for (auto object : objects) {
    // std::cout << "Predict for object #" << object << std::endl;
    result.push_back(tree.Predict(data.GetObjectsFeatures()[object]));
  }
  return result;
}

double GradientBoosting::EvaluateTree(const GradientBoostingTree& tree,
                                      const GradientBoostingLossFunction& loss_function,
                                      const InternalDataContainer& data,
                                      const vector<size_t>& objects) const {
  auto predicted_targets = Predict(tree, data, objects);
  double sum = 0.0;
  for (size_t index = 0; index < objects.size(); ++index){
    sum = loss_function.GetLoss(predicted_targets[index], data.GetTargetValues()[objects[index]]);
  }
  return sum;
}


pair<double, unique_ptr<GradientBoostingTree>> GradientBoosting::GetScoreAndTree(
    const GradientBoostingLossFunction& loss_function,
    const InternalDataContainer& data,
    const vector<size_t>& all_object,
    const vector<size_t>& all_features,
    const vector<double>& gradient) {

  size_t min_number_of_object = std::min(10Ul, data.GetNumberOfObject());
  // size_t min_number_of_object = data.GetNumberOfObject();
  // size_t sample_size = std::max(min_number_of_object,
  //                               size_t(sqrt(data.GetNumberOfObject())));
  size_t sample_size = data.GetNumberOfObject();

  pair<double, unique_ptr<GradientBoostingTree>> result;
  result.second = GetTree(config_, loss_function);
  std::vector<size_t> objects = GetSample(all_object,
                                          sample_size);
  result.second->Fit(data.GetFeaturesObjects(),
                 data.GetObjectsFeatures(),
                 gradient,
                 objects,
                 all_features,
                 thread_pool_);
  result.first = EvaluateTree(*result.second, loss_function, data, all_object);
  return result;
}

void GradientBoosting::UpdateGradient(vector<double>* gradient,
                                      const GradientBoostingTree& tree,
                                      const InternalDataContainer& data,
                                      const vector<size_t>& objects) const {
  auto prediction = Predict(tree, data, objects);
  for(size_t index = 0; index < objects.size(); ++index) {
    (*gradient)[objects[index]] -= learning_rate_ * prediction[index];
  }
}

void GradientBoosting::Fit(const DataContainer& data) {
  data_transformer_ = std::make_unique<DataTransformer>(config_);
  InternalDataContainer transformed_data = data_transformer_->FitAndTransform(data);
  Fit(transformed_data);
}


void GradientBoosting::Fit(const InternalDataContainer & data) {
  std::cout << "GradientBoosting::Fit" << std::endl;

  const vector<size_t> all_objects = GetNumberedVector(data.GetNumberOfObject());
  const vector<size_t> all_features = GetNumberedVector(data.GetNumberOfFeatures());
  vector<double> gradient = data.GetTargetValues();
  auto ptr_loss_function = GetLossFunction(config_, data);

  // size_t number_of_trees_per_tree = std::max(10Ul, size_t(log(data.GetNumberOfObject() + 1.0)));
  size_t number_of_trees_per_tree = 1;

  for (size_t index_tree = 0; index_tree < number_of_trees_; ++index_tree) {
    std::cout << "GradientBoosting::Fit learning tree #" << index_tree << std::endl;
    vector<pair<double, unique_ptr<GradientBoostingTree>>> trees_to_chose;
    size_t best_index = 0;
    for (size_t t = 0; t < number_of_trees_per_tree; ++t) {
      trees_to_chose.emplace_back(GetScoreAndTree(*ptr_loss_function,
                                                  data,
                                                  all_objects,
                                                  all_features,
                                                  gradient));
      if (trees_to_chose[t].first < trees_to_chose[best_index].first) {
        best_index = t;
      }
    }
    UpdateGradient(&gradient, *trees_to_chose[best_index].second, data, all_objects);
    forest_.emplace_back(std::move(trees_to_chose[best_index].second));
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
  std::vector<size_t> all_objects = GetNumberedVector(data.GetNumberOfObject());
  for(const auto& tree : forest_) {
    auto target_values = Predict(*tree, data, all_objects);
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
  auto predicted_probs = PredictProba(data);
  unordered_map<string, string> result;
  assert(!data_transformer_->GetTargetNames().empty());
  for (const auto& id_proba : predicted_probs) {
    size_t best_id = data_transformer_->GetTargetNames().begin()->first;
    for (const auto& el : data_transformer_->GetTargetNames()) {
      if (abs(best_id - id_proba.second) > abs(el.first - id_proba.second)) {
        best_id = el.first;
      }
    }
    result[id_proba.first] = data_transformer_->GetTargetNames().at(best_id);
  }
  return result;
};



}  // namespace gradient_boosting

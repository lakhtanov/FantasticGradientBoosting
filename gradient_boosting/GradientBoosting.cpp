#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gradient_boosting/data_transformer/DataTransformer.h"
#include "gradient_boosting/GradientBoosting.h"
#include "gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h"
#include "gradient_boosting/trees/GradientBoostingTreeOblivious.h"

namespace gradient_boosting {

using std::make_unique;
using std::pair;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

using DataContainer = utils::data_containers::DataContainer;
using DataTransformer = gradient_boosting::data_transformer::DataTransformer;
using GradientBoostingConfig =
    gradient_boosting::config::GradientBoostingConfig;
using InternalDataContainer =
    gradient_boosting::internal_data_container::InternalDataContainer;
using GradientBoostingLossFunction =
    gradient_boosting::loss_functions::GradientBoostingLossFunction;
using GradientBoostingMSELossFunction =
    gradient_boosting::loss_functions::GradientBoostingMSELossFunction;
using GradientBoostingTree = gradient_boosting::trees::GradientBoostingTree;
using GradientBoostingTreeOblivious =
    gradient_boosting::trees::GradientBoostingTreeOblivious;

GradientBoosting::GradientBoosting(
    const GradientBoostingConfig& config)
    : learning_rate_(config.GetLearningRate())
    , number_of_trees_(config.GetNumberOfTrees())
    , config_(config)
    , thread_pool_(config.GetNumberOfThreads())
    , fit_time_(0.0)
    , update_gradient_time_(0)
    , evaluate_time_(0)
    , build_tree_time_(0)
    , clear_build_tree_time_(0){
}

vector<size_t> GetNumberedVector(size_t size) {
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
  DataTransformer trans(config_);
  const auto res = trans.FitAndTransform(data);
  const auto train_objects = GetNumberedVector(60);
  const auto features = GetNumberedVector(30);
  GradientBoostingMSELossFunction loss_function(
      res.GetFeaturesObjects(),
      res.GetObjectsFeatures(),
      res.GetTargetValues());
  GradientBoostingTreeOblivious abcd(loss_function, 3);
  abcd.Fit(
      res.GetFeaturesObjects(),
      res.GetObjectsFeatures(),
      res.GetTargetValues(),
      train_objects,
      features,
      thread_pool_);
  std::unordered_map<double, size_t> cnt;
  for (size_t index = 0;
       index < 1 /*res.GetObjectsFeatures().size()*/;
       ++index) {
    ++cnt[abcd.Predict(res.GetObjectsFeatures()[index])];
  }
  for (auto el : cnt) {
    std::cout << el.first << " " << el.second << std::endl;
  }
}

unique_ptr<GradientBoostingLossFunction> GetLossFunction(
    const GradientBoostingConfig& config, const InternalDataContainer& data) {
  if (GradientBoostingConfig::LossFunction::MSE == config.GetLossFunction()) {
    return make_unique<GradientBoostingMSELossFunction>(
        data.GetFeaturesObjects(),
        data.GetObjectsFeatures(),
        data.GetTargetValues());
  } else {
    assert(false);
  }
}

unique_ptr<GradientBoostingTree> GetTree(
    const GradientBoostingConfig& config,
    const GradientBoostingLossFunction& loss_function) {
  // TODO(rialeksandrov) Get from config!
  if (GradientBoostingConfig::TreeType::ObliviousTree == config.GetTreeType()) {
    return make_unique<GradientBoostingTreeOblivious>(
        loss_function, config.GetHeight());
  } else {
    assert(false);
  }
}

vector<double> GradientBoosting::Predict(
    const GradientBoostingTree &tree,
    const InternalDataContainer& data,
    const vector<size_t>& objects) const {
  std::vector<double> result;
  result.reserve(objects.size());
  for (auto object : objects) {
    result.push_back(tree.Predict(data.GetObjectsFeatures()[object]));
  }
  return result;
}

double GradientBoosting::EvaluateTree(
    const GradientBoostingTree& tree,
    const GradientBoostingLossFunction& loss_function,
    const InternalDataContainer& data,
    const vector<size_t>& objects) {
  clock_t begin = clock();
  const auto predicted_targets = Predict(tree, data, objects);
  double sum = 0.0;
  for (size_t index = 0; index < objects.size(); ++index) {
    sum =
        loss_function.GetLoss(
            predicted_targets[index], data.GetTargetValues()[objects[index]]);
  }
  clock_t end = clock();
  evaluate_time_ += double(end - begin) / CLOCKS_PER_SEC;
  return sum;
}


pair<double, unique_ptr<GradientBoostingTree>>
GradientBoosting::GetScoreAndTree(
    const GradientBoostingLossFunction& loss_function,
    const InternalDataContainer& data,
    const vector<size_t>& all_object,
    const vector<size_t>& all_features,
    const vector<double>& gradient) {
  clock_t begin = clock();
  const size_t num_sample_object =
      std::min(
          size_t{200},
          data.GetNumberOfObject());
  const size_t num_sample_features =
      std::min(
          size_t{15},
          data.GetNumberOfFeatures());
  pair<double, unique_ptr<GradientBoostingTree>> result;
  result.second = GetTree(config_, loss_function);
  const vector<size_t> objects = GetSample(all_object, num_sample_object);
  const vector<size_t> features = GetSample(all_features, num_sample_features);
  clock_t begin_clear = clock();
  result.second->Fit(
      data.GetFeaturesObjects(),
      data.GetObjectsFeatures(),
      gradient,
      objects,
      features,
      thread_pool_);
  clear_build_tree_time_ +=  double(clock() - begin_clear) / CLOCKS_PER_SEC;
  result.first = EvaluateTree(*result.second, loss_function, data, all_object);
  clock_t end = clock();
  build_tree_time_ += double(end - begin) / CLOCKS_PER_SEC;
  return result;
}

void GradientBoosting::UpdateGradient(
    vector<double>* gradient,
    const GradientBoostingTree& tree,
    const InternalDataContainer& data,
    const vector<size_t>& objects) {
  clock_t begin = clock();
  const auto prediction = Predict(tree, data, objects);
  for (size_t index = 0; index < objects.size(); ++index) {
    (*gradient)[objects[index]] -= learning_rate_ * prediction[index];
  }
  clock_t end = clock();
  update_gradient_time_ += double(end - begin) / CLOCKS_PER_SEC;
}

void GradientBoosting::Fit(const DataContainer& data) {
  data_transformer_ = make_unique<DataTransformer>(config_);
  InternalDataContainer transformed_data =
      data_transformer_->FitAndTransform(data);
  Fit(transformed_data);
}


void GradientBoosting::Fit(const InternalDataContainer & data) {
  clock_t begin = clock();
  const vector<size_t> all_objects =
      GetNumberedVector(data.GetNumberOfObject());
  const vector<size_t> all_features =
      GetNumberedVector(data.GetNumberOfFeatures());
  vector<double> gradient = data.GetTargetValues();
  auto ptr_loss_function = GetLossFunction(config_, data);

  // TODO(lakhtanov): change number of trees per tree to something else.
  size_t number_of_trees_per_tree = 1;

  for (size_t index_tree = 0; index_tree < number_of_trees_; ++index_tree) {
    vector<pair<double, unique_ptr<GradientBoostingTree>>> trees_to_chose;
    size_t best_index = 0;
    for (size_t t = 0; t < number_of_trees_per_tree; ++t) {
      trees_to_chose.emplace_back(
          GetScoreAndTree(
              *ptr_loss_function,
              data,
              all_objects,
              all_features,
              gradient));
      if (trees_to_chose[t].first < trees_to_chose[best_index].first) {
        best_index = t;
      }
    }
    UpdateGradient(
        &gradient, *trees_to_chose[best_index].second, data, all_objects);
    forest_.emplace_back(std::move(trees_to_chose[best_index].second));
  }
  clock_t end = clock();
  fit_time_ += double(end - begin) / CLOCKS_PER_SEC;
  std::cout.precision(3);
  std::cout << std::fixed << fit_time_ << " " << update_gradient_time_ << " " << build_tree_time_
            << " " << evaluate_time_ << " " << clear_build_tree_time_ << std::endl;
}

unordered_map<string, double> GradientBoosting::PredictProba(
    const DataContainer& data) const {
  const auto transformed_data = data_transformer_->Transform(data);
  return PredictProba(transformed_data);
}

unordered_map<string, double> GradientBoosting::PredictProba(
    const InternalDataContainer& data) const {
  vector<double>accumulate(data.GetNumberOfObject());
  const vector<size_t> all_objects =
      GetNumberedVector(data.GetNumberOfObject());
  for (const auto& tree : forest_) {
    const auto target_values = Predict(*tree, data, all_objects);
    for (size_t index = 0; index < accumulate.size(); ++index) {
      accumulate[index] += target_values[index];
    }
  }
  unordered_map<string, double> result;
  for (size_t index = 0; index < accumulate.size(); ++index) {
    result[data.GetIdNames()[index]] = accumulate[index] / forest_.size();
  }

  return result;
}

unordered_map<string, string> GradientBoosting::PredictClassName(
    const DataContainer& data) const {
  const auto predicted_probs = PredictProba(data);
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
}

}  // namespace gradient_boosting

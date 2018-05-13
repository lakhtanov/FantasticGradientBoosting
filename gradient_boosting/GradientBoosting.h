#ifndef GRADIENT_BOOSTING_GRADIENTBOOSTING_H_
#define GRADIENT_BOOSTING_GRADIENTBOOSTING_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gradient_boosting/config/GradientBoostingConfig.h"
#include "gradient_boosting/data_transformer/DataTransformer.h"
#include "gradient_boosting/internal_data_container/InternalDataContainer.h"
#include "gradient_boosting/trees/GradientBoostingTreeOblivious.h"
#include "utils/data_containers/DataContainer.h"

#include "third_party/ctpl/ctpl_stl.h"

namespace gradient_boosting {

class GradientBoosting {
 public:
  explicit GradientBoosting(
      const gradient_boosting::config::GradientBoostingConfig& config);
  void TestGradientBoosting(const utils::data_containers::DataContainer& data);
  void Fit(const utils::data_containers::DataContainer& data);
  std::unordered_map<std::string, double> PredictProba(
      const utils::data_containers::DataContainer& data);
  std::unordered_map<std::string, std::string> PredictClassName(
      const utils::data_containers::DataContainer& data);

 private:
  void Fit(
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data);
  std::unordered_map<std::string, double> PredictProba(
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data);

  std::vector<double> Predict(
      const gradient_boosting::trees::GradientBoostingTree& tree,
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data,
      const std::vector<size_t>& objects);

  double EvaluateLoss(
      const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function,
      const std::vector<double>& predicted_targets,
      const std::vector<double>& targets,
      const std::vector<size_t>& objects);

  double EvaluateTree(
      const gradient_boosting::trees::GradientBoostingTree& tree,
      const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function,
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data,
      const std::vector<double>& gradient,
      const std::vector<size_t>& objects);

  std::pair<
      double,
      std::unique_ptr<gradient_boosting::trees::GradientBoostingTree>
  > GetScoreAndTree(
      const gradient_boosting::loss_functions::GradientBoostingLossFunction&
      loss_function,
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data,
      const std::vector<size_t>& all_object,
      const std::vector<size_t>& all_features,
      const std::vector<double>& gradient);

  void UpdateGradient(
      std::vector<double>* gradient,
      const gradient_boosting::trees::GradientBoostingTree& tree,
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data,
      const std::vector<size_t>& objects);

  void UpdatePrediction(
      std::vector<double>* gradient,
      const gradient_boosting::trees::GradientBoostingTree& tree,
      const gradient_boosting::internal_data_container::InternalDataContainer&
      data,
      const std::vector<size_t>& objects);

  double learning_rate_;
  size_t number_of_trees_;
  gradient_boosting::config::GradientBoostingConfig config_;
  ctpl::thread_pool thread_pool_;
  std::vector<std::unique_ptr<gradient_boosting::trees::GradientBoostingTree>>
      forest_;
  std::unique_ptr<gradient_boosting::data_transformer::DataTransformer>
      data_transformer_;
  double fit_time_;
  double update_gradient_time_;
  double update_prediction_time_;
  double evaluate_time_;
  double build_tree_time_;
  double clear_build_tree_time_;
};

/*
 * class SimpleSampler {
  SimpleSampler(size_t size);
  vector<size_t> GetSample(size_t sample_size) {
    vector<size_t> res = GetNumberedVector(number);
    std::random_shuffle(res.begin(), res.end());
    res.resize(sample_size);
    return res;
  }

 public:
  std::vector<int>

};*/

}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_GRADIENTBOOSTING_H_

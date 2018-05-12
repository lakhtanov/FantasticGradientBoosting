#ifndef GRADIENT_BOOSTING_GRADIENTBOOSTING_H_
#define GRADIENT_BOOSTING_GRADIENTBOOSTING_H_

#include <unordered_map>
#include <string>
#include <memory>

#include <gradient_boosting/config/GradientBoostingConfig.h>
#include <gradient_boosting/trees/GradientBoostingTreeOblivious.h>
#include <utils/data_containers/DataContainer.h>
#include <gradient_boosting/internal_data_container/InternalDataContainer.h>
#include <gradient_boosting/data_transformer/DataTransformer.h>

#include "third_party/ctpl/ctpl_stl.h"

namespace gradient_boosting {

// TODO(rialeksandrov,lakhtanov): class for learning gradient boosting ml model.
class GradientBoosting {
 public:
  explicit GradientBoosting(const gradient_boosting::config::GradientBoostingConfig& config);
  void TestGradientBoosting(const utils::data_containers::DataContainer& data);
  void Fit(const utils::data_containers::DataContainer& data);
  std::unordered_map<std::string, double> PredictProba(const utils::data_containers::DataContainer& data) const;
  std::unordered_map<std::string, std::string> PredictClassName(const utils::data_containers::DataContainer& data) const;
 private:
  void Fit(const gradient_boosting::internal_data_container::InternalDataContainer& data);
  std::unordered_map<std::string, double> PredictProba(
      const gradient_boosting::internal_data_container::InternalDataContainer& data) const;


  std::vector<double> Predict(const gradient_boosting::trees::GradientBoostingTree& tree,
                              const gradient_boosting::internal_data_container::InternalDataContainer& data,
                              const std::vector<size_t>& objects) const;

  double EvaluateTree(const gradient_boosting::trees::GradientBoostingTree& tree,
                      const gradient_boosting::loss_functions::GradientBoostingLossFunction& loss_function,
                      const gradient_boosting::internal_data_container::InternalDataContainer& data,
                      const std::vector<size_t>& objects) const;
  std::pair<double,
      std::unique_ptr<gradient_boosting::trees::GradientBoostingTree>> GetScoreAndTree(
      const gradient_boosting::loss_functions::GradientBoostingLossFunction& loss_function,
      const gradient_boosting::internal_data_container::InternalDataContainer& data,
      const std::vector<size_t>& all_object,
      const std::vector<size_t>& all_features,
      const std::vector<double>& gradient);
  void UpdateGradient(std::vector<double>* gradient,
                      const gradient_boosting::trees::GradientBoostingTree& tree,
                      const gradient_boosting::internal_data_container::InternalDataContainer& data,
                      const std::vector<size_t>& objects) const;

  double learning_rate_;
  size_t number_of_trees_;
  gradient_boosting::config::GradientBoostingConfig config_;
  ctpl::thread_pool thread_pool_;
  std::vector<std::unique_ptr<gradient_boosting::trees::GradientBoostingTree>> forest_;
  std::unique_ptr<gradient_boosting::data_transformer::DataTransformer > data_transformer_;
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

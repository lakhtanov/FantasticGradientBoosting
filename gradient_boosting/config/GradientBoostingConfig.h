#ifndef GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_
#define GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_

#include <string>
#include <unordered_map>

#include "third_party/json/single_include/nlohmann/json.hpp"

namespace gradient_boosting {
namespace config {

class GradientBoostingConfig {
 public:
  explicit GradientBoostingConfig(const nlohmann::json& config);
  enum class Verbose {v1, v2, v3};
  enum class LossFunction {MSE};
  enum class TaskType {Classification, Regression};

  LossFunction GetLossFunction() const;
  size_t GetNumberOfValueThresholds() const;
  size_t GetNumberOfStatisticsThresholds() const;
  std::string GetTargetValueName() const;
  TaskType GetTaskType() const;
  std::string GetTrainData() const;
  Verbose GetVerbose() const;
  size_t GetNumberOfThreads() const;
  size_t GetNumberOfTrees() const;
  size_t GetHeight() const;
  std::string GetIdValueName () const;
  double GetLearningRate() const;
 private:
  std::unordered_map<std::string, Verbose> GetVerboseMapping() const;
  std::unordered_map<std::string, LossFunction> GetLossFunctionMapping() const;
  std::unordered_map<std::string, TaskType> GetTaskTypeMapping() const;

  LossFunction GetLossFunction(const nlohmann::json& config) const;
  Verbose GetVerbose(const nlohmann::json& config) const;
  size_t GetNumberOfValueThresholds(const nlohmann::json& config) const;
  size_t GetNumberOfStatisticsThresholds(const nlohmann::json& config) const;
  std::string GetTargetValueName(const nlohmann::json& config) const;
  TaskType GetTaskType(const nlohmann::json& config) const;
  std::string GetTrainData(const nlohmann::json& config) const;

  // These maps should be higher than fields inited by them!
  // (C++ motherfucker, do you speak it?)
  const std::unordered_map<std::string, Verbose> to_verbose_;
  const std::unordered_map<std::string, LossFunction> to_loss_function_;
  const std::unordered_map<std::string, TaskType> to_task_type_;

  const LossFunction loss_function_;
  const size_t statistics_thresholds_;
  const std::string target_value_name_;
  const TaskType task_type_;
  const std::string train_data_;
  const size_t value_thresholds_;
  const Verbose verbose_;
};

}  // namespace config
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_
